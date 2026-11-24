# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "xgboost>=3.4,<3.5", "pyyaml"]
# ///
"""The DG-2 identification ladder (SPEC v2 §4): rungs R0-R8 on the validation split only.

  uv run --python 3.12 bench/dg2/ladder.py --rung R<k> [--candidate <id>] [--facts f.json]
  uv run --python 3.12 bench/dg2/ladder.py --status
  uv run --python 3.12 bench/dg2/ladder.py --smoke --ledger <scratch>.json --rung R<k>

Each candidate is the incumbent recipe with one change (§4.4). A fit trains XGBoost (Python 3.4.x,
D-26) on fit rows and scores validation rows (§3.3); a feature pass runs `blz mine` (R4 cpu, R5
oracle) or tools/gfp/dg2_features.py (R6, R7) under the limits of §4.4. Every fit and pass is
appended to results/dg2_ledger.json through ledger.add_entry(), which enforces the budget (§4.2);
when every candidate of a rung is run or skipped, ledger.decide() applies the selection rule
(§4.3). Test rows are never read: labels and features are sliced to the fit and validation
ranges only.

--smoke never touches the real ledger or the validation rows: it trains on eids [0, 1,000,000) and
scores [2,000,000, 3,000,000), both inside the fit split, and replaces every feature pass by a
stand-in built from the v1_default features (so it tests the plumbing, not the passes).
"""
import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xgboost as xgb
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ledger as L  # noqa: E402
from blzg import Blzg  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
GRAPH = "data/hi_small.blzg"
BLZ = "build/rel/blz"
GFP_PY = "tools/gfp/.venv/bin/python"
GT_JOIN = "results/gt_join.json"
XGB_CONFIG = "configs/xgb_default.yaml"
# R0's features: v1_default on the CPU backend, all rows (P7, AC-V2-0).
V1_FEATURES = "features/p7_cpu_default"
V1_PROFILE = "configs/profiles/v1_default.yaml"
V1_PROFILE_ID = "b21580d6e129821a"
V1_FNV1A = {"fan_in": 10768036772042710147, "fan_out": 904878896487262018, "deg_in": 3321375241280117763,
            "deg_out": 3341707662053112211, "sg": 18124683238158106329, "tcyc": 7737672036140136160}
BLOCK_ORDER = ["fan_in", "fan_out", "deg_in", "deg_out", "tcyc", "sg"]  # SPEC §10.1: the Table 2 order
# R4's profiles (D-25) and their candidates; R5 writes its own from the incumbent's.
PROFILES = {"W-B": "configs/profiles/dg2_wb.yaml", "W-C": "configs/profiles/dg2_wc.yaml"}
R5_PROFILE = "configs/profiles/dg2_v1.yaml"
WC_GROUPS = {("fan_in", "fan_out", "deg_in", "deg_out"): ["FAN-IN", "FAN-OUT"], ("sg",): ["SCATTER-GATHER"],
             ("tcyc",): ["CYCLE"]}
WC_CAP = 1_728_000
PASS_WALL_S, PASS_RSS_MB = 60 * 60, 3 * 1024  # §4.4: every feature pass
R4_PREFLIGHT_S = 45 * 60  # §4.4: R4's 1% preflight (seed 0), extrapolated to 12 threads
MGNN_TS_OFFSET = 10  # F9: seconds since the first row's midnight (2022/09/01) minus 10 s
SMOKE_SPLITS = {"fit": (0, 1_000_000), "val": (2_000_000, 3_000_000)}
RAW = {"S1": ["Timestamp", "Amount Sent", "Sent Currency", "Amount Received", "Received Currency", "Payment Format"]}
RAW["S2"] = RAW["S1"][1:]
IDS = {"v1": ["src_id", "dst_id"], "I1": ["from_id", "to_id"], "I2": ["from_id", "to_id", "EdgeID"],
       "I3": ["src_id", "dst_id", "eid"]}


def now_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_info():
    run = lambda *a: subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True).stdout.strip()
    return {"sha": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def machine():
    cpu = next((ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name")), "?")
    mem = next(int(ln.split()[1]) for ln in open("/proc/meminfo") if ln.startswith("MemTotal"))
    return f"local ({platform.node()}, {cpu}, {os.cpu_count()} threads, {mem // 1024} MB)"


# ---------------------------------------------------------------------------------------------
# Process accounting: peak RSS, the foreign-load log (SPEC v2 §2.5), monitored passes (§4.4).

def _proc_table():
    """pid -> (ppid, comm, utime + stime in ticks) for every process."""
    out = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                s = f.read()
        except OSError:
            continue
        r = s.rfind(")")
        comm = s[s.find("(") + 1:r]
        f = s[r + 2:].split()
        out[int(d)] = (int(f[1]), comm, int(f[11]) + int(f[12]))
    return out


def _tree(table, root):
    kids = {}
    for pid, (ppid, _, _) in table.items():
        kids.setdefault(ppid, []).append(pid)
    seen, stack = set(), [root]
    while stack:
        p = stack.pop()
        if p not in seen:
            seen.add(p)
            stack += kids.get(p, [])
    return seen


def hwm_kb(pid):
    """A process's VmHWM in kB (0 when it is gone). Unlike a child's ru_maxrss, which keeps the
    RSS that the forking (ladder) process had before exec, VmHWM starts afresh at exec."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for ln in f:
                if ln.startswith("VmHWM:"):
                    return int(ln.split()[1])
    except OSError:
        pass
    return 0


class ForeignLoad:
    """Every 10 s, the processes outside this ladder's process tree that used >= 5% of a core."""

    def __init__(self, period=10.0):
        self.period, self.samples, self.t0 = period, [], time.time()
        self.stop = threading.Event()
        self.th = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self.th.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.th.join()

    def _run(self):
        hz = os.sysconf("SC_CLK_TCK")
        prev, tprev = _proc_table(), time.time()
        while not self.stop.wait(self.period):
            cur, t = _proc_table(), time.time()
            mine = _tree(cur, os.getpid())
            top = []
            for pid, (_, comm, ticks) in cur.items():
                if pid in mine or pid not in prev or prev[pid][1] != comm:
                    continue
                pct = 100.0 * (ticks - prev[pid][2]) / hz / (t - tprev)
                if pct >= 5:
                    top.append([comm, pid, round(pct)])
            top.sort(key=lambda x: -x[2])
            self.samples.append({"t": round(t - self.t0), "top": top[:5]})
            prev, tprev = cur, t


def reset_peak_rss():
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")  # resets VmHWM (Linux >= 4.0)
    except OSError:
        pass


def own_peak_rss_mb():
    for ln in open("/proc/self/status"):
        if ln.startswith("VmHWM:"):
            return int(ln.split()[1]) / 1024
    return float("nan")


def run_monitored(cmd, log_path, wall_s, rss_mb_limit, early_abort=None, poll=0.2):
    """Runs `cmd`; kills its process tree past `wall_s` seconds or `rss_mb_limit` MB of peak RSS,
    or when early_abort(elapsed, log_text) returns a reason. Returns (returncode, elapsed, peak
    RSS MB, abort reason or None). Peak RSS: the sum over the tree's processes of the largest
    VmHWM seen for each, sampled every `poll` seconds (growth in a process's last `poll` seconds
    is missed)."""
    t0 = time.time()
    with open(log_path, "w") as log:
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, start_new_session=True)
    hwm, peak, reason, status = {}, 0.0, None, None
    while True:
        pid, status, _ = os.wait4(p.pid, os.WNOHANG)
        if pid == p.pid:
            break
        el = time.time() - t0
        for q in _tree(_proc_table(), p.pid):
            hwm[q] = max(hwm.get(q, 0), hwm_kb(q))
        peak = sum(hwm.values()) / 1024
        if el > wall_s:
            reason = f"wall time > {wall_s:.0f} s"
        elif peak > rss_mb_limit:
            reason = f"peak RSS {peak:.0f} MB > {rss_mb_limit} MB"
        elif early_abort is not None:
            reason = early_abort(el, Path(log_path).read_text(errors="replace"))
        if reason:
            os.killpg(p.pid, signal.SIGKILL)
            _, status, _ = os.wait4(p.pid, 0)
            break
        time.sleep(poll)
    p.returncode = os.waitstatus_to_exitcode(status)
    return p.returncode, time.time() - t0, peak, reason


# ---------------------------------------------------------------------------------------------
# Data: the graph, the splits, the encodings of R1 and R3, and feature directories.

class Data:
    def __init__(self, smoke):
        self.g = g = Blzg(str(ROOT / GRAPH))
        self.blzg_sha256 = file_sha256(ROOT / GRAPH)
        sp = g.splits()
        if sp["fit"] != (0, 3_047_216) or sp["val"] != (3_047_216, 4_062_676):  # SPEC v2 §3.3
            raise SystemExit(f"unexpected splits {sp}")
        self.splits = SMOKE_SPLITS if smoke else {"fit": sp["fit"], "val": sp["val"]}
        self._mgnn = None

    def labels(self, part):
        a, b = self.splits[part]
        return np.asarray(self.g.label[a:b], dtype=np.float32)

    def mgnn(self):
        """F9: Multi-GNN's codes. Accounts: key = bank string + account string, one dictionary, by
        first appearance in file order, the source before the destination within a row.
        Currencies: one dictionary for both columns, Receiving Currency before Payment Currency
        within a row. Payment Format: its own dictionary. All in file order (file_row)."""
        if self._mgnn is None:
            g = self.g
            keys = [b + a for b, a in zip(g.strings("node_bank"), g.strings("node_account"))]
            if len(set(keys)) != g.n:  # P7: the concatenated key gives D-2's 515,088 nodes
                raise SystemExit("bank + account keys are not one-to-one with the nodes")
            order = np.argsort(g.file_row, kind="stable")
            if not np.array_equal(g.file_row[order], np.arange(g.m, dtype=np.uint32)):
                raise SystemExit("file_row is not a permutation")

            def first_appearance(seq, size):
                uniq, first = np.unique(seq, return_index=True)
                code = np.full(size, np.iinfo(np.uint32).max, dtype=np.uint32)
                code[uniq[np.argsort(first)]] = np.arange(len(uniq), dtype=np.uint32)
                return code

            acc = np.empty(2 * g.m, dtype=np.uint32)
            acc[0::2], acc[1::2] = g.src[order], g.dst[order]
            ccy = np.empty(2 * g.m, dtype=np.uint8)
            ccy[0::2], ccy[1::2] = g.recv_ccy[order], g.pay_ccy[order]
            self._mgnn = {"acc": first_appearance(acc, g.n), "ccy": first_appearance(ccy, 256),
                          "fmt": first_appearance(g.format[order], 256)}
            if (self._mgnn["acc"] == np.iinfo(np.uint32).max).any():
                raise SystemExit("a node never appears")
            # Multi-GNN writes amounts with '%f' (6 decimals) and reads them back.
            for name, col in (("paid", g.amount_paid), ("received", g.amount_received)):
                self._mgnn[name] = np.array([float("%f" % v) for v in col.tolist()])
            self._mgnn["amounts_changed_by_%f"] = int((self._mgnn["paid"] != g.amount_paid).sum()
                                                      + (self._mgnn["received"] != g.amount_received).sum())
        return self._mgnn


class Features:
    """A feature directory written by `blz mine` (manifest_version 2) or tools/gfp/dg2_features.py."""

    def __init__(self, data, d):
        self.dir = ROOT / d
        self.meta = json.loads((self.dir / "features.json").read_text())
        m = self.meta
        if m.get("producer") == "tools/gfp/dg2_features.py":
            self.source, self.first, self.last = "gfp", m["rows"]["first"], m["rows"]["last"]
            self.gfp_recipe = m["gfp_recipe"]["id"]
            self.profile_id = None
        else:
            t, edges = m["triggers"], m.get("edges", data.g.m)  # edges < m: blz mine --edges (R5)
            if t.get("sampled") or t["first"] != 0 or t["last"] != edges:
                raise SystemExit(f"{d}: not a mining of every row of its graph ({t}, edges {edges})")
            self.source, self.first, self.last = "profile", 0, edges
            self.profile_id, self.gfp_recipe = m["profile"]["id"], None
        if m["blzg_sha256"] != data.blzg_sha256 and m.get("smoke_stand_in") is None:
            raise SystemExit(f"{d}: mined from another .blzg")
        self.blocks, self._cache = {}, {}
        for name, b in m["blocks"].items():
            arr = np.load(self.dir / b["file"], mmap_mode="r")  # header only; rows are read with pread
            if arr.shape[0] != self.last - self.first or arr.shape[1] != len(b["columns"]) or arr.ndim != 2 \
                    or not arr.flags.c_contiguous:
                raise SystemExit(f"{d}/{b['file']}: shape {arr.shape} does not match the manifest")
            self.blocks[name] = (self.dir / b["file"], arr.offset, arr.dtype, arr.shape[1], b["columns"])
            del arr

    def rows(self, name, a, b):
        """Rows for eids [a, b) of block `name`, read with pread (not mapped, so they do not count
        towards this process's RSS); the last range read is cached per block."""
        hit = self._cache.get(name)
        if hit and hit[0] == (a, b):
            return hit[1]
        path, off, dt, nc, _ = self.blocks[name]
        lo = a - self.first
        n = (b - a) * nc * dt.itemsize
        fd = os.open(path, os.O_RDONLY)
        try:
            buf = bytearray(n)
            view, got = memoryview(buf), 0
            while got < n:
                k = os.preadv(fd, [view[got:]], off + lo * nc * dt.itemsize + got)
                if k <= 0:
                    raise SystemExit(f"{path}: short read")
                got += k
        finally:
            os.close(fd)
        out = np.frombuffer(buf, dtype=dt).reshape(b - a, nc)
        self._cache[name] = ((a, b), out)
        return out

    def columns(self, name):
        cols = self.blocks[name][4]
        return [(c, (lambda a, b, j=j: self.rows(name, a, b)[:, j])) for j, c in enumerate(cols)]

    def covers(self, a, b):
        return self.first <= a and b <= self.last


def recipe_columns(data, recipe, set_name):
    """The (name, getter(a, b)) columns of `recipe` for set xgb_only or all (§4.4, §4.5 step 3):
    ids, raw columns and vertex statistics form the base; `all` adds the pattern blocks in
    v1's order (§10.1), then lcyc (R7 F2)."""
    g = data.g
    ids = recipe["ids"]
    if ids in ("v1", "I3"):
        cols = [("src_id", lambda a, b: g.src[a:b]), ("dst_id", lambda a, b: g.dst[a:b])]
    else:
        acc = data.mgnn()["acc"]
        cols = [("from_id", lambda a, b: acc[g.src[a:b]]), ("to_id", lambda a, b: acc[g.dst[a:b]])]
    if ids == "I2":
        cols.append(("EdgeID", lambda a, b: g.file_row[a:b]))
    if ids == "I3":
        cols.append(("eid", lambda a, b: np.arange(a, b)))
    if recipe["raw"]:
        mg = data.mgnn()
        raw = {"Timestamp": lambda a, b: g.ts[a:b].astype(np.int64) + MGNN_TS_OFFSET,
               "Amount Sent": lambda a, b: mg["paid"][a:b],
               "Sent Currency": lambda a, b: mg["ccy"][g.pay_ccy[a:b]],
               "Amount Received": lambda a, b: mg["received"][a:b],
               "Received Currency": lambda a, b: mg["ccy"][g.recv_ccy[a:b]],
               "Payment Format": lambda a, b: mg["fmt"][g.format[a:b]]}
        cols += [(n, raw[n]) for n in RAW[recipe["raw"]]]
    fam = recipe.get("family")
    if fam and fam["id"] == "F1":
        cols += Features(data, fam["dir"]).columns("vstats")
    if set_name == "all":
        f = Features(data, recipe["features"]["dir"])
        for b in BLOCK_ORDER:
            cols += f.columns(b)
        if fam and fam["id"] == "F2":
            cols += Features(data, fam["dir"]).columns("lcyc")
    names = [c for c, _ in cols]
    if len(set(names)) != len(names):
        raise SystemExit(f"duplicate column names in {names}")
    return cols


# ---------------------------------------------------------------------------------------------
# A fit: XGBoost on fit rows, scored on validation rows.

class Batches(xgb.DataIter):
    def __init__(self, cols, a, b, label, rows):
        self.cols, self.a, self.b, self.label, self.rows, self.it = cols, a, b, label, rows, a
        super().__init__()

    def next(self, input_data):
        if self.it >= self.b:
            return False
        j = min(self.it + self.rows, self.b)
        X = np.empty((j - self.it, len(self.cols)), dtype=np.float32)
        for k, (_, get) in enumerate(self.cols):
            X[:, k] = get(self.it, j)
        input_data(data=X, label=self.label[self.it - self.a:j - self.a])
        self.it = j
        return True

    def reset(self):
        self.it = self.a


def metrics(score, y, threshold):
    """As src/ml/metrics.cpp (SPEC §10.3): confusion at score > threshold, F1, precision, recall,
    and PR-AUC as average precision with tied scores forming one step."""
    pred, pos = score > threshold, y != 0
    tp, fp = int((pred & pos).sum()), int((pred & ~pos).sum())
    fn, tn = int((~pred & pos).sum()), int((~pred & ~pos).sum())
    pp, ap = tp + fp, tp + fn
    order = np.argsort(-score, kind="stable")
    s, hit = score[order], pos[order].astype(np.int64)
    last = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s) - 1]  # the end of each tie group
    hits = np.cumsum(hit)[last]
    rec = hits / ap if ap else np.zeros(len(last))
    prec = hits / (last + 1)
    pr_auc = float(np.sum(np.diff(np.r_[0.0, rec]) * prec)) if ap else 0.0
    return {"f1_x100": 100.0 * (2 * tp / (pp + ap) if pp + ap else 0.0), "precision": tp / pp if pp else 0.0,
            "recall": tp / ap if ap else 0.0, "pr_auc": pr_auc, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def xgb_setup(recipe):
    cfg = yaml.safe_load((ROOT / XGB_CONFIG).read_text())
    params = dict(cfg["params"])
    params["nthread"] = os.cpu_count()
    if recipe.get("base_score") is not None:
        params["base_score"] = recipe["base_score"]
    rounds = cfg["num_boost_round"]
    for k, v in recipe["xgb"].items():
        if k == "num_boost_round":
            rounds = v
        else:
            params[k] = v
    return params, rounds, cfg["threshold"], cfg["batch_rows"]


def resolved_base_score(config_json):
    v = json.loads(config_json)["learner"]["learner_model_param"]["base_score"]
    return float(v.strip("[]").split(",")[0]) if isinstance(v, str) else float(v)


def fit(data, recipe, set_name, cols=None):
    """Trains on data.splits["fit"] and scores data.splits["val"]. cols: the (name, getter)
    columns, for sets other than xgb_only and all (bench/dg2/e8.py); default recipe_columns()."""
    cols = recipe_columns(data, recipe, set_name) if cols is None else cols
    params, rounds, threshold, batch_rows = xgb_setup(recipe)
    (fa, fb), (va, vb) = data.splits["fit"], data.splits["val"]
    dirs = ([recipe["features"]["dir"]] if set_name != "xgb_only" else []) + \
        ([recipe["family"]["dir"]] if recipe.get("family") else [])
    for d in dirs:
        f = Features(data, d)
        if not (f.covers(fa, fb) and f.covers(va, vb)):
            raise SystemExit(f"{d} does not cover the fit and validation rows")
    yf, yv = data.labels("fit"), data.labels("val")
    t0 = time.time()
    dfit = xgb.QuantileDMatrix(Batches(cols, fa, fb, yf, batch_rows), max_bin=params["max_bin"],
                               nthread=params["nthread"])
    dval = xgb.QuantileDMatrix(Batches(cols, va, vb, yv, batch_rows), ref=dfit, nthread=params["nthread"])
    t_mat = time.time() - t0
    bst = xgb.train(params, dfit, num_boost_round=rounds)
    t_train = time.time() - t0 - t_mat
    score = bst.predict(dval)
    met = metrics(score, yv, threshold)
    cfg_json = bst.save_config()
    gain = bst.get_score(importance_type="gain")
    names = [c for c, _ in cols]
    top = sorted(((v, names[int(k[1:])]) for k, v in gain.items()), reverse=True)[:10]
    total_gain = {names[int(k[1:])]: v for k, v in bst.get_score(importance_type="total_gain").items()}
    return {"columns": {"names": names, "count": len(names)}, "validation": met,
            "rows": {"fit": [fa, fb], "val": [va, vb], "fit_positives": int(yf.sum()), "val_positives": int(yv.sum())},
            "xgb": {"version": xgb.__version__, "params": params, "num_boost_round": rounds, "threshold": threshold,
                    "batch_rows": batch_rows, "base_score_resolved": resolved_base_score(cfg_json),
                    "config": cfg_json},
            "top_gain": [[n, round(v, 3)] for v, n in top], "total_gain": total_gain,
            **({"raw_amounts_changed_by_%f": data.mgnn()["amounts_changed_by_%f"]} if recipe["raw"] else {}),
            "seconds": {"matrices": round(t_mat, 2), "train": round(t_train, 2)}}


# ---------------------------------------------------------------------------------------------
# Feature passes.

def profile_yaml_from(resolved, name, visibility):
    lines = [f"version: 2", f"profile: {name}", f"visibility: {visibility}", f"credit: {resolved['credit']}",
             f"k_max: {resolved['k_max']}", "blocks:"]
    for b in resolved["block_order"]:
        blk = resolved["blocks"][b]
        pats = ", ".join(p["name"] for p in blk["patterns"])
        lines.append(f"  {b}: {{patterns: [{pats}], window: {blk['window']}, bins: {blk['bins']}}}")
    return "\n".join(lines) + "\n"


def wc_windows():
    gt = json.loads((ROOT / GT_JOIN).read_text())
    out = {}
    for blocks, typs in WC_GROUPS.items():
        spans = [a["span"] for a in gt["attempts"] if a["typology"] in typs and a["status"] == "complete"
                 and a["split"] == "fit"]
        w = min(math.ceil(float(np.percentile(spans, 90)) / 3600) * 3600, WC_CAP)
        for b in blocks:
            out[b] = w
    return out


def blz_pass(data, profile, backend, out, preflight_s, budget_s, edges=None):
    """`blz mine` of `profile` on every row, after a 1% preflight (seed 0, SPEC v2 §2.5, §4.4) that
    aborts when its extrapolated wall time exceeds preflight_s or its extrapolated peak RSS 3 GB.
    edges: mine the stream prefix [0, edges) only (`blz mine --edges`). Neither run may go past
    budget_s seconds in total, what is left of the time cap (§4.2)."""
    out = ROOT / out
    pre = Path(str(out) + "_preflight")
    for d in (out, pre):
        shutil.rmtree(d, ignore_errors=True)
    m = edges or data.g.m
    base = [BLZ, "mine", "--graph", GRAPH, "--profile", profile, "--backend", backend, "--threads", "12"] + \
        (["--edges", str(edges)] if edges else [])
    started = {}

    def early(el, log):
        # After "index built ... N rows", the preflight's time scales with its rows. Its elapsed
        # time also holds output writes, so the extrapolation below over-estimates the exact one
        # (the mine seconds, after the preflight): it only stops a preflight that is clearly over,
        # at twice the limit, rather than wait out PASS_WALL_S.
        for ln in log.splitlines():
            if "index built in" in ln and "rows" in ln:
                if "t" not in started:
                    started["t"] = el
                    started["build"] = float(ln.split("index built in")[1].split()[0])
                    started["rows"] = int(ln.rsplit(":", 1)[1].split()[0])
                over = started["build"] + (el - started["t"]) * m / max(started["rows"], 1)
                if over > 2 * preflight_s:
                    return f"preflight: extrapolated wall time already > 2 x {preflight_s:.0f} s"
        return None

    rc, el_pre, rss_pre, why = run_monitored(base + ["--triggers", "sample:1:0", "--out", str(pre)],
                                             str(pre) + ".log", min(PASS_WALL_S, budget_s), PASS_RSS_MB, early)
    info = {"preflight": {"seconds": round(el_pre, 1), "peak_rss_mb": round(rss_pre), "log": str(pre) + ".log"}}
    if why is None and rc != 0:
        raise SystemExit(f"preflight failed (exit {rc}); see {pre}.log")
    if why is None:
        meta = json.loads((pre / "features.json").read_text())
        rows = int(np.load(pre / "triggers.npy", mmap_mode="r").shape[0])
        ext_s = meta["index_build_seconds"] + sum(b["extrapolated_mine_seconds"] for b in meta["blocks"].values())
        max_nb = max(len(b["columns"]) for b in meta["blocks"].values())
        # blz holds one block's rows at a time. Its own peak_rss_mb (ru_maxrss) includes this
        # process's RSS at fork, so the preflight's VmHWM from run_monitored is used instead.
        ext_rss = rss_pre + (m - rows) * max_nb * 4 / 2 ** 20
        info["preflight"].update({"rows": int(rows), "extrapolated_seconds": round(ext_s, 1),
                                  "extrapolated_peak_rss_mb": round(ext_rss), "profile_id": meta["profile"]["id"]})
        if ext_s > preflight_s:
            why = f"preflight: extrapolated wall time {ext_s:.0f} s > {preflight_s:.0f} s"
        elif ext_rss > PASS_RSS_MB:
            why = f"preflight: extrapolated peak RSS {ext_rss:.0f} MB > {PASS_RSS_MB} MB"
    info["profile_id"] = profile_id_of(str(pre) + ".log")
    if why:
        return None, el_pre, rss_pre, why, info
    rc, el, rss, why = run_monitored(base + ["--out", str(out)], str(out) + ".log",
                                     min(PASS_WALL_S, budget_s - el_pre), PASS_RSS_MB)
    info["full"] = {"seconds": round(el, 1), "peak_rss_mb": round(rss), "log": str(out) + ".log"}
    if why is None and rc != 0:
        raise SystemExit(f"blz mine failed (exit {rc}); see {out}.log")
    return (None if why else str(out.relative_to(ROOT))), el_pre + el, max(rss_pre, rss), why, info


def profile_id_of(log_path):
    for ln in Path(log_path).read_text(errors="replace").splitlines():
        if "index built in" in ln and "profile" in ln:
            return ln.split("(")[-1].split(")")[0] if ")" in ln else None
    return None


def gfp_pass(recipe_id, like_dir, out, budget_s):
    out_p = ROOT / out
    shutil.rmtree(out_p, ignore_errors=True)
    shutil.rmtree(str(out_p) + ".tmp", ignore_errors=True)
    cmd = [GFP_PY, "tools/gfp/dg2_features.py", "--graph", GRAPH, "--recipe", recipe_id,
           "--like", str(Path(like_dir) / "features.json"), "--out", out]
    rc, el, rss, why = run_monitored(cmd, str(out_p) + ".log", min(PASS_WALL_S, budget_s), PASS_RSS_MB)
    if why is None and rc != 0:
        raise SystemExit(f"dg2_features.py {recipe_id} failed (exit {rc}); see {out_p}.log")
    return (None if why else out), el, rss, why, {"log": str(out_p) + ".log", "command": cmd}


def smoke_stand_in(data, kind, cid, rung):
    """--smoke: a stand-in feature directory made from the v1_default features (no pass runs)."""
    src = ROOT / V1_FEATURES
    meta = json.loads((src / "features.json").read_text())
    d = Path(os.environ.get("TMPDIR", "/tmp")) / f"dg2_smoke_{rung}_{cid}"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    if kind == "profile":
        for b in meta["blocks"].values():
            os.symlink(src / b["file"], d / b["file"])
        meta["smoke_stand_in"] = True
    else:
        blocks = {}
        if cid == "F1":
            np.save(d / "vstats.npy", np.load(src / "fan_in.npy", mmap_mode="r")[:, :4].astype(np.float32))
            blocks["vstats"] = {"file": "vstats.npy", "columns": [f"vs_stand_in_{j}" for j in range(4)]}
        elif cid == "F2":
            os.symlink(src / "tcyc.npy", d / "lcyc.npy")
            blocks["lcyc"] = {"file": "lcyc.npy",
                              "columns": [c.replace("tcyc", "lcyc") for c in meta["blocks"]["tcyc"]["columns"]]}
        else:
            for name, b in meta["blocks"].items():
                os.symlink(src / b["file"], d / b["file"])
                blocks[name] = {"file": b["file"], "columns": b["columns"]}
        meta = {"producer": "tools/gfp/dg2_features.py", "gfp_recipe": {"id": cid}, "smoke_stand_in": True,
                "rows": {"first": 0, "last": data.g.m}, "blzg_sha256": data.blzg_sha256, "blocks": blocks}
    (d / "features.json").write_text(json.dumps(meta))
    return str(d)


# ---------------------------------------------------------------------------------------------
# The ladder.

def candidate_recipe(inc, rung, cid):
    """The incumbent recipe with rung `rung`'s change `cid` (SPEC v2 §4.4). Passes fill in the
    feature or family directory afterwards."""
    r = json.loads(json.dumps(inc))
    if rung == "R1":
        r["ids"] = cid
    elif rung == "R2":
        r["base_score"] = 0.5
    elif rung == "R3":
        r["raw"] = cid
    elif rung == "R8":
        r["xgb"] = dict(L.GRID[cid])
    return r


def r0_recipe():
    return {"ids": "v1", "base_score": None, "raw": None,
            "features": {"dir": V1_FEATURES, "source": "profile", "profile": V1_PROFILE,
                         "profile_id": V1_PROFILE_ID},
            "family": None, "xgb": {"scale_pos_weight": 1, "max_depth": 6, "num_boost_round": 100}}


def check_v1_features(data):
    f = Features(data, V1_FEATURES)
    m = f.meta
    if m["profile"]["id"] != V1_PROFILE_ID:
        raise SystemExit(f"{V1_FEATURES}: profile {m['profile']['id']}, not v1_default")
    got = {b: m["blocks"][b]["output_fnv1a"] for b in V1_FNV1A}
    if got != V1_FNV1A:
        raise SystemExit(f"{V1_FEATURES}: output_fnv1a {got} differ from AC-V2-0's")


class Ladder:
    def __init__(self, args):
        self.args = args
        self.path = ROOT / args.ledger
        self.led = L.load(self.path)
        self.data = Data(args.smoke)
        self.machine = machine()

    def save(self):
        L.save(self.led, str(self.path))
        Path(os.path.splitext(self.path)[0] + ".md").write_text(L.render(self.led))

    def record(self, entry):
        self.led = L.add_entry(self.led, entry)
        self.save()
        e = self.led["entries"][-1]
        v = e.get("validation")
        what = f"F1 {v['f1_x100']:.2f} P {v['precision']:.3f} R {v['recall']:.3f} PR-AUC {v['pr_auc']:.3f}" if v \
            else e.get("outcome", "")
        print(f"[{e['seq']}] {e['rung']}/{e['candidate']} {e['kind']} {e.get('set', '')}: {what} "
              f"({e['wall_seconds']:.0f} s, {e['peak_rss_mb']:.0f} MB; totals {e['totals_after']})", flush=True)

    def source_ids(self, recipe):
        f = recipe["features"]
        return {"gfp_recipe": f["gfp_recipe"]} if f["source"] == "gfp" else {"profile_id": f["profile_id"]}

    def do_fit(self, rung, cid, recipe, set_name, facts=None):
        if self.led["stopped"]:
            return False
        reset_peak_rss()
        t0 = time.time()
        with ForeignLoad() as fl:
            res = fit(self.data, recipe, set_name)
        entry = {"rung": rung, "candidate": cid, "kind": "fit", "set": set_name, **self.source_ids(recipe),
                 "recipe": recipe, **res, "wall_seconds": time.time() - t0, "peak_rss_mb": own_peak_rss_mb(),
                 "machine": self.machine, "foreign_load": fl.samples, "started": now_utc(), "git": git_info(),
                 "smoke": self.args.smoke}
        if facts:
            entry["facts"] = facts
        self.record(entry)
        return True

    def do_pass(self, rung, cid, inc):
        """Runs candidate cid's feature pass; returns its directory, or None if it was aborted
        (the candidate is then skipped: limit) or could not start (skipped: budget)."""
        if self.led["stopped"] or self.led["totals"]["passes"] >= L.BUDGET["passes"]:
            self.led = L.skip(self.led, rung, cid, "budget")
            self.save()
            return None
        out = f"features/dg2_{rung}_{cid}"
        gfp_recipe = cid if rung in ("R6", "R7") else None
        budget_s = (L.BUDGET["machine_hours"] - self.led["totals"]["machine_hours"]) * 3600
        cols, info = [], {}
        with ForeignLoad() as fl:
            if self.args.smoke:
                kind = "gfp" if gfp_recipe else "profile"
                d, el, rss, why = smoke_stand_in(self.data, kind, cid, rung), 0.0, 0.0, None
                info = {"smoke_stand_in": True}
                prof_id = V1_PROFILE_ID
            elif rung in ("R4", "R5"):
                profile = PROFILES.get(cid, R5_PROFILE)
                backend, limit = ("cpu", R4_PREFLIGHT_S) if rung == "R4" else ("oracle", PASS_WALL_S)
                # R5: the validation/test boundary splits a timestamp (§3.3), so under timestamp
                # visibility a validation row could see test edges; the graph ends at the boundary.
                edges = self.data.splits["val"][1] if rung == "R5" else None
                d, el, rss, why, info = blz_pass(self.data, profile, backend, out, limit, budget_s, edges)
                info.update({"profile": profile, "backend": backend, "preflight_limit_s": limit, "edges": edges})
                prof_id = info.get("profile_id")
            else:
                d, el, rss, why, info = gfp_pass(cid, inc["features"]["dir"], out, budget_s)
                prof_id = None
        if d:
            f = Features(self.data, d)
            cols = [c for name in f.blocks for c, _ in f.columns(name)]
        ids = {"gfp_recipe": gfp_recipe} if gfp_recipe else {"profile_id": prof_id or "unknown"}
        entry = {"rung": rung, "candidate": cid, "kind": "pass", **ids, "dir": d,
                 "outcome": "ok" if d else f"aborted: {why}", "pass": info,
                 "columns": {"names": cols, "count": len(cols)}, "wall_seconds": el, "peak_rss_mb": rss,
                 "machine": self.machine, "foreign_load": fl.samples, "started": now_utc(), "git": git_info(),
                 "smoke": self.args.smoke}
        if not d and why.startswith("wall time") and el >= budget_s - 5:  # killed at the time cap
            entry["outcome"] = f"aborted: time cap ({why})"
        self.record(entry)
        if not d:
            self.led = L.skip(self.led, rung, cid, "budget" if "time cap" in entry["outcome"] else "limit")
            self.save()
        return d

    def run_rung(self, rung, only=None, facts=None):
        r = L._rung(self.led, rung)
        if r["carried"] is not None:
            raise SystemExit(f"{rung} is decided ({r['carried']})")
        order = [x["rung"] for x in self.led["rungs"]]
        for prev in self.led["rungs"][:order.index(rung)]:
            if prev["carried"] is None:
                raise SystemExit(f"{prev['rung']} is not decided yet")
        status = lambda cid: L._candidate(self.led, rung, cid)["status"]  # self.led is replaced on each add
        if rung == "R0":
            if status("v1") != "pending":
                raise SystemExit("R0 has run")
            check_v1_features(self.data)
            for s in r["sets_fitted"]:
                self.do_fit("R0", "v1", r0_recipe(), s)
        else:
            inc = self.led["incumbent"]["recipe"]
            if rung == "R2":
                facts = [self.intercept_fact()] + (facts or [])
            for cid in [c["id"] for c in r["candidates"]]:
                if status(cid) != "pending" or (only and cid != only):
                    continue
                if self.led["stopped"]:
                    self.led = L.skip(self.led, rung, cid, "budget")
                    self.save()
                    continue
                recipe = candidate_recipe(inc, rung, cid)
                if rung == "R4":
                    self.check_wc()
                if rung == "R5" and not self.args.smoke:
                    self.write_r5_profile(inc)
                if rung in ("R4", "R5", "R6"):
                    d = self.do_pass(rung, cid, inc)
                    if d is None:
                        continue
                    f = Features(self.data, d)
                    recipe["features"] = {"dir": d, "source": f.source}
                    if f.source == "gfp":
                        recipe["features"]["gfp_recipe"] = f.gfp_recipe
                    else:
                        recipe["features"].update({"profile": PROFILES.get(cid, R5_PROFILE), "profile_id": f.profile_id})
                if rung == "R7":
                    d = self.do_pass(rung, cid, inc)
                    if d is None:
                        continue
                    recipe["family"] = {"id": cid, "dir": d}
                cfacts = facts if rung == "R2" else None
                for s in r["sets_fitted"]:
                    if not self.do_fit(rung, cid, recipe, s, cfacts):
                        break
                if self.led["stopped"] and status(cid) == "pending":
                    self.led = L.skip(self.led, rung, cid, "budget")
                    self.save()
        r = L._rung(self.led, rung)
        if all(c["status"] != "pending" for c in r["candidates"]):
            self.led = L.decide(self.led, rung)
            self.save()
            print(f"{rung} decided: carried {L._rung(self.led, rung)['carried']}; "
                  f"incumbent {self.led['incumbent']['source']} {self.led['incumbent']['scores']}")

    def intercept_fact(self):
        """R2's [R] fact, checked on this XGBoost: R0's resolved base_score (SPEC v2 §4.4 R2)."""
        r0 = [e for e in self.led["entries"] if e["rung"] == "R0" and e["kind"] == "fit"]
        vals = sorted({e["xgb"]["base_score_resolved"] for e in r0})
        ver = sorted({e["xgb"]["version"] for e in r0})
        return {"fact": "[R] XGBoost >= 2.0 estimates the intercept (base_score) from the labels when it is not set; "
                        "earlier versions used 0.5 (SPEC v2 §4.4 R2)",
                "check": f"learner_model_param.base_score in Booster.save_config() of R0's fits (XGBoost {ver}, "
                         "base_score not set)",
                "result": f"{vals}: " + ("confirmed for this version (not 0.5)" if all(v != 0.5 for v in vals)
                                         else "refuted for this version (0.5)")}

    def check_wc(self):
        prof = yaml.safe_load((ROOT / PROFILES["W-C"]).read_text())
        got = {b: v["window"] for b, v in prof["blocks"].items()}
        if got != wc_windows():
            raise SystemExit(f"{PROFILES['W-C']} windows {got} differ from gt_join.json's {wc_windows()}")

    def write_r5_profile(self, inc):
        if inc["features"]["source"] != "profile":
            raise SystemExit("R5 needs an incumbent mined from a profile")
        meta = json.loads((ROOT / inc["features"]["dir"] / "features.json").read_text())
        res = meta["profile"]["resolved"]
        if res["visibility"] != "stream":
            raise SystemExit("the incumbent already has timestamp visibility")
        text = (f"# DG-2 rung R5, candidate V1 (SPEC v2 §4.4): the incumbent profile {res['profile']} "
                f"({meta['profile']['id']},\n# {inc['features']['dir']}) with visibility: timestamp (§6.1). "
                "Written by bench/dg2/ladder.py.\n" + profile_yaml_from(res, "dg2_v1", "timestamp"))
        p = ROOT / R5_PROFILE
        if p.exists() and p.read_text() != text:
            raise SystemExit(f"{R5_PROFILE} exists with other content")
        p.write_text(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rung")
    ap.add_argument("--candidate")
    ap.add_argument("--facts", help="R2: a JSON list of [R]-fact checks to record (SPEC v2 §3.6)")
    ap.add_argument("--ledger", default=L.LEDGER)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    if a.smoke and Path(a.ledger).resolve() == (ROOT / L.LEDGER).resolve():
        sys.exit("--smoke needs a scratch --ledger")
    if a.smoke and not (ROOT / a.ledger).exists():
        L.save(L.init_ledger(), str(ROOT / a.ledger))
    if a.status:
        print(L.render(L.load(str(ROOT / a.ledger))))
        return 0
    if not a.rung:
        sys.exit("--rung or --status")
    facts = json.loads(Path(a.facts).read_text()) if a.facts else None
    lad = Ladder(a)
    if a.rung == "R2" and not a.smoke and not facts:
        sys.exit("R2 records the check of its [R] fact: pass --facts")
    lad.run_rung(a.rung, a.candidate, facts)
    return 0


if __name__ == "__main__":
    sys.exit(main())

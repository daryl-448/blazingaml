# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "xgboost>=3.4,<3.5", "pyyaml"]
# ///
"""E16 (SPEC v3 §3.6): what each clean rule changes on HI-Small, against its quirk.

  uv run --python 3.12 bench/e16.py mine [name...]  # every configuration (or these), one at a time
  uv run --python 3.12 bench/e16.py report          # per-block effects, the v1 check, validation fits
  uv run --python 3.12 bench/e16.py clean           # delete the .npy outputs under features/e16/
  ... --dataset transxion [--smoke]                 # the same on TransXion (P15); --smoke: a prefix

Configurations (features/e16/<dir>/, all rows of data/hi_small.blzg, `blz mine --backend oracle`):
clean_e16 clean; clean_e16 with each oracle quirk alone (Q1-Q6, Q9, Q10) and with all of them;
the profile variants clean_e16_q7 (no sg_ord) and clean_e16_q8 (fan and degree as histograms);
clean_e16_q8 with every quirk, which must reproduce v1_e16 (SPEC v3 F6); and v1_e16 on the legacy
oracle, with its vertex statistics from `blz vstats --window 86400` (cpu: legacy vstats have no
oracle). `mine` skips a directory whose features.json exists.

`report`:
- Per block and configuration, on the fit rows and the validation rows separately: instances,
  active rows, the share of positive rows that are active, and the rows that differ from clean.
- The v1 check: every block of clean_e16_q8 under `--quirks all` against v1_e16's (and sg_ord
  against sg_ordered_v1w, V2's legacy sg_ord), vstats bit for bit against `blz vstats`.
- Validation fits: V2 R0's XGBoost settings (configs/xgb_default.yaml), fit on the fit rows, for
  clean, each quirk alone, all oracle quirks, and v1_e16. Every fit has the same base columns, the
  DG-2 ladder's carried ids and raw columns (v1 `src_id`, `dst_id`; S2), then vstats and the
  pattern blocks. Each is fitted once unweighted and TRAIN_REPLICATES times on block-bootstrapped
  fit rows (the same weights for every configuration), so a quirk's paired ΔF1 is read against the
  training noise. A fit is cached in features/e16/fits.json under the hashes of its inputs.
- Validation ends at the first eid of the timestamp holding eid 4,062,676 when that eid does not
  start a timestamp (SPEC v3 §6.3); the rows in between are scored nowhere. Test rows are never read.
Writes results/e16_clean.{json,md}.

`--dataset transxion` (SPEC v3 §3.6, §7 P15): features/e16_transxion/, results/e16_clean_transxion.
Every run mines only the fit and validation rows (`--triggers train`, [0, b_test)), and the splits
are the .blzg's (`blz prepare`'s, §6.3). sg_ord is checked against sg_ordered_v1w mined here on the
legacy oracle; the ledger reproduction and the AC-V2-0 hashes are HI-Small's and are not run.
`--smoke` mines and fits the prefix [0, SMOKE_ROWS) instead (fit the first 3/4, validate the rest),
under build/e16_<dataset>_smoke/.
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent / "dg2"))
import e8 as E8  # noqa: E402
import ladder as Lad  # noqa: E402
import ledger as L  # noqa: E402

ROOT = Lad.ROOT
FEAT = ROOT / "features/e16"
OUT = ROOT / "results/e16_clean.json"
FIT = (0, 3_047_216)  # SPEC v2 §3.3
VAL_END_LEGACY = 4_062_676
VSTATS_WINDOW = 86_400  # V2 F1 (SPEC v2 §4.4 R7): one window, time_window = vertex_stats_tw = 86,400
P10_SG_ORD = "features/p10_sg_ordered_v1w_oracle"  # V2's sg_ordered, stream visibility, v1 windows
# The dataset (set by configure()): HI-Small mines every row; TransXion only its fit and validation rows.
DATASETS = {
    "hi_small": {"name": "HI-Small", "graph": Lad.GRAPH, "blz": Lad.BLZ, "feat": "features/e16",
                 "out": "results/e16_clean.json", "triggers": None},
    "transxion": {"name": "TransXion", "graph": "data/transxion.blzg", "blz": "build/rel-cuda/blz",
                  "feat": "features/e16_transxion", "out": "results/e16_clean_transxion.json", "triggers": "train"},
}
DS = DATASETS["hi_small"]
GRAPH, BLZ = DS["graph"], DS["blz"]
ROWS = None  # the eids every run mines, [a, b)
SPLIT = None  # {"fit", "val"} when the .blzg's (or the smoke prefix's) splits are used, None on HI-Small
SMOKE = False
SMOKE_ROWS = 100_000  # TransXion: 52 fit and 20 validation positives
REF_SG_ORD = ("sg_ordered_v1w", "sg_ordered_v1w")  # dir, profile: V2's legacy sg_ord, mined where P10's is not
F32_MAX = float(np.finfo(np.float32).max)
CHUNK = 1 << 20
BOOT_CHUNKS, BOOT_RESAMPLES = 100, 10_000
TRAIN_REPLICATES = 3  # refits on block-bootstrapped fit rows (chunk weights), R0's settings unchanged
VAL_RANGE = None  # set by report(): SPEC v3 §6.3's end on HI-Small

# name -> directory, profile, quirks ("" clean; None: the legacy path)
CONFIGS = {
    "clean": ("clean", "clean_e16", ""),
    **{q: (q.lower(), "clean_e16", q) for q in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6")},
    "Q7": ("q7", "clean_e16_q7", ""),
    "Q8": ("q8", "clean_e16_q8", ""),
    **{q: (q.lower(), "clean_e16", q) for q in ("Q9", "Q10")},
    "all": ("all", "clean_e16", "all"),
    "all+Q8": ("all_q8", "clean_e16_q8", "all"),
    "v1_e16": ("v1_e16", "v1_e16", None),
}
FITS = ["clean", "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10", "all", "v1_e16"]
FIT_ORDER = ["vstats"] + Lad.BLOCK_ORDER + ["sg_ord", "lcyc"]  # the ladder's: vstats, v1's six, then the rest
BASE = {"ids": "v1", "raw": "S2"}  # the DG-2 ladder's carried ids and raw columns (SPEC v2 P8)

# SPEC v3 §3.1, one row per quirk: how E16 runs it, and what it touches.
LEDGER_ROWS = [
    ("Q1", "oracle quirk", "sg and sg_ord see self-loops, with v1's role-dependent endpoint exclusion (A10, A30)"),
    ("Q2", "oracle quirk", "vertex statistics count self-loops; fan and degree drop them in both, self-loop "
                           "triggers emit no pattern rows in both, and batch mode (A41) is deleted, so only "
                           "vstats can change"),
    ("Q3", "oracle quirk", "closed lower bound t' >= t - delta for sg, sg_ord, tcyc, lcyc (and tchain, not in "
                           "clean_e16); fan and degree are open in both"),
    ("Q4", "oracle quirk", "stream visibility eid' <= e instead of ts' <= t, every block and vstats"),
    ("Q5", "oracle quirk", "strict time order in tcyc and sg_ord (and tchain and gs's A, not in clean_e16)"),
    ("Q6", "oracle quirk", "lcyc drops every edge tied with the trigger"),
    ("Q7", "profile variant", "sg only, no sg_ord block (clean_e16_q7)"),
    ("Q8", "profile variant", "fan and degree as v1's one-hot histograms with min size 2, not one count column "
                              "(clean_e16_q8)"),
    ("Q9", "oracle quirk", "vstats window [ts[e-1] - W, t], anchored at the previous edge, closed"),
    ("Q10", "oracle quirk", "vstats over absolute ts, eid-order accumulation, GFP's 1e-5 variance guard (A44)"),
    ("Q11", "not applicable", "GFP batch mode and credit: participants: deleted in P12 (D-55), no value to compare"),
    ("Q12", "not applicable", "GFP harness plumbing in the runtime (ts_offset, fuzz knobs): harness only, removed in P12"),
    ("Q13", "not applicable", "input range fixed to HI-Small: an input check, the same rows on HI-Small"),
    ("Q14", "not applicable", "GFP's bin rule (A38): harness only, blz keeps [b_i, b_(i+1)) in both"),
    ("Q15", "kept", "trigger credit re-reports unchanged instances: clean keeps it (R9), nothing to switch"),
    ("Q16", "kept", "min size 2 for histograms: clean keeps it (R10); count blocks have none, which is Q8"),
    ("Q17", "not applicable", "windows and k_max copied from GFP's setup: parameters, re-chosen on validation "
                              "later; clean_e16 and v1_e16 share v1's windows"),
]
NEVER_COPIED = [
    ("A29", "GFP's fit_transform double-counts re-sent edges (snapml >= 1.14.1): blz never copied it"),
    ("A33", "GFP's ts-0 self-loop artifact in tcyc (D-12's ts + 1): harness only, blz never copied it"),
    ("A45", "GFP's running-sum drift after eviction in vertex statistics: v2 computes each window exactly (D-37)"),
]


def configure(dataset, smoke):
    """Point the globals at `dataset`: its graph, folders, mined rows and splits (or the smoke prefix's)."""
    global DS, GRAPH, BLZ, FEAT, OUT, FIT, ROWS, SPLIT, SMOKE
    DS = DATASETS[dataset]
    GRAPH, BLZ, SMOKE = DS["graph"], DS["blz"], smoke
    FEAT, OUT = ROOT / DS["feat"], ROOT / DS["out"]
    g = Lad.Blzg(str(ROOT / GRAPH))
    if smoke:
        FEAT = ROOT / f"build/e16_{dataset}_smoke"
        OUT = FEAT / "result.json"
        ROWS, FIT = (0, SMOKE_ROWS), (0, SMOKE_ROWS * 3 // 4)
        SPLIT = {"fit": FIT, "val": (FIT[1], SMOKE_ROWS)}
    elif DS["triggers"] is None:
        ROWS = (0, g.m)
    else:
        sp = g.splits()
        ROWS, FIT, SPLIT = sp[DS["triggers"]], sp["fit"], {"fit": sp["fit"], "val": sp["val"]}


def triggers_arg():
    """`--triggers` for every run: None on HI-Small, whose runs mine every row."""
    return f"range:{ROWS[0]}:{ROWS[1]}" if SMOKE else DS["triggers"]


def mined_configs():
    """CONFIGS, plus V2's legacy sg_ord where features/p10_sg_ordered_v1w_oracle does not cover the rows."""
    return CONFIGS if SPLIT is None else {**CONFIGS, REF_SG_ORD[0]: (*REF_SG_ORD, "")}


class GraphData(Lad.Data):
    """Lad.Data on GRAPH with SPLIT's rows (Lad.Data itself requires HI-Small's SPEC v2 splits)."""

    def __init__(self):
        self.g = Lad.Blzg(str(ROOT / GRAPH))
        self.blzg_sha256 = Lad.file_sha256(ROOT / GRAPH)
        self.csv_sha256 = self.g.csv_sha256
        self.splits = dict(SPLIT)
        self._mgnn = None


def load_data():
    return Lad.Data(False) if SPLIT is None else GraphData()


def log_load(path):
    """uptime, free -g and the busiest processes, written to `path` before a heavy job."""
    up = subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()
    free = subprocess.run(["free", "-g"], capture_output=True, text=True).stdout
    top = subprocess.run(["ps", "--sort=-pcpu", "-eo", "pid,pcpu,etime,comm"], capture_output=True, text=True).stdout
    path.write_text(f"{Lad.now_utc()}\n{up}\n{free}" + "\n".join(top.splitlines()[:8]) + "\n")
    print(f"load: {up}", flush=True)
    return up


def timed(cmd, log):
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.run(["/usr/bin/time", "-v"] + cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc:
        raise SystemExit(f"{' '.join(cmd)}: exit {rc}, see {log}")
    return round(time.time() - t0, 1)


def mine_all(names):
    cfg, tr = mined_configs(), triggers_arg()
    rows = ["--triggers", tr] if tr else []
    for name in names or cfg:
        d, prof, quirks = cfg[name]
        out = FEAT / d
        if (out / "features.json").exists():
            continue
        FEAT.mkdir(parents=True, exist_ok=True)
        log_load(FEAT / f"{d}.load")
        cmd = [BLZ, "mine", "--graph", GRAPH, "--profile", f"configs/profiles/{prof}.yaml", "--backend",
               "oracle", "--out", str(out.relative_to(ROOT))] + (["--quirks", quirks] if quirks else []) + rows
        print("+", " ".join(cmd), flush=True)
        print(f"{name}: {timed(cmd, FEAT / f'{d}.log')} s", flush=True)
        if quirks is None:  # the legacy vertex statistics (Q9, Q10): blz vstats, GFP's window and guards
            vs = FEAT / f"{d}_vstats"
            log_load(FEAT / f"{d}_vstats.load")
            cmd = [BLZ, "vstats", "--graph", GRAPH, "--window", str(VSTATS_WINDOW), "--backend", "cpu",
                   "--out", str(vs.relative_to(ROOT))] + rows
            print("+", " ".join(cmd), flush=True)
            print(f"{name} vstats: {timed(cmd, FEAT / f'{d}_vstats.log')} s", flush=True)


def time_log(log):
    t = log.read_text()
    wall = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", t).group(1)
    secs = sum(float(x) * 60 ** i for i, x in enumerate(reversed(wall.split(":"))))
    rss = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", t).group(1))
    load = (log.with_suffix(".load")).read_text().splitlines()[1]
    return {"wall_s": round(secs, 1), "peak_rss_mb": round(rss / 1024), "load_before": load}


# ---------------------------------------------------------------------------------------------
# Reading a configuration's blocks.

def check_manifest(d, m, data):
    """A folder E16 reads: mined from this graph's CSV (D-89) over exactly ROWS, not sampled."""
    t = m["triggers"]
    if Lad.csv_sha256_of(m) != data.csv_sha256 or (t["first"], t["last"]) != ROWS or t.get("sampled"):
        raise SystemExit(f"{d}: not rows [{ROWS[0]:,}, {ROWS[1]:,}) of {GRAPH}")


class Blocks:
    """A configuration's blocks (vstats included), each a read-only memory map of the mined rows."""

    def __init__(self, data, name):
        cfg = mined_configs()
        d0 = FEAT / cfg[name][0]
        dirs = [d0] + ([FEAT / f"{d0.name}_vstats"] if cfg[name][2] is None else [])
        self.arr, self.bins, self.cols, self.hash = {}, {}, {}, {}
        for d in dirs:
            m = json.loads((d / "features.json").read_text())
            check_manifest(d, m, data)
            for b, v in m["blocks"].items():
                self.arr[b] = np.load(d / v["file"], mmap_mode="r")
                self.bins[b] = v.get("bins", [])
                self.cols[b] = v["columns"]
                self.hash[b] = v.get("output_fnv1a", v.get("output_sha256"))
        self._cache = {}

    def kind(self, b):
        return "vstats" if b == "vstats" else ("count" if not self.bins[b] else "histogram")

    def rows(self, b, a, z):
        hit = self._cache.get(b)
        if hit and hit[0] == (a, z):
            return hit[1]
        x = np.asarray(self.arr[b][a:z])
        if b == "vstats":  # XGBoost reads float32: clip as dg2_features.py did; NaN stays missing
            x = np.clip(x, -F32_MAX, F32_MAX)
        self._cache[b] = ((a, z), x)
        return x

    def columns(self):
        return [(c, (lambda a, z, b=b, j=j: self.rows(b, a, z)[:, j]))
                for b in FIT_ORDER if b in self.arr for j, c in enumerate(self.cols[b])]


def histogram_of(counts, bins):
    """A count block's column as the one-hot histogram v1's bins give it ([b_i, b_(i+1)), min b_0)."""
    idx = np.searchsorted(np.asarray(bins), counts, side="right") - 1
    h = np.zeros((len(counts), len(bins)), dtype=np.uint32)
    ok = idx >= 0
    h[np.flatnonzero(ok), idx[ok]] = 1
    return h


VS_GROUPS = {"fan_deg_ratio": [0, 1, 2], "time": [3, 4, 5, 6, 7], "amount": [8, 9, 10, 11, 12]}


def block_stats(blocks, b, ref, y, parts):
    """Per part: instances, active rows, active positive rows and, against `ref` (clean's blocks),
    the rows that differ. Count blocks: one instance per trigger with a count >= 1; `active_ge2` is
    what a histogram with min size 2 marks. vstats: no instances; rows changed per column group."""
    kind = blocks.kind(b)
    arr = blocks.arr[b]
    rarr = ref.arr.get(b) if ref is not None else None
    keys = ["active", "active_pos"] + {"histogram": ["instances"], "vstats": ["nan", "clipped"],
                                       "count": ["instances", "count_sum", "active_ge2", "active_ge2_pos"]}[kind]
    if rarr is not None:
        keys += ["changed", "changed_pos"] + ([f"changed_{g}" for g in VS_GROUPS] if kind == "vstats" else [])
    out = {"kind": kind, "columns": len(blocks.cols[b]), "hash": blocks.hash[b]}
    for part, (a, z) in parts.items():
        acc = dict.fromkeys(keys, 0)
        for i in range(a, z, CHUNK):
            j = min(i + CHUNK, z)
            x = np.asarray(arr[i:j])
            pos = y[i:j] != 0
            if kind == "count":
                c = x[:, 0].astype(np.uint64)
                act, ge2 = c > 0, c >= 2
                acc["instances"] += int(act.sum())
                acc["count_sum"] += int(c.sum())
                acc["active_ge2"] += int(ge2.sum())
                acc["active_ge2_pos"] += int((ge2 & pos).sum())
            else:
                act = (x != 0).any(axis=1)
                if kind == "histogram":
                    acc["instances"] += int(x.sum(dtype=np.uint64))
                else:
                    acc["nan"] += int(np.isnan(x).sum())
                    acc["clipped"] += int((np.abs(x) > F32_MAX).sum())
            acc["active"] += int(act.sum())
            acc["active_pos"] += int((act & pos).sum())
            if rarr is None:
                continue
            r = np.asarray(rarr[i:j])
            if ref.kind(b) == "count" and kind == "histogram":  # Q8: clean's count, binned as this block
                r = histogram_of(r[:, 0], blocks.bins[b])
            if r.shape != x.shape:
                raise SystemExit(f"block {b}: shape {x.shape} vs clean's {r.shape}")
            if kind == "vstats":
                d = x.view(np.uint64) != r.view(np.uint64)  # bit for bit
                for g, js in VS_GROUPS.items():
                    acc[f"changed_{g}"] += int(d[:, [k + 13 * side for side in range(4) for k in js]].any(axis=1).sum())
            else:
                d = x != r
            ch = d.any(axis=1)
            acc["changed"] += int(ch.sum())
            acc["changed_pos"] += int((ch & pos).sum())
        npos = int(y[a:z].sum())
        acc["active_pos_share"] = round(acc["active_pos"] / npos, 4)
        if kind == "count":
            acc["active_ge2_pos_share"] = round(acc["active_ge2_pos"] / npos, 4)
        out[part] = {"rows": z - a, "positives": npos, **acc}
    return out


# ---------------------------------------------------------------------------------------------
# The v1 check (SPEC v3 F6): every quirk on and histogram bins give the legacy rows.

def v1_check(data):
    got, want = Blocks(data, "all+Q8"), Blocks(data, "v1_e16")
    if SPLIT is None:
        sgo = json.loads((ROOT / P10_SG_ORD / "features.json").read_text())
        sg_ref = (np.load(ROOT / P10_SG_ORD / "sg_ord.npy", mmap_mode="r"), sgo["blocks"]["sg_ord"]["output_fnv1a"],
                  f"{P10_SG_ORD} (sg_ordered_v1w, oracle)")
    else:
        sgo = Blocks(data, REF_SG_ORD[0])
        sg_ref = (sgo.arr["sg_ord"], sgo.hash["sg_ord"], f"{(FEAT / REF_SG_ORD[0]).relative_to(ROOT)} (sg_ordered_v1w, oracle)")
    res = {}
    for b in got.arr:
        if b == "sg_ord":
            ref, rhash, rname = sg_ref
        else:
            ref, rhash, rname = want.arr[b], want.hash[b], "v1_e16" + (" (blz vstats)" if b == "vstats" else "")
        x = got.arr[b]
        if x.shape != ref.shape:
            res[b] = {"against": rname, "equal": False, "why": f"shape {x.shape} vs {ref.shape}"}
            continue
        if b != "vstats" and got.hash[b] == rhash:
            res[b] = {"against": rname, "equal": True, "by": "output_fnv1a", "hash": rhash}
            continue
        bad = 0
        for i in range(0, x.shape[0], CHUNK):
            xi, ri = np.asarray(x[i:i + CHUNK]), np.asarray(ref[i:i + CHUNK])
            if b == "vstats":
                xi, ri = xi.view(np.uint64), ri.view(np.uint64)
            bad += int((xi != ri).any(axis=1).sum())
        res[b] = {"against": rname, "equal": bad == 0, "by": "rows, bit for bit", "rows_differing": bad}
    if SPLIT is None:  # AC-V2-0's hashes are of HI-Small's rows
        v1_hashes = {b: want.hash[b] for b in Lad.V1_FNV1A}
        res["v1_e16 six blocks vs v1_default (AC-V2-0 hashes)"] = {"equal": v1_hashes == Lad.V1_FNV1A}
    return res


# ---------------------------------------------------------------------------------------------
# Fits.

def recipe():
    r = Lad.r0_recipe()  # XGBoost: V2 R0's settings, configs/xgb_default.yaml (spw 1, depth 6, 100 rounds)
    r.update({"raw": BASE["raw"], "features": None})
    return r


def fit_key(blocks):
    h = hashlib.sha256(json.dumps({"base": BASE, "order": FIT_ORDER, "hash": blocks.hash, "val": VAL_RANGE,
                                   "xgb": Lad.file_sha256(ROOT / Lad.XGB_CONFIG)}, sort_keys=True).encode())
    return h.hexdigest()[:16]


class WeightedBatches(Lad.Batches):
    def __init__(self, cols, a, b, label, rows, weight):
        self.weight = weight
        super().__init__(cols, a, b, label, rows)

    def next(self, input_data):
        if self.it >= self.b:
            return False
        j = min(self.it + self.rows, self.b)
        X = np.empty((j - self.it, len(self.cols)), dtype=np.float32)
        for k, (_, get) in enumerate(self.cols):
            X[:, k] = get(self.it, j)
        input_data(data=X, label=self.label[self.it - self.a:j - self.a], weight=self.weight[self.it - self.a:j - self.a])
        self.it = j
        return True


def fit_weights(r):
    """Training replicate r: each of BOOT_CHUNKS contiguous chunks of the fit rows weighted by its
    multinomial draw count (a block bootstrap of the fit rows as weights)."""
    edges = np.linspace(FIT[0], FIT[1], BOOT_CHUNKS + 1).astype(np.int64)
    k = np.random.default_rng(1000 + r).multinomial(BOOT_CHUNKS, [1 / BOOT_CHUNKS] * BOOT_CHUNKS)
    return np.repeat(k, np.diff(edges)).astype(np.float32)


def xgb_fit(data, cols, weight=None):
    """Lad.fit's training and scoring (QuantileDMatrix in batches, the validation matrix on the fit
    matrix's cuts), returning the booster and the validation scores as well."""
    params, rounds, threshold, batch_rows = Lad.xgb_setup(recipe())
    (fa, fb), (va, vb) = data.splits["fit"], data.splits["val"]
    yf, yv = data.labels("fit"), data.labels("val")
    it = Lad.Batches(cols, fa, fb, yf, batch_rows) if weight is None else WeightedBatches(cols, fa, fb, yf, batch_rows, weight)
    dfit = xgb.QuantileDMatrix(it, max_bin=params["max_bin"], nthread=params["nthread"])
    dval = xgb.QuantileDMatrix(Lad.Batches(cols, va, vb, yv, batch_rows), ref=dfit, nthread=params["nthread"])
    bst = xgb.train(params, dfit, num_boost_round=rounds)
    score = bst.predict(dval)
    return bst, score, Lad.metrics(score, yv, threshold)


def run_fit(data, name, cache):
    blocks = Blocks(data, name)
    key = fit_key(blocks)
    hit = cache.get(name)
    if hit and hit["key"] == key and (FEAT / f"scores_{CONFIGS[name][0]}.npy").exists():
        return hit
    up = log_load(FEAT / f"fit_{CONFIGS[name][0]}.load")
    t0 = time.time()
    Lad.reset_peak_rss()
    cols = Lad.recipe_columns(data, recipe(), "xgb_only") + blocks.columns()
    names = [c for c, _ in cols]
    bst, score, met = xgb_fit(data, cols)
    gain = {}
    for k, v in bst.get_score(importance_type="total_gain").items():
        b = E8.block_of(names[int(k[1:])])
        gain[b] = gain.get(b, 0.0) + v
    tot = sum(gain.values()) or 1.0
    top = sorted(((v, names[int(k[1:])]) for k, v in bst.get_score(importance_type="gain").items()), reverse=True)[:10]
    out = {"key": key, "metrics": met, "columns": len(names),
           "gain_share_by_block": {b: round(v / tot, 4) for b, v in sorted(gain.items(), key=lambda x: -x[1])},
           "top_gain": [[n, round(v, 3)] for v, n in top],
           "base_score_resolved": Lad.resolved_base_score(bst.save_config()),
           "seconds": round(time.time() - t0, 1), "peak_rss_mb": round(Lad.own_peak_rss_mb()), "load_before": up}
    np.save(FEAT / f"scores_{CONFIGS[name][0]}.npy", score)
    print(f"fit {name}: F1 {met['f1_x100']:.2f} P {met['precision']:.3f} R {met['recall']:.3f} "
          f"PR-AUC {met['pr_auc']:.3f} ({out['seconds']} s)", flush=True)
    cache[name] = out
    (FEAT / "fits.json").write_text(json.dumps(cache, indent=1))
    return out


def run_replicates(data, name, cache):
    """F1 of TRAIN_REPLICATES refits on reweighted fit rows; the same weights for every configuration."""
    blocks = Blocks(data, name)
    key = fit_key(blocks)
    out = []
    for r in range(1, TRAIN_REPLICATES + 1):
        tag = f"{name}#r{r}"
        hit = cache.get(tag)
        if not (hit and hit["key"] == key):
            log_load(FEAT / f"fit_{CONFIGS[name][0]}_r{r}.load")
            t0 = time.time()
            cols = Lad.recipe_columns(data, recipe(), "xgb_only") + blocks.columns()
            met = xgb_fit(data, cols, fit_weights(r))[2]
            hit = cache[tag] = {"key": key, "metrics": met, "seconds": round(time.time() - t0, 1)}
            (FEAT / "fits.json").write_text(json.dumps(cache, indent=1))
            print(f"fit {tag}: F1 {met['f1_x100']:.2f} ({hit['seconds']} s)", flush=True)
        out.append(hit["metrics"])
    return out


def reproduce_ledger(data):
    """xgb_fit against the DG-2 ledger: v1_default's blocks with R3/S2 on the legacy validation
    range must give the ledger's confusion counts."""
    r = Lad.r0_recipe()
    r["raw"] = "S2"
    saved = data.splits["val"]
    data.splits["val"] = (FIT[1], VAL_END_LEGACY)
    try:
        got = xgb_fit(data, Lad.recipe_columns(data, r, "all"))[2]
    finally:
        data.splits["val"] = saved
    led = L.load(str(ROOT / L.LEDGER))
    e = next(x for x in led["entries"] if x["rung"] == "R3" and x.get("candidate") == "S2" and x.get("set") == "all")
    keys = ("tp", "fp", "fn", "tn")
    return {"fit": "v1_default (features/p7_cpu_default) + R3/S2, validation [3,047,216, 4,062,676)",
            "got": {k: got[k] for k in keys}, "ledger": {k: e["validation"][k] for k in keys},
            "reproduces": all(got[k] == e["validation"][k] for k in keys)}


def boot_delta(y, s0, s1, threshold):
    """Paired block bootstrap of F1(s1) - F1(s0) x100 over BOOT_CHUNKS contiguous chunks of the
    validation rows (laundering rows cluster in time, so rows are not resampled one by one)."""
    edges = np.linspace(0, len(y), BOOT_CHUNKS + 1).astype(np.int64)
    pos = y != 0

    def per_chunk(s):
        pred = s > threshold
        return np.stack([np.add.reduceat((pred & pos).astype(np.int64), edges[:-1]),
                         np.add.reduceat((pred & ~pos).astype(np.int64), edges[:-1]),
                         np.add.reduceat((~pred & pos).astype(np.int64), edges[:-1])])

    c0, c1 = per_chunk(s0), per_chunk(s1)
    idx = np.random.default_rng(0).integers(0, BOOT_CHUNKS, size=(BOOT_RESAMPLES, BOOT_CHUNKS))

    def f1(c):
        tp, fp, fn = (c[k][idx].sum(axis=1) for k in range(3))
        d = 2 * tp + fp + fn
        return np.where(d > 0, 200.0 * tp / np.maximum(d, 1), 0.0)

    d = f1(c1) - f1(c0)
    return {"lo95": round(float(np.percentile(d, 2.5)), 2), "hi95": round(float(np.percentile(d, 97.5)), 2),
            "sd": round(float(d.std()), 2)}


# ---------------------------------------------------------------------------------------------
# The report.

def replicate_f1s(f):
    return ", ".join(f"{r['f1_x100']:.2f}" for r in f["train_replicates"])


def fmt_int(v):
    return f"{v:,}" if isinstance(v, int) else "—"


def render(doc):
    sp = doc["split"]
    mined = sp.get("mined")
    rows = f"all {sp['m']:,} rows" if mined is None else f"rows [{mined[0]:,}, {mined[1]:,}) of {sp['m']:,}"
    L_ = [f"# E16: what the clean semantics change (SPEC v3 §3.6), {doc['dataset']}", "",
          f"git {doc['git']['sha'][:7]}{' (dirty)' if doc['git']['dirty'] else ''}, {doc['machine']}, "
          f"{doc['started']}. Every configuration mined on the oracle (`blz mine --backend oracle`), {rows}; "
          f"nothing here changes a recipe, profile or default (E16 informs §3.5 only).", ""]
    L_ += [f"- Rows: fit [0, {sp['fit'][1]:,}) ({sp['fit_positives']:,} positives); validation "
           f"[{sp['val'][0]:,}, {sp['val'][1]:,}) ({sp['val_positives']:,} positives). {sp['note']}",
           f"- Classifier: {doc['classifier']['note']}"]
    if "reproduction" in doc:
        L_.append(f"- Fit plumbing: {doc['reproduction']['fit']} gives {doc['reproduction']['got']}, the ledger "
                  f"{doc['reproduction']['ledger']}: {'reproduces' if doc['reproduction']['reproduces'] else 'DIFFERS'}.")
    L_.append("")
    L_ += ["## The v1 check", "", doc["v1_check_summary"], ""]
    L_ += ["## Quirks", "",
           "Rows changed = rows of the configuration that differ from clean_e16's (vstats bit for bit; Q8's "
           "histograms against clean's counts binned the same way). ΔF1 = validation F1 ×100 minus clean_e16's, "
           "with a 95% paired block-bootstrap interval.", "",
           "| Quirk | How | Blocks with changed rows (fit / validation rows changed) | F1 ×100 | ΔF1 [95%] | "
           "ΔF1, training resamples |", "|---|---|---|---:|---|---|"]
    for q in doc["quirks"]:
        f = doc["validation"].get(q["id"])
        L_.append(f"| {q['id']} | {q['how']} | {q['effect']} | {f['metrics']['f1_x100']:.2f} | "
                  f"{f['delta_f1_vs_clean']:+.2f} [{f['delta_ci']['lo95']:+.2f}, {f['delta_ci']['hi95']:+.2f}] | "
                  f"{', '.join(f'{d:+.2f}' for d in f['delta_f1_train_replicates'])} |"
                  if f else f"| {q['id']} | {q['how']} | {q['effect']} | — | — | — |")
    for a, why in NEVER_COPIED:
        L_.append(f"| {a} | not applicable | {why} | — | — | — |")
    L_ += ["", "## Validation fits", "",
           "| Configuration | Columns | F1 ×100 | Precision | Recall | PR-AUC | TP | FP | FN | ΔF1 vs clean [95%] | "
           "F1, training resamples |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for n in FITS:
        f = doc["validation"][n]
        m = f["metrics"]
        ci = "" if n == "clean" else f"{f['delta_f1_vs_clean']:+.2f} [{f['delta_ci']['lo95']:+.2f}, {f['delta_ci']['hi95']:+.2f}]"
        L_.append(f"| {n} | {f['columns']} | {m['f1_x100']:.2f} | {m['precision']:.3f} | {m['recall']:.3f} | "
                  f"{m['pr_auc']:.3f} | {m['tp']} | {m['fp']} | {m['fn']} | {ci} | "
                  f"{replicate_f1s(f)} |")
    L_ += ["", "## Blocks: clean_e16 against v1_e16", "",
           "Instances, active rows and the share of positive rows that are active, fit / validation. Count blocks: "
           "one instance per trigger with a count of at least 1; the `≥ 2` share is what a min-size-2 histogram "
           "marks active.", "",
           "| Block | clean kind | clean instances | clean active | clean pos. active | v1_e16 instances | v1_e16 active | "
           "v1_e16 pos. active | rows changed (v1_e16 vs clean) |", "|---|---|---|---|---|---|---|---|---|"]
    bs = doc["blocks"]
    for b in FIT_ORDER[1:] + ["vstats"]:
        c, v = bs["clean"].get(b), bs["v1_e16"].get(b)
        if not c:
            continue

        def cell(s, k):
            return " / ".join(fmt_int(s[p].get(k)) for p in ("fit", "val")) if s else "—"

        def share(s):
            if not s:
                return "—"
            t = " / ".join(f"{s[p]['active_pos_share']:.3f}" for p in ("fit", "val"))
            if s["kind"] == "count":
                t += " (≥ 2: " + " / ".join(f"{s[p]['active_ge2_pos_share']:.3f}" for p in ("fit", "val")) + ")"
            return t

        L_.append(f"| {b} | {c['kind']} | {cell(c, 'instances')} | {cell(c, 'active')} | {share(c)} | "
                  f"{cell(v, 'instances')} | {cell(v, 'active')} | {share(v)} | {cell(v, 'changed')} |")
    L_ += ["", "## Per-block effects", "",
           "Each block a configuration changes, against clean_e16, as fit / validation: instances, active rows, the "
           "share of positive rows that are active, and rows changed. A count block shows its rows with a count of at "
           "least 2 (every non-loop trigger has a count of at least 1), which is also what a histogram marks; Q4's "
           "count blocks show the sum of the counts as instances.", "",
           "| Configuration | Block | Instances, clean → quirk | Active rows, clean → quirk | Pos. active, clean → quirk "
           "| Rows changed (positives) |", "|---|---|---|---|---|---|"]

    def view(st, kind, count_inst):
        if kind == "count":
            return {p: {"inst": st[p]["count_sum"] if count_inst else st[p]["active_ge2"], "act": st[p]["active_ge2"],
                        "share": st[p]["active_ge2_pos_share"]} for p in ("fit", "val")}
        return {p: {"inst": st[p].get("instances"), "act": st[p]["active"], "share": st[p]["active_pos_share"]}
                for p in ("fit", "val")}

    for n in CONFIGS:
        if n == "clean":
            continue
        for b, s in bs[n].items():
            c = bs["clean"].get(b)
            if not c or not (s["fit"]["changed"] or s["val"]["changed"]):
                continue
            both_counts = c["kind"] == s["kind"] == "count"
            cv, sv = view(c, c["kind"], both_counts), view(s, s["kind"], both_counts)
            inst = "—" if s["kind"] == "vstats" else " / ".join(
                f"{fmt_int(cv[p]['inst'])} → {fmt_int(sv[p]['inst'])}" for p in ("fit", "val"))
            act = " / ".join(f"{cv[p]['act']:,} → {sv[p]['act']:,}" for p in ("fit", "val"))
            sh = " / ".join(f"{cv[p]['share']:.3f} → {sv[p]['share']:.3f}" for p in ("fit", "val"))
            ch = " / ".join(f"{s[p]['changed']:,} ({s[p]['changed_pos']:,})" for p in ("fit", "val"))
            L_.append(f"| {n} | {b} | {inst} | {act} | {sh} | {ch} |")
    L_ += ["", f"Every (block, configuration) pair, with the same counts, is in `{OUT.relative_to(ROOT)}` (`blocks`).", "",
           "## Runs", "", "| Configuration | Profile | Quirks | Wall | Peak RSS | Load before |", "|---|---|---|---:|---:|---|"]
    for n, r in doc["runs"].items():
        L_.append(f"| {n} | {r['profile']} | {r['quirks']} | {r['wall_s']:.0f} s | {r['peak_rss_mb']} MB | "
                  f"{r['load_before'].split('load average: ')[-1]} |")
    L_ += ["", "## Notes", "", *[f"- {n}" for n in doc["notes"]], ""]
    return "\n".join(L_)


def evidence(data, y, parts):
    """The facts behind the large effects, from fit and validation rows only."""
    g = data.g
    ts, src, dst = np.asarray(g.ts), np.asarray(g.src), np.asarray(g.dst)
    fmt = g.strings("format_dict")
    ev = {"file_order": {}, "self_loops": {}, "absolute_time": {}}
    for part, (a, z) in parts.items():
        t = ts[a:z]
        first, last = np.searchsorted(ts, t, "left"), np.searchsorted(ts, t, "right")
        size, e = last - first, np.arange(a, z)
        m, pos = size > 1, y[a:z] != 0
        r = (e - first) / np.maximum(size - 1, 1)
        ev["file_order"][part] = {
            "mean_rank_positive": round(float(r[m & pos].mean()), 4), "mean_rank_negative": round(float(r[m & ~pos].mean()), 4),
            "positives_by_rank_decile": np.histogram(r[m & pos], bins=10, range=(0, 1))[0].tolist()}
        loop = src[a:z] == dst[a:z]
        f = np.asarray(g.format[a:z])
        ev["self_loops"][part] = {"self_loops": int(loop.sum()), "laundering_self_loops": int((loop & pos).sum())}
        if "Reinvestment" in fmt:  # AMLworld's formats; TransXion has none
            ev["self_loops"][part].update({
                "reinvestment_self_loops": int((loop & (f == fmt.index("Reinvestment"))).sum()),
                "laundering_reinvestment": int((pos & (f == fmt.index("Reinvestment"))).sum())})
    (fa, fb), (va, vb) = parts["fit"], parts["val"]
    for n in ("clean", "Q10", "v1_e16"):
        b = Blocks(data, n)
        col = b.cols["vstats"][3]  # u_out's time mean: the age (clean) or the absolute ts (Q10, legacy)
        x = b.arr["vstats"]
        hi = float(np.asarray(x[fa:fb, 3]).max())
        ev["absolute_time"][n] = {"column": col, "fit_max": hi,
                                  "val_share_above_fit_max": round(float((np.asarray(x[va:vb, 3]) > hi).mean()), 4)}
    return ev


def effect_text(doc, name):
    parts = []
    for b, s in doc["blocks"][name].items():
        f, v = s["fit"].get("changed"), s["val"].get("changed")
        if f or v:
            parts.append(f"{b} {f:,} / {v:,}")
    return ", ".join(parts) if parts else "none"


def report():
    data = load_data()
    g = data.g
    global VAL_RANGE
    y = np.asarray(g.label[:ROWS[1]], dtype=np.uint8)  # the mined rows' labels only: no test label is read
    if SPLIT is None:
        t = int(g.ts[VAL_END_LEGACY])
        starts = g.ts[VAL_END_LEGACY - 1] != t
        val_end = VAL_END_LEGACY if starts else int(np.searchsorted(g.ts, t, side="left"))
        VAL_RANGE = (FIT[1], val_end)
        hi = int(np.searchsorted(g.ts, t, side="right"))
        extra = {"eid_4062676_starts_a_timestamp": bool(starts),
                 "note": ("Eid 4,062,676 starts a new timestamp, so validation keeps its legacy end." if starts else
                          f"Eid 4,062,676 does not start a timestamp: ts {t:,} holds eids [{val_end:,}, {hi:,}), so "
                          f"clean validation ends at {val_end:,} (SPEC v3 §6.3). Its {VAL_END_LEGACY - val_end} "
                          f"rows below 4,062,676 ({int(y[val_end:VAL_END_LEGACY].sum())} positive) are scored "
                          f"nowhere; every fit here, legacy ones included, uses the same range.")}
    else:
        VAL_RANGE = SPLIT["val"]
        extra = {"mined": list(ROWS), "note": (
            f"Smoke: the prefix [0, {ROWS[1]:,}), its first three quarters fitted, the rest validated; not a result."
            if SMOKE else
            f"The .blzg's fit and validation rows, as `blz prepare` stored them with V2 D-17's boundary rule at both "
            f"cuts (SPEC v3 §6.3). Every run mines [0, {ROWS[1]:,}) (`--triggers {DS['triggers']}`); the test rows "
            f"[{ROWS[1]:,}, {g.m:,}) are neither mined nor read.")}
    data.splits = {"fit": FIT, "val": VAL_RANGE}
    parts = {"fit": FIT, "val": VAL_RANGE}
    doc = {"experiment": "E16", "spec": "SPEC v3 §3.6", "dataset": DS["name"], "git": Lad.git_info(),
           "machine": Lad.machine(), "started": Lad.now_utc(), "blzg_sha256": data.blzg_sha256,
           "split": {"m": g.m, "fit": list(FIT), "val": list(VAL_RANGE), "fit_positives": int(y[slice(*FIT)].sum()),
                     "val_positives": int(y[slice(*VAL_RANGE)].sum()), **extra},
           "vstats_window": VSTATS_WINDOW}
    if SMOKE:
        doc["smoke"] = True
    doc["runs"] = {}
    feat, tr = FEAT.relative_to(ROOT), triggers_arg()
    for n, (d, prof, q) in mined_configs().items():
        r = {"dir": f"{feat}/{d}", "profile": prof, "quirks": "legacy path" if q is None else (q or "none"),
             "profile_id": json.loads((FEAT / d / "features.json").read_text())["profile"]["id"], **time_log(FEAT / f"{d}.log")}
        if q is None:
            r["vstats"] = {"dir": f"{feat}/{d}_vstats", "command": f"blz vstats --window {VSTATS_WINDOW} --backend cpu"
                                                                   + (f" --triggers {tr}" if tr else ""),
                           **time_log(FEAT / f"{d}_vstats.log")}
            r["wall_s"] = round(r["wall_s"] + r["vstats"]["wall_s"], 1)
        doc["runs"][n] = r
    clean = Blocks(data, "clean")
    doc["blocks"] = {}
    for n in CONFIGS:
        bl = Blocks(data, n)
        doc["blocks"][n] = {b: block_stats(bl, b, None if n == "clean" else clean, y, parts) for b in bl.arr}
        print(f"stats {n}", flush=True)
    doc["v1_check"] = v1_check(data)
    bad = [b for b, v in doc["v1_check"].items() if not v["equal"]]
    by_hash = [b for b, v in doc["v1_check"].items() if v.get("by") == "output_fnv1a" and b != "sg_ord"]
    doc["v1_check_summary"] = (
        f"MISMATCH in {bad}: see `v1_check` in the JSON." if bad else
        "clean_e16_q8 under `--quirks all` (every oracle quirk, plus Q8's histograms) reproduces v1_e16 on every "
        "block: the six v1 blocks and lcyc by output hash, the vertex statistics bit for bit against `blz vstats "
        "--window 86400`, and sg_ord against V2's sg_ordered (sg_ordered_v1w, oracle). v1_e16's six blocks carry "
        "v1_default's AC-V2-0 hashes." if SPLIT is None else
        "clean_e16_q8 under `--quirks all` (every oracle quirk, plus Q8's histograms) reproduces v1_e16 on every "
        f"block of these rows: {', '.join(by_hash) or 'no block'} against v1_e16 by output hash, the others row by row (the vertex "
        f"statistics bit for bit against `blz vstats --window {VSTATS_WINDOW}`), and sg_ord against V2's sg_ordered "
        f"(sg_ordered_v1w) mined on the same rows on the oracle. AC-V2-0's hashes are HI-Small's and are not checked.")
    doc["evidence"] = evidence(data, y, parts)
    if SPLIT is None:  # the DG-2 ledger's fit is HI-Small's
        doc["reproduction"] = reproduce_ledger(data)
        print("reproduction:", doc["reproduction"], flush=True)
    cache = json.loads((FEAT / "fits.json").read_text()) if (FEAT / "fits.json").exists() else {}
    doc["validation"] = {n: run_fit(data, n, cache) for n in FITS}
    for n in FITS:
        doc["validation"][n]["train_replicates"] = [
            {k: m[k] for k in ("f1_x100", "precision", "recall", "pr_auc", "tp", "fp", "fn")}
            for m in run_replicates(data, n, cache)]
    yv = y[slice(*VAL_RANGE)]
    s0 = np.load(FEAT / "scores_clean.npy")
    base = doc["validation"]["clean"]["metrics"]["f1_x100"]
    for n in FITS[1:]:
        f = doc["validation"][n]
        f["delta_f1_vs_clean"] = round(f["metrics"]["f1_x100"] - base, 2)
        f["delta_ci"] = boot_delta(yv, s0, np.load(FEAT / f"scores_{CONFIGS[n][0]}.npy"), 0.5)
        f["delta_f1_train_replicates"] = [round(a["f1_x100"] - b["f1_x100"], 2) for a, b in
                                          zip(f["train_replicates"], doc["validation"]["clean"]["train_replicates"])]
    for f in doc["validation"].values():
        f.pop("key", None)
    doc["classifier"] = {
        "xgb_config": Lad.XGB_CONFIG, "xgb": "V2 R0's settings: scale_pos_weight 1, max_depth 6, 100 rounds, eta 0.3, "
                                             "hist, base_score estimated from the fit labels",
        "base_columns": [c for c, _ in Lad.recipe_columns(data, recipe(), "xgb_only")],
        "column_order": ["ids", "raw"] + FIT_ORDER,
        "note": ("V2 R0's XGBoost settings (`configs/xgb_default.yaml`: library defaults, 100 rounds, threshold 0.5), "
                 "fit on the fit rows. Every fit has the same base, the DG-2 ladder's carried ids and raw columns: "
                 "v1 `src_id`, `dst_id` and S2 (Amount Sent, Sent Currency, Amount Received, Received Currency, "
                 "Payment Format, with Multi-GNN's codes). Then the vertex statistics (52 columns, clipped to the "
                 "float32 range) and the pattern blocks in v1's order, sg_ord and lcyc last.")}
    doc["quirks"] = []
    for qid, how, what in LEDGER_ROWS:
        row = {"id": qid, "how": how, "changes": what}
        if qid in doc["blocks"]:
            row["effect"] = effect_text(doc, qid) if qid != "Q7" else "sg_ord removed (its other blocks: " + \
                effect_text(doc, qid) + ")"
            if qid == "Q8":
                row["effect"] = "fan and degree become histograms (rows changed counted after binning clean's counts: " + \
                    effect_text(doc, qid) + ")"
        else:
            row["effect"] = what
        doc["quirks"].append(row)
    doc["never_copied"] = [{"id": a, "status": "not applicable", "why": w} for a, w in NEVER_COPIED]
    doc["notes"] = notes(doc) if SPLIT is None else measured_notes(doc)
    doc["finished"] = Lad.now_utc()
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    OUT.with_suffix(".md").write_text(render(doc))
    print(f"-> {OUT}, {OUT.with_suffix('.md')}")
    return 0 if not bad and doc.get("reproduction", {"reproduces": True})["reproduces"] else 2


def draws(f, k):
    return [f["metrics"][k]] + [r[k] for r in f["train_replicates"]]


def notes(doc):
    V, ev, bs = doc["validation"], doc["evidence"], doc["blocks"]
    d = {n: [a - b for a, b in zip(draws(V[n], "f1_x100"), draws(V["clean"], "f1_x100"))] for n in FITS}
    dp = {n: [a - b for a, b in zip(draws(V[n], "pr_auc"), draws(V["clean"], "pr_auc"))] for n in FITS}
    lst = lambda xs, f="+.2f": ", ".join(format(x, f) for x in xs)  # noqa: E731
    same = [n for n in FITS[1:] if all(x == 0 for x in d[n])]
    pattern = ("sg", "sg_ord", "tcyc", "lcyc")
    gmax = max(V[n]["gain_share_by_block"].get(b, 0.0) for n in FITS for b in pattern)
    first_tenth = sum(ev["file_order"][p]["positives_by_rank_decile"][0] for p in ("fit", "val"))
    ranked = sum(sum(ev["file_order"][p]["positives_by_rank_decile"]) for p in ("fit", "val"))
    sl = {k: sum(ev["self_loops"][p][k] for p in ("fit", "val")) for k in ev["self_loops"]["fit"]}
    at = ev["absolute_time"]
    lower = lambda n: sum(x < 0 for x in dp[n])  # noqa: E731
    return [
        f"Training noise dominates single fits. F1 at threshold 0.5 under R0's settings moves a lot when only the "
        f"training sample changes: clean_e16's four draws (the unweighted fit, then {TRAIN_REPLICATES} refits on "
        f"block-bootstrapped fit rows, {BOOT_CHUNKS} chunks, the same weights for every configuration) give F1 "
        f"{lst(draws(V['clean'], 'f1_x100'), '.2f')}. The [95%] column covers only the validation sample's noise. "
        f"Here a quirk has an effect only when its paired ΔF1 keeps one sign over all four draws.",
        f"Only Q2 has one. Its ΔF1 over the four draws is {lst(d['Q2'])} and its ΔPR-AUC {lst(dp['Q2'], '+.3f')}. "
        f"The fit and validation rows hold {sl['self_loops']:,} self-loops, {sl['reinvestment_self_loops']:,} of them "
        f"Reinvestment, and {sl['laundering_self_loops']} laundering ones ({sl['laundering_reinvestment']} Reinvestment). "
        f"Every Reinvestment edge is a self-loop. Q2 puts each account's reinvestment history into its vertex "
        f"statistics, and R1 takes it out: R1 keeps self-loops as classifier rows, but no feature carries an "
        f"account's self-loops into its other rows. Q2's effect is the vstats part only: fan and degree drop "
        f"self-loops under both semantics, self-loop triggers have zero pattern rows under both, and batch mode "
        f"(A41) is deleted.",
        f"{', '.join(same)} leave every fit identical to clean's in all four draws, and Q1 and Q3 move F1 by at most "
        f"{max(abs(x) for n in ('Q1', 'Q3') for x in d[n]):.2f} in one draw. The blocks they change (sg, sg_ord, tcyc, "
        f"lcyc) hold at most {100 * gmax:.2f}% of the gain in any fit, and HI-Small barely has a cycle or a "
        f"scatter-gather with two hops in one minute: Q5 changes {bs['Q5']['sg_ord']['fit']['changed']} sg_ord rows "
        f"and no tcyc row, and Q6 changes no lcyc row. sg_ord has {bs['clean']['sg_ord']['fit']['active']} active fit "
        f"rows, {bs['clean']['sg_ord']['fit']['active_pos']} of them positive, and none of the {len(FITS)} unweighted "
        f"fits splits on sg or sg_ord, so Q7 "
        f"(dropping the block) changes nothing.",
        f"Q4 (stream visibility) changes fan, degree and vstats rows, but its ΔF1 takes both signs ({lst(d['Q4'])}). "
        f"Within a timestamp, file order carries the label: {first_tenth:,} of the {ranked:,} fit and validation "
        f"positives that share their minute sit in the first tenth of its rows (mean relative rank "
        f"{ev['file_order']['fit']['mean_rank_positive']:.3f} on fit and {ev['file_order']['val']['mean_rank_positive']:.3f} "
        f"on validation, against 0.500 for negatives). Under stream visibility a row's windows end at its file "
        f"position, so its features can encode that rank; R3 removes this.",
        f"Q10 (absolute timestamps) puts {100 * at['Q10']['val_share_above_fit_max']:.1f}% of validation rows above the "
        f"fit maximum of u_out's time mean ({100 * at['v1_e16']['val_share_above_fit_max']:.1f}% for v1_e16; clean's "
        f"ages: {100 * at['clean']['val_share_above_fit_max']:.1f}%), so the trees extrapolate on time. Its ΔF1 takes "
        f"both signs ({lst(d['Q10'])}); its PR-AUC is lower in {lower('Q10')} of 4 draws.",
        f"All the oracle quirks together score below clean in {sum(x < 0 for x in d['all'])} of 4 draws on F1 and "
        f"{lower('all')} of 4 on PR-AUC, and v1_e16 in {sum(x < 0 for x in d['v1_e16'])} and {lower('v1_e16')}, even "
        f"though Q2 alone raises both every time. The quirks do not add up, and which of them cancels Q2's gain was "
        f"not tested. The all-quirks unweighted fit predicts {V['all']['metrics']['fp']:,} false positives.",
        f"Q8 (histograms instead of counts) is information-preserving here (clean's counts, binned, equal Q8's rows "
        f"exactly), and its ΔF1 takes both signs ({lst(d['Q8'])}). So does Q9's ({lst(d['Q9'])}).",
        "R0's settings make a weak classifier on these columns, and the absolute levels are not comparable to E8: "
        "DG-2's tuned recipe reached F1 53.02 on GFP features. Only the paired differences are read here.",
        "No test label is read. Labels are sliced to the fit and validation rows, and the evidence above uses those "
        "rows only. Each run mines every row, like every other HI-Small run, but no test row is scored.",
        "Count blocks: instances = rows with a count of at least 1 (one instance per trigger, R7). The ≥ 2 share is "
        "what v1's histogram marks active.",
        "Timings are informal (P13). No foreign process used more than 20% of a core before any run. The load "
        "average logged before a run still includes the previous E16 run.",
    ]


def measured_notes(doc):
    """Notes for a dataset other than HI-Small: what was measured, without HI-Small's reading of it."""
    V, ev, bs, sp = doc["validation"], doc["evidence"], doc["blocks"], doc["split"]
    k = 1 + TRAIN_REPLICATES
    d = {n: [a - b for a, b in zip(draws(V[n], "f1_x100"), draws(V["clean"], "f1_x100"))] for n in FITS}
    dp = {n: [a - b for a, b in zip(draws(V[n], "pr_auc"), draws(V["clean"], "pr_auc"))] for n in FITS}
    lst = lambda xs, f="+.2f": ", ".join(format(x, f) for x in xs)  # noqa: E731
    same = [n for n in FITS[1:] if all(x == 0 for x in d[n])]
    signed = [n for n in FITS[1:] if n not in same and (all(x > 0 for x in d[n]) or all(x < 0 for x in d[n]))]
    unchanged = [n for n in CONFIGS if n != "clean"
                 and not any(s[p].get("changed") for s in bs[n].values() for p in ("fit", "val"))]
    first_tenth = sum(ev["file_order"][p]["positives_by_rank_decile"][0] for p in ("fit", "val"))
    ranked = sum(sum(ev["file_order"][p]["positives_by_rank_decile"]) for p in ("fit", "val"))
    sl = {key: sum(ev["self_loops"][p][key] for p in ("fit", "val")) for key in ev["self_loops"]["fit"]}
    at = ev["absolute_time"]
    return [
        f"clean_e16's {k} draws (the unweighted fit, then {TRAIN_REPLICATES} refits on block-bootstrapped fit rows, "
        f"{BOOT_CHUNKS} chunks, the same weights for every configuration) give F1 {lst(draws(V['clean'], 'f1_x100'), '.2f')} "
        f"and PR-AUC {lst(draws(V['clean'], 'pr_auc'), '.3f')}. The [95%] column covers only the validation sample's noise.",
        "Paired ΔF1 keeps one sign over all draws for " + (
            "; ".join(f"{n} ({lst(d[n])}; ΔPR-AUC {lst(dp[n], '+.3f')})" for n in signed) if signed else "no configuration")
        + ". " + (f"{', '.join(same)} give fits identical to clean's in every draw." if same else ""),
        f"Configurations that change no block row on the fit or validation rows: {', '.join(unchanged) or 'none'} "
        "(Q7 drops sg_ord and counts as unchanged when its other blocks are).",
        f"The fit and validation rows hold {sl['self_loops']:,} self-loops, {sl['laundering_self_loops']:,} of them "
        f"laundering.",
        f"{first_tenth:,} of the {ranked:,} fit and validation positives that share their timestamp sit in the first "
        f"tenth of its rows (mean relative rank {ev['file_order']['fit']['mean_rank_positive']:.3f} on fit and "
        f"{ev['file_order']['val']['mean_rank_positive']:.3f} on validation; negatives "
        f"{ev['file_order']['fit']['mean_rank_negative']:.3f} and {ev['file_order']['val']['mean_rank_negative']:.3f}).",
        f"u_out's time mean: {100 * at['Q10']['val_share_above_fit_max']:.1f}% of validation rows lie above its fit "
        f"maximum under Q10 (absolute timestamps), {100 * at['v1_e16']['val_share_above_fit_max']:.1f}% under v1_e16 "
        f"and {100 * at['clean']['val_share_above_fit_max']:.1f}% under clean's ages.",
        f"No test label is read. Every run mines [{sp['mined'][0]:,}, {sp['mined'][1]:,}), labels are read for those "
        "rows only, and only validation rows are scored.",
        "R0's settings (`configs/xgb_default.yaml`), not a tuned recipe: only the paired differences are read.",
        "Count blocks: instances = rows with a count of at least 1 (one instance per trigger, R7). The ≥ 2 share is "
        "what v1's histogram marks active.",
        "Timings are informal (P13); the load before each run is logged with it.",
    ]


def clean_outputs():
    for p in sorted(FEAT.rglob("*.npy")):
        if p.name.startswith("scores_"):
            continue
        print("rm", p.relative_to(ROOT))
        p.unlink()
    for d in FEAT.glob("smoke_*"):
        shutil.rmtree(d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["mine", "report", "clean"])
    ap.add_argument("names", nargs="*", help=f"mine: these configurations only, of {', '.join(CONFIGS)} "
                                             f"(and {REF_SG_ORD[0]} off HI-Small)")
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="hi_small")
    ap.add_argument("--smoke", action="store_true", help=f"the prefix [0, {SMOKE_ROWS:,}) under build/")
    a = ap.parse_args()
    configure(a.dataset, a.smoke)
    if a.phase == "mine":
        if set(a.names) - set(mined_configs()):
            ap.error(f"unknown configuration {sorted(set(a.names) - set(mined_configs()))}")
        mine_all(a.names)
        return 0
    if a.phase == "clean":
        clean_outputs()
        return 0
    return report()


if __name__ == "__main__":
    sys.exit(main())

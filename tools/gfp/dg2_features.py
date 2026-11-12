"""GFP-produced feature passes for the DG-2 ladder (SPEC v2 §4.4 R6, R7; D-24). Never used for parity.

This is the only place where GFP batch mode and fit_transform are allowed (SPEC v2 §4.1). Every
recipe uses a fresh preprocessor, vertex_stats off unless the recipe is F1, time_window = -1,
and ts + GFP_TS_OFFSET (D-12). The pattern windows and bins come from the incumbent's
features.json (--like), so G1-G3 run "at the incumbent windows and bins".

  G1  transform() on consecutive batches of 128 edges in stream order, cut short at b_fv and at
      the train/test boundary; one preprocessor throughout. Rows: every edge.
  G2  the same with batches of 2,048.
  G3  fit_transform(fit rows), then transform(validation rows), one call each. Rows: fit + val.
  F1  vertex statistics, stream mode (one transform() per edge), on Amount Received and
      Timestamp, vertex_stats_tw = 86,400, the default vertex_stats_feats. Rows: every edge.
  F2  the lc-cycle block, stream mode, lc-cycle_len = 10, window 86,400, the incumbent's tcyc
      bins. Rows: every edge.

Bins: GFP is given gfp_bins(profile bins), so that its blocks are binned as SPEC §4.2 bins blz's
(A38, D-32): the recipes change the call semantics, not the binning.

What a row may see: G1, G2: the later edges of its own batch (A28), never an edge of a later
batch or split (A34); instances are credited to every participant in the batch (A28, A35). G3:
every fit row sees every fit row, and re-sent fit edges are double-counted (A29, A36);
validation rows see every train edge and the later validation edges. F1, F2: stream visibility.

Outputs (written to <out>.tmp, renamed to <out> when complete): <block>.npy (uint32, the blz
layout, rows from eid 0) or vstats.npy (float32), and features.json.

Usage: .venv/bin/python dg2_features.py --graph ../../data/hi_small.blzg --recipe G1
           --like ../../features/<incumbent>/features.json --out ../../features/dg2_G1 [--limit N]
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
from snapml import GraphFeaturePreprocessor

from gfp_common import GFP_TS_OFFSET, PATTERN_BLOCKS, gfp_bins, make_preprocessor

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench" / "dg2"))
from blzg import Blzg  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
V1_PATTERNS = ["fan", "degree", "scatter-gather", "temp-cycle"]
# GFP pattern -> the profile block whose window and bins it takes (fan_out, deg_out must agree).
PATTERN_OF_BLOCK = {"fan": ("fan_in", "fan_out"), "degree": ("deg_in", "deg_out"),
                    "scatter-gather": ("sg",), "temp-cycle": ("tcyc",)}
VS_STATS = {0: "fan", 1: "deg", 2: "ratio", 3: "avg", 4: "sum", 5: "min", 6: "max", 7: "median",
            8: "var", 9: "skew", 10: "kurtosis"}
VS_FEATS = [0, 1, 2, 3, 4, 8, 9, 10]   # GFP's default vertex_stats_feats (F7)
F32_MAX = float(np.finfo(np.float32).max)
# Input columns: [eid, src, dst, ts, amount_received]. GFP emits each column's statistics in input
# column order, whatever the order of vertex_stats_cols (check_vs_order), so this list is sorted.
VS_COLS = [("ts", 3), ("amount_received", 4)]
WINDOW_F = 86400                       # F1, F2 (Altman, F8)
LC_LEN = 10
RECIPES = ("G1", "G2", "G3", "F1", "F2")


def column_names(block, bins):
    """As blz's column_names (SPEC §4.2): <block>[b_i,b_{i+1}) ... <block>[b_last,inf)."""
    return [f"{block}[{a},{b})" for a, b in zip(bins, bins[1:])] + [f"{block}[{bins[-1]},inf)"]


def vstats_names():
    """GFP's vertex-statistics layout: <src out> <src in> <dst out> <dst in>, each fan, degree,
    ratio, then per input column its selected statistics (transform() docstring)."""
    names = []
    for vert in ("src", "dst"):
        for d in ("out", "in"):
            names += [f"vs_{vert}_{d}_{VS_STATS[f]}" for f in VS_FEATS if f <= 2]
            for col, _ in VS_COLS:
                names += [f"vs_{vert}_{d}_{col}_{VS_STATS[f]}" for f in VS_FEATS if f > 2]
    return names


def windows_from(like):
    """Pattern -> (window, bins) from a features.json: blz's resolved profile, or the blocks of a
    manifest written by this script (an incumbent that R6 made GFP-produced)."""
    blocks = like["blocks"] if like.get("producer") == "tools/gfp/dg2_features.py" else like["profile"]["resolved"]["blocks"]
    out = {}
    for pat, bs in PATTERN_OF_BLOCK.items():
        wb = {(blocks[b]["window"], tuple(blocks[b]["bins"])) for b in bs}
        if len(wb) != 1:
            sys.exit(f"--like: GFP has one window and bin list per pattern; {bs} differ: {wb}")
        w, bins = wb.pop()
        out[pat] = (int(w), [int(x) for x in bins])
    return out


def check_vs_order(params, skip, names):
    """One edge 1 -> 2 at ts 101 with amount 7: the source's out-statistics must name 101 as the
    timestamp average and 7 as the amount average, else the column names are wrong."""
    gp = GraphFeaturePreprocessor()
    gp.set_params(params)
    row = dict(zip(names, gp.transform(np.array([[0, 1, 2, 101, 7]], dtype=np.float64))[0, skip:]))
    got = (row["vs_src_out_ts_avg"], row["vs_src_out_amount_received_avg"], row["vs_dst_in_ts_avg"])
    if got != (101, 7, 101):
        raise RuntimeError(f"GFP vertex statistics are not in the order of vstats_names(): {got}")


class NpyWriter:
    """Appends rows to a .npy file whose shape is fixed in advance (no memory map, so the rows do
    not count towards the process's RSS)."""

    def __init__(self, path, rows, cols, dtype):
        self.path, self.rows, self.cols, self.dtype = path, rows, cols, np.dtype(dtype)
        self.f = open(path, "wb")
        np.lib.format.write_array_header_1_0(
            self.f, {"descr": self.dtype.str, "fortran_order": False, "shape": (rows, cols)})
        self.written = 0
        self.sha = hashlib.sha256()
        self.buf, self.nbuf = [], 0

    def add(self, block):
        block = np.ascontiguousarray(block, dtype=self.dtype)
        assert block.ndim == 2 and block.shape[1] == self.cols, block.shape
        self.buf.append(block)
        self.nbuf += block.shape[0]
        if self.nbuf >= 65536:
            self.flush()

    def flush(self):
        if self.buf:
            data = np.concatenate(self.buf).tobytes()
            self.f.write(data)
            self.sha.update(data)
            self.written += self.nbuf
            self.buf, self.nbuf = [], 0

    def close(self):
        self.flush()
        self.f.close()
        if self.written != self.rows:
            raise RuntimeError(f"{self.path}: wrote {self.written} rows, expected {self.rows}")
        return self.sha.hexdigest()


def as_counts(a):
    """GFP histogram counts (float64) -> uint32, checking they are whole and in range."""
    if not (np.all(a >= 0) and np.all(a <= np.iinfo(np.uint32).max) and np.all(a == np.floor(a))):
        raise RuntimeError("GFP returned a count that is not a uint32")
    return a.astype(np.uint32)


def vm_hwm_mb():
    """VmHWM, not ru_maxrss: when this runs under the ladder, ru_maxrss keeps the RSS the ladder had
    when it forked this process."""
    for ln in open("/proc/self/status"):
        if ln.startswith("VmHWM:"):
            return int(ln.split()[1]) / 1024
    return float("nan")


def git_info():
    def run(*a):
        return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True).stdout.strip()
    return {"sha": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


class Progress:
    def __init__(self, total):
        self.total, self.t0, self.last = total, time.time(), time.time()

    def __call__(self, done):
        now = time.time()
        if now - self.last > 30:
            self.last = now
            rate = done / (now - self.t0)
            print(f"{done}/{self.total} edges, {rate:.0f}/s, eta {(self.total - done) / max(rate, 1e-9) / 60:.1f} min, "
                  f"peak RSS {vm_hwm_mb():.0f} MB",
                  file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--recipe", required=True, choices=RECIPES)
    ap.add_argument("--like", required=True, help="the incumbent's features.json (windows and bins)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=12, help="GFP num_threads for the batch recipes")
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke test: the first N edges, with fit/val/test cut at N/2 and 3N/4 (not a pass)")
    ap.add_argument("--batch", type=int, default=None,
                    help="smoke test with --limit only: G1/G2 with this batch size (1 = stream mode, = blz rows)")
    ap.add_argument("--force", action="store_true", help="replace an existing --out")
    args = ap.parse_args()

    if args.batch is not None and (args.limit is None or args.recipe not in ("G1", "G2")):
        sys.exit("--batch is a smoke-test option: it needs --limit and recipe G1 or G2")
    t_start = time.time()
    g = Blzg(args.graph)
    sp = g.splits()
    m = g.m
    b_fv, train = sp["fit"][1], sp["train"][1]
    if args.limit is not None:
        m = min(args.limit, g.m)
        b_fv, train = m // 2, 3 * m // 4
    like = json.loads(Path(args.like).read_text())
    win = windows_from(like)
    out = Path(args.out)
    if out.exists() and not args.force:
        sys.exit(f"{out} exists (--force replaces it)")
    tmp = Path(str(out) + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)

    X = np.empty((m, 5), dtype=np.float64)  # eid, src, dst, ts + offset, amount_received
    X[:, 0] = np.arange(m)
    X[:, 1] = g.src[:m]
    X[:, 2] = g.dst[:m]
    X[:, 3] = g.ts[:m].astype(np.float64) + GFP_TS_OFFSET
    X[:, 4] = g.amount_received[:m]
    r = args.recipe
    blocks = {}  # name -> (writer, columns, extra json)
    progress = Progress(m)

    if r in ("G1", "G2", "G3"):
        gp = make_preprocessor(V1_PATTERNS, {p: w for p, (w, _) in win.items()},
                               {p: gfp_bins(b) for p, (_, b) in win.items()}, num_threads=args.threads)
        rows = m if r != "G3" else train
        for p in V1_PATTERNS:
            for b in PATTERN_BLOCKS[p]:
                bins = win[p][1]
                blocks[b] = (NpyWriter(tmp / f"{b}.npy", rows, len(bins), np.uint32), column_names(b, bins),
                             {"window": win[p][0], "bins": bins, "gfp_pattern": p})

        def emit(feats):
            k = 0
            for p in V1_PATTERNS:
                nb = len(win[p][1])
                for b in PATTERN_BLOCKS[p]:
                    blocks[b][0].add(as_counts(feats[:, k:k + nb]))
                    k += nb
            assert k == feats.shape[1], (k, feats.shape)

        X4 = np.ascontiguousarray(X[:, :4])
        if r in ("G1", "G2"):
            batch = args.batch or (128 if r == "G1" else 2048)
            calls = []
            for a0, a1 in ((0, b_fv), (b_fv, train), (train, m)):  # no batch mixes two splits
                for i in range(a0, a1, batch):
                    j = min(i + batch, a1)
                    emit(gp.transform(X4[i:j])[:, 4:])
                    calls.append(j - i)
                    progress(j)
            call = {"mode": "batches", "batch": batch, "cuts": [b_fv, train], "transform_calls": len(calls)}
        else:
            emit(gp.fit_transform(X4[:b_fv])[:, 4:])
            progress(b_fv)
            emit(gp.transform(X4[b_fv:train])[:, 4:])
            call = {"mode": "fit_transform(fit rows), then transform(validation rows)", "cuts": [b_fv, train]}
    elif r == "F1":
        # GFP rejects vertex statistics with no pattern enabled, so temp-cycle (the cheapest in
        # stream mode) runs alongside with one bin and its column is dropped. Vertex statistics
        # do not depend on the companion pattern (checked against degree and fan on 30,000 edges).
        gp = GraphFeaturePreprocessor()
        params = {"num_threads": 1, "time_window": -1, "vertex_stats": True, "vertex_stats_tw": WINDOW_F,
                  "vertex_stats_cols": [c for _, c in VS_COLS], "vertex_stats_feats": VS_FEATS}
        for pat in PATTERN_BLOCKS:
            params.update({pat: pat == "temp-cycle", pat + "_tw": WINDOW_F})
        params["temp-cycle_bins"] = [2]
        gp.set_params(params)
        skip = 5 + 1  # the input columns, then the companion's one bin
        names = vstats_names()
        check_vs_order(params, skip, names)
        vs_clipped = vs_nan = 0
        rows = m
        blocks["vstats"] = (NpyWriter(tmp / "vstats.npy", rows, len(names), np.float32), names,
                            {"window": WINDOW_F, "columns_of_input": [c for c, _ in VS_COLS]})
        for i in range(m):
            f = gp.transform(X[i:i + 1])[:, skip:]
            if f.shape[1] != len(names):
                raise RuntimeError(f"GFP returned {f.shape[1]} vertex statistics, expected {len(names)}")
            big = np.abs(f) > F32_MAX  # inf included; NaN compares False and stays NaN (missing)
            if big.any():
                f = np.clip(f, -F32_MAX, F32_MAX)
                vs_clipped += int(big.sum())
            vs_nan += int(np.isnan(f).sum())
            blocks["vstats"][0].add(f.astype(np.float32))
            progress(i + 1)
        call = {"mode": "stream (one transform() per edge)",
                "companion": "temp-cycle, window 86,400, bins [2]: GFP needs one pattern; its column is dropped"}
        blocks["vstats"][2].update({
            "clipped_to_float32": vs_clipped, "nan": vs_nan,
            "float32": "GFP's float64 values beyond the float32 range (kurtosis and skew of near-constant inputs "
                       "blow up) are clipped to +-3.4028235e38, which XGBoost, working in float32, accepts; "
                       "NaN is kept and is missing to XGBoost"})
    else:  # F2
        bins = win["temp-cycle"][1]
        gp = make_preprocessor(["lc-cycle"], WINDOW_F, gfp_bins(bins), lc_len=LC_LEN, num_threads=1)
        rows = m
        blocks["lcyc"] = (NpyWriter(tmp / "lcyc.npy", rows, len(bins), np.uint32), column_names("lcyc", bins),
                          {"window": WINDOW_F, "bins": bins, "lc_cycle_len": LC_LEN, "gfp_pattern": "lc-cycle"})
        X4 = np.ascontiguousarray(X[:, :4])
        for i in range(m):
            blocks["lcyc"][0].add(as_counts(gp.transform(X4[i:i + 1])[:, 4:]))
            progress(i + 1)
        call = {"mode": "stream (one transform() per edge)"}

    meta_blocks = {}
    for name, (w, cols, extra) in blocks.items():
        sha = w.close()
        meta_blocks[name] = dict(extra, file=f"{name}.npy", columns=cols, dtype=str(w.dtype), output_sha256=sha)
    sees = {"G1": "the later edges of its own batch of 128 (A28), never an edge of a later batch or split (A34); "
                  "instances credit every participant in the batch (A28, A35)",
            "G2": "the later edges of its own batch of 2,048 (A28), never an edge of a later batch or split (A34); "
                  "instances credit every participant in the batch (A28, A35)",
            "G3": "fit rows: every fit edge, re-sent edges double-counted (A29, A36); validation rows: every train "
                  "edge, including the later validation edges (one batch)",
            "F1": "stream visibility (eid' <= eid)", "F2": "stream visibility (eid' <= eid)"}[r]
    meta = {
        "manifest_version": 2,
        "producer": "tools/gfp/dg2_features.py",
        "gfp_recipe": {"id": r, "call": call, "ts_offset": GFP_TS_OFFSET, "params": gp.get_params(),
                       "snapml": version("snapml")},
        "rows": {"first": 0, "last": rows, "note": "row r is eid r"},
        "causal": r in ("F1", "F2"),
        "row_sees": sees,
        "graph": args.graph,
        "blzg_sha256": file_sha256(args.graph),
        "csv_sha256": g.csv_sha256,
        "like": {"file": args.like, "profile_id": like.get("profile", {}).get("id"),
                 "gfp_recipe": like.get("gfp_recipe", {}).get("id")},
        "smoke_limit": args.limit,
        "blocks": meta_blocks,
        "seconds": time.time() - t_start,
        "peak_rss_mb": vm_hwm_mb(),
        "python": platform.python_version(),
        "git": git_info(),
    }
    (tmp / "features.json").write_text(json.dumps(meta, indent=1) + "\n")
    if out.exists():
        shutil.rmtree(out)
    os.rename(tmp, out)
    print(f"{r}: {rows} rows, {', '.join(meta_blocks)} in {meta['seconds']:.1f} s, "
          f"peak RSS {meta['peak_rss_mb']:.0f} MB -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()

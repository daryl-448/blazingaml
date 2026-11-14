"""blz's vertex statistics against GFP's (SPEC v2 §8 S2).

Fuzz mode: `blz fuzz --vstats` writes the fuzz graphs, their amounts (fuzz_amounts) and blz's
statistics (window = the graph's window, ts offset GFP_TS_OFFSET). Each graph is replayed into a
fresh GFP in stream mode (one transform() per edge, ts + GFP_TS_OFFSET) with vertex_stats on and
temp-cycle as the companion pattern (window = the graph's window, bins [2]), as F1 calls it.

HI-Small mode (--blz-dir): blz's vstats.npy (float64) against F1's (features/dg2_R7_F1,
float32, GFP's values clipped to the float32 range); blz's values are rounded to float32 and
clipped the same way first.

The fan and degree columns must be equal. Every other value must be within a relative error of
1e-6 of GFP's, or 1e-9 absolute (§8 S2); NaN only matches NaN. Each mismatch is classified by
whether its side (node, direction) had lost an edge to eviction before the row: GFP keeps
running sums that it adds to and subtracts from (A45), and only those rows can carry its
rounding residue. The report also has a histogram of the mismatches' relative error and the count
of negative variances in GFP's output. The summary goes to the report under "vstats_vs_gfp_fuzz" or
"vstats_vs_gfp_hi_small".

Usage: uv run vstats_vs_gfp.py --seeds 0:2000 [--work ../../data/fuzz_vs]
       uv run vstats_vs_gfp.py --blz-dir ../../features/<run> [--gfp-dir ../../features/dg2_R7_F1]
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from gfp_common import GFP_TS_OFFSET, PATTERN_BLOCKS
from snapml import GraphFeaturePreprocessor

VS_FEATS = [0, 1, 2, 3, 4, 8, 9, 10]
PER_SIDE = 13
SIDES = ["src_out", "src_in", "dst_out", "dst_in"]
STATS = ["fan", "deg", "ratio"] + [f"{c}_{s}" for c in ("ts", "amount_received")
                                   for s in ("avg", "sum", "var", "skew", "kurtosis")]
NAMES = [f"vs_{side}_{s}" for side in SIDES for s in STATS]
INT_COLS = [i for i, n in enumerate(NAMES) if n.endswith("_fan") or n.endswith("_deg")]
VAR_COLS = [i for i, n in enumerate(NAMES) if n.endswith("_var")]
REL_EDGES = [1e-4, 1e-2, 1.0, np.inf]  # upper edges of the mismatches' relative-error histogram
F32_MAX = float(np.finfo(np.float32).max)


def gfp_vstats(window):
    g = GraphFeaturePreprocessor()
    p = {"num_threads": 1, "time_window": -1, "vertex_stats": True, "vertex_stats_tw": int(window),
         "vertex_stats_cols": [3, 4], "vertex_stats_feats": VS_FEATS}
    for pat in PATTERN_BLOCKS:
        p.update({pat: pat == "temp-cycle", pat + "_tw": int(window)})
    p["temp-cycle_bins"] = [2]
    g.set_params(p)
    return g


def evicted_before(src, dst, ts, window):
    """bool[m, 4]: whether side k of row e had an edge of its node and direction evicted before the
    row, i.e. with ts < ts[e - 1] - window. The node's earliest edge in that direction decides,
    since ts is sorted; an edge with eid >= e has ts >= ts[e - 1] and never qualifies."""
    m = len(src)
    lo = np.full(m, np.iinfo(np.int64).min, dtype=np.int64)
    lo[1:] = ts[:-1].astype(np.int64) - window
    n = int(max(src.max(), dst.max())) + 1 if m else 0
    out = np.zeros((m, 4), dtype=bool)
    for k, (nodes, own) in enumerate(((src, src), (src, dst), (dst, src), (dst, dst))):
        first_ts = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
        u, idx = np.unique(own, return_index=True)  # each node's first edge in this direction
        first_ts[u] = ts[idx]
        out[:, k] = first_ts[nodes] < lo
    return out


def compare(blz, gfp, evicted):
    """Counts per column and class; blz, gfp float64 [rows, 52]; evicted bool [rows, 4]."""
    with np.errstate(invalid="ignore", over="ignore"):
        close = (blz == gfp) | (np.isnan(blz) & np.isnan(gfp)) | (
            np.isfinite(blz) & np.isfinite(gfp) & (np.abs(blz - gfp) <= np.maximum(1e-9, 1e-6 * np.abs(gfp))))
    close[:, INT_COLS] = blz[:, INT_COLS] == gfp[:, INT_COLS]  # fan and degree: exact
    bad = ~close
    bad_int = bad[:, INT_COLS].sum()
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        rel = np.abs(blz[bad] - gfp[bad]) / np.abs(gfp[bad])
    rel = np.where(np.isnan(rel), np.inf, rel)
    rel_hist = np.histogram(rel, bins=[0.0] + REL_EDGES)[0]
    side_ev = np.repeat(evicted, PER_SIDE, axis=1)
    cols = {}
    for j, n in enumerate(NAMES):
        if bad[:, j].any():
            cols[n] = {"after_eviction": int((bad[:, j] & side_ev[:, j]).sum()),
                       "no_eviction": int((bad[:, j] & ~side_ev[:, j]).sum())}
    return {"rows": int(blz.shape[0]), "values": int(bad.size), "mismatched_values": int(bad.sum()),
            "mismatched_int_values": int(bad_int),
            "mismatched_after_eviction": int((bad & side_ev).sum()),
            "mismatched_no_eviction": int((bad & ~side_ev).sum()),
            "rows_with_eviction_side": int(evicted.any(axis=1).sum()),
            "rows_mismatched": int(bad.any(axis=1).sum()),
            "mismatch_relative_error": {f"<={e:g}": int(c) for e, c in zip(REL_EDGES, rel_hist)},
            "gfp_negative_variances": int((gfp[:, VAR_COLS] < 0).sum()),
            "by_column": cols}, bad


def fuzz_mode(args, root):
    t0 = time.time()
    subprocess.run([args.blz, "fuzz", "--seeds", args.seeds, "--vstats", "--out", args.work], check=True)
    t_blz = time.time() - t0
    work = Path(args.work)
    graphs = np.load(work / "graphs.npy")  # seed, eid, src, dst, ts, window
    amounts = np.load(work / "amounts.npy").reshape(-1)
    mine = np.load(work / "vstats.npy")
    assert mine.shape == (len(graphs), len(NAMES)), mine.shape
    gfp = np.empty_like(mine)
    evicted = np.zeros((len(graphs), 4), dtype=bool)
    seeds = graphs[:, 0]
    starts = np.flatnonzero(np.r_[True, seeds[1:] != seeds[:-1]])
    ends = np.r_[starts[1:], len(seeds)]
    t1 = time.time()
    for a, z in zip(starts, ends):
        window = int(graphs[a, 5])
        g = gfp_vstats(window)
        for r in range(a, z):
            X = np.array([[graphs[r, 1], graphs[r, 2], graphs[r, 3], graphs[r, 4] + GFP_TS_OFFSET, amounts[r]]],
                         dtype=np.float64)
            gfp[r] = g.transform(X)[0, 6:]  # the input columns, then temp-cycle's one bin
        evicted[a:z] = evicted_before(graphs[a:z, 2], graphs[a:z, 3], graphs[a:z, 4], window)
    t_gfp = time.time() - t1
    summary, bad = compare(mine, gfp, evicted)
    rows_bad = np.flatnonzero(bad.any(axis=1))
    examples = []
    for r in rows_bad[:10]:
        j = int(np.flatnonzero(bad[r])[0])
        examples.append({"seed": int(graphs[r, 0]), "eid": int(graphs[r, 1]), "column": NAMES[j],
                         "gfp": float(gfp[r, j]), "blz": float(mine[r, j])})
    summary.update({"generator": "blz fuzz --vstats (SPEC §13.3 graphs, fuzz_amounts)",
                    "gfp": "snapml==1.17.2, stream mode, temp-cycle companion, fresh instance per graph",
                    "gfp_ts_offset": GFP_TS_OFFSET, "seeds": args.seeds, "graphs": int(len(starts)),
                    "first_mismatches": examples,
                    "seconds": {"blz_fuzz": round(t_blz, 1), "gfp": round(t_gfp, 1)}})
    return "vstats_vs_gfp_fuzz", summary


def hi_small_mode(args, root):
    sys.path.insert(0, str(root / "bench/dg2"))
    from blzg import Blzg  # noqa: E402
    blz_dir, gfp_dir = Path(args.blz_dir), Path(args.gfp_dir)
    meta = json.loads((gfp_dir / "features.json").read_text())
    names = meta["blocks"]["vstats"]["columns"]
    if names != NAMES:
        sys.exit(f"{gfp_dir}: vstats columns differ from {NAMES[:3]}...")
    window = int(meta["blocks"]["vstats"]["window"])
    gfp = np.load(gfp_dir / "vstats.npy", mmap_mode="r")
    mine = np.load(blz_dir / "vstats.npy", mmap_mode="r")
    if mine.shape != gfp.shape:
        sys.exit(f"shapes differ: {mine.shape} vs {gfp.shape}")
    g = Blzg(str(root / args.graph))
    m = mine.shape[0]
    t0 = time.time()
    evicted = evicted_before(g.src[:m], g.dst[:m], g.ts[:m], window)
    tot = None
    step = 1 << 18
    for a in range(0, m, step):
        b = min(m, a + step)
        mine32 = np.clip(np.asarray(mine[a:b]), -F32_MAX, F32_MAX).astype(np.float32).astype(np.float64)
        s, _ = compare(mine32, np.asarray(gfp[a:b]).astype(np.float64), evicted[a:b])
        if tot is None:
            tot = s
        else:
            for k in ("rows", "values", "mismatched_values", "mismatched_int_values", "mismatched_after_eviction",
                      "mismatched_no_eviction", "rows_with_eviction_side", "rows_mismatched", "gfp_negative_variances"):
                tot[k] += s[k]
            for k, v in s["mismatch_relative_error"].items():
                tot["mismatch_relative_error"][k] += v
            for c, v in s["by_column"].items():
                d = tot["by_column"].setdefault(c, {"after_eviction": 0, "no_eviction": 0})
                d["after_eviction"] += v["after_eviction"]
                d["no_eviction"] += v["no_eviction"]
    tot.update({"blz_dir": str(blz_dir), "gfp_dir": str(gfp_dir), "window": window,
                "note": "blz float64 rounded to float32 and clipped to the float32 range, as F1 stored GFP's values",
                "seconds": round(time.time() - t0, 1)})
    return "vstats_vs_gfp_hi_small", tot


def main():
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--blz", default=str(root / "build/rel/blz"))
    ap.add_argument("--seeds", default="0:2000")
    ap.add_argument("--work", default=str(root / "data/fuzz_vs"))
    ap.add_argument("--blz-dir", default=None, help="HI-Small mode: a `blz vstats` output directory")
    ap.add_argument("--gfp-dir", default=str(root / "features/dg2_R7_F1"))
    ap.add_argument("--graph", default="data/hi_small.blzg")
    ap.add_argument("--report", default=str(root / "results/s2_vstats.json"))
    args = ap.parse_args()
    key, summary = hi_small_mode(args, root) if args.blz_dir else fuzz_mode(args, root)
    report = Path(args.report)
    doc = json.loads(report.read_text()) if report.exists() else {}
    doc[key] = summary
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("by_column", "first_mismatches")}, indent=1))
    print(json.dumps(summary["by_column"], indent=1))


if __name__ == "__main__":
    main()

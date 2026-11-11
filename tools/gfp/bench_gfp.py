"""GFP timing baseline (SPEC §11.2) for E4.

Times only transform(), with one pattern group enabled at a time. Each timed run is one complete
stream traversal (every edge sent exactly once, in stream order, in batches) on a fresh, empty GFP
instance; construction and set_params happen outside the timer, and fit() is never called
(SPEC Appendix A, A29). GFP is fed ts + GFP_TS_OFFSET, as in the parity runs (D-12).

The grid is reduced (D-14): every batch size at the highest thread count, which defines
T_GFP_best (the minimum over batch sizes there), with --reps runs each; the other thread counts
use one batch size (--scaling-batch) and --scaling-reps runs. There is no warm-up run: the edges
are in memory before any timer starts, and each run lasts tens of seconds.

Merges {"gfp": ...} into --out (the E4 results file).

Usage: uv run bench_gfp.py --edges ../../data/edges.npy --out ../../results/e4_cpu_local.json
"""

import argparse
import json
import platform
import resource
import statistics
import sys
import time
from pathlib import Path

from importlib.metadata import version

import numpy as np

from gfp_common import GFP_TS_OFFSET, make_preprocessor

# GFP pattern -> (window, bins, our blocks): configs/features_default.yaml (SPEC §4.8).
BINS_FD = [2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 30, 50]
PATTERNS = {
    "fan": (86400, BINS_FD, ["fan_in", "fan_out"]),
    "degree": (86400, BINS_FD, ["deg_in", "deg_out"]),
    "scatter-gather": (21600, [2, 3, 4, 5, 6, 8, 10, 13, 16, 20], ["sg"]),
    "temp-cycle": (86400, [2, 3, 4, 5, 6, 7, 8, 9, 10], ["tcyc"]),
}


def ints(s):
    return [int(x) for x in s.split(",")]


def traverse(edges, pattern, threads, batch):
    tw, bins, _ = PATTERNS[pattern]
    g = make_preprocessor([pattern], tw, bins, num_threads=threads)
    m = edges.shape[0]
    t0 = time.perf_counter()
    for i in range(0, m, batch):
        g.transform(edges[i:i + batch])
    secs = time.perf_counter() - t0
    del g
    return secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", required=True, help="int64[m,4] eid,src,dst,ts (blz export-edges)")
    ap.add_argument("--patterns", default=",".join(PATTERNS))
    ap.add_argument("--threads", default="1,2,4,6,12")
    ap.add_argument("--batch", default="2048,65536", help="batch sizes at the highest thread count")
    ap.add_argument("--reps", type=int, default=3, help="runs per point at the highest thread count")
    ap.add_argument("--scaling-batch", type=int, default=65536, help="batch size at the other thread counts")
    ap.add_argument("--scaling-reps", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="gfp")
    args = ap.parse_args()

    edges = np.load(args.edges).astype(np.float64)
    assert edges.ndim == 2 and edges.shape[1] == 4, edges.shape
    edges[:, 3] += GFP_TS_OFFSET
    threads = sorted(ints(args.threads))
    top = threads[-1]
    grid = [(top, b, args.reps) for b in ints(args.batch)]
    grid += [(t, args.scaling_batch, args.scaling_reps) for t in reversed(threads[:-1])]

    points, best = [], {}
    for pattern in args.patterns.split(","):
        for t, b, reps in grid:
            secs = [traverse(edges, pattern, t, b) for _ in range(reps)]
            p = {"pattern": pattern, "threads": t, "batch": b, "seconds": secs,
                 "median": statistics.median(secs), "min": min(secs), "max": max(secs)}
            points.append(p)
            print(f"{pattern:15s} {t:2d} threads batch {b:6d}: median {p['median']:8.2f} s "
                  f"(min {p['min']:.2f}, max {p['max']:.2f})", file=sys.stderr, flush=True)
        top_pts = [p for p in points if p["pattern"] == pattern and p["threads"] == top]
        b = min(top_pts, key=lambda p: p["median"])
        best[pattern] = {"seconds": b["median"], "batch": b["batch"], "threads": top,
                         "blocks": PATTERNS[pattern][2]}

    res = {
        "snapml": version("snapml"),
        "python": platform.python_version(),
        "edges": int(edges.shape[0]),
        "ts_offset": GFP_TS_OFFSET,
        "params": {k: {"window": v[0], "bins": v[1], "blocks": v[2]} for k, v in PATTERNS.items()},
        "grid": "every --batch at the highest thread count (--reps runs), --scaling-batch elsewhere "
                "(--scaling-reps runs); no warm-up (D-14)",
        "reps": args.reps,
        "scaling_reps": args.scaling_reps,
        "points": points,
        "best": best,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "caveat": "GFP batch mode credits every edge of an instance and lets edges see later edges "
                  "of the same batch (SPEC Appendix A, A28): the timings compare equal work, not "
                  "identical outputs (SPEC §11.2).",
    }
    out = Path(args.out)
    doc = json.loads(out.read_text()) if out.exists() else {}
    doc[args.name] = res
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {out} (/{args.name})", file=sys.stderr)


if __name__ == "__main__":
    main()

"""GFP parity run in stream mode (SPEC §11.1).

One fresh GFP instance with one pattern enabled, num_threads=1, vertex_stats=False; transform()
is called once per edge in stream order. Writes <out>/<block>.npy (uint32[m, bins]) per output
block of the pattern, the same layout as `blz mine`.

Usage: uv run parity.py --edges data/edges.npy --block tcyc --tw 86400 --bins 2..33 --out data/gfp/
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from gfp_common import PATTERN_BLOCKS, make_preprocessor

PATTERNS = {"fan": "fan", "degree": "degree", "sg": "scatter-gather", "tcyc": "temp-cycle"}


def parse_bins(s):
    if ".." in s:
        lo, hi = s.split("..")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", required=True, help="int64[m,4] eid,src,dst,ts (blz export-edges)")
    ap.add_argument("--block", required=True, choices=list(PATTERNS))
    ap.add_argument("--tw", required=True, type=int)
    ap.add_argument("--bins", default="2..33")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--limit", type=int, default=None, help="only the first N edges (testing)")
    args = ap.parse_args()

    edges = np.load(args.edges, mmap_mode="r")
    assert edges.ndim == 2 and edges.shape[1] == 4, edges.shape
    m = edges.shape[0] if args.limit is None else min(args.limit, edges.shape[0])
    assert (np.asarray(edges[:m, 0]) == np.arange(m)).all(), "edges must be in eid order"
    bins = parse_bins(args.bins)
    pattern = PATTERNS[args.block]
    blocks = PATTERN_BLOCKS[pattern]
    nb = len(bins)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    outs = [np.lib.format.open_memmap(out_dir / f"{b}.npy", mode="w+", dtype=np.uint32,
                                      shape=(m, nb)) for b in blocks]

    g = make_preprocessor([pattern], args.tw, bins, num_threads=1)
    row = np.empty((1, 4), dtype=np.float64)
    t0 = last = time.time()
    for i in range(m):
        row[0] = edges[i]
        feats = g.transform(row)[0, 4:]
        for k, o in enumerate(outs):
            o[i] = feats[k * nb:(k + 1) * nb]
        if time.time() - last > 30:
            last = time.time()
            rate = (i + 1) / (last - t0)
            print(f"{i + 1}/{m} edges, {rate:.0f}/s, eta {(m - i - 1) / rate / 60:.1f} min",
                  file=sys.stderr, flush=True)
    for o in outs:
        o.flush()
    print(f"{m} edges in {time.time() - t0:.1f} s -> {', '.join(blocks)} in {out_dir}")


if __name__ == "__main__":
    main()

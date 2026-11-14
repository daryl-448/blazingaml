"""Fuzz the oracle against GFP in stream mode (SPEC §11.3), or in batch mode (SPEC v2 §8 S1).

`blz fuzz` generates the SPEC §13.3 graphs and the oracle's features for every v1 block (bins
2..33, one window per graph, tcyc uncapped). This script replays each graph into a fresh GFP, one
transform() per edge with ts + GFP_TS_OFFSET (D-12), and diffs every row. The summary is merged
into the report JSON under "fuzz_vs_gfp".

With --batch B, `blz fuzz --batch B` computes the batch oracle's features (batches of B edges, no
cuts, time_window = the graph's window) and GFP gets one transform() per batch of B edges. The
summary goes under "fuzz_vs_gfp_batch<B>" (default report: results/s1_batch.json).

Usage: uv run fuzz_vs_gfp.py --blz ../../build/rel/blz --seeds 0:2000 --work ../../data/fuzz [--batch B]
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from gfp_common import GFP_TS_OFFSET, edge_row, enabled_blocks, make_preprocessor, split_blocks

PATTERNS = ["fan", "degree", "scatter-gather", "temp-cycle"]
BINS = list(range(2, 34))


def main():
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--blz", default=str(root / "build/rel/blz"))
    ap.add_argument("--seeds", default="0:2000", help="a:b")
    ap.add_argument("--work", default=str(root / "data/fuzz"))
    ap.add_argument("--report", default=None, help="default: results/e1_parity.json, or results/s1_batch.json with --batch")
    ap.add_argument("--batch", type=int, default=0, help="GFP batch mode with transform() batches of this many edges")
    args = ap.parse_args()
    report_path = args.report or str(root / ("results/s1_batch.json" if args.batch else "results/e1_parity.json"))

    t0 = time.time()
    subprocess.run([args.blz, "fuzz", "--seeds", args.seeds, "--bins", "2..33", "--batch", str(args.batch),
                    "--out", args.work], check=True)
    t_oracle = time.time() - t0
    work = Path(args.work)
    graphs = np.load(work / "graphs.npy")  # seed, eid, src, dst, ts, window
    blocks = enabled_blocks(PATTERNS)
    oracle = {b: np.load(work / f"{b}.npy") for b in blocks}

    seeds = graphs[:, 0]
    starts = np.flatnonzero(np.r_[True, seeds[1:] != seeds[:-1]])
    ends = np.r_[starts[1:], len(seeds)]
    mismatches = {b: 0 for b in blocks}
    bad_seeds, examples = set(), []
    t1 = time.time()
    for a, z in zip(starts, ends):
        seed, window = int(graphs[a, 0]), int(graphs[a, 5])
        g = make_preprocessor(PATTERNS, window, BINS)
        step = args.batch or 1
        for c in range(a, z, step):
            X = graphs[c:min(c + step, z), 1:5].astype(np.float64)  # eid, src, dst, ts
            X[:, 3] += GFP_TS_OFFSET
            out = g.transform(X)
            for k, r in enumerate(range(c, min(c + step, z))):
                eid = int(graphs[r, 1])
                got = split_blocks(out[k], PATTERNS, len(BINS))
                for b in blocks:
                    want = [int(x) for x in oracle[b][r]]
                    if got[b] != want:
                        mismatches[b] += 1
                        bad_seeds.add(seed)
                        if len(examples) < 20:
                            examples.append({"seed": seed, "eid": eid, "block": b,
                                             "gfp": got[b], "oracle": want})
    t_gfp = time.time() - t1

    summary = {
        "generator": "blz fuzz (SPEC §13.3), oracle bins 2..33, tcyc uncapped"
                     + (f", batch oracle with batches of {args.batch} (SPEC v2 §8 S1)" if args.batch else ""),
        "gfp": "snapml==1.17.2, " + (f"one transform() per batch of {args.batch} edges" if args.batch
                                     else "stream mode") + ", fresh instance per graph",
        "gfp_ts_offset": GFP_TS_OFFSET,
        "seeds": args.seeds,
        "graphs": int(len(starts)),
        "edges": int(len(graphs)),
        "mismatched_rows_by_block": mismatches,
        "mismatched_seeds": sorted(bad_seeds)[:100],
        "first_mismatches": examples,
        "seconds": {"blz_fuzz": round(t_oracle, 1), "gfp": round(t_gfp, 1)},
    }
    report = Path(report_path)
    doc = json.loads(report.read_text()) if report.exists() else {}
    doc["fuzz_vs_gfp" + (f"_batch{args.batch}" if args.batch else "")] = summary
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("graphs", "edges", "mismatched_rows_by_block",
                                              "seconds")}, indent=1))
    if bad_seeds:
        print(f"{len(bad_seeds)} graphs differ; first: {examples[0]}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

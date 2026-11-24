"""Fuzz the oracle against GFP in stream mode (SPEC §11.3).

`blz fuzz` generates the SPEC §13.3 graphs and the oracle's features for every v1 block (bins
2..33, one window per graph, tcyc uncapped). This script replays each graph into a fresh GFP, one
transform() per edge with ts + GFP_TS_OFFSET (D-12), and diffs every row. The summary is merged
into the report JSON under "fuzz_vs_gfp".

Usage: uv run fuzz_vs_gfp.py --blz ../../build/rel/blz --seeds 0:2000 --work ../../data/fuzz
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
    ap.add_argument("--report", default=str(root / "results/e1_parity.json"))
    args = ap.parse_args()

    t0 = time.time()
    subprocess.run([args.blz, "fuzz", "--seeds", args.seeds, "--bins", "2..33", "--out", args.work], check=True)
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
        for r in range(a, z):
            eid, src, dst, ts = (int(x) for x in graphs[r, 1:5])
            got = split_blocks(g.transform(edge_row(eid, src, dst, ts + GFP_TS_OFFSET))[0], PATTERNS, len(BINS))
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
        "generator": "blz fuzz (SPEC §13.3), oracle bins 2..33, tcyc uncapped",
        "gfp": "snapml==1.17.2, stream mode, fresh instance per graph",
        "gfp_ts_offset": GFP_TS_OFFSET,
        "seeds": args.seeds,
        "graphs": int(len(starts)),
        "edges": int(len(graphs)),
        "mismatched_rows_by_block": mismatches,
        "mismatched_seeds": sorted(bad_seeds)[:100],
        "first_mismatches": examples,
        "seconds": {"blz_fuzz": round(t_oracle, 1), "gfp": round(t_gfp, 1)},
    }
    report = Path(args.report)
    doc = json.loads(report.read_text()) if report.exists() else {}
    doc["fuzz_vs_gfp"] = summary
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("graphs", "edges", "mismatched_rows_by_block",
                                              "seconds")}, indent=1))
    if bad_seeds:
        print(f"{len(bad_seeds)} graphs differ; first: {examples[0]}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

"""GFP structural eligibility for E9 (SPEC v2 §5): a block that the frozen recipe takes from GFP
has its eligibility computed by that same GFP recipe on the attempt's edges alone. Never used for
parity.

For each complete attempt of results/gt_join.json: a fresh preprocessor with the recipe's
parameters, copied from the features' manifest (gfp_recipe.params: windows, gfp_bins() bins,
vertex_stats off, time_window -1), ts + GFP_TS_OFFSET (D-12), and the original eids and node ids.
The call pattern is the recipe's, with each edge in the batch it had in the full stream [D-33]:
G1 cuts the stream at b_fv and at the train/test boundary and batches 128 consecutive eids from
each cut, so the attempt's edges are sent in groups of equal full-stream batch, in eid order.

Self-check, every run: the same code on the stream prefix [0, CHECK_EDGES), which is whole
batches inside the fit split, must give the features' rows for those eids exactly (A34: a batch
never sees a later batch).

Output (--out, JSON): producer, gfp_recipe (the manifest's, verbatim), call, gt_join_sha256,
features, blocks (GFP column order), the self-check, and attempts: [{id, rows}], rows = the
number of the attempt's edges with a non-zero row, per block.

Usage: .venv/bin/python coverage_gfp.py --graph ../../data/hi_small.blzg --gt ../../results/gt_join.json
           --features ../../features/dg2_R6_G1 --out ../../features/e9/gfp_eligibility_G1.json
"""

import argparse
import hashlib
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
from snapml import GraphFeaturePreprocessor

from gfp_common import GFP_TS_OFFSET, PATTERN_BLOCKS, gfp_bins

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench" / "dg2"))
from blzg import Blzg  # noqa: E402

CHECK_EDGES = 4096  # 32 batches of 128, inside the fit split


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


class Recipe:
    """A batch recipe of tools/gfp/dg2_features.py (G1, G2), as its manifest records it."""

    def __init__(self, fm, g):
        rec = fm["gfp_recipe"]
        call = rec["call"]
        if call.get("mode") != "batches":
            sys.exit(f"recipe {rec['id']}: only the batch recipes (G1, G2) are supported, not {call}")
        if rec["snapml"] != version("snapml"):
            sys.exit(f"recipe {rec['id']} ran on snapml {rec['snapml']}, this is {version('snapml')}")
        if rec["ts_offset"] != GFP_TS_OFFSET:
            sys.exit(f"recipe {rec['id']}: ts_offset {rec['ts_offset']}")
        sp = g.splits()
        b_fv, train = sp["fit"][1], sp["train"][1]
        if call["cuts"] != [b_fv, train]:
            sys.exit(f"recipe {rec['id']}: cuts {call['cuts']}, not [{b_fv}, {train}]")
        self.id, self.params, self.batch, self.call = rec["id"], rec["params"], call["batch"], call
        self.segments = [(0, b_fv), (b_fv, train), (train, g.m)]
        self.patterns = [p for p in PATTERN_BLOCKS if self.params.get(p)]
        self.blocks, self.widths = [], []
        for p in self.patterns:
            for b in PATTERN_BLOCKS[p]:
                fb = fm["blocks"][b]
                if gfp_bins(fb["bins"]) != self.params[p + "_bins"] or fb["window"] != self.params[p + "_tw"]:
                    sys.exit(f"block {b}: the manifest's window and bins are not the recipe's (D-32)")
                self.blocks.append(b)
                self.widths.append(len(fb["bins"]))

    def batch_of(self, e):
        """The first eid of e's batch in the full stream."""
        for a, b in self.segments:
            if a <= e < b:
                return a + (e - a) // self.batch * self.batch
        raise ValueError(f"eid {e} outside the stream")

    def rows(self, g, eids):
        """GFP's rows (one array per block) for the stream that holds only `eids` (ascending)."""
        eids = np.asarray(eids, dtype=np.int64)
        assert np.all(np.diff(eids) > 0), "eids must be strictly ascending"
        X = np.empty((len(eids), 4), dtype=np.float64)
        X[:, 0] = eids
        X[:, 1] = g.src[eids]
        X[:, 2] = g.dst[eids]
        X[:, 3] = g.ts[eids].astype(np.float64) + GFP_TS_OFFSET
        gp = GraphFeaturePreprocessor()
        gp.set_params(self.params)
        starts = np.array([self.batch_of(int(e)) for e in eids])
        cut = np.flatnonzero(np.diff(starts)) + 1
        feats = np.concatenate([gp.transform(X[i:j])[:, 4:] for i, j in zip(np.r_[0, cut], np.r_[cut, len(eids)])])
        out, k = [], 0
        for w in self.widths:
            out.append(feats[:, k:k + w])
            k += w
        if k != feats.shape[1]:
            raise RuntimeError(f"GFP returned {feats.shape[1]} feature columns, expected {k}")
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--features", required=True, help="a GFP batch pass of tools/gfp/dg2_features.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    g = Blzg(args.graph)
    fdir = Path(args.features)
    fm = json.loads((fdir / "features.json").read_text())
    if fm.get("producer") != "tools/gfp/dg2_features.py":
        sys.exit(f"{fdir}: not written by tools/gfp/dg2_features.py")
    if fm["blzg_sha256"] != file_sha256(args.graph):
        sys.exit(f"{fdir}: mined from another .blzg")
    rec = Recipe(fm, g)

    # Self-check: the prefix [0, CHECK_EDGES) reproduces the full pass's rows.
    prefix = rec.rows(g, np.arange(CHECK_EDGES))
    for b, got in zip(rec.blocks, prefix):
        full = np.load(fdir / fm["blocks"][b]["file"], mmap_mode="r")[:CHECK_EDGES]
        if not np.array_equal(got, full):
            sys.exit(f"self-check: block {b} on the prefix [0, {CHECK_EDGES}) differs from {fdir}'s rows")
    check = {"prefix_edges": CHECK_EDGES, "equals_full_pass": True,
             "nonzero_rows": {b: int((x != 0).any(axis=1).sum()) for b, x in zip(rec.blocks, prefix)}}

    gt = json.loads(Path(args.gt).read_text())
    attempts = []
    for a in gt["attempts"]:
        if a["status"] != "complete":
            continue
        eids = sorted(r["eid"] for r in a["rows"] if r["status"] in ("unique", "resolved-duplicate"))
        rows = rec.rows(g, eids)
        attempts.append({"id": a["id"], "rows": [int((x != 0).any(axis=1).sum()) for x in rows]})
    doc = {"producer": "tools/gfp/coverage_gfp.py", "spec": "SPEC v2 §5, D-33",
           "gfp_recipe": fm["gfp_recipe"],
           "call": {"graph": "the attempt's edges alone, eids, node ids and ts + 1 kept",
                    "batches": f"each edge in its full-stream batch ({rec.batch} eids from each cut "
                               f"{rec.call['cuts']}), groups sent in eid order to a fresh preprocessor"},
           "features": str(fdir), "gt_join": args.gt, "gt_join_sha256": file_sha256(args.gt),
           "blocks": rec.blocks, "self_check": check, "attempts": attempts,
           "seconds": round(time.time() - t0, 2), "snapml": version("snapml")}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n")
    elig = {b: sum(1 for a in attempts if a["rows"][i] > 0) for i, b in enumerate(rec.blocks)}
    print(f"{rec.id}: {len(attempts)} attempts; eligible per block {elig}; self-check ok "
          f"({doc['seconds']} s) -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

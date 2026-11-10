"""Re-run every SPEC Appendix A probe against snapml GFP and write the golden JSON (SPEC §11.4).

Each probe asserts the facts that its Appendix A row states (edge, block -> {size: count}; an
empty dict means "no instance"). Every probe except A28, A29 and A33 (GFP behaviours the C++
code does not reproduce) is also written, with GFP's full
per-edge output, to tests/golden/gfp_probes.json for the C++ tests.

Usage: uv run probes.py [--out ../../tests/golden/gfp_probes.json] [--check-only]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from gfp_common import PATTERN_BLOCKS, edge_row, make_preprocessor, split_blocks

BINS = list(range(1, 9))
LC_LEN = 8
ALL_PATTERNS = list(PATTERN_BLOCKS)
V1_BLOCKS = ["fan_in", "fan_out", "deg_in", "deg_out", "sg", "tcyc"]
NOTHING = {b: {} for b in V1_BLOCKS}


def hist(sizes):
    """{size: count} -> GFP bin counts for BINS (last bin open)."""
    out = [0] * len(BINS)
    for size, count in sizes.items():
        idx = max(i for i, b in enumerate(BINS) if b <= size)
        out[idx] += count
    return out


def E(i, blocks=None, **kw):
    """Expectation for edge i: block -> {size: count}."""
    d = dict(blocks or {})
    d.update(kw)
    return (i, d)


def edges(spec):
    """'1>9@1 2>9@2' -> [(1, 9, 1), (2, 9, 2)]"""
    out = []
    for tok in spec.split():
        sd, t = tok.split("@")
        s, d = sd.split(">")
        out.append((int(s), int(d), int(t)))
    return out


A1 = "1>9@1 2>9@2 1>9@3 3>9@4 4>9@50"
A5 = "0>1@1 0>2@2 0>3@3 1>5@4 2>5@5 3>5@6"
FANDEG_EMPTY = {"fan_in": {}, "fan_out": {}, "deg_in": {}, "deg_out": {}}

# (id, edges, tw, [expectations])
CASES = [
    ("A1", A1, 100, [
        E(0, FANDEG_EMPTY),
        E(1, FANDEG_EMPTY, fan_in={2: 1}, deg_in={2: 1}),
        E(2, FANDEG_EMPTY, fan_in={2: 1}, deg_in={3: 1}, deg_out={2: 1}),
        E(3, FANDEG_EMPTY, fan_in={3: 1}, deg_in={4: 1}),
        E(4, FANDEG_EMPTY, fan_in={4: 1}, deg_in={5: 1}),
    ]),
    ("A2", A1, 3, [E(3, deg_in={3: 1}), E(4, NOTHING)]),
    *[(f"A3/tw={tw}", "1>9@0 2>9@10", tw, [E(1, fan_in={2: 1} if tw == 11 else {})])
      for tw in (9, 10, 11)],
    ("A4", "1>9@5 2>9@5 3>9@5", 100, [E(1, fan_in={2: 1}), E(2, fan_in={3: 1})]),
    ("A5", A5, 100, [E(i, sg={}) for i in range(4)] + [E(4, sg={2: 1}), E(5, sg={3: 1})]),
    ("A6", "1>5@1 2>5@2 0>1@3 0>2@4", 100,
     [E(i, sg={}) for i in range(3)] + [E(3, sg={2: 1})]),
    *[(f"A7/tw={tw}", "0>1@0 0>2@5 1>5@6 2>5@10", tw, [E(3, sg={2: 1} if tw >= 10 else {})])
      for tw in (9, 10, 11)],
    ("A8", "0>1@1 0>1@2 1>5@3 2>5@4 0>2@5 1>5@6", 100,
     [E(i, sg={}) for i in range(4)] + [E(4, sg={2: 1}), E(5, sg={2: 1})]),
    ("A9", "0>1@1 0>2@2 1>0@3 2>0@4", 100,
     [E(0, sg={}, tcyc={}), E(1, sg={}, tcyc={}), E(2, sg={}, tcyc={2: 1}),
      E(3, sg={}, tcyc={2: 1})]),
    ("A10", "0>0@1 0>1@2 0>5@3 1>5@4", 100,
     [E(i, sg={}) for i in range(3)] + [E(3, sg={2: 1})]),
    ("A11", "0>1@2 0>5@3 1>5@4", 100, [E(i, sg={}) for i in range(3)]),
    ("A12", "5>5@1 0>1@2 0>5@3 1>5@4", 100, [E(i, sg={}) for i in range(4)]),
    ("A13", "9>1@1 9>2@2 3>9@3 4>9@4 9>9@5", 100, [E(4, NOTHING)]),
    ("A14", "0>1@1 1>5@2 0>0@3 0>5@4 0>0@6", 100,
     [E(3, fan_in={2: 1}, fan_out={2: 1}, deg_in={2: 1}, deg_out={2: 1}, sg={})]),
    ("A15", "0>1@5 0>2@5 1>5@5 2>5@5", 100,
     [E(i, sg={}) for i in range(3)] + [E(3, sg={2: 1})]),
    ("A16", "1>5@1 2>5@2 1>6@3 2>6@4 0>1@5 0>2@6", 100,
     [E(i, sg={}) for i in range(5)] + [E(5, sg={2: 2})]),
    ("A17", "0>1@1 0>2@2 3>1@3 3>2@4 1>5@5 2>5@6", 100,
     [E(i, sg={}) for i in range(5)] + [E(5, sg={2: 2})]),
    ("A18", "10>11@1 11>12@2 12>10@3", 100,
     [E(0, tcyc={}), E(1, tcyc={}), E(2, tcyc={3: 1}, lcyc={3: 1})]),
    *[(f"A19/ts={a},{b},{c}", f"10>11@{a} 11>12@{b} 12>10@{c}", 100,
       [E(i, tcyc={}) for i in range(3)])
      for a, b, c in ((1, 1, 2), (1, 2, 2), (1, 1, 1))],
    *[(f"A20/tw={tw}", "10>11@0 11>12@5 12>10@10", tw, [E(2, tcyc={3: 1} if tw >= 10 else {})])
      for tw in (9, 10, 11)],
    ("A21", "10>11@1 11>10@2", 100, [E(0, tcyc={}), E(1, tcyc={2: 1})]),
    ("A22", "10>10@1 10>11@2", 100, [E(0, NOTHING), E(1, NOTHING)]),
    ("A23", "10>11@1 11>12@2 10>11@3 11>12@4 12>10@5", 100, [E(4, tcyc={3: 1})]),
    ("A24", "10>11@1 10>13@1 11>12@2 13>12@2 12>10@3", 100, [E(4, tcyc={3: 2})]),
    ("A25", "10>11@1 11>12@3 10>11@4 12>10@5", 100, [E(3, tcyc={3: 1})]),
    ("A26", "30>31@1 31>32@2 32>33@3 33>34@4 34>35@5 35>30@6", 100, [E(5, tcyc={6: 1})]),
    ("A27", "40>41@1 41>40@2 40>42@3 42>40@4", 100,
     [E(0, tcyc={}), E(1, tcyc={2: 1}), E(2, tcyc={}), E(3, tcyc={2: 1})]),
    ("A30", "5>5@1 0>5@2 1>5@3 0>1@4", 100,
     [E(i, sg={}) for i in range(3)] + [E(3, sg={2: 1})]),
    ("A31", "0>5@2 1>5@3 0>1@4", 100, [E(i, sg={}) for i in range(3)]),
    ("A32", "0>0@1 0>5@2 1>5@3 0>1@4", 100, [E(i, sg={}) for i in range(4)]),
]


def fresh():
    return make_preprocessor(ALL_PATTERNS, 100, BINS, lc_len=LC_LEN)


def run_stream(es, tw):
    g = make_preprocessor(ALL_PATTERNS, tw, BINS, lc_len=LC_LEN)
    return [split_blocks(g.transform(edge_row(i, s, d, t))[0], ALL_PATTERNS, len(BINS))
            for i, (s, d, t) in enumerate(es)]


def run_batch(g, es):
    X = np.array([[i, s, d, t] for i, (s, d, t) in enumerate(es)], dtype=np.float64)
    return [split_blocks(row, ALL_PATTERNS, len(BINS)) for row in g.transform(X)]


def check(case_id, out, expects, failures):
    for i, blocks in expects:
        for block, sizes in blocks.items():
            want, got = hist(sizes), out[i][block]
            if want != got:
                failures.append(f"{case_id} edge {i} {block}: want {want} got {got}")


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--out", default=str(here.parents[1] / "tests/golden/gfp_probes.json"))
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    failures, golden = [], []
    for case_id, spec, tw, expects in CASES:
        es = edges(spec)
        # Streams are in ts order (SPEC §3.3); the C++ consumers rely on it.
        assert [t for _, _, t in es] == sorted(t for _, _, t in es), case_id
        out = run_stream(es, tw)
        check(case_id, out, expects, failures)
        golden.append({"id": case_id, "edges": [list(e) for e in es], "tw": tw, "output": out})

    # A28: batch mode credits every participating edge and lets edges see later batch edges.
    check("A28/A5", run_batch(fresh(), edges(A5)), [E(i, sg={3: 1}) for i in range(6)], failures)
    check("A28/A1", run_batch(fresh(), edges(A1)), [E(0, fan_in={2: 2, 3: 1, 4: 1})], failures)

    # A29: snapml >= 1.14.1 double-counts edges re-sent to transform(). Asserts the bug is present
    # as the ledger row records it: row 0's sg and tcyc counts are exactly doubled versus a fresh
    # transform(), and the exported graph still holds the 8 edges. (Appendix A does not record
    # the original 8-edge fixture, so this one is new.)
    es = edges("0>1@1 0>2@2 0>3@3 1>5@4 2>5@5 3>5@6 5>0@7 1>0@8")
    clean = run_batch(fresh(), es)
    g = fresh()
    X = np.array([[i, s, d, t] for i, (s, d, t) in enumerate(es)], dtype=np.float64)
    g.fit(X)
    resent = [split_blocks(row, ALL_PATTERNS, len(BINS)) for row in g.transform(X)]
    for b in ("sg", "tcyc"):
        r, c = resent[0][b], clean[0][b]
        if not any(c) or r != [2 * x for x in c]:
            failures.append(f"A29: row 0 {b} after fit+transform {r} is not 2x fresh {c}")
    if g.__getstate__()[0].shape[0] != len(es):  # exported graph: one row per stored edge
        failures.append("A29: exported graph does not hold exactly the 8 edges")

    # A33: snapml drops every temporal cycle through a node other than the trigger's source that
    # has an in-window self-loop at exactly ts 0. The same streams shifted by +1, or with the loop
    # at ts 1, keep the cycle. Asserts the artifact as the ledger row records it.
    for spec, tw, want in [
        ("13>9@3 9>13@4", 100, {2: 1}),
        ("13>13@0 13>9@3 9>13@4", 100, {}),          # loop at v
        ("13>13@1 13>9@4 9>13@5", 100, {2: 1}),      # the same, shifted by +1
        ("13>13@1 13>9@3 9>13@4", 100, {2: 1}),      # loop at ts 1
        ("13>13@0 13>9@3 9>13@4", 3, {2: 1}),        # loop outside the window
        ("13>13@0 13>9@3 9>13@4", 4, {}),            # closed lower bound reaches ts 0
        ("9>9@0 13>9@3 9>13@4", 100, {2: 1}),        # loop at u: no effect
        ("11>11@0 10>11@1 11>12@2 12>10@3", 100, {}),  # loop at an intermediate
        ("11>11@0 10>11@1 10>13@1 11>12@2 13>12@2 12>10@3", 100, {3: 1}),  # other cycle kept
        ("5>5@0 13>5@1 13>9@3 9>13@4", 100, {2: 1}),  # loop off the cycle: no effect
    ]:
        es = edges(spec)
        check(f"A33/{spec}/tw={tw}", run_stream(es, tw), [E(len(es) - 1, tcyc=want)], failures)

    for f in failures:
        print("FAIL", f, file=sys.stderr)
    print(f"{len(CASES)} stream probes + A28 + A29 + A33: {len(failures)} failures")
    if failures:
        sys.exit(1)
    if not args.check_only:
        doc = {
            "generator": "tools/gfp/probes.py",
            "gfp": "snapml==1.17.2, stream mode (one transform() per edge, fresh instance)",
            "params": {"num_threads": 1, "vertex_stats": False, "bins": BINS,
                       "lc_cycle_len": LC_LEN, "patterns": ALL_PATTERNS},
            "blocks": [b for p in ALL_PATTERNS for b in PATTERN_BLOCKS[p]],
            "note": "edges are [src, dst, ts] with eid = index; tw applies to every pattern; "
                    "output[i][block] is the per-bin count vector of edge i.",
            "cases": golden,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        cases = doc.pop("cases")
        head = json.dumps(doc, indent=1)[:-2]  # one case per line keeps diffs readable
        body = ",\n".join("  " + json.dumps(c, separators=(",", ":")) for c in cases)
        Path(args.out).write_text(f'{head},\n "cases": [\n{body}\n ]\n}}\n')
        print("wrote", args.out)


if __name__ == "__main__":
    main()

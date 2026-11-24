"""Re-run every SPEC Appendix A probe against snapml GFP and write the golden JSON (SPEC §11.4).

Each probe asserts the facts that its Appendix A row states (edge, block -> {size: count}; an
empty dict means "no instance"). Every probe except A28, A29 and A33 (GFP behaviours the C++
code does not reproduce) is also written, with GFP's full
per-edge output, to tests/golden/gfp_probes.json for the C++ tests.

The vertex-statistics probes A43-A45 (S2) assert GFP's window, formulas and guard against the S2
rule (vs_model) and each rejected alternative, and A45 its running-accumulator drift after
eviction. GFP's rows go to tests/golden/gfp_vstats_probes.json (test_vstats.cpp).

The lc-cycle probes A46-A49 (S3) assert GFP's window, length bound, cycle shape and tie rule
against the S3 rule (lcyc_vs_gfp.lc_model) and each rejected alternative. GFP's rows and the
rule's without ties (blz's lcyc) go to tests/golden/gfp_lcyc_probes.json (test_lcyc.cpp).

Usage: uv run probes.py [--out ../../tests/golden/gfp_probes.json]
                        [--vstats-out ../../tests/golden/gfp_vstats_probes.json]
                        [--lcyc-out ../../tests/golden/gfp_lcyc_probes.json] [--check-only]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from gfp_common import PATTERN_BLOCKS, edge_row, gfp_bins, make_preprocessor, split_blocks
from lcyc_vs_gfp import BINS as LCYC_BINS, LC_LEN as LCYC_LEN, binned, gfp_lc, lc_model, run_gfp
from vstats_vs_gfp import INT_COLS as VS_INT_COLS, NAMES as VS_NAMES, gfp_vstats

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


def bin_rule_probe(failures):
    """A38: GFP's bin rule. Edge i of a fan-in star (distinct sources, one destination, stream
    mode) has one fan_in instance of size i + 1. GFP puts size S in the first bin i with
    S <= b_i, or in the last bin, not in [b_i, b_{i+1}) as its docstring says; given
    gfp_bins(bins), it bins as SPEC §4.2. Returns {bins given: [bin index per size 2..20]}."""
    bins = [2, 3, 4, 5, 7, 9, 12, 15]  # the start of D-9's fan bins
    es = [(100 + i, 9, i + 1) for i in range(20)]
    sizes = range(2, 21)
    first_le = [next((i for i, b in enumerate(bins) if s <= b), len(bins) - 1) for s in sizes]
    spec = [max(i for i, b in enumerate(bins) if b <= s) for s in sizes]
    got = {}
    for given, expect in ((bins, first_le), (gfp_bins(bins), spec)):
        g = make_preprocessor(["fan"], 1000, given)
        rows = [split_blocks(g.transform(edge_row(i, s, d, t))[0], ["fan"], len(bins))["fan_in"]
                for i, (s, d, t) in enumerate(es)]
        idx = [[j for j, c in enumerate(r) if c] for r in rows[1:]]  # edge i: size i + 1
        flat = [x[0] if len(x) == 1 else x for x in idx]
        got[str(given)] = flat
        if flat != expect or any(sum(r) != 1 for r in rows[1:]):
            failures.append(f"A38: bins {given}: bin per size 2..20 {flat}, want {expect}")
    return got


# S2 (SPEC v2 §8 S2, Appendix A' A43-A45): GFP's vertex statistics in stream mode, F1's call
# (vstats_vs_gfp.gfp_vstats: amount as the second statistics column, temp-cycle as the companion).
# vs_model() is the S2 addendum's rule; its keyword arguments are the rejected alternatives:
#   lower=trigger: the window is [t_e - W, ...], not GFP's graph after eviction at ts[e - 1] - W;
#   closed=False: an open lower bound; W=<x>: vertex_stats_tw, or another pattern's window, as W;
#   loops=False: self-loops left out; ratio=fan/deg; ddof=1: sample variance; excess: kurtosis - 3;
#   guard=False: no cancellation guard.
# Each fixture: (probe, rejected alternatives, edges src>dst@ts:amount, the rule's W, GFP params).
S2_FIXTURES = [
    ("A43", ["lower=trigger", "W=vertex_stats_tw"], "1>9@1:2 2>9@5:3 3>9@20:5 4>9@40:1", 10,
     {"vertex_stats_tw": 1000}),
    ("A43", ["closed=False"], "1>9@1:2 2>9@11:3 3>9@21:5", 10, {}),
    ("A43", ["W=temp-cycle_tw"], "1>9@1:2 2>9@5:3 3>9@20:5 4>9@40:1", 100,
     {"temp-cycle_tw": 10, "fan": True, "fan_tw": 100, "fan_bins": [2]}),
    ("A43", ["loops=False"], "9>9@1:5 1>9@2:3 9>1@3:4", 100, {}),
    ("A44", ["ratio=fan/deg", "ddof=1", "excess"], "1>9@1:1 2>9@2:2 1>9@3:6", 100, {}),
    ("A44", ["guard=False"], "1>9@1:1000 2>9@2:1006", 100, {}),
    ("A44", [], "1>9@1:1000 2>9@2:1007", 100, {}),
    ("A44", ["guard=False"], "1>9@1000001:2 2>9@1000006:3", 100, {}),
]
# A43 (split): a vertex_stats_tw below W shrinks the window of the value statistics (sum, avg,
# var, ...), with the same rule (ts >= ts[e - 1] - vertex_stats_tw), but not of fan and degree,
# which stay GFP's graph at W. The S2 rule needs vertex_stats_tw >= W (F1: both 86,400).
S2_SPLIT = ("A43", "1>9@1:2 2>9@5:3 3>9@20:5 4>9@40:1", 10, 1)
# A45: GFP's running accumulators after eviction. In row 3 node 1's out-edges hold the single
# amount 4.0 (eids 0 and 1 are evicted at 42 - 15), whose variance is 0; GFP reports a negative one.
S2_DRIFT = [("A45", "1>1@1:73.47 1>1@3:3.82 1>1@42:4 0>1@72:6.92", 15, 3, "vs_dst_out_amount_received_var"),
            ("A45", "1>2@1:73.47 1>2@3:3.82 1>2@42:4 0>1@72:6.92", 15, 3, "vs_dst_out_amount_received_var")]


def vs_edges(spec):
    """'1>9@1:2.5' -> [(1, 9, 1, 2.5)]"""
    return [(s, d, t, float(a)) for (s, d, t), a in
            zip(edges(" ".join(tok.split(":")[0] for tok in spec.split())), (tok.split(":")[1] for tok in spec.split()))]


def run_vstats(es, tw, params):
    """A fresh GFP with F1's call at window tw, `params` overriding; one transform() per edge.
    Returns each row's 52 statistics (the input columns and pattern bins are dropped)."""
    g = gfp_vstats(tw)
    p = g.get_params()
    p.update(params)
    g.set_params(p)
    out = []
    for i, (s, d, t, a) in enumerate(es):
        row = g.transform(np.array([[i, s, d, t, a]], dtype=np.float64))[0]
        out.append([float(x) for x in row[len(row) - len(VS_NAMES):]])
    return out


def vs_moments(xs, ddof=0, excess=False, guard=True):
    n = len(xs)
    if n == 0:
        return [0.0] * 5
    s = sum(xs)
    mu = s / n
    d = [x - mu for x in xs]
    m2 = sum(v * v for v in d) / (n - ddof) if n > ddof else 0.0
    m3, m4 = sum(v ** 3 for v in d) / n, sum(v ** 4 for v in d) / n
    if not m2 > 0 or (guard and m2 < 1e-5 * mu * mu):
        return [mu, s, 0.0, 0.0, 0.0]
    return [mu, s, m2, m3 / m2 ** 1.5, m4 / m2 ** 2 - (3 if excess else 0)]


def vs_model(es, W, lower="prev", closed=True, loops=True, ratio="deg/fan", **mom):
    """Each row's 52 statistics by the S2 addendum's rule (the defaults) or an alternative."""
    rows = []
    for e, (u, v, t, _) in enumerate(es):
        lo = (es[e - 1][2] - W if e else -float("inf")) if lower == "prev" else t - W
        row = []
        for node in (u, v):
            for out_dir in (True, False):
                win = [f for f in range(e + 1) if (es[f][0] if out_dir else es[f][1]) == node
                       and (es[f][2] >= lo if closed else es[f][2] > lo) and (loops or es[f][0] != es[f][1])]
                fan, n = len({es[f][1] if out_dir else es[f][0] for f in win}), len(win)
                row += [fan, n, (n / fan if fan else 0.0) if ratio == "deg/fan" else (fan / n if n else 0.0)]
                for k in (2, 3):
                    row += vs_moments([float(es[f][k]) for f in win], **mom)
        rows.append(row)
    return rows


def vs_equal(gfp, model):
    """S2's acceptance test: integer columns equal, the rest within 1e-6 relative or 1e-9 absolute."""
    a, b = np.array(model, dtype=np.float64), np.array(gfp, dtype=np.float64)
    ok = np.abs(a - b) <= np.maximum(1e-9, 1e-6 * np.abs(b))
    ok[:, VS_INT_COLS] = a[:, VS_INT_COLS] == b[:, VS_INT_COLS]
    return bool(ok.all())


def vstats_probes(failures):
    """SPEC v2 Appendix A' (A43-A45). Returns the golden cases: edges, the rule's W, GFP's rows."""
    cases = []
    for probe, rejected, spec, W, params in S2_FIXTURES:
        es = vs_edges(spec)
        gfp = run_vstats(es, W, params)
        cid = f"{probe}/{spec}/W={W}/{','.join(f'{k}={v}' for k, v in params.items()) or 'F1'}"
        if not vs_equal(gfp, vs_model(es, W)):
            failures.append(f"{cid}: GFP differs from the rule")
        for alt in rejected:
            key, _, val = alt.partition("=")
            if key == "W":
                kw = {"W": params[val]}
            elif key == "lower":
                kw = {"W": W, "lower": val}
            elif key in ("closed", "loops", "guard"):
                kw = {"W": W, key: False}
            elif key == "ratio":
                kw = {"W": W, "ratio": val}
            elif key == "ddof":
                kw = {"W": W, "ddof": int(val)}
            else:
                kw = {"W": W, key: True}
            if vs_equal(gfp, vs_model(es, **kw)):
                failures.append(f"{cid}: does not separate the rule from {alt}")
        cases.append({"id": cid, "edges": [list(e) for e in es], "window": W, "params": params,
                      "kind": "rule", "output": gfp})
    probe, spec, W, vtw = S2_SPLIT
    es = vs_edges(spec)
    gfp = run_vstats(es, W, {"vertex_stats_tw": vtw})
    cid = f"{probe}/{spec}/W={W}/vertex_stats_tw={vtw}"
    at_w, at_v = vs_model(es, W), vs_model(es, vtw)
    sums = [j for j, n in enumerate(VS_NAMES) if n.endswith("_sum")]
    if not ([[r[c] for c in VS_INT_COLS] for r in gfp] == [[r[c] for c in VS_INT_COLS] for r in at_w]
            and [[r[c] for c in sums] for r in gfp] == [[r[c] for c in sums] for r in at_v] and at_v != at_w):
        failures.append(f"{cid}: fan and degree are not GFP's graph at W, or the sums not the window at vertex_stats_tw")
    cases.append({"id": cid, "edges": [list(e) for e in es], "window": W, "params": {"vertex_stats_tw": vtw},
                  "kind": "split", "output": gfp})
    for probe, spec, W, row, col in S2_DRIFT:
        es = vs_edges(spec)
        gfp = run_vstats(es, W, {})
        j = VS_NAMES.index(col)
        cid = f"{probe}/{spec}/W={W}"
        exact = vs_model(es, W)
        if not (gfp[row][j] < 0 and exact[row][j] == 0 and vs_equal(gfp[:row], exact[:row])):
            failures.append(f"{cid}: row {row} {col}: GFP {gfp[row][j]}, want < 0 (the rule gives {exact[row][j]})")
        if [gfp[r][c] for r in range(len(es)) for c in VS_INT_COLS] != [exact[r][c] for r in range(len(es)) for c in VS_INT_COLS]:
            failures.append(f"{cid}: integer statistics differ from the rule")
        cases.append({"id": cid, "edges": [list(e) for e in es], "window": W, "params": {}, "kind": "drift",
                      "output": gfp})
    return cases


# S3 (SPEC v2 §8 S3, Appendix A' A46-A49): GFP's lc-cycle in stream mode, lc-cycle_len 10, bins
# 2..10 (F2's). lc_model() is the S3 rule with GFP's tie rule (ties="rank"); its keyword arguments
# are the rejected alternatives:
#   A46 window: closed=False (open lower bound); lower=prev (the window starts at ts[e - 1] - W,
#       GFP's eviction rule of A43); W=other (another pattern's window, i.e. GFP's time_window).
#   A47 length: L=9, L=11 (the bound is lc-cycle_len, inclusive); min_len=3 (no 2-cycles).
#   A48 shape: ordered=True (increasing ts along the path, as tcyc); through=u (every cycle through
#       u, not only the trigger edge's); nodes=False (edge sequences); simple=False (walks).
#   A49 ties: ties=none (every tie left out: blz's lcyc, D-38); ties=all; key=src (a tie x -> y
#       iff rank(x) > rank(u)); order=dst_first, rank=id, rank=window (other ranks).
# Each fixture: (probe, rejected alternatives, edges src>dst@ts, W[, other pattern's window]).
S3_FIXTURES = [
    ("A46", ["closed=False"], "10>11@5 11>12@6 12>10@10", 5),
    ("A46", ["lower=prev"], "10>11@5 11>12@6 12>10@10", 4),
    ("A46", ["W=other"], "10>11@5 11>12@6 12>10@10", 4, 100),
    ("A47", ["L=9"], " ".join(f"{30 + i}>{30 + (i + 1) % 10}@{i + 1}" for i in range(10)), 100),
    ("A47", ["L=11"], " ".join(f"{30 + i}>{30 + (i + 1) % 11}@{i + 1}" for i in range(11)), 100),
    ("A47", ["min_len=3"], "10>11@1 11>10@2", 100),
    ("A48", ["ordered=True"], "11>12@1 10>11@2 12>10@3", 100),
    ("A48", ["through=u"], "10>12@1 12>10@2 10>11@3", 100),
    ("A48", ["nodes=False"], "10>11@1 11>12@2 11>12@3 12>10@4", 100),
    ("A48", ["simple=False"], "10>11@1 11>10@2 10>12@3 12>10@4", 100),
    ("A49", ["ties=none", "rank=window"], "10>12@1 11>10@100 10>11@100", 5),
    ("A49", ["ties=all", "order=dst_first", "rank=id"], "11>10@5 10>11@5", 100),
    ("A49", ["key=src"], "12>13@1 10>14@2 11>12@3 12>10@9 10>11@9", 100),
    # O-2 (SPEC Appendix A): (1,1,2) is found, (1,2,2) is not, and the latter depends on the order
    # in which GFP first saw the nodes: with 12 seen first, the same cycle is found.
    ("A49", [], "10>11@1 11>12@1 12>10@2", 100),
    ("A49", ["ties=all"], "10>11@1 11>12@2 12>10@2", 100),
    ("A49", ["ties=none", "rank=id"], "12>13@1 10>11@1 11>12@2 12>10@2", 100),
]


def lc_alt(alt, W, other):
    """'key=value' -> lc_model keyword arguments."""
    key, _, val = alt.partition("=")
    if key == "W":
        return {"W": other}
    if key == "L":
        return {"W": W, "L": int(val)}
    if key in ("closed", "simple", "nodes"):
        return {"W": W, key: False}
    if key == "ordered":
        return {"W": W, "ordered": True}
    if key == "min_len":
        return {"W": W, "min_len": int(val)}
    return {"W": W, key: val}


def lcyc_probes(failures):
    """SPEC v2 Appendix A' (A46-A49). Returns the golden cases: edges, W, GFP's rows, blz's."""
    cases = []
    for fx in S3_FIXTURES:
        probe, rejected, spec, W = fx[:4]
        other = fx[4] if len(fx) > 4 else None
        es = edges(spec)
        assert [t for _, _, t in es] == sorted(t for _, _, t in es), spec  # SPEC §3.3, as the C++ tests need
        gfp = run_gfp(es, W, other_tw=other)
        cid = f"{probe}/{spec}/W={W}" + (f"/fan_tw={other}" if other else "")
        rule = [binned(r, LCYC_BINS) for r in lc_model(es, W)]
        if gfp != rule:
            failures.append(f"{cid}: GFP {gfp} differs from the rule {rule}")
        for alt in rejected:
            if [binned(r, LCYC_BINS) for r in lc_model(es, **lc_alt(alt, W, other))] == gfp:
                failures.append(f"{cid}: does not separate the rule from {alt}")
        blz = [binned(r, LCYC_BINS) for r in lc_model(es, W, ties="none")]
        cases.append({"id": cid, "edges": [list(e) for e in es], "window": W, "other_tw": other,
                      "kind": "rule" if blz == gfp else "tie", "output": gfp, "blz": blz})
    # A47: GFP rejects lc-cycle_len = 1 (set_params raises SystemError, the check's Error as its cause).
    try:
        gfp_lc(10, L=1).transform(np.array([[0, 1, 2, 1]], dtype=np.float64))
        failures.append("A47: GFP accepted lc-cycle_len = 1")
    except Exception as e:  # noqa: BLE001
        if "greater than one" not in f"{e} {e.__cause__}":
            failures.append(f"A47: lc-cycle_len = 1 failed with {e!r}, not the length check")
    return cases


def write_golden(path, doc, cases):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    head = json.dumps(doc, indent=1)[:-2]  # one case per line keeps diffs readable
    body = ",\n".join("  " + json.dumps(c, separators=(",", ":")) for c in cases)
    Path(path).write_text(f'{head},\n "cases": [\n{body}\n ]\n}}\n')
    print("wrote", path)


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
    ap.add_argument("--vstats-out", default=str(here.parents[1] / "tests/golden/gfp_vstats_probes.json"))
    ap.add_argument("--lcyc-out", default=str(here.parents[1] / "tests/golden/gfp_lcyc_probes.json"))
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

    bin_rule_probe(failures)
    vstats_cases = vstats_probes(failures)
    lcyc_cases = lcyc_probes(failures)

    for f in failures:
        print("FAIL", f, file=sys.stderr)
    print(f"{len(CASES)} stream probes + A28 + A29 + A33 + A38"
          f" + vertex-statistics probes A43-A45 ({len(vstats_cases)} cases)"
          f" + lc-cycle probes A46-A49 ({len(lcyc_cases)} cases): {len(failures)} failures")
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
        write_golden(args.out, doc, doc.pop("cases"))
        vdoc = {
            "generator": "tools/gfp/probes.py (SPEC v2 Appendix A', A43-A45)",
            "gfp": "snapml==1.17.2, stream mode (one transform() per edge, fresh instance), F1's call "
                   "(vstats_vs_gfp.gfp_vstats(window), then params)",
            "columns": VS_NAMES,
            "note": "edges are [src, dst, ts, amount] with eid = index, ts as GFP saw it (no offset). window = "
                    "the S2 rule's W (GFP's time_window, the largest pattern window). output[i] = row i's 52 "
                    "statistics. kind: rule = the S2 rule gives GFP's rows (within S2's tolerance); split = "
                    "vertex_stats_tw < W (A43), drift = GFP's running accumulators after eviction (A45): only the "
                    "integer columns follow the rule.",
        }
        write_golden(args.vstats_out, vdoc, vstats_cases)
        ldoc = {
            "generator": "tools/gfp/probes.py (SPEC v2 Appendix A', A46-A49)",
            "gfp": "snapml==1.17.2, stream mode (one transform() per edge, fresh instance), lc-cycle only (with fan "
                   "at other_tw where given), lc-cycle_len 10, bins 2..10",
            "bins": LCYC_BINS,
            "lc_len": LCYC_LEN,
            "note": "edges are [src, dst, ts] with eid = index, ts as GFP saw it (no offset). window = lc-cycle_tw. "
                    "output[i] = GFP's row i; blz[i] = the S3 rule's without tie edges (blz's lcyc). kind: rule = "
                    "they are equal; tie = they differ by A49's tie rule (D-38).",
        }
        write_golden(args.lcyc_out, ldoc, lcyc_cases)


if __name__ == "__main__":
    main()

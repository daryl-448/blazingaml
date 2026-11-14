"""blz's lcyc against GFP's lc-cycle (SPEC v2 §8 S3).

The model (cycles, lc_model) is the S3 addendum's rule. Trigger e = (u -> v, t), u != v, counts the
simple cycles through the trigger edge, as node sequences v = c_0, c_1, ..., c_h, u (distinct
nodes, parallel edges collapse), of length h + 2 in 2..L, whose every other hop has an edge of an
earlier eid in lc-cycle's own window [t - W, t] (closed lower bound), in any time order. An edge
tied with the trigger (ts = t) is where blz and GFP part (A49): blz leaves every tie out
(ties="none"); GFP uses a tie edge x -> y iff rank(y) < rank(v), rank = the node's first
appearance in the stream, src before dst within an edge, kept after eviction (ties="rank").
The keyword arguments of cycles() and lc_model() are the rejected alternatives that the probes
A46-A49 (tools/gfp/probes.py) separate from GFP.

Fuzz mode: `blz fuzz --lcyc` writes the fuzz graphs and the oracle's lcyc rows (length <= 10, bins
2..10, the graph's window; graphs over the work limit have UINT32_MAX rows and are skipped). Each
graph is replayed into a fresh GFP in stream mode (lc-cycle only, lc-cycle_len 10, window = the
graph's, ts + GFP_TS_OFFSET). HI-Small mode (--blz-dir): a `blz mine` lcyc block against F2's
(features/dg2_R7_F2).

Every mismatching row is checked against both rules: it is a tie row when GFP's row is the
ties="rank" model's and blz's the ties="none" model's (D-38); any other mismatch is unexplained.
The summary goes to the report under "lcyc_vs_gfp_fuzz" or "lcyc_vs_gfp_hi_small".

Usage: uv run lcyc_vs_gfp.py --seeds 0:2000 [--work ../../data/fuzz_lc] [--work-limit 1000000] [--jobs 4]
                             [--reuse]
       uv run lcyc_vs_gfp.py --blz-dir ../../features/<run> [--gfp-dir ../../features/dg2_R7_F2]
"""

import argparse
import json
import multiprocessing
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from gfp_common import GFP_TS_OFFSET, gfp_bins, make_preprocessor

LC_LEN = 10
BINS = list(range(2, 11))  # F2's lcyc bins; the last is [10, inf)
U32_MAX = np.iinfo(np.uint32).max


def first_seen(edges, order="src_first"):
    """node -> its rank: the position of its first appearance in `edges` [(s, d, ...)], src before
    dst within an edge (order="dst_first": the other way)."""
    rank = {}
    for e in edges:
        for x in ((e[0], e[1]) if order == "src_first" else (e[1], e[0])):
            rank.setdefault(x, len(rank))
    return rank


class NodeId:
    """rank="id": the node's id as its rank."""

    def __getitem__(self, x):
        return x


def cycles(win, u, v, t, L, rank, ties="rank", key="dst", nodes=True, through="trigger", simple=True,
           min_len=2, ordered=False):
    """{size: count} of the trigger u -> v at t, given its window edges `win` [(s, d, ts)].

    ties: rank (GFP: a tie edge x -> y iff rank[y] < rank[v]; key="src": iff rank[x] > rank[u]),
    none (blz: no tie edge), all. Alternatives: nodes=False counts edge sequences (parallel edges
    multiply); through="u" counts every cycle through u, not only the trigger edge's; simple=False
    lets a walk repeat nodes; min_len=3 drops 2-cycles; ordered=True needs increasing ts along the
    path v -> ... -> u, as tcyc."""
    adj = {}
    for s, d, ts in win:
        if ts == t:
            if ties == "none":
                continue
            if ties == "rank" and not (rank[d] < rank[v] if key == "dst" else rank[s] > rank[u]):
                continue
        adj.setdefault(s, {}).setdefault(d, []).append(ts)
    if through == "u":
        adj.setdefault(u, {}).setdefault(v, []).append(t)
    start, seen0, base = (v, {u, v}, 1) if through == "trigger" else (u, {u}, 0)
    res = Counter()

    def go(x, seen, n, prev, w):  # n: edges walked from start; w: the paths' multiplicity
        for y, tss in adj.get(x, {}).items():
            ok = [s for s in tss if s > prev] if ordered else tss
            if not ok:
                continue
            ww = w * (1 if nodes else len(ok))
            if y == u:
                if min_len <= base + n + 1 <= L:
                    res[base + n + 1] += ww
            elif base + n + 2 <= L and (not simple or y not in seen):
                go(y, seen | {y}, n + 1, min(ok) if ordered else prev, ww)

    go(start, seen0, 0, -float("inf"), 1)
    return dict(res)


def lc_model(es, W, L=LC_LEN, closed=True, lower="trigger", order="src_first", rank="stream", **kw):
    """Every row's {size: count} for a stream es [(s, d, ts)] at window W. lower="prev": the window
    starts at ts[e - 1] - W (GFP's eviction rule for vertex statistics, A43), not at t - W;
    closed=False: an open lower bound; rank="window": ranks from the window's edges only (lost
    on eviction), rank="id": node ids. Other keywords go to cycles()."""
    out = []
    for e, (u, v, t) in enumerate(es):
        if u == v:
            out.append({})
            continue
        lo = t - W if lower == "trigger" else (es[e - 1][2] - W if e else -float("inf"))
        win = [es[f] for f in range(e) if (es[f][2] >= lo if closed else es[f][2] > lo) and es[f][2] <= t]
        r = (first_seen(es[:e + 1], order) if rank == "stream" else
             first_seen(win + [es[e]], order) if rank == "window" else NodeId())
        out.append(cycles(win, u, v, t, L, r, **kw))
    return out


def counts_of(row, bins):
    """A bin row -> {bin's lower edge: count} (the last bin is open)."""
    return {int(bins[i]): int(c) for i, c in enumerate(row) if c}


def binned(sizes, bins):
    """{size: count} -> bin row (SPEC §4.2: the last b_i <= size; the last bin open)."""
    row = [0] * len(bins)
    for size, c in sizes.items():
        row[max(i for i, b in enumerate(bins) if b <= size)] += c
    return row


def gfp_lc(W, L=LC_LEN, bins=BINS, other_tw=None):
    """A fresh GFP with lc-cycle only (window W), or with fan at other_tw as well (A46)."""
    if other_tw is None:
        return make_preprocessor(["lc-cycle"], W, gfp_bins(bins), lc_len=L)
    return make_preprocessor(["fan", "lc-cycle"], {"fan": other_tw, "lc-cycle": W}, gfp_bins(bins), lc_len=L)


def run_gfp(es, W, L=LC_LEN, bins=BINS, other_tw=None, ts_offset=0):
    """Stream mode, one transform() per edge: every row's lc-cycle bins."""
    g = gfp_lc(W, L, bins, other_tw)
    out = []
    for i, (s, d, t) in enumerate(es):
        row = g.transform(np.array([[i, s, d, t + ts_offset]], dtype=np.float64))[0]
        out.append([int(x) for x in row[len(row) - len(bins):]])
    return out


def explain(win, u, v, t, rank, gfp_row, blz_row, bins):
    """'tie' if GFP's row is the rank rule's and blz's the no-tie rule's, else 'unexplained'."""
    g = binned(cycles(win, u, v, t, LC_LEN, rank, ties="rank"), bins)
    b = binned(cycles(win, u, v, t, LC_LEN, rank, ties="none"), bins)
    return "tie" if g == list(gfp_row) and b == list(blz_row) else "unexplained"


def replay(job):
    """One fuzz graph into a fresh GFP (a worker process)."""
    es, W = job
    return run_gfp(es, W, ts_offset=GFP_TS_OFFSET)


def fuzz_mode(args, root):
    t0 = time.time()
    work = Path(args.work)
    s0, s1 = (int(x) for x in args.seeds.split(":"))
    if args.reuse:  # the outputs of an earlier `blz fuzz --lcyc` run over the same seeds
        g = np.load(work / "graphs.npy", mmap_mode="r")
        if (int(g[0, 0]), int(g[-1, 0])) != (s0, s1 - 1):
            sys.exit(f"--reuse: {work} holds seeds {int(g[0, 0])}..{int(g[-1, 0])}, not {args.seeds}")
    else:
        subprocess.run([args.blz, "fuzz", "--seeds", args.seeds, "--bins", "2..10", "--lcyc",
                        "--work-limit", str(args.work_limit), "--out", args.work], check=True)
    t_blz = time.time() - t0
    graphs = np.load(work / "graphs.npy")  # seed, eid, src, dst, ts, window
    mine = np.load(work / "lcyc.npy")
    seeds = graphs[:, 0]
    starts = np.flatnonzero(np.r_[True, seeds[1:] != seeds[:-1]])
    ends = np.r_[starts[1:], len(seeds)]
    t1 = time.time()
    rows = mismatched = limited = 0
    kinds, examples, cyc_rows = Counter(), [], 0
    spans = []
    for a, z in zip(starts, ends):
        if (mine[a:z] == U32_MAX).all():
            limited += 1
        else:
            spans.append((a, z))
    jobs = [([(int(s), int(d), int(t)) for s, d, t in graphs[a:z, 2:5]], int(graphs[a, 5])) for a, z in spans]
    with multiprocessing.Pool(args.jobs) as pool:
        replies = pool.map(replay, jobs, chunksize=1)
    for (a, z), (es, W), gfp in zip(spans, jobs, replies):
        rank = first_seen(es)
        rows += z - a
        for e in range(z - a):
            cyc_rows += any(gfp[e])
            if list(mine[a + e]) == gfp[e]:
                continue
            mismatched += 1
            u, v, t = es[e]
            win = [es[f] for f in range(e) if t - W <= es[f][2] <= t]
            k = explain(win, u, v, t, rank, gfp[e], mine[a + e], BINS)
            kinds[k] += 1
            if k == "unexplained" and len(examples) < 10:
                examples.append({"seed": int(graphs[a, 0]), "eid": e, "gfp": counts_of(gfp[e], BINS),
                                 "blz": counts_of(mine[a + e], BINS)})
    summary = {"generator": "blz fuzz --lcyc (SPEC §13.3 graphs; the oracle, length <= 10, bins 2..10)",
               "gfp": "snapml==1.17.2, stream mode, lc-cycle only, lc-cycle_len 10, fresh instance per graph",
               "gfp_ts_offset": GFP_TS_OFFSET, "seeds": args.seeds, "work_limit": args.work_limit,
               "gfp_jobs": args.jobs,
               "graphs": int(len(starts)), "graphs_limited_skipped": limited, "rows": int(rows),
               "rows_with_a_gfp_cycle": int(cyc_rows), "mismatched_rows": mismatched,
               "mismatched_tie_rows": kinds["tie"], "mismatched_unexplained": kinds["unexplained"],
               "first_unexplained": examples,
               "seconds": {"blz_fuzz": round(t_blz, 1), "gfp": round(time.time() - t1, 1)}}
    return "lcyc_vs_gfp_fuzz", summary


def hi_small_mode(args, root):
    sys.path.insert(0, str(root / "bench/dg2"))
    from blzg import Blzg  # noqa: E402
    blz_dir, gfp_dir = Path(args.blz_dir), Path(args.gfp_dir)
    meta = json.loads((gfp_dir / "features.json").read_text())["blocks"]["lcyc"]
    bmeta = json.loads((blz_dir / "features.json").read_text())["blocks"]["lcyc"]
    bins, W = [int(b) for b in meta["bins"]], int(meta["window"])
    if [int(b) for b in bmeta["bins"]] != bins or int(bmeta["window"]) != W:
        sys.exit(f"blz's lcyc block (bins {bmeta['bins']}, window {bmeta['window']}) is not F2's ({bins}, {W})")
    gfp = np.load(gfp_dir / "lcyc.npy", mmap_mode="r")
    mine = np.load(blz_dir / "lcyc.npy", mmap_mode="r")
    if mine.shape != gfp.shape:
        sys.exit(f"shapes differ: {mine.shape} vs {gfp.shape}")
    g = Blzg(str(root / args.graph))
    m = mine.shape[0]
    src, dst, ts = np.asarray(g.src[:m]), np.asarray(g.dst[:m]), np.asarray(g.ts[:m]).astype(np.int64)
    t0 = time.time()
    bad = []
    step = 1 << 20
    for a in range(0, m, step):
        b = min(m, a + step)
        bad += list(a + np.flatnonzero((np.asarray(mine[a:b]) != np.asarray(gfp[a:b])).any(axis=1)))
    # Ranks: first appearance over the whole stream (F2 is one GFP instance from eid 0), src first.
    inter = np.empty(2 * m, dtype=np.int64)
    inter[0::2], inter[1::2] = src, dst
    nodes, first = np.unique(inter, return_index=True)
    rank_arr = np.full(int(nodes.max()) + 1, -1, dtype=np.int64)
    rank_arr[nodes] = first
    kinds, examples = Counter(), []
    for e in bad:
        u, v, t = int(src[e]), int(dst[e]), int(ts[e])
        lo = int(np.searchsorted(ts, t - W, side="left"))
        s, d, tt = src[lo:e], dst[lo:e], ts[lo:e]
        keep = (tt < t) | (rank_arr[d] < rank_arr[v])  # a tie edge only where GFP's rule may use it
        win = list(zip(s[keep].tolist(), d[keep].tolist(), tt[keep].tolist()))
        rank = {x: int(rank_arr[x]) for x in {u, v} | {x for x, _, _ in win} | {y for _, y, _ in win}}
        k = explain(win, u, v, t, rank, gfp[e], mine[e], bins)
        kinds[k] += 1
        if k == "unexplained" and len(examples) < 10:
            examples.append({"eid": int(e), "gfp": counts_of(gfp[e], bins), "blz": counts_of(mine[e], bins)})
    rows_cyc = sum(int(np.asarray(gfp[a:min(m, a + step)]).any(axis=1).sum()) for a in range(0, m, step))
    summary = {"blz_dir": str(blz_dir), "gfp_dir": str(gfp_dir), "window": W, "bins": bins, "lc_len": LC_LEN,
               "rows": int(m), "rows_with_a_gfp_cycle": rows_cyc, "mismatched_rows": len(bad),
               "mismatched_tie_rows": kinds["tie"], "mismatched_unexplained": kinds["unexplained"],
               "first_mismatched_eids": [int(e) for e in bad[:20]], "first_unexplained": examples,
               "seconds": round(time.time() - t0, 1)}
    return "lcyc_vs_gfp_hi_small", summary


def main():
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--blz", default=str(root / "build/rel/blz"))
    ap.add_argument("--seeds", default="0:2000")
    ap.add_argument("--work", default=str(root / "data/fuzz_lc"))
    ap.add_argument("--work-limit", type=int, default=1_000_000)
    ap.add_argument("--jobs", type=int, default=4, help="GFP worker processes (fuzz mode)")
    ap.add_argument("--reuse", action="store_true", help="fuzz mode: reuse --work's blz fuzz outputs")
    ap.add_argument("--blz-dir", default=None, help="HI-Small mode: a `blz mine` output directory with lcyc.npy")
    ap.add_argument("--gfp-dir", default=str(root / "features/dg2_R7_F2"))
    ap.add_argument("--graph", default="data/hi_small.blzg")
    ap.add_argument("--report", default=str(root / "results/s3_lcyc.json"))
    args = ap.parse_args()
    key, summary = hi_small_mode(args, root) if args.blz_dir else fuzz_mode(args, root)
    report = Path(args.report)
    doc = json.loads(report.read_text()) if report.exists() else {}
    doc[key] = summary
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

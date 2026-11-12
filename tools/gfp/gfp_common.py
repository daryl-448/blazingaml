"""Shared GFP helpers (SPEC §11). Stream mode only: a fresh preprocessor, one transform() per edge.

Never call fit()/fit_transform() and never re-send an edge: snapml >= 1.14.1 double-counts
edges that are already in the graph (SPEC Appendix A, A29).

Parity runs feed GFP ts + GFP_TS_OFFSET (D-12): snapml drops every temporal cycle through a node
with an in-window self-loop at exactly ts 0 (A33). Every SPEC §4 rule is translation-invariant,
and ts 0 is only this repo's epoch (D-1), so the shift removes the artifact without changing
any other output.
"""

import numpy as np
from snapml import GraphFeaturePreprocessor

# GFP pattern name -> output blocks, in GFP's column order (SPEC §4.2).
PATTERN_BLOCKS = {
    "fan": ["fan_in", "fan_out"],
    "degree": ["deg_in", "deg_out"],
    "scatter-gather": ["sg"],
    "temp-cycle": ["tcyc"],
    "lc-cycle": ["lcyc"],
}
BLOCK_ORDER = [b for blocks in PATTERN_BLOCKS.values() for b in blocks]
GFP_TS_OFFSET = 1


def gfp_bins(bins):
    """The GFP bin list that reproduces SPEC §4.2's binning for `bins`.

    GFP puts an instance of size S in the first bin i with S <= b_i, or in the last bin (A38,
    SPEC Appendix A), not in the bin with b_i <= S < b_{i+1} that its docstring states. The two
    rules agree for unit-width bins (2..33, 1..8: every parity run and probe), not for D-9's.
    Passing b_i = c_{i+1} - 1 (and the last value unchanged) makes GFP bin as §4.2 does, given
    c_0 = 2 (min size 2, SPEC §4.2) or unit-width bins.
    """
    c = [int(x) for x in bins]
    if any(b <= a for a, b in zip(c, c[1:])):
        raise ValueError(f"bins must be strictly increasing: {c}")
    if c[0] != 2 and any(b != a + 1 for a, b in zip(c, c[1:])):
        raise ValueError(f"gfp_bins needs bins starting at 2, or unit-width bins: {c}")
    return [c[i + 1] - 1 for i in range(len(c) - 1)] + [c[-1]]


def make_preprocessor(patterns, tw, bins, lc_len=None, num_threads=1):
    """A fresh GFP with only `patterns` enabled, each with window `tw` and `bins`.

    `tw` is an int (all patterns) or a dict pattern -> int; `bins` a list or a dict pattern -> list.
    """
    params = {"num_threads": num_threads, "vertex_stats": False, "time_window": -1}
    for p in PATTERN_BLOCKS:
        params[p] = p in patterns
        if p in patterns:
            params[p + "_tw"] = int(tw[p] if isinstance(tw, dict) else tw)
            params[p + "_bins"] = list(bins[p] if isinstance(bins, dict) else bins)
    if lc_len is not None:
        params["lc-cycle_len"] = int(lc_len)
    g = GraphFeaturePreprocessor()
    g.set_params(params)
    return g


def enabled_blocks(patterns):
    return [b for p, blocks in PATTERN_BLOCKS.items() if p in patterns for b in blocks]


def split_blocks(row, patterns, nbins):
    """Split one transform() output row (input cols + features) into {block: [counts]}."""
    feats = row[4:]
    blocks = enabled_blocks(patterns)
    assert feats.shape[0] == nbins * len(blocks), (feats.shape, nbins, blocks)
    return {b: [int(x) for x in feats[i * nbins:(i + 1) * nbins]] for i, b in enumerate(blocks)}


def edge_row(eid, src, dst, ts):
    return np.array([[eid, src, dst, ts]], dtype=np.float64)

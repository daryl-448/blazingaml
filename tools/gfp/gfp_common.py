"""Shared GFP helpers (SPEC §11). Stream mode only: a fresh preprocessor, one transform() per edge.

Never call fit()/fit_transform() and never re-send an edge: snapml >= 1.14.1 double-counts
edges that are already in the graph (SPEC Appendix A, A29).
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


def make_preprocessor(patterns, tw, bins, lc_len=None, num_threads=1):
    """A fresh GFP with only `patterns` enabled, each with window `tw` and `bins`.

    `tw` is an int (all patterns) or a dict pattern -> int.
    """
    params = {"num_threads": num_threads, "vertex_stats": False, "time_window": -1}
    for p in PATTERN_BLOCKS:
        params[p] = p in patterns
        if p in patterns:
            params[p + "_tw"] = int(tw[p] if isinstance(tw, dict) else tw)
            params[p + "_bins"] = list(bins)
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

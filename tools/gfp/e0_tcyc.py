"""E0: GFP tcyc length histogram (SPEC §12.2) and the DG-1 check (SPEC §17).

Reads the tcyc block written by `parity.py --block tcyc --bins 2..33` and merges a `gfp_tcyc`
section into the E0 profile JSON written by `blz stats`.

Usage: uv run e0_tcyc.py --tcyc data/gfp/tcyc.npy --profile results/e0_profile.json --tw 86400
"""

import argparse
import json
from pathlib import Path

import numpy as np

from parity import parse_bins

CHUNK = 1 << 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tcyc", required=True)
    ap.add_argument("--bins", default="2..33", help="bins parity.py was run with")
    ap.add_argument("--tw", type=int, required=True, help="window parity.py was run with")
    ap.add_argument("--k-max", type=int, default=10)
    ap.add_argument("--profile", required=True)
    args = ap.parse_args()

    bins = parse_bins(args.bins)
    assert bins == list(range(bins[0], bins[-1] + 1)), "bins must be one size each"
    assert bins[0] <= args.k_max < bins[-1], "k_max must be below the open last bin"
    x = np.load(args.tcyc, mmap_mode="r")
    m, nb = x.shape
    assert nb == len(bins), (nb, len(bins))

    instances = np.zeros(nb, dtype=np.uint64)
    max_len = np.zeros(nb, dtype=np.uint64)  # triggers by their longest cycle's bin
    for a in range(0, m, CHUNK):
        c = np.asarray(x[a:a + CHUNK], dtype=np.uint64)
        instances += c.sum(axis=0)
        nz = c > 0
        has = nz.any(axis=1)
        last = nb - 1 - np.argmax(nz[:, ::-1], axis=1)
        max_len += np.bincount(last[has], minlength=nb).astype(np.uint64)

    with_any = int(max_len.sum())
    over = int(max_len[bins.index(args.k_max) + 1:].sum())
    cum = np.cumsum(max_len)
    # smallest k whose cap leaves <= 0.1% of all triggers with a longer cycle
    k999 = next((bins[i] for i in range(nb) if m - with_any + cum[i] >= 0.999 * m), None)
    label = [str(b) for b in bins[:-1]] + [f"{bins[-1]}+"]
    gfp = {
        "source": "tools/gfp/parity.py --block tcyc (snapml 1.17.2, stream mode)",
        "tw": args.tw,
        "bins": label,
        "instances_by_length": {k: int(v) for k, v in zip(label, instances)},
        "triggers_by_max_length": {k: int(v) for k, v in zip(label, max_len)},
        "triggers": int(m),
        "triggers_with_cycle": with_any,
        "k_max": args.k_max,
        "triggers_with_cycle_longer_than_k_max": over,
        "frac_triggers_longer_than_k_max": over / m,
        "dg1_open": over / m > 0.001,
        "k_covering_99_9pct_of_triggers": k999,
        "last_bin_is_open": True,
    }
    p = Path(args.profile)
    doc = json.loads(p.read_text()) if p.exists() else {}
    doc["gfp_tcyc"] = gfp
    p.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(gfp, indent=1))


if __name__ == "__main__":
    main()

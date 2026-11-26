# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pyyaml"]
# ///
"""Independent count of a dataset's `blz prepare` invariants (SPEC v3 §6.1, P15).

  uv run --python 3.12 bench/p15_counts.py configs/datasets/<id>.yaml [--check]

Reads the descriptor for the transactions file, its schema, the split rule and the node tables,
then streams the CSV with its own parsing (str.split, datetime) and prints one JSON object:
"invariants", the names and values that the descriptor pins and `blz prepare` checks, and
"info" (epoch, timestamp range, examples of unresolved accounts). --check exits 1 when a pinned
value differs from the count, or a name is pinned that is not counted or the other way round.
Paths in the descriptor are relative to the working directory (the repository root).

The rules, written from SPEC v3 §6.1-§6.3 and SPEC v2 §3.3, §4.4 R1, not from the C++:
- a node is the exact (bank, account) string pair; a self-loop has equal pairs on both sides;
- the stream is the CSV rows stably sorted by timestamp; the epoch is midnight (no time zone) of
  the earliest timestamp's day;
- start(e) is the first index >= e whose timestamp differs from its predecessor's (index 0
  counts), or m. legacy_hi_small: b_test = floor(0.8 m), b_fv = min(start(floor(0.6 m)), b_test).
  temporal_60_20_20: b_fv = start(floor(0.6 m)), b_test = start(floor(0.8 m)).
  fit = [0, b_fv), val = [b_fv, b_test), test = [b_test, m);
- concatenated_keys (ibm): distinct bank + account strings with no separator;
- unresolved_accounts (node tables): tx nodes that no node-table row has; unmatched_node_rows:
  node-table rows whose (bank, account) is no tx node.
"""
import csv
import datetime as dt
import json
import sys

import numpy as np
import yaml

SCHEMAS = {
    "ibm": ("Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,"
            "Amount Paid,Payment Currency,Payment Format,Is Laundering", "%Y/%m/%d %H:%M"),
    "transxion": ("Timestamp,From Bank,From Account,To Bank,To Account,Amount Received,Receiving Currency,"
                  "Amount Paid,Payment Currency,Payment Format,Is Laundering", "%Y-%m-%d %H:%M:%S"),
}
UNIX = dt.datetime(1970, 1, 1)


def parse_ts(s, fmt, cache):
    v = cache.get(s)
    if v is None:
        t = dt.datetime.strptime(s, fmt)
        if t.strftime(fmt) != s:  # strptime accepts "2022/9/1 0:0"; the files use fixed widths
            raise ValueError(f"bad timestamp {s!r}")
        v = cache[s] = (t - UNIX) // dt.timedelta(seconds=1)
    return v


def start(ts, e):
    m = len(ts)
    if e == 0 or e >= m or ts[e] != ts[e - 1]:
        return min(e, m)
    return int(np.searchsorted(ts, ts[e - 1], side="right"))


def count(desc):
    header, fmt = SCHEMAS[desc["schema"]]
    ts, label = [], bytearray()
    nodes, concat = set(), set()
    self_loops = 0
    cache = {}
    with open(desc["transactions"], encoding="utf-8", newline="") as f:
        if f.readline().rstrip("\r\n") != header:
            raise SystemExit("unexpected header")
        chunk = []
        for i, line in enumerate(f, start=2):
            line = line.rstrip("\r\n")
            if '"' in line:
                raise SystemExit(f"line {i}: a quoted field")
            p = line.split(",")
            if len(p) != 11:
                raise SystemExit(f"line {i}: {len(p)} fields")
            chunk.append(parse_ts(p[0], fmt, cache))
            if p[10] not in ("0", "1"):
                raise SystemExit(f"line {i}: label {p[10]!r}")
            label.append(p[10] == "1")
            a, b = p[1] + "," + p[2], p[3] + "," + p[4]
            nodes.add(a)
            nodes.add(b)
            self_loops += a == b
            concat.add(p[1] + p[2])
            concat.add(p[3] + p[4])
            if len(chunk) == 1 << 20:
                ts.append(np.array(chunk, np.int64))
                chunk = []
        ts.append(np.array(chunk, np.int64))
    ts = np.concatenate(ts)
    lab = np.frombuffer(bytes(label), np.uint8)
    del label
    order = np.argsort(ts, kind="stable")
    ts, lab = ts[order], lab[order]
    del order
    m = len(ts)
    if desc["split"] == "legacy_hi_small":
        b_test = m * 4 // 5
        b_fv = min(start(ts, m * 3 // 5), b_test)
    elif desc["split"] == "temporal_60_20_20":
        b_fv, b_test = start(ts, m * 3 // 5), start(ts, m * 4 // 5)
    else:
        raise SystemExit(f"unknown split {desc['split']!r}")
    epoch = int(ts[0]) // 86400 * 86400 if m else 0
    inv = {
        "edges": m,
        "nodes": len(nodes),
        "self_loops": self_loops,
        "distinct_timestamps": int(np.count_nonzero(np.diff(ts))) + 1 if m else 0,
        "positives": int(lab.sum()),
        "b_fv": b_fv,
        "b_test": b_test,
        "fit_positives": int(lab[:b_fv].sum()),
        "val_positives": int(lab[b_fv:b_test].sum()),
        "test_positives": int(lab[b_test:].sum()),
    }
    info = {
        "epoch_unix": epoch,
        "epoch_date": (UNIX + dt.timedelta(seconds=epoch)).date().isoformat(),
        "ts_max": int(ts[-1]) - epoch if m else 0,
    }
    if desc["schema"] == "ibm":
        inv["concatenated_keys"] = len(concat)
    if desc.get("node_tables"):
        listed = set()
        unmatched = 0
        for t in desc["node_tables"]:
            with open(t["file"], encoding="utf-8", newline="") as f:
                r = csv.reader(f)
                cols = next(r)
                bi, ai = cols.index(t["bank"]), cols.index(t["account"])
                for row in r:
                    k = row[bi] + "," + row[ai]
                    listed.add(k)
                    unmatched += k not in nodes
        unresolved = sorted(nodes - listed)
        inv["unresolved_accounts"] = len(unresolved)
        inv["unmatched_node_rows"] = unmatched
        info["unresolved_examples"] = unresolved[:10]
    return {"dataset": desc["id"], "invariants": dict(sorted(inv.items())), "info": info}


def main():
    args = [a for a in sys.argv[1:] if a != "--check"]
    if len(args) != 1:
        raise SystemExit(__doc__)
    with open(args[0]) as f:
        desc = yaml.safe_load(f)
    out = count(desc)
    print(json.dumps(out))
    if "--check" in sys.argv:
        pinned = desc.get("invariants") or {}
        got = out["invariants"]
        bad = [f"{k}: pinned {pinned.get(k)}, counted {got.get(k)}"
               for k in sorted(set(pinned) | set(got)) if pinned.get(k) != got.get(k)]
        for b in bad:
            print(b, file=sys.stderr)
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

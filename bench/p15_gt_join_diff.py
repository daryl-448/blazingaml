#!/usr/bin/env python3
"""Compare two `blz gt-join` outputs apart from their git and path fields (SPEC v3 P15).

    bench/p15_gt_join_diff.py results/gt_join.json NEW.json

Ignored: "git", and inputs.graph, inputs.csv and inputs.patterns (paths). Everything else, the
CSV's and Patterns.txt's SHA-256 and m included, must be equal. Exits 1 on any difference, listing
the first ones by JSON path.
"""
import json
import sys

IGNORED = {("git",), ("inputs", "graph"), ("inputs", "csv"), ("inputs", "patterns")}


def diffs(a, b, path=()):
    if path in IGNORED:
        return
    if isinstance(a, dict) and isinstance(b, dict):
        if list(a) != list(b):
            yield path, f"keys {list(a)} != {list(b)}"
        for k in a:
            if k in b:
                yield from diffs(a[k], b[k], path + (k,))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            yield path, f"length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            yield from diffs(x, y, path + (i,))
    elif type(a) is not type(b) or a != b:
        yield path, f"{a!r} != {b!r}"


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    a, b = (json.load(open(p)) for p in sys.argv[1:])
    found = list(diffs(a, b))
    for path, what in found[:20]:
        print("/".join(map(str, path)) or "/", what)
    if found:
        sys.exit(f"{len(found)} differences")
    print(f"{sys.argv[1]} and {sys.argv[2]} are equal apart from git and paths")


if __name__ == "__main__":
    main()

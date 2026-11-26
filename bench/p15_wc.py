# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pyyaml"]
# ///
"""W-C windows from a dataset's Patterns.txt join (SPEC v3 §3.3 `clean_wc`, V2 D-25, P15).

  uv run --python 3.12 bench/p15_wc.py [--check]

Per pattern group: the p90 (numpy's linear interpolation) of the spans of the join's complete
attempts in the fit split, rounded up to the hour, capped at 20 days. FAN-IN and FAN-OUT give
fan_in, fan_out, deg_in and deg_out; SCATTER-GATHER gives sg and sg_ord; CYCLE gives tcyc. The
other typologies (GATHER-SCATTER included) set no window, as in V2 D-25. A group with no complete
fit attempt keeps clean_default's window.

Without --check it writes LI-Small's and HI-Medium's profiles: clean_wc_hi_small.yaml with the
dataset's windows, name and header. HI-Small's clean_wc_hi_small.yaml is hand-written (V2's
dg2_wc windows) and only checked. TransXion has no attempt file and borrows HI-Small's profile
(SPEC v3 §3.3). --check writes nothing and exits 1 when any profile's windows differ from its
join's, or a written profile's text differs from what this tool would write.
"""
import json
import math
import re
import sys
import textwrap
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
BASE = "configs/profiles/clean_wc_hi_small.yaml"
DEFAULT = "configs/profiles/clean_default.yaml"
DATASETS = [  # id, name, join, profile, written by this tool
    ("hi_small", "HI-Small", "results/gt_join.json", BASE, False),
    ("li_small", "LI-Small", "results/gt_join_li_small.json", "configs/profiles/clean_wc_li_small.yaml", True),
    ("hi_medium", "HI-Medium", "results/gt_join_hi_medium.json", "configs/profiles/clean_wc_hi_medium.yaml", True),
]
GROUPS = [  # typologies, their blocks, and how a header names the blocks
    (("FAN-IN", "FAN-OUT"), ("fan_in", "fan_out", "deg_in", "deg_out"), "fan and degree"),
    (("SCATTER-GATHER",), ("sg", "sg_ord"), "sg and sg_ord"),
    (("CYCLE",), ("tcyc",), "tcyc"),
]
CAP = 20 * 86400


def block_windows(path):
    return {b: v["window"] for b, v in yaml.safe_load((ROOT / path).read_text())["blocks"].items()}


def wc_groups(join_path, default):
    """[(typologies, blocks, label, window, attempts)]; attempts 0 means clean_default's window."""
    attempts = json.loads((ROOT / join_path).read_text())["attempts"]
    out = []
    for typs, blocks, label in GROUPS:
        spans = [a["span"] for a in attempts if a["typology"] in typs and a["status"] == "complete"
                 and a["split"] == "fit"]
        if spans:
            w = min(math.ceil(float(np.percentile(spans, 90)) / 3600) * 3600, CAP)
        else:
            w = default[blocks[0]]
            assert all(default[b] == w for b in blocks)
        out.append((typs, blocks, label, w, len(spans)))
    return out


def profile_text(ds_id, name, join_path, groups):
    parts = []
    for typs, _, label, w, n in groups:
        src = (f"{' + '.join(typs)}, {n} attempt{'s' if n != 1 else ''}" if n
               else f"clean_default's, no complete {' + '.join(typs)} fit attempt")
        parts.append(f"{label} {w:,} s ({w // 3600} h; {src})")
    header = (f"Semantic profile, clean family (SPEC v3 §3.3 `clean_wc`, {name}): clean_wc_hi_small with "
              f"{name}'s W-C windows (V2 D-25), the p90 of the spans of {join_path}'s complete fit-split "
              f"attempts per pattern group, rounded up to the hour, capped at 20 days: {', '.join(parts)}. "
              "Bins, k_max and the count blocks as clean_default. 33 columns. Written by bench/p15_wc.py.")
    body = (ROOT / BASE).read_text()
    body = body[body.index("\nversion:") + 1:]
    body, n = re.subn(r"^profile: clean_wc_hi_small$", f"profile: clean_wc_{ds_id}", body, flags=re.M)
    assert n == 1
    window = {b: w for _, blocks, _, w, _ in groups for b in blocks}

    def sub(mt):
        return mt.group(1) + str(window.pop(mt.group(2)))
    body = re.sub(r"^(  (\w+):\s*\{.*window: )\d+", sub, body, flags=re.M)
    assert not window, f"blocks missing from {BASE}: {sorted(window)}"
    return "".join(f"# {ln}\n" for ln in textwrap.wrap(header, 98, break_on_hyphens=False)) + body


def main():
    check = "--check" in sys.argv[1:]
    if [a for a in sys.argv[1:] if a != "--check"]:
        raise SystemExit(__doc__)
    default = block_windows(DEFAULT)
    bad = []
    for ds_id, name, join_path, profile, written in DATASETS:
        groups = wc_groups(join_path, default)
        want = {b: w for _, blocks, _, w, _ in groups for b in blocks}
        print(f"{name}: {', '.join(f'{label} {w} ({n})' for _, _, label, w, n in groups)} <- {join_path}")
        if written:
            text = profile_text(ds_id, name, join_path, groups)
            assert {b: v["window"] for b, v in yaml.safe_load(text)["blocks"].items()} == want
            path = ROOT / profile
            if not check:
                path.write_text(text)
            elif not path.exists() or path.read_text() != text:
                bad.append(f"{profile}: its text differs from what bench/p15_wc.py writes")
                continue
        got = block_windows(profile)
        if got != want:
            bad.append(f"{profile}: windows {got}, {join_path} gives {want}")
    for b in bad:
        print("FAIL", b)
    if bad:
        raise SystemExit(1)
    print("checked" if check else "written")


if __name__ == "__main__":
    main()

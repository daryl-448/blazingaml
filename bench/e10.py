# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "xgboost>=3.4,<3.5", "pyyaml"]
# ///
"""E10 (SPEC v2 §6.3, §9): instance counts and F1 effect of sg_ordered and timestamp visibility.

  uv run --python 3.12 bench/e10.py            # instance counts and validation fits
  uv run --python 3.12 bench/e10.py --test     # also the test split, as sensitivity only

Profiles: v1_default, sg_ordered_v1w (v1_default + sg_ord, stream visibility) and
sg_ordered_tsvis (the same under timestamp visibility). Features are the cpu backend's
(`blz mine`); a missing directory is mined first.

- Instance counts: per block, the sum of its bin columns (instances of size >= max(2, b_0)) and the
  rows with at least one instance, per split.
- Validation (fit -> val, SPEC v2 §3.3): two classifiers at v1 windows from the DG-2 ladder, R0
  (v1: ids and the pattern blocks) and R3/S2 (plus the Multi-GNN columns without Timestamp). The
  set is `all`: the six v1 blocks in v1's order, then sg_ord. v1_default must reproduce the
  ledger's R0 and R3/S2 confusion counts. sg_ordered_tsvis is mined on the prefix [0, 4,062,676)
  (`--edges`, as R5): the validation/test boundary splits a timestamp.
- --test (after the freeze, sensitivity only, never used to select; SPEC v2 §6.3): train on the
  train split, score the test split. sg_ordered_tsvis takes its train rows from the prefix mining
  and its test rows from the full mining.

Writes results/e10_semantics.{json,md}.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "dg2"))
import e8 as E8  # noqa: E402
import ladder as Lad  # noqa: E402
import ledger as L  # noqa: E402

ROOT = Lad.ROOT
OUT = "results/e10_semantics.json"
CUT = 4_062_676  # the train/test boundary (SPEC v2 §3.3)
PROFILES = {
    "v1_default": {"full": Lad.V1_FEATURES},
    "sg_ordered_v1w": {"full": "features/p10_sg_ordered_v1w_cpu"},
    "sg_ordered_tsvis": {"full": "features/p10_sg_ordered_tsvis_cpu", "cut": "features/e10_sg_ordered_tsvis_cut"},
}
BLOCKS = Lad.BLOCK_ORDER + ["sg_ord"]
RECIPES = {"R0": None, "R3/S2": "S2"}  # name -> raw columns
LEDGER_REF = {"R0": ("R0", "v1"), "R3/S2": ("R3", "S2")}


def mine(profile, out, edges=None):
    if (ROOT / out / "features.json").exists():
        return
    cmd = [Lad.BLZ, "mine", "--graph", Lad.GRAPH, "--profile", f"configs/profiles/{profile}.yaml", "--backend", "cpu",
           "--threads", "12", "--out", out] + (["--edges", str(edges)] if edges else [])
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def instance_counts(data, d):
    """Per block and split: instances (the sum of the bin columns) and rows with an instance."""
    f = Lad.Features(data, d)
    m = data.g.m
    parts = {"fit": (0, 3_047_216), "val": (3_047_216, CUT), "test": (CUT, m)}
    out = {}
    for name, (path, off, dt, nc, _) in f.blocks.items():
        arr = np.memmap(path, dtype=dt, mode="r", offset=off, shape=(f.last - f.first, nc))
        out[name] = {}
        for part, (a, b) in parts.items():
            inst = rows = 0
            for i in range(a, b, 1 << 20):
                x = np.asarray(arr[i:min(i + (1 << 20), b)])
                inst += int(x.sum(dtype=np.uint64))
                rows += int((x != 0).any(axis=1).sum())
            out[name][part] = {"instances": inst, "rows": rows}
        out[name]["all"] = {k: sum(out[name][p][k] for p in parts) for k in ("instances", "rows")}
        del arr
    return out


def block_columns(data, dirs, boundary=None):
    """The pattern-block columns, in BLOCKS order (those the features have). dirs = [d] or
    [prefix, full]: rows below `boundary` from the prefix mining, the rest from the full one."""
    fs = [Lad.Features(data, d) for d in dirs]
    cols = []
    for b in BLOCKS:
        if b not in fs[0].blocks:
            continue
        for j, c in enumerate(fs[0].blocks[b][4]):
            if len(fs) == 1:
                cols.append((c, lambda a, z, b=b, j=j: fs[0].rows(b, a, z)[:, j]))
            else:
                def get(a, z, b=b, j=j):
                    if z <= boundary:
                        return fs[0].rows(b, a, z)[:, j]
                    assert a >= boundary, (a, z)
                    return fs[1].rows(b, a, z)[:, j]
                cols.append((c, get))
    return cols


def run_fit(data, recipe, cols):
    t0 = time.time()
    Lad.reset_peak_rss()
    res = Lad.fit(data, recipe, "all", Lad.recipe_columns(data, recipe, "xgb_only") + cols)
    gain = {}
    for n, v in res.pop("total_gain").items():
        gain[E8.block_of(n)] = gain.get(E8.block_of(n), 0.0) + v
    tot = sum(gain.values()) or 1.0
    return {"metrics": res["validation"], "columns": res["columns"]["count"],
            "gain_share_by_block": {b: round(v / tot, 4) for b, v in sorted(gain.items(), key=lambda x: -x[1])},
            "top_gain": res["top_gain"], "rows": res["rows"],
            "xgb": {k: v for k, v in res["xgb"].items() if k != "config"},
            "seconds": round(time.time() - t0, 1), "peak_rss_mb": round(Lad.own_peak_rss_mb())}


def recipe_for(profile, raw, d):
    r = Lad.r0_recipe()
    r["raw"] = raw
    meta = json.loads((ROOT / d / "features.json").read_text())
    r["features"] = {"dir": d, "source": "profile", "profile": f"configs/profiles/{profile}.yaml",
                     "profile_id": meta["profile"]["id"]}
    return r


def confusion(m):
    return {k: m[k] for k in ("tp", "fp", "fn", "tn")}


def fmt_metrics(m):
    return f"{m['f1_x100']:.2f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['pr_auc']:.3f} | {m['tp']} | {m['fp']} | {m['fn']}"


def render(doc):
    L_ = ["# E10: new temporal semantics (SPEC v2 §6.3, §9)", "",
          f"git {doc['git']['sha'][:7]}{' (dirty)' if doc['git']['dirty'] else ''}, {doc['machine']}, {doc['started']}.", "",
          "Profiles: `v1_default`; `sg_ordered_v1w` = v1_default + `sg_ord` (stream visibility); `sg_ordered_tsvis` = "
          "the same under timestamp visibility. Features from the cpu backend; every profile's rows are bit-identical "
          "across oracle, cpu and CUDA (`results/p10_semantics.json`).", "",
          "## Instance counts (HI-Small, all rows)", "",
          "Instances = the sum of a block's bin columns; rows = triggers with at least one instance.", "",
          "| Block | " + " | ".join(f"{p} instances | {p} rows" for p in PROFILES) + " |",
          "|---|" + "---:|" * (2 * len(PROFILES))]
    ic = doc["instance_counts"]
    for b in BLOCKS:
        cells = []
        for p in PROFILES:
            v = ic[p].get(b, {}).get("all")
            cells += [f"{v['instances']:,}", f"{v['rows']:,}"] if v else ["—", "—"]
        L_.append(f"| {b} | " + " | ".join(cells) + " |")
    L_ += ["", "By split, `sg_ord` vs `sg` (instances):", "", "| Profile | Split | sg | sg_ord | sg_ord / sg |", "|---|---|---:|---:|---:|"]
    for p in ("sg_ordered_v1w", "sg_ordered_tsvis"):
        for part in ("fit", "val", "test", "all"):
            s, o = ic[p]["sg"][part]["instances"], ic[p]["sg_ord"][part]["instances"]
            L_.append(f"| {p} | {part} | {s:,} | {o:,} | {o / s:.4f} |" if s else f"| {p} | {part} | 0 | {o:,} | — |")
    for split, title in (("validation", "Validation (fit → val), set `all`"),
                         ("test", "Test (train → test), set `all`: sensitivity only, after the freeze, never used to select")):
        if split not in doc:
            continue
        L_ += ["", f"## {title}", "", "| Classifier | Profile | Columns | F1 ×100 | Precision | Recall | PR-AUC | TP | FP | FN | ΔF1 vs v1_default |",
               "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for rname, rows in doc[split].items():
            base = rows["v1_default"]["metrics"]["f1_x100"]
            for p, r in rows.items():
                d = r["metrics"]["f1_x100"] - base
                L_.append(f"| {rname} | {p} | {r['columns']} | {fmt_metrics(r['metrics'])} | {'' if p == 'v1_default' else f'{d:+.2f}'} |")
        L_ += ["", "Gain share of `sg` and `sg_ord`:", ""]
        for rname, rows in doc[split].items():
            for p, r in rows.items():
                g = r["gain_share_by_block"]
                L_.append(f"- {rname}, {p}: sg {g.get('sg', 0):.4f}, sg_ord {g.get('sg_ord', 0):.4f}")
    L_ += ["", "## Reproduction", ""] + [f"- {k}: {v}" for k, v in doc["reproduction"].items()]
    L_ += ["", "## Notes", "", *[f"- {n}" for n in doc["notes"]], ""]
    return "\n".join(L_)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", action="store_true", help="also score the test split (sensitivity only)")
    a = ap.parse_args()
    data = Lad.Data(False)
    Lad.check_v1_features(data)
    mine("sg_ordered_v1w", PROFILES["sg_ordered_v1w"]["full"])
    mine("sg_ordered_tsvis", PROFILES["sg_ordered_tsvis"]["full"])
    mine("sg_ordered_tsvis", PROFILES["sg_ordered_tsvis"]["cut"], CUT)
    doc = {"experiment": "E10", "spec": "SPEC v2 §6.3, §9", "git": Lad.git_info(), "machine": Lad.machine(),
           "started": Lad.now_utc(), "profiles": {}, "instance_counts": {}}
    for p, dirs in PROFILES.items():
        doc["profiles"][p] = {k: {"dir": d, "profile_id": json.loads((ROOT / d / "features.json").read_text())["profile"]["id"],
                                  "output_fnv1a": {b: v["output_fnv1a"] for b, v in
                                                   json.loads((ROOT / d / "features.json").read_text())["blocks"].items()}}
                              for k, d in dirs.items()}
        doc["instance_counts"][p] = instance_counts(data, dirs["full"])
        print(p, {b: v["all"] for b, v in doc["instance_counts"][p].items()}, flush=True)

    def log(tag, r):
        mm = r["metrics"]
        print(f"{tag}: F1 {mm['f1_x100']:.2f} P {mm['precision']:.3f} R {mm['recall']:.3f} PR-AUC {mm['pr_auc']:.3f} "
              f"tp {mm['tp']} fp {mm['fp']} fn {mm['fn']} ({r['seconds']} s)", flush=True)

    doc["validation"] = {}
    for rname, raw in RECIPES.items():
        doc["validation"][rname] = {}
        for p, dirs in PROFILES.items():
            d = dirs.get("cut", dirs["full"])
            r = run_fit(data, recipe_for(p, raw, d), block_columns(data, [d]))
            doc["validation"][rname][p] = r
            log(f"val {rname} {p}", r)
    led = L.load(str(ROOT / L.LEDGER))
    repro = {}
    for rname, (rung, cid) in LEDGER_REF.items():
        e = next(x for x in led["entries"] if x["rung"] == rung and x.get("candidate") == cid and x.get("set") == "all")
        got = confusion(doc["validation"][rname]["v1_default"]["metrics"])
        repro[f"validation {rname} v1_default vs ledger {rung}/{cid}"] = "reproduces" if got == confusion(e["validation"]) \
            else f"DIFFERS: {got} vs {confusion(e['validation'])}"
    if a.test:
        train, test = (0, CUT), (CUT, data.g.m)
        assert data.g.splits()["train"] == train and data.g.splits()["test"] == test
        data.splits = {"fit": train, "val": test}
        doc["test"] = {}
        for rname, raw in RECIPES.items():
            doc["test"][rname] = {}
            for p, dirs in PROFILES.items():
                ds = [dirs["cut"], dirs["full"]] if "cut" in dirs else [dirs["full"]]
                r = run_fit(data, recipe_for(p, raw, dirs["full"]), block_columns(data, ds, CUT))
                doc["test"][rname][p] = r
                log(f"test {rname} {p}", r)
        e8 = json.loads((ROOT / E8.OUT).read_text())
        got = confusion(doc["test"]["R0"]["v1_default"]["metrics"])
        want = confusion(e8["v1_default"]["all"]["metrics"])
        repro["test R0 v1_default vs E8 v1_default all"] = "reproduces" if got == want else f"DIFFERS: {got} vs {want}"
    doc["reproduction"] = repro
    doc["notes"] = [
        "Classifiers: the DG-2 ladder's R0 recipe (v1 ids, XGBoost scale_pos_weight 1, max_depth 6, 100 rounds, "
        "base_score estimated) and R3/S2 (the same plus the Multi-GNN columns without Timestamp), both at v1 windows. "
        "The frozen DG-2 recipe uses GFP-produced blocks, so it has no compiled sg_ord to add.",
        f"sg_ordered_tsvis on validation is mined on the stream prefix [0, {CUT:,}) (blz mine --edges), so the 213 "
        "validation rows at timestamp 663,120 do not see its 105 test edges (SPEC v2 §4.4, R5). On test, its train rows "
        "come from that prefix mining and its test rows from the full mining.",
        "Test scores are sensitivity only: DG-2 is closed and frozen (E8), and nothing here selects a recipe.",
    ]
    fits = [r for s in ("validation", "test") if s in doc for rows in doc[s].values() for r in rows.values()]
    top = max(r["gain_share_by_block"].get("sg_ord", 0.0) for r in fits)
    fit_inst = doc["instance_counts"]["sg_ordered_v1w"]["sg_ord"]["fit"]["instances"]
    doc["notes"].append(
        f"At v1's sg window (21,600 s) sg_ord is sparse: {fit_inst} instances in the fit split. Its largest gain share "
        f"in any fit here is {top:.4f}, so it cannot move F1 at these windows. Wider windows (R4's W-C) were not "
        "pre-registered for E10 and are not run.")
    doc["finished"] = Lad.now_utc()
    out = ROOT / OUT
    out.write_text(json.dumps(doc, indent=1) + "\n")
    out.with_suffix(".md").write_text(render(doc))
    print(f"-> {out}, {out.with_suffix('.md')}")
    for k, v in repro.items():
        print(f"{k}: {v}")
    return 0 if all(v == "reproduces" for v in repro.values()) else 2


if __name__ == "__main__":
    sys.exit(main())

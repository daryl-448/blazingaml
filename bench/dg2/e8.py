# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "xgboost>=3.4,<3.5", "pyyaml"]
# ///
"""E8 (SPEC v2 §4.5 step 3, §4.6): the final Table 2/3 for v1_default and the frozen DG-2 recipe.

  uv run --python 3.12 bench/dg2/e8.py --dry-run [--out <scratch>.json]
  uv run --python 3.12 bench/dg2/e8.py

For v1_default and for configs/dg2_frozen.yaml, each fit trains on the train split
[0, 4,062,676) and scores the test split. It runs:
- the five cumulative Table 2 sets (SPEC §10.1). Base columns (ids, raw columns, vertex statistics)
  are in every set, and the pattern blocks are added in v1's order;
- the Table 3 matrix (the `all` confusion);
- per-block gain importances;
- for the frozen recipe, the v1 E6 / R8 grid on `all`, reported as sensitivity and never used to
  select.

v1_default is also trained by `blz train` (R11). Both tools must reproduce v1's E2/E3 confusion
counts (results/e2_table2.json). The verdict follows §4.6 [D-21]. Writes
results/e8_table2_v2.{json,md}.

The test split is scored once: the real run refuses when configs/dg2_frozen.yaml is not committed
unchanged, when the features differ from the frozen output_sha256, or when its output exists.

--dry-run uses the fit split as "train" and the validation split as "test", so the frozen recipe
never sees a test row. It must reproduce the ledger's numbers: R0's `xgb_only` and `all`, and
R8/X7's `all`. It also runs `blz train` once, for v1_default's xgb_only (a test score E2 already
reported), to check that path. It writes to a scratch path (default $TMPDIR/e8_dry.json).
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ladder as Lad  # noqa: E402
import ledger as L  # noqa: E402

ROOT = Lad.ROOT
FROZEN = "configs/dg2_frozen.yaml"
OUT = "results/e8_table2_v2.json"
E2 = "results/e2_table2.json"
SETS = {"xgb_only": [], "fan": Lad.BLOCK_ORDER[:2], "fan_deg": Lad.BLOCK_ORDER[:4],
        "fan_deg_cyc": Lad.BLOCK_ORDER[:5], "all": Lad.BLOCK_ORDER}
PAPER_T2 = {"xgb_only": 11.1, "fan": 23.2, "fan_deg": 38.4, "fan_deg_cyc": 42.6, "all": 46.6}  # [P Table 2]
PAPER_T3 = {"tp": 558, "fn": 1239, "fp": 31, "tn": 1013841, "precision": 0.947, "recall": 0.311}  # [P §7.2]
GRID = [Lad.r0_recipe()["xgb"]] + list(L.GRID.values())  # v1 E6: R8's seven points and its incumbent's


def set_columns(data, recipe, set_name):
    """Base columns (recipe_columns' xgb_only: ids, raw, vertex statistics), then SETS' blocks."""
    cols = Lad.recipe_columns(data, recipe, "xgb_only")
    f = Lad.Features(data, recipe["features"]["dir"])
    for b in SETS[set_name]:
        cols += f.columns(b)
    if set_name == "all":
        want = [c for c, _ in Lad.recipe_columns(data, recipe, "all")]
        if [c for c, _ in cols] != want:
            raise SystemExit("set_columns(all) differs from the ladder's all")
    return cols


def block_of(name):
    if name in ("src_id", "dst_id", "from_id", "to_id", "EdgeID", "eid"):
        return "ids"
    if name in Lad.RAW["S1"]:
        return "raw"
    if name.startswith("vs_"):
        return "vstats"
    return name.split("[")[0]


def run_fit(data, recipe, set_name):
    t0 = time.time()
    Lad.reset_peak_rss()
    res = Lad.fit(data, recipe, set_name, set_columns(data, recipe, set_name))
    gain = {}
    for n, v in res.pop("total_gain").items():
        gain[block_of(n)] = gain.get(block_of(n), 0.0) + v
    tot = sum(gain.values()) or 1.0
    return {"set": set_name, "recipe": recipe, "metrics": res.pop("validation"), "columns": res["columns"],
            "gain_share_by_block": {b: round(v / tot, 4) for b, v in sorted(gain.items(), key=lambda x: -x[1])},
            "top_gain": res["top_gain"], "rows": res["rows"],
            "xgb": {k: v for k, v in res["xgb"].items() if k != "config"}, "xgb_config": res["xgb"]["config"],
            "fit_seconds": res["seconds"], "seconds": round(time.time() - t0, 1),
            "peak_rss_mb": round(Lad.own_peak_rss_mb())}


def confusion(m):
    return {k: m[k] for k in ("tp", "fp", "fn", "tn")}


def blz_train(set_name):
    """v1_default through `blz train` (the C API path, R11): train on train, score test."""
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "t.json")
        subprocess.run([Lad.BLZ, "train", "--graph", Lad.GRAPH, "--features", Lad.V1_FEATURES, "--set", set_name,
                        "--threads", "12", "--out", out], cwd=ROOT, check=True, capture_output=True)
        r = json.loads(Path(out).read_text())[set_name]
    m = r["metrics"]
    return {"set": set_name, "metrics": {"f1_x100": m["f1_x100"], "precision": m["precision"], "recall": m["recall"],
                                         "pr_auc": m["pr_auc"], **m["confusion"]},
            "columns": len(r["columns"]), "xgb": r["xgb"]}


def data_sha256(path):
    """sha256 of a .npy file's array data, without its header (tools/gfp/dg2_features.py NpyWriter)."""
    off = np.load(path, mmap_mode="r").offset
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(off)
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def check_frozen(data, real):
    """The frozen recipe: equal to the ledger's incumbent, committed unchanged (real run), and its
    feature files hash to the frozen output_sha256."""
    fz = yaml.safe_load((ROOT / FROZEN).read_text())
    led = L.load(str(ROOT / L.LEDGER))
    if fz["recipe"] != led["incumbent"]["recipe"] or fz["source"] != led["incumbent"]["source"]:
        raise SystemExit(f"{FROZEN} is not the ledger's incumbent")
    if real:
        st = subprocess.run(["git", "status", "--porcelain", "--", FROZEN, L.LEDGER, "bench/dg2"], cwd=ROOT,
                            capture_output=True, text=True).stdout
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", FROZEN], cwd=ROOT, capture_output=True)
        if st.strip() or tracked.returncode != 0:
            raise SystemExit(f"{FROZEN}, the ledger and bench/dg2 must be committed unchanged before E8 (§4.5)")
    want = {f"{fz['gfp_recipe']['dir']}/{b}.npy": v["output_sha256"] for b, v in fz["gfp_recipe"]["blocks"].items()}
    want[f"{fz['families']['F1']['dir']}/vstats.npy"] = fz["families"]["F1"]["vstats"]["output_sha256"]
    for path, sha in want.items():
        if data_sha256(ROOT / path) != sha:
            raise SystemExit(f"{path}: sha256 differs from {FROZEN}")
    return fz


def verdict(rows):
    """§4.6 [D-21] on the frozen recipe's five Table 2 cells and its Table 3 precision/recall."""
    d = {s: rows[s]["metrics"]["f1_x100"] - PAPER_T2[s] for s in SETS}
    m = rows["all"]["metrics"]
    dp, dr = m["precision"] - PAPER_T3["precision"], m["recall"] - PAPER_T3["recall"]
    if all(abs(v) <= 3.0 for v in d.values()) and abs(dp) <= 0.03 and abs(dr) <= 0.03:
        v = "AGREE"
    elif abs(d["all"]) <= 5.0:
        v = "PARTIAL"
    else:
        v = "GAP"
    return {"verdict": v, "f1_minus_paper": {s: round(x, 2) for s, x in d.items()},
            "precision_minus_paper": round(dp, 4), "recall_minus_paper": round(dr, 4)}


def render(doc):
    t2 = ["xgb_only", "fan", "fan_deg", "fan_deg_cyc", "all"]
    hdr = "| | XGB Only | Fan | Fan+Degree | Fan+Degree+Cycle | Fan+Degree+Cycle+SG |"
    sep = "|---|---:|---:|---:|---:|---:|"
    f1 = lambda rows: " | ".join(f"{rows[s]['metrics']['f1_x100']:.1f}" for s in t2)
    out = [f"# E8: final Table 2/3, v1_default and the frozen DG-2 recipe (SPEC v2 §4.5, §4.6){' — DRY RUN' if doc['dry_run'] else ''}", "",
           f"Train rows {doc['splits']['train']}, scored rows {doc['splits']['test']}. Frozen recipe: `{FROZEN}` "
           f"({doc['frozen']['source']}). Git: `{doc['git']['sha']}`.", "",
           "Every set holds the recipe's base columns (ids, raw columns, vertex statistics; §4.5 step 3), so "
           "the frozen recipe's XGB Only cell is not the paper's raw-attribute XGBoost.", "",
           "## Table 2 (F1 ×100)", "", hdr, sep,
           "| Paper | " + " | ".join(f"{PAPER_T2[s]:.1f}" for s in t2) + " |"]
    if doc.get("v1_default_blz"):
        out.append("| v1_default, `blz train` | " + f1(doc["v1_default_blz"]) + " |")
    out += ["| v1_default, Python | " + f1(doc["v1_default"]) + " |",
            "| **Frozen recipe** | " + " | ".join(f"**{doc['frozen_recipe'][s]['metrics']['f1_x100']:.1f}**" for s in t2) + " |"]
    for k, lab in (("precision", "Precision"), ("recall", "Recall"), ("pr_auc", "PR-AUC")):
        out.append(f"| Frozen {lab} | " + " | ".join(f"{doc['frozen_recipe'][s]['metrics'][k]:.3f}" for s in t2) + " |")
    out.append("| Frozen columns | " + " | ".join(str(doc["frozen_recipe"][s]["columns"]["count"]) for s in t2) + " |")
    m, v = doc["frozen_recipe"]["all"]["metrics"], doc["v1_default"]["all"]["metrics"]
    out += ["", "## Table 3 (`all`; paper, then v1_default, then the frozen recipe)", "",
            "| | Predicted laundering | Predicted non-laundering |", "|---|---:|---:|",
            f"| Actual laundering | {PAPER_T3['tp']} / {v['tp']} / **{m['tp']}** | {PAPER_T3['fn']} / {v['fn']} / **{m['fn']}** |",
            f"| Actual non-laundering | {PAPER_T3['fp']} / {v['fp']} / **{m['fp']}** | {PAPER_T3['tn']} / {v['tn']} / **{m['tn']}** |",
            "", f"Frozen: precision {m['precision']:.3f} (paper 0.947), recall {m['recall']:.3f} (paper 0.311)."]
    if doc.get("verdict"):
        vd = doc["verdict"]
        out += ["", f"## Verdict (§4.6, D-21): **{vd['verdict']}**", "",
                "F1 minus paper: " + ", ".join(f"{s} {vd['f1_minus_paper'][s]:+.1f}" for s in t2) +
                f"; precision {vd['precision_minus_paper']:+.3f}, recall {vd['recall_minus_paper']:+.3f}."]
    out += ["", "## v1_default reproduction (R11)", ""]
    for k, r in doc["reproduction"].items():
        out.append(f"- {k}: {r}")
    out += ["", "## Sensitivity: the R8 grid on the frozen recipe, `all` (never used to select)", "",
            "| scale_pos_weight | max_depth | rounds | F1 ×100 | precision | recall | PR-AUC |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for g in doc["grid"]:
        x, mm = g["xgb_point"], g["metrics"]
        star = " (frozen)" if g.get("frozen_point") else ""
        out.append(f"| {x['scale_pos_weight']} | {x['max_depth']} | {x['num_boost_round']}{star} | {mm['f1_x100']:.1f} | "
                   f"{mm['precision']:.3f} | {mm['recall']:.3f} | {mm['pr_auc']:.3f} |")
    out += ["", "## Gain share by block, frozen recipe", ""]
    blocks = ["ids", "raw", "vstats"] + Lad.BLOCK_ORDER
    out += ["| set | " + " | ".join(blocks) + " |", "|---|" + "---:|" * len(blocks)]
    for s in t2:
        g = doc["frozen_recipe"][s]["gain_share_by_block"]
        out.append(f"| {s} | " + " | ".join(f"{g.get(b, 0):.2f}" if b in g else "-" for b in blocks) + " |")
    out += ["", "DG-3: the frozen recipe is identified with GFP-produced features (G1 blocks, F1 vertex "
            "statistics); not yet compiled (SPEC v2 §13)."]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    real = not a.dry_run
    out = Path(a.out) if a.out else (ROOT / OUT if real else Path(os.environ.get("TMPDIR", "/tmp")) / "e8_dry.json")
    if real and (a.out or (ROOT / OUT).exists()):
        raise SystemExit(f"E8 writes {OUT} once (§4.5); it exists or --out was given")
    data = Lad.Data(False)
    fz = check_frozen(data, real)
    m = data.g.m
    train, test = (0, 4_062_676), (4_062_676, m)
    if real:
        assert data.g.splits()["train"] == train and data.g.splits()["test"] == test
        data.splits = {"fit": train, "val": test}
    else:
        train, test = data.splits["fit"], data.splits["val"]
    print(f"E8{' dry run' if not real else ''}: train {train}, scored {test}", flush=True)
    doc = {"experiment": "E8", "spec": "SPEC v2 §4.5, §4.6", "dry_run": not real,
           "splits": {"train": list(train), "test": list(test)}, "frozen": {"file": FROZEN, "source": fz["source"],
                                                                            "recipe": fz["recipe"]},
           "git": Lad.git_info(), "machine": Lad.machine(), "started": Lad.now_utc()}

    def log(tag, r):
        mm = r["metrics"]
        print(f"{tag}: F1 {mm['f1_x100']:.2f} P {mm['precision']:.3f} R {mm['recall']:.3f} PR-AUC {mm['pr_auc']:.3f} "
              f"tp {mm['tp']} fp {mm['fp']} fn {mm['fn']} ({r.get('seconds', 0)} s, {r.get('peak_rss_mb', 0)} MB)",
              flush=True)

    v1 = Lad.r0_recipe()
    Lad.check_v1_features(data)
    doc["v1_default"] = {}
    for s in SETS:
        doc["v1_default"][s] = run_fit(data, v1, s)
        log(f"v1_default {s}", doc["v1_default"][s])
    repro = {}
    e2 = json.loads((ROOT / E2).read_text())
    if not real:  # the blz train path, on v1's xgb_only only: a test score E2 already reported
        r = blz_train("xgb_only")
        repro["blz train xgb_only vs E2 (test split)"] = "reproduces" if confusion(r["metrics"]) == \
            e2["xgb_only"]["metrics"]["confusion"] else f"DIFFERS: {confusion(r['metrics'])}"
    if real:
        doc["v1_default_blz"] = {}
        for s in SETS:
            doc["v1_default_blz"][s] = blz_train(s)
            log(f"v1_default blz {s}", doc["v1_default_blz"][s])
        for s in SETS:
            want = e2[s]["metrics"]["confusion"]
            repro[f"python {s}"] = "reproduces E2" if confusion(doc["v1_default"][s]["metrics"]) == want else \
                f"DIFFERS: {confusion(doc['v1_default'][s]['metrics'])} vs E2 {want}"
            repro[f"blz train {s}"] = "reproduces E2" if confusion(doc["v1_default_blz"][s]["metrics"]) == want else \
                f"DIFFERS: {confusion(doc['v1_default_blz'][s]['metrics'])} vs E2 {want}"
    else:
        led = L.load(str(ROOT / L.LEDGER))
        for s in ("xgb_only", "all"):
            e = next(x for x in led["entries"] if x["rung"] == "R0" and x.get("set") == s)
            repro[f"python {s} vs ledger R0"] = "reproduces" if confusion(doc["v1_default"][s]["metrics"]) == \
                confusion(e["validation"]) else f"DIFFERS: {confusion(doc['v1_default'][s]['metrics'])} vs {confusion(e['validation'])}"
    doc["frozen_recipe"] = {}
    for s in SETS:
        doc["frozen_recipe"][s] = run_fit(data, fz["recipe"], s)
        log(f"frozen {s}", doc["frozen_recipe"][s])
    if not real:
        led = L.load(str(ROOT / L.LEDGER))
        src_rung, src_cid = fz["source"].split("/")
        e = next(x for x in led["entries"] if x["rung"] == src_rung and x["candidate"] == src_cid and x.get("set") == "all")
        repro["frozen all vs ledger " + fz["source"]] = "reproduces" if confusion(doc["frozen_recipe"]["all"]["metrics"]) == \
            confusion(e["validation"]) else f"DIFFERS: {confusion(doc['frozen_recipe']['all']['metrics'])} vs {confusion(e['validation'])}"
    doc["reproduction"] = repro
    doc["grid"] = []
    for x in GRID:
        if x == fz["recipe"]["xgb"]:
            r = dict(doc["frozen_recipe"]["all"], frozen_point=True)
        else:
            r = run_fit(data, dict(fz["recipe"], xgb=x), "all")
            log(f"grid {x}", r)
        doc["grid"].append({"xgb_point": x, **{k: r[k] for k in ("metrics", "seconds", "peak_rss_mb")},
                            **({"frozen_point": True} if r.get("frozen_point") else {})})
    if real:
        doc["verdict"] = verdict(doc["frozen_recipe"])
        doc["verdict_v1_default"] = verdict(doc["v1_default"])
    doc["finished"] = Lad.now_utc()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    out.with_suffix(".md").write_text(render(doc))
    print(f"-> {out}, {out.with_suffix('.md')}")
    for k, r in repro.items():
        print(f"{k}: {r}")
    if doc.get("verdict"):
        print("verdict:", doc["verdict"])
    return 0 if all(v.startswith("reproduces") for v in repro.values()) else 2


if __name__ == "__main__":
    sys.exit(main())

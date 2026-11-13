# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "xgboost>=3.4,<3.5", "pyyaml"]
# ///
"""E9 detection inputs (SPEC v2 §5): per-edge predictions of the models E9 scores attempts with.

  uv run --python 3.12 bench/e9/predict.py [--out features/e9] [--profiles v1_default,frozen]

Neither the ladder nor E8 kept a model or its per-row predictions, so this refits them with the
same code (bench/dg2/ladder.py fit(), configs/xgb_default.yaml, nthread = every core) on the
`all` set:

- validation rows: trained on fit, scored on validation. For v1_default that is ledger R0's
  `all` fit; for the frozen recipe, the validation-selected model (configs/dg2_frozen.yaml
  `source`, R8/X7).
- test rows: trained on train, scored on test: E8's model.

Each refit must give exactly the confusion counts its original recorded (results/dg2_ledger.json,
results/e8_table2_v2.json), or nothing is written and the exit code is 2. No recipe changes and
nothing is selected here: E9 is descriptive and never feeds back (§5, §4.5 step 4).

Writes, per profile p:
- <out>/pred_<p>.npy: u8[m], 0 = predicted negative, 1 = predicted positive (score > threshold),
  255 = not scored. Fit rows are training data and are never scored.
- <out>/scores_<p>.npy: f32[m] scores, NaN where not scored.
- <out>/pred_<p>.json: the models, row ranges, threshold, confusion counts and checks.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench" / "dg2"))
import e8 as E8  # noqa: E402
import ladder as Lad  # noqa: E402
import ledger as L  # noqa: E402

E8_OUT = "results/e8_table2_v2.json"
NOT_SCORED = 255


def confusion(m):
    return {k: int(m[k]) for k in ("tp", "fp", "fn", "tn")}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="features/e9")
    ap.add_argument("--profiles", default="v1_default,frozen")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)

    data = Lad.Data(False)
    g, m = data.g, data.g.m
    sp = g.splits()
    fit_r, val_r, test_r, train_r = sp["fit"], sp["val"], sp["test"], sp["train"]
    fz = E8.check_frozen(data, real=False)  # the ledger's incumbent; features hash to the frozen sha256
    Lad.check_v1_features(data)
    led = L.load(str(ROOT / L.LEDGER))
    e8 = json.loads((ROOT / E8_OUT).read_text())
    src_rung, src_cid = fz["source"].split("/")

    def ledger_entry(rung, cid):
        return next(x for x in led["entries"] if x["rung"] == rung and x.get("candidate") == cid and x.get("set") == "all")

    profiles = {
        "v1_default": {"recipe": Lad.r0_recipe(), "val_ref": ("ledger R0 all", ledger_entry("R0", "v1")["validation"]),
                       "test_ref": ("E8 v1_default all", e8["v1_default"]["all"]["metrics"])},
        "frozen": {"recipe": fz["recipe"],
                   "val_ref": (f"ledger {fz['source']} all", ledger_entry(src_rung, src_cid)["validation"]),
                   "test_ref": ("E8 frozen_recipe all", e8["frozen_recipe"]["all"]["metrics"])},
    }
    if e8["frozen"]["recipe"] != fz["recipe"]:
        raise SystemExit("E8's frozen recipe differs from configs/dg2_frozen.yaml")
    status = 0
    for name in a.profiles.split(","):
        p = profiles[name]
        pred = np.full(m, NOT_SCORED, dtype=np.uint8)
        scores = np.full(m, np.nan, dtype=np.float32)
        doc = {"producer": "bench/e9/predict.py", "spec": "SPEC v2 §5", "profile": name, "recipe": p["recipe"],
               "set": "all", "codes": {"0": "predicted negative", "1": "predicted positive",
                                       "255": "not scored (fit rows are training data)"},
               "git": Lad.git_info(), "machine": Lad.machine(), "started": Lad.now_utc(), "ranges": {}}
        ok = True
        for part, (tr, sc), ref in (("val", (fit_r, val_r), p["val_ref"]), ("test", (train_r, test_r), p["test_ref"])):
            data.splits = {"fit": tr, "val": sc}
            t0 = time.time()
            Lad.reset_peak_rss()
            res = Lad.fit(data, p["recipe"], "all", keep_scores=True)
            s = res.pop("scores")
            thr = res["xgb"]["threshold"]
            got, want = confusion(res["validation"]), confusion(ref[1])
            same = got == want
            ok &= same
            pred[sc[0]:sc[1]] = (s > thr).astype(np.uint8)
            scores[sc[0]:sc[1]] = s
            doc["ranges"][part] = {
                "rows": list(sc), "trained_on": list(tr), "threshold": thr,
                "model": ("the validation-selected model" if part == "val" else "E8's model") + f" ({ref[0]})",
                "confusion": got, "reference": {"source": ref[0], "confusion": want},
                "reproduces": same, "metrics": {k: res["validation"][k] for k in ("f1_x100", "precision", "recall", "pr_auc")},
                "columns": res["columns"]["count"], "xgb_params": res["xgb"]["params"],
                "num_boost_round": res["xgb"]["num_boost_round"], "base_score_resolved": res["xgb"]["base_score_resolved"],
                "seconds": round(time.time() - t0, 1), "peak_rss_mb": round(Lad.own_peak_rss_mb())}
            print(f"{name} {part}: {got} vs {ref[0]} {want}: {'reproduces' if same else 'DIFFERS'} "
                  f"({doc['ranges'][part]['seconds']} s, {doc['ranges'][part]['peak_rss_mb']} MB)", flush=True)
        if (pred[fit_r[0]:fit_r[1]] != NOT_SCORED).any():
            raise SystemExit("a fit row was scored")
        if not ok:
            print(f"{name}: a refit does not reproduce its original; nothing written", flush=True)
            status = 2
            continue
        np.save(out / f"pred_{name}.npy", pred)
        np.save(out / f"scores_{name}.npy", scores)
        doc["file"] = f"pred_{name}.npy"
        doc["sha256"] = sha256(out / f"pred_{name}.npy")
        doc["scores_file"] = f"scores_{name}.npy"
        doc["finished"] = Lad.now_utc()
        (out / f"pred_{name}.json").write_text(json.dumps(doc, indent=1) + "\n")
        print(f"-> {out / doc['file']}", flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())

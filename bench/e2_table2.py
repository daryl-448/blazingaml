"""Render results/e2_table2.json (blz train, SPEC §12.2 E2/E3) as markdown next to the paper."""
import json
import sys

PAPER_F1 = {"xgb_only": 11.1, "fan": 23.2, "fan_deg": 38.4, "fan_deg_cyc": 42.6, "all": 46.6}
PAPER_CM = {"tp": 558, "fn": 1239, "fp": 31, "tn": 1013841}  # Table 3
LABELS = {"xgb_only": "XGB Only", "fan": "Fan", "fan_deg": "Fan+Degree",
          "fan_deg_cyc": "Fan+Degree+Cycle", "all": "Fan+Degree+Cycle+SG"}

doc = json.load(open(sys.argv[1]))
sets = [s for s in PAPER_F1 if s in doc]
out = ["# E2/E3: HI-Small, oracle features, default XGBoost (SPEC §10)", ""]
out += ["| | " + " | ".join(LABELS[s] for s in sets) + " |", "|---|" + "---:|" * len(sets)]
out.append("| Paper F1 (Table 2) | " + " | ".join(f"{PAPER_F1[s]:.1f}" for s in sets) + " |")
row = lambda name, f: out.append(f"| {name} | " + " | ".join(f(doc[s]["metrics"]) for s in sets) + " |")
row("**This repo F1**", lambda m: f"**{m['f1_x100']:.1f}**")
row("Precision", lambda m: f"{m['precision']:.3f}")
row("Recall", lambda m: f"{m['recall']:.3f}")
row("PR-AUC", lambda m: f"{m['pr_auc']:.3f}")
out.append("| Columns | " + " | ".join(str(len(doc[s]["columns"])) for s in sets) + " |")
if "all" in doc:
    cm = doc["all"]["metrics"]["confusion"]
    out += ["", "Confusion matrix, `all` (Table 3 layout; paper in parentheses):", "",
            "| | Predicted laundering | Predicted non-laundering |", "|---|---:|---:|",
            f"| Actual laundering | {cm['tp']} ({PAPER_CM['tp']}) | {cm['fn']} ({PAPER_CM['fn']}) |",
            f"| Actual non-laundering | {cm['fp']} ({PAPER_CM['fp']}) | {cm['tn']} ({PAPER_CM['tn']}) |"]
any_run = doc[sets[0]]
x = any_run["xgb"]
out += ["", f"XGBoost {x['version']}, `{x['config']}`, {x['num_boost_round']} rounds, threshold "
        f"{x['threshold']}, {x['nthread']} threads. Test positives: "
        f"{any_run['split']['test_positives']}. Git: `{(any_run.get('git') or {}).get('sha', '?')}`."]
print("\n".join(out))

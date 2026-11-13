"""E9 (SPEC v2 §5): merge the `blz coverage` outputs of v1_default and the frozen recipe into
results/e9_coverage.{json,md}. Standard library only.

  python3 bench/e9/report.py [--dir features/e9] [--out results/e9_coverage.json]

Reads <dir>/coverage_<profile>.json (blz coverage), <dir>/pred_<profile>.json
(bench/e9/predict.py) and, through the coverage files, the GFP eligibility
(tools/gfp/coverage_gfp.py). E9 is descriptive: it is computed after the freeze and never feeds
back into the DG-2 ladder (§5).
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROFILES = {"v1_default": "v1_default", "frozen": "frozen recipe"}
FROZEN = "configs/dg2_frozen.yaml"
V1_PROFILE_ID = "b21580d6e129821a"
SUMMARY_KEYS = ("experiment", "spec", "inputs", "profile", "features", "eligibility_source", "pred", "git",
                "blocks", "eligibility", "splits", "scored", "by_typology", "positives", "base_rates", "seconds",
                "peak_rss_mb")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_info():
    run = lambda *a: subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True).stdout.strip()
    return {"sha": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def pct(k, n):
    return "–" if not n else f"{100.0 * k / n:.0f}"


def share(x, digits=2):
    return "–" if x is None else f"{x:.{digits}f}"


def dump(doc):
    """JSON with one line per attempt."""
    parts = []
    for k, v in doc.items():
        if k == "attempts":
            body = ",\n".join("  " + json.dumps(a, ensure_ascii=False) for a in v)
            parts.append(f' "attempts": [\n{body}\n ]')
        else:
            parts.append(f" {json.dumps(k)}: " + json.dumps(v, indent=1, ensure_ascii=False).replace("\n", "\n "))
    return "{\n" + ",\n".join(parts) + "\n}\n"


def models(pred):
    return {part: {k: r[k] for k in ("model", "rows", "trained_on", "threshold", "confusion", "reference", "reproduces",
                                     "metrics", "columns")}
            for part, r in pred["ranges"].items()}


def render(doc):
    P = doc["profiles"]
    v1, fz = P["v1_default"], P["frozen"]
    blocks = v1["blocks"]
    if fz["blocks"] != blocks:
        raise SystemExit("the two profiles have different blocks")
    typs = list(v1["by_typology"])
    out = ["# E9: ground-truth coverage (SPEC v2 §5)", "",
           f"Git `{doc['git']['sha'][:12]}`{' (dirty)' if doc['git']['dirty'] else ''}. Attempts: "
           f"`{doc['inputs']['gt_join']}` (§3.5). Descriptive only: computed after the freeze, never fed back (§5).", "",
           "- **Structurally eligible**: the block gives a non-zero row to at least one of the attempt's edges when "
           "the graph holds only those edges (eids kept).",
           "- **Activated**: the block gives a non-zero row to at least one attempt edge in the full HI-Small mining.",
           "- **Detected**: the share of the attempt's validation (test) edges that the validation-selected "
           "(E8's) model predicts positive. Fit edges are training data and are never scored.", "",
           "| Profile | Pattern blocks | Eligibility | Models |", "|---|---|---|---|"]
    for name, p in P.items():
        f = p["features"]
        blk = (f"`blz mine` ({f['backend']}), profile `{p['profile']['name']}` {p['profile']['id']}" if f["source"] == "blz"
               else f"GFP {f['gfp_recipe']} (`{f['dir']}`), windows and bins of `{p['profile']['name']}` "
                    f"{p['profile']['id']}")
        el = p["eligibility_source"]
        eli = ("the oracle, profile `" + p["profile"]["name"] + "`" if el["source"] == "oracle"
               else f"GFP {el['gfp_recipe']}, each edge in its full-stream batch (D-33)")
        mods = "; ".join(f"{part}: {m['reference']['source']}, TP {m['confusion']['tp']} FP {m['confusion']['fp']} "
                         f"FN {m['confusion']['fn']} ({'reproduces' if m['reproduces'] else 'DIFFERS'})"
                         for part, m in p["models"].items())
        out.append(f"| {PROFILES[name]} | {blk} | {eli} | {mods} |")
    out += ["", "The frozen recipe's F1 vertex statistics are aggregates, not pattern blocks: they have no "
            "eligibility or activation here, but they are in its detection model.", ""]

    # Attempts.
    out += ["## Attempts", "", "| Typology | Attempts | Complete | fit | val | test | straddles | Edges | "
            "Span p10 (h) | p50 | p90 | max |", "|---|" + "---:|" * 11]
    for t in typs:
        x = v1["by_typology"][t]
        s, sp = x["split"], x["span_hours"]
        f1 = lambda v: "–" if v is None else f"{v:.0f}"
        out.append(f"| {'**ALL**' if t == 'ALL' else t} | {x['attempts']} | {x['complete']} | {s['fit']} | {s['val']} | "
                   f"{s['test']} | {s['straddles']} | {x['edges']} | {f1(sp['p10'])} | {f1(sp['p50'])} | "
                   f"{f1(sp['p90'])} | {f1(sp['max'])} |")

    # Eligibility and activation.
    for name, p in P.items():
        out += ["", f"## Eligible / activated, {PROFILES[name]} (% of complete attempts)", "",
                "| Typology | " + " | ".join(blocks) + " | any |", "|---|" + "---:|" * (len(blocks) + 1)]
        for t in typs:
            x = p["by_typology"][t]
            n = x["complete"]
            cells = [f"{pct(x['eligible'][b], n)} / {pct(x['activated'][b], n)}" for b in blocks + ["any"]]
            out.append(f"| {'**ALL**' if t == 'ALL' else t} | " + " | ".join(cells) + " |")
        a = p["by_typology"]["ALL"]
        ena = {b: a["eligible_not_activated"][b] for b in blocks + ["any"] if a["eligible_not_activated"][b]}
        out += ["", f"Eligible but not activated: {ena if ena else 'none'} (each pattern's count only grows when "
                "edges are added, so none is expected)."]
        if "eligible_oracle" in a:
            diff = sum(1 for at in doc["attempts"] if name in at and "eligible" in at[name]
                       for e, o in zip(at[name]["eligible"], at[name]["eligible_oracle"]) if (e > 0) != (o > 0))
            pairs = a["complete"] * len(blocks)
            out.append(f"GFP eligibility against blz's (the oracle, profile `{p['profile']['name']}`, stream mode): "
                       f"{diff} of {pairs} (attempt, block) pairs differ. Oracle, ALL: " +
                       ", ".join(f"{b} {pct(a['eligible_oracle'][b], a['complete'])}" for b in blocks + ["any"]) + " %.")

    # Detection.
    out += ["", "## Detection", "",
            "Mean of the per-attempt shares, over the complete attempts with edges in the split; n = attempts / edges. "
            "The pooled share (predicted edges / edges) is in the JSON.", "",
            "| Typology | v1_default val | v1_default test | frozen val | frozen test | n val | n test |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for t in typs:
        a, b = v1["by_typology"][t], fz["by_typology"][t]
        out.append(f"| {'**ALL**' if t == 'ALL' else t} | {share(a['det_val']['mean'])} | {share(a['det_test']['mean'])} | "
                   f"{share(b['det_val']['mean'])} | {share(b['det_test']['mean'])} | "
                   f"{a['det_val']['attempts']} / {a['det_val']['edges']} | {a['det_test']['attempts']} / {a['det_test']['edges']} |")

    # Positives.
    pv = v1["positives"]
    out += ["", "## Positives", "", "| | fit | val | test | all |", "|---|---:|---:|---:|---:|"]
    row = lambda lab, d: f"| {lab} | {d['fit']} | {d['val']} | {d['test']} | {d['all']} |"
    out += [row("listed (in an attempt)", pv["listed"]), row("unlisted", pv["unlisted"])]
    for name, p in P.items():
        nb = p["positives"]["no_block_activates"]
        out += [row(f"listed, no block activates ({PROFILES[name]})", nb["listed"]),
                row(f"unlisted, no block activates ({PROFILES[name]})", nb["unlisted"])]
    out += ["", "Recall on listed and unlisted positives (predicted / positives):", "",
            "| | val listed | val unlisted | test listed | test unlisted |", "|---|---:|---:|---:|---:|"]
    for name, p in P.items():
        pr = p["positives"]["predicted"]
        c = lambda s, k: (f"{pr[s][k]['predicted']} / {pr[s][k]['positives']} "
                          f"({pct(pr[s][k]['predicted'], pr[s][k]['positives'])} %)") if pr.get(s) else "–"
        out.append(f"| {PROFILES[name]} | {c('val', 'listed')} | {c('val', 'unlisted')} | {c('test', 'listed')} | "
                   f"{c('test', 'unlisted')} |")

    # Base rates.
    out += ["", "## Base rates: non-zero rows (% of edges, all splits)", "",
            "| Block | v1_default, all edges | v1_default, positives | frozen, all edges | frozen, positives |",
            "|---|---:|---:|---:|---:|"]
    for b in blocks + ["any"]:
        cells = []
        for p in (v1, fz):
            br = p["base_rates"]
            cells += [f"{100.0 * br['nonzero'][b]['edges']['all'] / br['edges']['all']:.2f}",
                      f"{100.0 * br['nonzero'][b]['positives']['all'] / br['positives']['all']:.1f}"]
        out.append(f"| {b} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="features/e9")
    ap.add_argument("--out", default="results/e9_coverage.json")
    a = ap.parse_args()
    d = ROOT / a.dir
    doc = {"experiment": "E9", "spec": "SPEC v2 §5", "git": git_info(), "inputs": {}, "profiles": {}, "attempts": []}
    cov = {}
    for name in PROFILES:
        c = json.loads((d / f"coverage_{name}.json").read_text())
        pred = json.loads((d / f"pred_{name}.json").read_text())
        if c["pred"].get("sha256") != pred["sha256"]:
            raise SystemExit(f"coverage_{name}.json scored another prediction file than pred_{name}.json")
        if not all(r["reproduces"] for r in pred["ranges"].values()):
            raise SystemExit(f"pred_{name}.json: a refit does not reproduce its original")
        cov[name] = c
        doc["profiles"][name] = dict({k: c[k] for k in SUMMARY_KEYS if k in c}, models=models(pred),
                                     coverage_file=str(Path(a.dir) / f"coverage_{name}.json"))
    # The frozen coverage is the frozen recipe's: its pattern blocks hash to configs/dg2_frozen.yaml's.
    frozen_text = (ROOT / FROZEN).read_text()
    if cov["frozen"]["features"]["source"] != "gfp" or not all(
            sha in frozen_text for sha in cov["frozen"]["features"]["blocks"].values()):
        raise SystemExit(f"coverage_frozen.json: its blocks are not {FROZEN}'s (output_sha256)")
    if cov["v1_default"]["profile"]["id"] != V1_PROFILE_ID:
        raise SystemExit("coverage_v1_default.json is not profile v1_default")
    gtj = {c["inputs"]["gt_join"] for c in cov.values()}
    if len(gtj) != 1 or len({c["inputs"]["gt_join_sha256"] for c in cov.values()}) != 1:
        raise SystemExit("the coverage files use different gt-joins")
    doc["inputs"] = {"gt_join": gtj.pop(), "gt_join_sha256": cov["v1_default"]["inputs"]["gt_join_sha256"],
                     "graph": cov["v1_default"]["inputs"]["graph"], "blzg_sha256": cov["v1_default"]["inputs"]["blzg_sha256"]}
    gt = json.loads((ROOT / doc["inputs"]["gt_join"]).read_text())
    heads = {x["id"]: x for x in gt["attempts"]}
    by_id = {name: {x["id"]: x for x in c["attempts"]} for name, c in cov.items()}
    for i, base in by_id["v1_default"].items():
        h = heads[i]
        row = {"id": i, "typology": base["typology"], "header": h["header"], "status": base["status"],
               "split": base["split"], "span": base["span"], "edges": base["edges"]}
        for name in PROFILES:
            x = by_id[name][i]
            row[name] = {k: x[k] for k in ("eligible", "eligible_oracle", "activated", "predicted") if k in x}
        doc["attempts"].append(row)
    out = ROOT / a.out
    out.write_text(dump(doc))
    out.with_suffix(".md").write_text(render(doc))
    print(f"-> {out}, {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()

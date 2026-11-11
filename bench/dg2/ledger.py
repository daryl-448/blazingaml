#!/usr/bin/env python3
"""The DG-2 experiment ledger (SPEC v2 §3.6, §4): results/dg2_ledger.json and its rendering.

  python3 bench/dg2/ledger.py init     [--ledger results/dg2_ledger.json]   # the pre-registered ladder
  python3 bench/dg2/ledger.py add      <entry.json> [--ledger ...]           # one fit or feature pass
  python3 bench/dg2/ledger.py skip     <rung> <candidate> <budget|limit> [--ledger ...]
  python3 bench/dg2/ledger.py render   [--ledger ...] [--out results/dg2_ledger.md]
  python3 bench/dg2/ledger.py selftest

P7 builds the skeleton: the ladder of §4.4 with every candidate `pending`, the budget of §4.2 and
the selection rule of §4.3. P8's bench/dg2/ladder.py appends entries through add_entry(). Entries
are never edited or removed; the running totals and the stop rule are recomputed from them.
Python standard library only.
"""
import argparse
import copy
import json
import os
import sys
import tempfile

LEDGER = "results/dg2_ledger.json"
BUDGET = {"fits": 40, "passes": 8, "machine_hours": 4.0}  # §4.2 [D-20]
SELECTION = {"metric": "validation F1 x100 of the laundering class at threshold 0.5",
             "min_gain": 2.0, "tie": 0.1,
             "tie_break": ["fewer columns", "listed first in SPEC v2 §4.4"]}  # §4.3 [D-19]
PASS_LIMITS = {"wall_minutes": 60, "peak_rss_gb": 3.0}  # §4.4: an exceeding pass is aborted
# §4.4: rung -> hypothesis, candidates, sets fitted, selection set, planned fits and passes.
RUNGS = [
    ("R0", "the v1 baseline (incumbent)", ["v1"], ["xgb_only", "all"], None, 2, 0),
    ("R1", "input encoding (F3, F9)", ["I1", "I2", "I3"], ["xgb_only", "all"], "xgb_only", 6, 0),
    ("R2", "classifier contract", ["C1"], ["xgb_only", "all"], "all", 2, 0),
    ("R3", "raw schema", ["S1", "S2"], ["xgb_only", "all"], "all", 4, 0),
    ("R4", "windows (F5-F8)", ["W-B", "W-C"], ["all"], "all", 2, 2),
    ("R5", "visibility (Fig. 5 literal)", ["V1"], ["all"], "all", 1, 1),
    ("R6", "GFP call semantics (F8, A28, A29)", ["G1", "G2", "G3"], ["all"], "all", 3, 3),
    ("R7", "feature families (F7, F8)", ["F1", "F2"], ["all"], "all", 2, 2),
    ("R8", "XGBoost grid (the v1 E6 grid)", [f"X{i}" for i in range(1, 8)], ["all"], "all", 7, 0),
]
KINDS = ("fit", "pass")
METRICS = ("f1_x100", "precision", "recall", "pr_auc", "tp", "fp", "fn", "tn")


def init_ledger():
    rungs = []
    for rid, hyp, cands, sets, sel, fits, passes in RUNGS:
        rungs.append({"rung": rid, "hypothesis": hyp, "sets_fitted": sets, "selection_set": sel,
                      "planned_fits": fits, "planned_passes": passes,
                      "candidates": [{"id": c, "status": "pending"} for c in cands],
                      "carried": None})
    return {"ledger_version": 1, "spec": "SPEC_V2.md §3.6, §4", "budget": BUDGET, "selection": SELECTION,
            "pass_limits": PASS_LIMITS, "rungs": rungs, "incumbent": None, "entries": [],
            "totals": {"fits": 0, "passes": 0, "machine_hours": 0.0}, "stopped": None}


def _rung(ledger, rid):
    for r in ledger["rungs"]:
        if r["rung"] == rid:
            return r
    raise ValueError(f"unknown rung {rid!r}")


def _candidate(ledger, rid, cid):
    for c in _rung(ledger, rid)["candidates"]:
        if c["id"] == cid:
            return c
    raise ValueError(f"rung {rid} has no candidate {cid!r}")


def totals(entries):
    t = {"fits": 0, "passes": 0, "machine_hours": 0.0}
    for e in entries:
        t["fits" if e["kind"] == "fit" else "passes"] += 1
        t["machine_hours"] += e["wall_seconds"] / 3600.0
    t["machine_hours"] = round(t["machine_hours"], 4)
    return t


def stop_state(t):
    """§4.2: the fit or time cap stops the ladder; the pass cap stops passes only."""
    if t["fits"] >= BUDGET["fits"]:
        return "fit cap"
    if t["machine_hours"] >= BUDGET["machine_hours"]:
        return "time cap"
    return None


def validate_entry(e):
    need = ["rung", "candidate", "kind", "columns", "wall_seconds", "peak_rss_mb", "machine", "foreign_load"]
    missing = [k for k in need if k not in e]
    if missing:
        raise ValueError(f"entry lacks {missing}")
    if e["kind"] not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    if ("profile_id" in e) == ("gfp_recipe" in e):
        raise ValueError("an entry names exactly one of profile_id and gfp_recipe")
    cols = e["columns"]
    if not isinstance(cols.get("names"), list) or cols.get("count") != len(cols["names"]):
        raise ValueError("columns needs names and a matching count")
    if e["kind"] == "fit":
        if e.get("set") not in ("xgb_only", "all"):
            raise ValueError("a fit names its set: xgb_only or all")
        m = e.get("validation", {})
        if [k for k in METRICS if k not in m]:
            raise ValueError(f"a fit records validation {list(METRICS)}")
    for f in e.get("facts", []):
        if not {"fact", "check", "result"} <= set(f):
            raise ValueError("a fact records fact, check and result")


def add_entry(ledger, entry):
    """Appends one fit or feature pass; returns the new ledger. Refuses past the budget (§4.2)."""
    validate_entry(entry)
    led = copy.deepcopy(ledger)
    cand = _candidate(led, entry["rung"], entry["candidate"])
    if led["stopped"]:
        raise ValueError(f"the ladder stopped ({led['stopped']}); record the rest with skip")
    t = led["totals"]
    if entry["kind"] == "pass" and t["passes"] >= BUDGET["passes"]:
        raise ValueError("the pass cap is reached; record the candidate as skipped: budget")
    e = dict(entry, seq=len(led["entries"]))
    led["entries"].append(e)
    led["totals"] = totals(led["entries"])
    e["totals_after"] = dict(led["totals"])
    led["stopped"] = stop_state(led["totals"])
    cand["status"] = "run"
    return led


def skip(ledger, rid, cid, why):
    if why not in ("budget", "limit"):
        raise ValueError("a skip is 'budget' or 'limit' (SPEC v2 §4.2, §4.4)")
    led = copy.deepcopy(ledger)
    _candidate(led, rid, cid)["status"] = f"skipped: {why}"
    return led


def render(ledger):
    t, b = ledger["totals"], ledger["budget"]
    out = ["# DG-2 ledger", "",
           "Generated by `bench/dg2/ledger.py render` from `results/dg2_ledger.json` (SPEC v2 §3.6). "
           "The ladder is pre-registered in SPEC v2 §4.4 [D-18].", "",
           "## Budget (SPEC v2 §4.2)", "",
           "| | used | cap |", "|---|---:|---:|",
           f"| fits | {t['fits']} | {b['fits']} |",
           f"| feature passes | {t['passes']} | {b['passes']} |",
           f"| machine-hours | {t['machine_hours']:.2f} | {b['machine_hours']:.0f} |", "",
           f"Stopped: {ledger['stopped'] or 'no'}. Incumbent: {ledger['incumbent'] or 'none yet (R0 not fitted)'}.", "",
           "## Rungs (SPEC v2 §4.4)", "",
           "| rung | hypothesis | candidates | selection set | planned fits | planned passes | carried |",
           "|---|---|---|---|---:|---:|---|"]
    for r in ledger["rungs"]:
        cands = ", ".join(f"{c['id']} ({c['status']})" for c in r["candidates"])
        out.append(f"| {r['rung']} | {r['hypothesis']} | {cands} | {r['selection_set'] or '-'} | "
                   f"{r['planned_fits']} | {r['planned_passes']} | {r['carried'] or '-'} |")
    out += ["", "## Entries", ""]
    if not ledger["entries"]:
        out.append("None yet: P8 runs the ladder.")
    else:
        out += ["| # | rung | candidate | kind | set | source | columns | F1 x100 | precision | recall | PR-AUC "
                "| wall s | peak RSS MB | fits | passes | hours |",
                "|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for e in ledger["entries"]:
            v = e.get("validation", {})
            fmt = lambda k, f="{:.3f}": f.format(v[k]) if k in v else "-"
            src = f"profile {e['profile_id']}" if "profile_id" in e else f"GFP {e['gfp_recipe']}"
            ta = e["totals_after"]
            out.append(f"| {e['seq']} | {e['rung']} | {e['candidate']} | {e['kind']} | {e.get('set', '-')} | {src} | "
                       f"{e['columns']['count']} | {fmt('f1_x100', '{:.2f}')} | {fmt('precision')} | "
                       f"{fmt('recall')} | {fmt('pr_auc')} | {e['wall_seconds']:.0f} | {e['peak_rss_mb']:.0f} | "
                       f"{ta['fits']} | {ta['passes']} | {ta['machine_hours']:.2f} |")
    return "\n".join(out) + "\n"


def load(path):
    with open(path) as f:
        return json.load(f)


def save(ledger, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(ledger, f, indent=1)
        f.write("\n")


def selftest():
    led = init_ledger()
    assert sum(r["planned_fits"] for r in led["rungs"]) == 29  # §4.4 "Total with R0"
    assert sum(r["planned_passes"] for r in led["rungs"]) == 8
    fit = {"rung": "R0", "candidate": "v1", "kind": "fit", "set": "all", "profile_id": "0123456789abcdef",
           "columns": {"names": ["src_id", "dst_id"], "count": 2},
           "validation": {"f1_x100": 30.0, "precision": 0.5, "recall": 0.2, "pr_auc": 0.3,
                          "tp": 1, "fp": 1, "fn": 4, "tn": 10},
           "wall_seconds": 1800, "peak_rss_mb": 2000, "machine": "local", "foreign_load": []}
    led2 = add_entry(led, fit)
    assert led["entries"] == [] and led2["totals"] == {"fits": 1, "passes": 0, "machine_hours": 0.5}
    assert _candidate(led2, "R0", "v1")["status"] == "run"
    for bad in [dict(fit, kind="x"), dict(fit, set="raw"), {k: v for k, v in fit.items() if k != "machine"},
                dict(fit, gfp_recipe="G1"), dict(fit, columns={"names": ["a"], "count": 2}),
                dict(fit, validation={"f1_x100": 1})]:
        try:
            add_entry(led, bad)
            raise AssertionError(f"accepted {bad}")
        except ValueError:
            pass
    p = dict(fit, kind="pass", rung="R4", candidate="W-B", wall_seconds=600)
    p.pop("validation")
    p.pop("set")
    for _ in range(8):
        led2 = add_entry(led2, p)
    try:
        add_entry(led2, p)
        raise AssertionError("a ninth pass was accepted")
    except ValueError:
        pass
    led2 = add_entry(led2, dict(fit, wall_seconds=3600 * 2.2))  # 0.5 + 8/6 + 2.2 > 4 hours
    assert led2["stopped"] == "time cap"
    led2 = skip(led2, "R8", "X7", "budget")
    md = render(led2)
    assert "skipped: budget" in md and "| machine-hours | 4.03 | 4 |" in md, md
    with tempfile.TemporaryDirectory() as d:
        save(led2, os.path.join(d, "l.json"))
        assert load(os.path.join(d, "l.json")) == json.loads(json.dumps(led2))
    print("ledger selftest: ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["init", "add", "skip", "render", "selftest"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--ledger", default=LEDGER)
    ap.add_argument("--out", default=None, help="render: the Markdown file (default: the ledger's .md)")
    ap.add_argument("--force", action="store_true", help="init: overwrite an existing ledger")
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
        return 0
    if a.cmd == "init":
        if os.path.exists(a.ledger) and not a.force:
            sys.exit(f"{a.ledger} exists (entries are never discarded; --force to overwrite)")
        led = init_ledger()
    else:
        led = load(a.ledger)
    if a.cmd == "add":
        (path,) = a.args
        led = add_entry(led, load(path))
    elif a.cmd == "skip":
        rid, cid, why = a.args
        led = skip(led, rid, cid, why)
    if a.cmd != "render":
        save(led, a.ledger)
    out = a.out or os.path.splitext(a.ledger)[0] + ".md"
    with open(out, "w") as f:
        f.write(render(led))
    print(f"{a.ledger}: {led['totals']}, stopped: {led['stopped']} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

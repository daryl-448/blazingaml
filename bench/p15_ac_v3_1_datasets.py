#!/usr/bin/env python3
"""AC-V3-1's dataset parts (SPEC v3 §2.3, §7 P15 Live): the oracle, cpu and cuda give the same clean
rows, block by block, on LI-Small and TransXion under clean_default, the dataset's clean_wc profile
and clean_all; the oracle and cpu on HI-Medium in trigger ranges under clean_default and
clean_wc_hi_medium. Rows are compared by the row hash per block of `blz measure --reps 1` (the
bounded sink, SPEC v3 §5.1) over the same triggers.

  bench/p15_ac_v3_1_datasets.py [--smoke] [--budget 3600] [--plan]

The oracle's wall time is budgeted (D-92): about 1 h in all. The plan comes from the preflight's
sampled oracle rates (results/p15_preflight_hi_medium.json, oracle_timing), written before any run:
  - HI-Medium clean_default runs whole if its estimate is at most 20 min, else in 3 ranges;
  - HI-Medium clean_wc_hi_medium runs in 3 ranges (start, middle, end), never whole;
  - every other (dataset, profile) runs whole if it fits.
A cap c is found so that the estimates sum to the budget: an item whose whole estimate is at most c
runs whole; the others get c, as 3 ranges (start, middle, end) of at most 2,000,000 triggers.
cpu and cuda run on the oracle's selections, and on LI-Small and TransXion also over every trigger.
During the run, an item's later ranges shrink by how much slower than planned its earlier ones
were, and once the oracle has used 1.25 x the budget the remaining oracle runs are skipped and
recorded. Steps already done (build/p15_ac_v3_1[_smoke]/steps.json) are not run again. Writes
results/ac_v3_1_datasets_local.{json,md} (--smoke: build/p15_ac_v3_1_smoke/result.{json,md}) and
exits 1 unless every comparison is equal and nothing failed or was skipped.
--smoke runs each item on two tiny ranges (start and end: 4,000 triggers on HI-Medium, 20,000 on LI-Small
and TransXion), and no whole cpu and cuda runs.
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLZ = "build/rel-cuda/blz"
PREFLIGHT = "results/p15_preflight_hi_medium.json"
ITEMS = [("transxion", "clean_default"), ("transxion", "clean_wc_hi_small"), ("transxion", "clean_all"),
         ("li_small", "clean_default"), ("li_small", "clean_wc_li_small"), ("li_small", "clean_all"),
         ("hi_medium", "clean_default"), ("hi_medium", "clean_wc_hi_medium")]
BACKENDS = {"li_small": ["oracle", "cpu", "cuda"], "transxion": ["oracle", "cpu", "cuda"], "hi_medium": ["oracle", "cpu"]}
MAX_RANGE = 2_000_000
WHOLE_LIMIT = {("hi_medium", "clean_default"): 1200}  # s: run whole only under this (D-92)
FORCED_RANGES = {("hi_medium", "clean_wc_hi_medium")}


def load(path, default):
    return json.load(open(path)) if os.path.exists(path) else default


def save(path, doc):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")


def three_ranges(m, r):
    r = max(1, min(r, m // 3))
    mid = (m - r) // 2
    return [f"range:0:{r}", f"range:{mid}:{mid + r}", f"range:{m - r}:{m}"]


def plan(budget, smoke):
    pf = load(os.path.join(ROOT, PREFLIGHT), None)
    if pf is None:
        sys.exit(f"no {PREFLIGHT}: run bench/p15_preflight.py first")
    F, T = pf["facts"], pf["oracle_timing"]
    items = []
    for ds, prof in ITEMS:
        t = T[f"{ds}/{prof}"]
        m, build, rate = F[ds]["m"], t["build_seconds"], t["per_trigger_us"] * 1e-6
        items.append({"dataset": ds, "profile": prof, "m": m, "build_s": build, "rate_s": rate,
                      "whole_s": build + rate * m, "forced_ranges": (ds, prof) in FORCED_RANGES or
                      build + rate * m > WHOLE_LIMIT.get((ds, prof), float("inf"))})
    if smoke:
        for it in items:
            r = 20_000 if it["dataset"] != "hi_medium" else 4_000
            it["selections"] = three_ranges(it["m"], r)[::2]
            it["planned_s"] = round(len(it["selections"]) * (it["build_s"] + it["rate_s"] * r), 1)
        return {"budget_s": None, "cap_s": None, "items": items}

    def cost(it, c):
        if not it["forced_ranges"] and it["whole_s"] <= c:
            return it["whole_s"]
        return min(c, 3 * (it["build_s"] + it["rate_s"] * MAX_RANGE))

    lo, hi = 0.0, max(it["whole_s"] for it in items)
    for _ in range(100):
        c = (lo + hi) / 2
        lo, hi = (c, hi) if sum(cost(it, c) for it in items) < budget else (lo, c)
    c = lo
    for it in items:
        if not it["forced_ranges"] and it["whole_s"] <= c:
            it["selections"], it["planned_s"] = ["all"], round(it["whole_s"], 1)
        else:
            r = round(min(MAX_RANGE, max(1000, (cost(it, c) / 3 - it["build_s"]) / it["rate_s"])))
            it["selections"] = three_ranges(it["m"], r)
            it["planned_s"] = round(3 * (it["build_s"] + it["rate_s"] * r), 1)
    return {"budget_s": budget, "cap_s": round(c, 1), "items": items,
            "rule": "D-92: HI-Medium clean_default whole only if <= 20 min, clean_wc_hi_medium in 3 ranges; a cap c with "
                    "the estimates summing to the budget; whole if <= c, else 3 ranges (start, middle, end) sized to c, "
                    f"at most {MAX_RANGE:,} triggers"}


def measure(work, ds, prof, sel, be):
    name = f"{ds}/{prof}/{sel.replace(':', '_')}/{be}"
    out = os.path.join(work, name.replace("/", "__") + ".json")  # fresh per run: no stale result is read
    if os.path.exists(out):
        os.remove(out)
    cmd = [BLZ, "measure", "--graph", f"data/{ds}.blzg", "--profile", f"configs/profiles/{prof}.yaml", "--backend", be,
           "--reps", "1", "--triggers", sel, "--out", os.path.relpath(out, ROOT), "--name", name]
    log = os.path.join(work, name.replace("/", "__") + ".log")
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    wall = time.time() - t0
    res = load(out, {})
    for k in name.split("/"):
        res = res.get(k, {}) if isinstance(res, dict) else {}
    run = (res.get("runs") or [{}])[0]
    return {"rc": rc, "wall_s": round(wall, 1), "seconds": run.get("seconds"), "error": run.get("error"),
            "errors": res.get("errors"), "peak_rss_mib": round(res.get("peak_rss_mb", 0)),
            "hashes": res.get("output_fnv1a", {}), "rows": res.get("triggers", {}).get("rows"),
            "log": os.path.relpath(log, ROOT)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--budget", type=float, default=3600.0, help="oracle wall seconds in all")
    ap.add_argument("--plan", action="store_true", help="print the plan and run nothing")
    a = ap.parse_args()
    if a.plan:
        print(json.dumps(plan(a.budget, a.smoke), indent=1))
        return
    work = os.path.join(ROOT, "build", "p15_ac_v3_1_smoke" if a.smoke else "p15_ac_v3_1")
    state = load(os.path.join(work, "steps.json"), {})
    if "plan" not in state:  # written once, before any run
        state["plan"] = plan(a.budget, a.smoke)
        save(os.path.join(work, "steps.json"), state)
    P = state["plan"]
    runs = state.setdefault("runs", {})
    oracle_used = sum(r["wall_s"] for k, r in runs.items() if k.endswith("/oracle") and "wall_s" in r)
    for it in P["items"]:
        ds, prof = it["dataset"], it["profile"]
        sels = list(it["selections"])
        if ds != "hi_medium" and "all" not in sels and not a.smoke:
            sels.append("all")  # cpu = cuda over every trigger
        slow = 1.0
        for i, sel in enumerate(sels):
            for be in BACKENDS[ds]:
                if be == "oracle" and sel not in it["selections"]:
                    continue
                key = f"{ds}/{prof}/{sel}/{be}"
                if key in runs and runs[key].get("rc") == 0:
                    continue
                if be == "oracle" and P["budget_s"] and oracle_used > 1.25 * P["budget_s"]:
                    runs[key] = {"skipped": f"oracle budget: {oracle_used:.0f} s used of {P['budget_s']:.0f}"}
                    continue
                print(f"{key} ...", flush=True)
                r = runs[key] = measure(work, ds, prof, sel, be)
                print(f"  rc {r['rc']}, {r['wall_s']} s, peak {r['peak_rss_mib']} MiB", flush=True)
                if be == "oracle":
                    oracle_used += r["wall_s"]
                save(os.path.join(work, "steps.json"), state)
            # An item slower than planned: its later ranges shrink by the ratio (the plan's sizes stay recorded).
            o = runs.get(f"{ds}/{prof}/{sel}/oracle", {})
            if o.get("seconds") and sel.startswith("range:") and not a.smoke:
                a0, b0 = map(int, sel.split(":")[1:])
                want = it["build_s"] + it["rate_s"] * (b0 - a0)
                slow = max(slow, o["seconds"] / want)
                if slow > 1.3 and i + 1 < len(it["selections"]):
                    r_new = max(1000, int((b0 - a0) / slow))
                    rest = three_ranges(it["m"], r_new)[i + 1:len(it["selections"])]
                    it.setdefault("resized", []).append({"after": sel, "slower_by": round(slow, 2), "to": rest})
                    it["selections"][i + 1:] = rest
                    sels[i + 1:len(it["selections"])] = rest
                    save(os.path.join(work, "steps.json"), state)
    report(state, work, a.smoke)


def report(state, work, smoke):
    P, runs = state["plan"], state["runs"]
    items, ok = [], True
    for it in P["items"]:
        ds, prof = it["dataset"], it["profile"]
        sels = list(dict.fromkeys(it["selections"] + (["all"] if ds != "hi_medium" and not smoke else [])))
        rows = []
        for sel in sels:
            got = {be: runs.get(f"{ds}/{prof}/{sel}/{be}") for be in BACKENDS[ds]}
            want = [be for be in BACKENDS[ds] if not (be == "oracle" and sel not in it["selections"])]
            got = {be: r for be, r in got.items() if r is not None and be in want}
            bad = [be for be in want if be not in got or got[be].get("rc") != 0 or got[be].get("errors")
                   or not got[be].get("hashes")]
            blocks = sorted({b for r in got.values() for b in r.get("hashes", {})})
            unequal = [b for b in blocks if len({r.get("hashes", {}).get(b) for r in got.values()}) != 1]
            if len({r.get("rows") for r in got.values()}) != 1:
                unequal.append("triggers")
            row = {"selection": sel, "backends": sorted(got), "triggers": next((r["rows"] for r in got.values() if r.get("rows")), None),
                   "equal": not bad and not unequal and len(got) >= 2, "unequal_blocks": unequal, "failed": bad,
                   "skipped": [be for be, r in got.items() if "skipped" in r],
                   "seconds": {be: r.get("seconds") for be, r in got.items()},
                   "peak_rss_mib": {be: r.get("peak_rss_mib") for be, r in got.items()},
                   "errors": {be: r.get("error") for be, r in got.items() if r.get("error")},
                   "hashes": next((r["hashes"] for be, r in got.items() if r.get("hashes")), {})}
            ok &= row["equal"]
            rows.append(row)
        o = [runs.get(f"{ds}/{prof}/{s}/oracle", {}) for s in it["selections"]]
        checked = sum(r.get("rows") or 0 for r in o if r.get("rc") == 0)
        items.append({**{k: it[k] for k in ("dataset", "profile", "m", "selections", "planned_s")},
                      "resized": it.get("resized", []), "oracle_seconds": round(sum(r.get("wall_s", 0) for r in o), 1),
                      "oracle_triggers_checked": checked, "oracle_share": round(checked / it["m"], 4), "comparisons": rows})
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    doc = {"experiment": "AC-V3-1, dataset parts", "spec": "SPEC v3 §2.3 AC-V3-1, §7 P15 Live, D-92", "git": git,
           "smoke": smoke, "plan": {k: P[k] for k in ("budget_s", "cap_s", "rule") if k in P},
           "oracle_seconds_total": round(sum(i["oracle_seconds"] for i in items), 1), "items": items,
           "verdict": "PASS" if ok else "FAIL"}
    out = os.path.join(work, "result.json") if smoke else os.path.join(ROOT, "results", "ac_v3_1_datasets_local.json")
    save(out, doc)
    with open(out.replace(".json", ".md"), "w") as f:
        f.write(markdown(doc))
    print(f"{doc['verdict']}: oracle {doc['oracle_seconds_total']} s; wrote {os.path.relpath(out, ROOT)}")
    sys.exit(0 if ok else 1)


def markdown(doc):
    L = [f"# AC-V3-1, dataset parts{' (smoke)' if doc['smoke'] else ''}", "",
         f"git {doc['git']}. Row hash per block of `blz measure --reps 1` (SPEC v3 §5.1's sink) over the same triggers. "
         f"Oracle wall time in all: {doc['oracle_seconds_total']:,} s" +
         (f" of a {doc['plan']['budget_s']:,.0f} s budget (cap {doc['plan']['cap_s']:,} s per item; {doc['plan']['rule']})."
          if doc["plan"].get("budget_s") else "."), "",
         f"Verdict: **{doc['verdict']}**.", "",
         "| Dataset | Profile | Selection | Triggers | Backends | Equal | Seconds |", "|---|---|---|---:|---|---|---|"]
    for it in doc["items"]:
        for r in it["comparisons"]:
            secs = ", ".join(f"{b} {s:.1f}" for b, s in r["seconds"].items() if s is not None)
            eq = "yes" if r["equal"] else f"NO {r['unequal_blocks'] or ''} {r['failed'] or ''} {r['errors'] or ''}"
            trig = f"{r['triggers']:,}" if r["triggers"] else "-"
            L.append(f"| {it['dataset']} | {it['profile']} | {r['selection']} | {trig} | "
                     f"{', '.join(r['backends'])} | {eq} | {secs} |")
    L += ["", "| Dataset | Profile | Oracle selections | Oracle triggers checked | Share | Oracle s (planned) |",
          "|---|---|---|---:|---:|---:|"]
    for it in doc["items"]:
        L.append(f"| {it['dataset']} | {it['profile']} | {', '.join(it['selections'])} | {it['oracle_triggers_checked']:,} | "
                 f"{it['oracle_share']:.1%} | {it['oracle_seconds']} ({it['planned_s']}) |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()

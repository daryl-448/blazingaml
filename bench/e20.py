#!/usr/bin/env python3
"""E20 (SPEC v3 §8): the verification stack in one run, per check its layer, target, bound, cases,
seconds, path hits and result, per mutant what caught it, and the foreign load during each step.

    bench/e20.py run [STEP ...]   # the steps in order (default all), into build/e20/
    bench/e20.py report           # results/e20_verification.{json,md} from build/e20/

Steps: quick-l1 (verify/algorithms.sh), quick-rel and quick-rel-cuda (ctest --preset, label fast:
the per-change subset), l1 (verify/algorithms.sh --mutants), l2 (verify/codegen.sh, every build
present), l3-<build> (the [l3] properties at their fixed seeds on rel, rel-cuda and dev),
l3-random-<build> (fresh seeds at ten times the cases, as ctest l3_random, on rel and rel-cuda),
mutants (verify/mutants.py --escape). Each step's output goes to build/e20/<step>.log and its
record, with the revision it ran at, to build/e20/steps.json; a sampler reads /proc every 15 s and
splits CPU use between this run's processes and everything else (the foreign load). Both modes
refuse uncommitted tracked changes; the report refuses steps from different commits, and exits 1
unless every step ran, exited 0 and passed (the L2 rows take verify/codegen.sh's verdicts).
"""
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "build", "e20")
TICK = os.sysconf("SC_CLK_TCK")
L3 = lambda b: [os.path.join(ROOT, "build", b, "tests", "blz_unit_tests"), "[l3]", "--durations", "yes"]
RANDOM = {"BLZ_L3_SEED": "random", "BLZ_L3_SCALE": "10"}
STEPS = {  # name: (command, extra environment)
    "quick-l1": (["verify/algorithms.sh"], {}),
    "quick-rel": (["ctest", "--preset", "rel", "-j6"], {}),
    "quick-rel-cuda": (["ctest", "--preset", "rel-cuda", "-j4"], {}),
    "l1": (["verify/algorithms.sh", "--mutants"], {}),
    "l2": (["verify/codegen.sh", "rel", "rel-cuda", "dev", "rel-noopt"], {}),  # a missing build fails
    "l3-rel": (L3("rel"), {}),
    "l3-rel-cuda": (L3("rel-cuda"), {}),
    "l3-dev": (L3("dev"), {}),
    "l3-random-rel": (L3("rel"), RANDOM),
    "l3-random-rel-cuda": (L3("rel-cuda"), RANDOM),
    "mutants": (["verify/mutants.py", "--escape", "--report", os.path.join(OUT, "mutants.json")], {}),
}


def procs():
    """pid -> (ppid, cpu ticks, name) for every process."""
    out = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                s = f.read()
        except OSError:
            continue
        name, rest = s[s.index("(") + 1:s.rindex(")")], s[s.rindex(")") + 2:].split()
        out[int(d)] = (int(rest[1]), int(rest[11]) + int(rest[12]), name)
    return out


class Sampler(threading.Thread):
    """Every `every` s: CPU % (of one core) of this run's process tree and of the rest, per process."""

    def __init__(self, every=15):
        super().__init__(daemon=True)
        self.every, self.samples, self.stop = every, [], threading.Event()

    def run(self):
        me, prev, t_prev = os.getpid(), procs(), time.monotonic()
        while True:  # one sample every `every` s, and one at the end
            stopped = self.stop.wait(self.every)
            cur, t = procs(), time.monotonic()
            if t - t_prev < 1:
                break
            kids, ours = {}, {me}
            for pid, (ppid, _, _) in cur.items():
                kids.setdefault(ppid, []).append(pid)
            stack = [me]
            while stack:
                for c in kids.get(stack.pop(), []):
                    ours.add(c)
                    stack.append(c)
            mine = foreign = 0.0
            top = []
            for pid, (_, ticks, name) in cur.items():
                pct = 100.0 * (ticks - prev.get(pid, (0, ticks, ""))[1]) / TICK / (t - t_prev)
                if pid in ours:
                    mine += pct
                else:
                    foreign += pct
                    if pct >= 5:
                        top.append((round(pct), name))
            with open("/proc/meminfo") as f:
                avail = int(re.search(r"MemAvailable:\s+(\d+)", f.read()).group(1)) / 2**20
            self.samples.append({"t": round(t), "loadavg1": os.getloadavg()[0], "ours_pct": round(mine),
                                 "foreign_pct": round(foreign), "mem_available_gb": round(avail, 1),
                                 "foreign_top": sorted(top, reverse=True)[:4]})
            prev, t_prev = cur, t
            if stopped:
                break


def summary(samples):
    if not samples:
        return {}
    f = [s["foreign_pct"] for s in samples]
    names = {}
    for s in samples:
        for pct, n in s["foreign_top"]:
            names[n] = max(names.get(n, 0), pct)
    return {"samples": len(samples), "foreign_pct_mean": round(sum(f) / len(f)), "foreign_pct_max": max(f),
            "ours_pct_mean": round(sum(s["ours_pct"] for s in samples) / len(samples)),
            "loadavg1_max": max(s["loadavg1"] for s in samples),
            "mem_available_gb_min": min(s["mem_available_gb"] for s in samples),
            "foreign_processes_max_pct": dict(sorted(names.items(), key=lambda kv: -kv[1])[:6])}


def revision():
    """HEAD, with -dirty when tracked files have uncommitted changes: then no commit describes the results."""
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    return git("rev-parse", "HEAD") + ("-dirty" if git("status", "--porcelain", "--untracked-files=no") else "")


def run(names):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "steps.json")
    steps = json.load(open(path)) if os.path.exists(path) else {}
    if revision().endswith("-dirty"):
        sys.exit("e20 run: commit or stash the tracked changes first; E20 records a commit")
    for name in names or STEPS:
        cmd, env = STEPS[name]
        if cmd[0].startswith("/") and not os.path.exists(cmd[0]):
            print(f"{name}: skipped, no {cmd[0]}", flush=True)
            continue
        sampler = Sampler()
        sampler.start()
        t0, start, rev = time.monotonic(), time.strftime("%Y-%m-%d %H:%M:%S %z"), revision()
        with open(os.path.join(OUT, f"{name}.log"), "w") as log:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **env}).returncode
        sampler.stop.set()
        sampler.join()
        if revision() != rev:
            rc = rc or "the tree changed during the step"
        steps[name] = {"cmd": " ".join(f"{k}={v}" for k, v in env.items()).split() + cmd, "rc": rc, "rev": rev,
                       "started": start,
                       "seconds": round(time.monotonic() - t0, 1), "load": summary(sampler.samples),
                       "load_samples": sampler.samples}
        if name == "l2":  # codegen.sh writes build/verify; keep this run's copy
            for f in ("codegen.json", "codegen.log"):
                shutil.copy(os.path.join(ROOT, "build", "verify", f), os.path.join(OUT, f))
        with open(path, "w") as f:
            json.dump(steps, f, indent=1)
        print(f"{name}: exit {rc}, {steps[name]['seconds']} s, foreign CPU mean "
              f"{steps[name]['load'].get('foreign_pct_mean', '-')} % of a core", flush=True)


# ---- report -------------------------------------------------------------------------------------

COVERS = {  # L2 check -> (target, what it covers)
    "cpu": ("T1, T2", "generated CPU code, both schedules, reach pruning both ways, sweep drivers, vstats"),
    "cpu-limit1": ("T1", "R11's work limit: lcyc's stop eids, each graph alone at a limit of 1 binding"),
    "cuda": ("T1, T2", "generated CUDA kernels at H = 0, 4, all (light, heavy and mixed)"),
    "cuda-fuse-pack": ("T2", "scan fusion (members at the same and at different windows) and packed D2H"),
    "hybrid": ("T2", "the hybrid: heavy triggers on the pruned CPU, light on the GPU, H x C in {0, 4, all}"),
    "cuda-limit1": ("T1", "R11 on CUDA: stop eids, each graph alone, limit 1"),
    "cpu-alone": ("T1", "ASan + UBSan: the CPU code with each graph alone (no batch)"),
    "index-contracts": ("T4", "index primitives against naive definitions; ReachScratch::search against ldJ"),
}


def l2_rows(report):
    rows = []

    def walk(node, path):
        if isinstance(node, dict) and "seconds" in node and "graphs" in node:
            rows.append((path, node))
        elif isinstance(node, dict):
            for k, v in node.items():
                walk(v, path + [k])
    walk(report, [])
    out = []
    for path, r in rows:
        build, check = path[0], path[1]
        target, what = COVERS[check]
        if check == "index-contracts":
            searches = r["by_contract"].get("search", {}).get("checks", 0)
            cases = f"{r['queries']:,} queries ({searches:,} searches)"
            hit = f"search reached {r.get('search_nodes', 0):,} nodes besides the target"
            bad = r["violations"]
            layer, profile = "T4", "-"
        else:
            cases = f"{r['trigger_checks']:,} trigger checks, {r['graphs']:,} graphs"
            op = r["path_hits"]["optimised_paths"]
            nz = r["path_hits"]["nonzero_rows_by_block"]
            parts = [f"nonzero rows in {sum(v > 0 for v in nz.values())}/{len(nz)} blocks"]
            for key, label in (("sweep_rows", "sweep"), ("reach_pruned_rows", "reach-pruned"), ("fused_rows", "fused"),
                               ("hybrid_cpu_triggers", "hybrid cpu triggers")):
                if op.get(key):
                    parts.append(f"{label} {op[key]:,}")
            packed = op.get("packed_rows_by_width", {})
            if any(packed.values()):
                parts.append("packed 1/2/4 B " + "/".join(f"{packed[w]:,}" for w in ("1", "2", "4")))
            if "limit1" in check:
                parts.append(f"lcyc stopped in {r['limited_batches_by_block'].get('lcyc', 0):,} graphs")
            hit = "; ".join(parts)
            bad = sum(r["mismatched_rows_by_block"].values())
            layer, profile = "L2", path[2]
        out.append({"layer": layer, "target": target, "build": build, "check": check, "profile": profile,
                    "covers": what, "bound": r["bound"] if isinstance(r["bound"], str) else
                    f"m={r['bound']['m_max']},n={r['bound']['n_max']},t={r['bound']['t']}",
                    "cases": cases, "seconds": round(r["seconds"], 1), "path_hit": hit,
                    "result": r.get("verdict", "FAIL (no verdict from verify/codegen.sh)")})
    return out


L3_TARGET = {"k_max": "T1", "R10": "T1", "packing": "T2", "arithmetic": "T1", "position": "T1, T2",
             "random": "T1, T2"}


def l3_rows(step, log, build):
    out = []
    text = open(log).read()
    for m in re.finditer(r"^(L3 [^\n]*?): seed (\d+), OK, passed (\d+) tests?\n((?:  [^\n]*\n)*)", text, re.M):
        name, seed, cases, extra = m.groups()
        secs = re.search(r"^([\d.]+) s: " + re.escape(name) + "$", text, re.M)
        hits = re.search(r"hits: ([^\n]*)", extra)
        tags = [l.strip() for l in extra.splitlines() if l.strip() and not l.strip().startswith("hits:")]
        key = name.split()[1].rstrip(":")
        out.append({"layer": "L3", "target": L3_TARGET.get(key, "T1"), "build": build, "check": step,
                    "profile": "clean_all, l2_legacy_stream, l2_legacy_tsvis", "covers": name[3:],
                    "bound": "fresh seeds, 10x the cases" if step == "l3_random" else "the fixed seeds",
                    "seed": seed, "n_cases": int(cases), "cases": f"{int(cases):,} cases",
                    "seconds": float(secs.group(1)) if secs else None,
                    "path_hit": "; ".join(([hits.group(1)] if hits else []) + tags[:6]), "result": "pass"})
    for m in re.finditer(r"^(L3 [^\n]*?)\n[-]+\n.*?FAILED", text, re.M | re.S):
        out.append({"layer": "L3", "build": build, "check": step, "covers": m.group(1), "result": "FAIL"})
    return out


NOT_COVERED = [
    "Packing at 2 and 4 bytes is L3's only: no cell passes 255 at L2's bounds (m <= 6), so L2's packed rows "
    "at 2 and 4 bytes are 0; L3's packing properties reach both widths.",
    "CUDA under compute-sanitizer: not run. dev's ASan and UBSan cover the host code only.",
    "The hybrid's work limit: L2 checks R11's stop eids on the CPU and on cuda, not on the hybrid "
    "(about 170 ms a graph alone).",
    "sg_ord's count len_I has no single L1 theorem: order_test decides each intermediate, Scan's lemmas say "
    "the scan visits each once, and L2 checks the count.",
    "The DFS over node sequences (simple paths, U and V excluded) is L2's and L3's, not L1's.",
    "Machine width is L3's (timestamps at the int32 limits, windows to INT32_MAX), not L1's: L1's integers "
    "are unbounded.",
    "Reachability pruning is legacy only: proved for Q5's strict order, planned only for legacy tcyc; "
    "clean tcyc never prunes.",
    "ReachScratch::search meets Search only up to L2's bound (walks of at most 5 edges); the 9-round end is "
    "L3's k_max property.",
    "Batch position: batched L2 runs put only a batch's first and last member at node 0 and the arrays' ends; "
    "the runs with each graph alone (dev, the limit-1 checks) cover position at smaller bounds.",
    "CUDA's heavy-kernel clamp at UINT32_MAX, fused slots at 4 bytes, the 4 MiB pack staging chunks: L3 and "
    "test_pack_fuse at most, not L2.",
    "CPU races: L2 runs 12 threads over batches, L3 two threads per case; no thread sanitizer.",
]


def markdown(out):
    agg = {}
    for c in out["checks"]:
        target = "T1, T2" if c["layer"] == "L3" else c.get("target", "")
        key = (c["layer"], target, c["build"], c["check"], c.get("bound", ""))
        a = agg.setdefault(key, {"n": 0, "seconds": 0.0, "profiles": [], "results": set(), "hit": c.get("path_hit", ""),
                                 "cases": c.get("cases", ""), "covers": c.get("covers", "")})
        a["n"] += 1
        a["seconds"] += c.get("seconds") or 0
        if c["layer"] == "L3":
            a["l3_cases"] = a.get("l3_cases", 0) + c.get("n_cases", 0)
            a["cases"] = f"{a['l3_cases']:,} cases in {a['n']} properties"
            a["covers"] = "k_max cycles and chains, R10 saturation, packing widths, int32 limits, array ends, random"
        a["results"].add(c["result"])
        if c.get("profile", "-") != "-" and c["layer"] == "L2":
            a["profiles"].append(c["profile"])
    lines = [f"# E20: the verification stack (SPEC v3 §8)\n",
             f"Commit `{out['rev'][:7]}`, {out['host']}. "
             "Generated by `bench/e20.py report` from build/e20/; results/e20_verification.json has every check, "
             "its path hits and each step's load.\n",
             f"Result: {out['result']} (every step ran at that commit and exited 0, and every check passed).\n",
             "| Layer | Target | Build | Check | Bound or scope | Cases (one row) | Runs | Seconds | Result |",
             "|---|---|---|---|---|---|---|---|---|"]
    for (layer, target, build, check, bound), a in agg.items():
        res = "pass" if a["results"] == {"pass"} else ", ".join(sorted(a["results"]))
        runs = f"{a['n']} ({', '.join(a['profiles'])})" if a["profiles"] else str(a["n"])
        lines.append(f"| {layer} | {target} | {build} | {check} | {bound} | {a['cases']} | {runs} | "
                     f"{a['seconds']:.1f} | {res} |")
    lines += ["", "## Mutants", "", "| Kind | Mutant | Caught by | Layer | Counterexample (edges) | Check s | Escapes |",
              "|---|---|---|---|---|---|---|"]
    proofs = [m for m in out["mutants"] if m["kind"] == "proof"]
    if proofs:
        killed = sum(m["status"] == "KILLED" for m in proofs)
        lines.append(f"| proof | {killed} of {len(proofs)} (verify/lean/mutants.py) | lake build, at the recorded "
                     "declarations | L1 | - | - | - |")
    for m in out["mutants"]:
        if m["kind"] != "code":
            continue
        esc = ", ".join(f"{k} {v}" for k, v in m["escapes"].items()) or "-"
        by = f"{m['caught_by']} ({m['part']})" if m["status"] == "CAUGHT" else m["status"]
        lines.append(f"| code | `{m['name']}`: {m['what']} | {by} | {m['layer']} | "
                     f"{m['counterexample_edges'] if m['counterexample_edges'] is not None else '-'} | "
                     f"{m['check_seconds']} | {esc} |")
    p = out["perf"]
    lines += ["", "## Perf (P14)", "",
              "Quick per-change subset: " + ", ".join(f"{k} {v:.0f} s" for k, v in p["quick_seconds"].items()) + ".",
              f"Full run: {p['full_total_seconds'] / 60:.0f} min (" +
              ", ".join(f"{k} {v:.0f} s" for k, v in p["full_seconds"].items()) + ").", "",
              "## Load", ""]
    for k, v in out["steps"].items():
        ld = v["load"]
        if ld:
            procs = ", ".join(f"{n} {pct}%" for n, pct in ld["foreign_processes_max_pct"].items()) or "none at 5% or more"
            lines.append(f"- {k}: foreign CPU mean {ld['foreign_pct_mean']}% of a core, max {ld['foreign_pct_max']}%; "
                         f"available memory at least {ld['mem_available_gb_min']} GB; foreign peaks: {procs}.")
    lines += ["", "## What is not covered", ""] + [f"- {x}" for x in out["not_covered"]]
    return "\n".join(lines) + "\n"


def report():
    steps = json.load(open(os.path.join(OUT, "steps.json")))
    revs = {v.get("rev") for v in steps.values()}
    if len(revs) != 1 or None in revs or any(str(r).endswith("-dirty") for r in revs):
        sys.exit(f"e20 report: the steps ran at different, uncommitted or unrecorded revisions "
                 f"{sorted(map(str, revs))}; rerun them at one commit")
    rev = revs.pop()
    checks = []
    if "quick-l1" in steps or "l1" in steps:
        log = open(os.path.join(OUT, "l1.log" if "l1" in steps else "quick-l1.log")).read()
        n = re.search(r"L1 proofs: ok, (\d+) main theorems", log)
        checks.append({"layer": "L1", "target": "T3", "build": "-", "check": "verify/algorithms.sh",
                       "covers": "the generated loop nests' shortcuts against SPEC v3 §3.2 (verify/lean/COVERAGE.md)",
                       "bound": "all sizes (unbounded integers)", "cases": f"{n.group(1) if n else '?'} main theorems",
                       "seconds": steps.get("quick-l1", steps.get("l1"))["seconds"],
                       "path_hit": "none at runtime: COVERAGE.md pins each theorem to a generated function and snapshot",
                       "result": "pass" if n else "FAIL"})
    proof_mutants = []
    if "l1" in steps:
        log = open(os.path.join(OUT, "l1.log")).read()
        for m in re.finditer(r"^(KILLED|SURVIVED|MOVED|BAD)\s+(\d+) (.*)\n\s+fails at ([^\n]*)", log, re.M):
            proof_mutants.append({"kind": "proof", "status": m.group(1), "name": m.group(3),
                                  "caught_by": "lake build: " + m.group(4)})
        k = re.search(r"(\d+) of (\d+) mutants killed where recorded", log)
        checks.append({"layer": "L1", "target": "T3", "build": "-", "check": "verify/algorithms.sh --mutants",
                       "covers": "proof mutants: a wrong model, contract or claim must fail the build",
                       "bound": "-", "cases": f"{k.group(2) if k else '?'} proof mutants",
                       "seconds": steps["l1"]["seconds"], "path_hit": "each fails at its recorded declarations",
                       "result": "pass" if k and k.group(1) == k.group(2) else "FAIL"})
    if os.path.exists(os.path.join(OUT, "codegen.json")):
        checks += l2_rows(json.load(open(os.path.join(OUT, "codegen.json"))))
    for name in steps:
        if name.startswith("l3-"):
            build = name.replace("l3-random-", "").replace("l3-", "")
            checks += l3_rows("l3_random" if "random" in name else "l3 fixed seeds", os.path.join(OUT, f"{name}.log"),
                              build)
    code_mutants = []
    if os.path.exists(os.path.join(OUT, "mutants.json")):
        mj = json.load(open(os.path.join(OUT, "mutants.json")))
        for name, r in mj["mutants"].items():
            cx = r.get("counterexample") or {}
            code_mutants.append({"kind": "code", "name": name, "file": f"{r['file']}:{r.get('line')}", "build": r["build"],
                                 "what": r["what"], "origin": r["origin"], "status": r["status"],
                                 "caught_by": r.get("caught_by"), "layer": r.get("layer"), "part": r.get("part"),
                                 "counterexample_edges": cx.get("edges"), "check_seconds": r.get("check_seconds"),
                                 "seconds_with_builds": r["seconds"],
                                 "escapes": {k: "passes" if not v.get("caught") else "CAUGHT"
                                             for k, v in r.get("escapes", {}).items()}})
    quick = {k: steps[k]["seconds"] for k in ("quick-l1", "quick-rel", "quick-rel-cuda") if k in steps}
    full_steps = [k for k in steps if not k.startswith("quick")]
    out = {"experiment": "E20", "rev": rev, "host": "local: Ryzen 5 3600 (6 cores, 12 threads), 16 GB, GTX 1650 SUPER",
           "steps": {k: {kk: v[kk] for kk in ("cmd", "rc", "started", "seconds", "load")} for k, v in steps.items()},
           "checks": checks, "mutants": proof_mutants + code_mutants,
           "perf": {"quick_seconds": quick, "full_seconds": {k: steps[k]["seconds"] for k in full_steps},
                    "full_total_seconds": round(sum(steps[k]["seconds"] for k in full_steps), 1)}}
    out["not_covered"] = NOT_COVERED
    # Every step ran and exited 0, and every check passed.
    failed = [k for k in STEPS if k not in steps or steps[k]["rc"] != 0]
    failed += [c["check"] for c in checks if c["result"] != "pass"]
    out["result"] = "FAIL: " + ", ".join(failed) if failed else "pass"
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "e20_verification.json"), "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    with open(os.path.join(ROOT, "results", "e20_verification.md"), "w") as f:
        f.write(markdown(out))
    print(f"results/e20_verification.json at {rev}: {len(checks)} checks, {len(proof_mutants)} proof and "
          f"{len(code_mutants)} code mutants; result {out['result']}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("run", "report"):
        sys.exit(__doc__)
    unknown = [s for s in sys.argv[2:] if s not in STEPS]
    if unknown:
        sys.exit(f"unknown steps {unknown}; steps: {', '.join(STEPS)}")
    run(sys.argv[2:]) if sys.argv[1] == "run" else report()

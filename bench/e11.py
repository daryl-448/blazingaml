#!/usr/bin/env python3
"""E11 (SPEC v2 §7.2): the optimization ablation of the generated CPU code.

  python3 bench/e11.py [--host local] [--threads 1,12] [--reps 3] [--jobs 6] [--deps-from build/rel]
  python3 bench/e11.py --report-only [--host local]

1. Builds each variant in build/e11/<variant>: Release, -DBLZ_WITH_XGBOOST=OFF, no tests, and
   -DBLZ_DISABLE_OPTS=<variant> (blzc --disable); "full" disables nothing. The dependency sources
   come from --deps-from, an existing build tree, so nothing is downloaded.
2. Gates each variant: `blz fuzz-diff` on --fuzz-seeds graphs, every block against the oracle.
3. Waits for an idle machine (SPEC v2 §2.5): the other processes together below --idle-cpu %
   CPU over 10 s, for at most --max-wait-min minutes, else stops without timing anything.
4. Per block of the profile, thread count and variant: a preflight on a 1 % trigger sample
   (`blz mine --triggers sample:1:11`, its extrapolated_mine_seconds) estimates one run. A cell
   whose warm-up and timed runs would exceed --cell-limit-s is skipped and recorded (aborted,
   never truncated, §2.5). Otherwise `blz bench`: 1 warm-up and --reps timed runs (median, min,
   max), dynamic schedule, OMP_PROC_BIND=close, OMP_PLACES=threads.
5. Every variant's output_fnv1a must equal the full build's, block by block.
6. Logs the top CPU consumers every 10 s into the JSON (§2.5).
The frozen DG-2 recipe is not compiled (DG-3: GFP-produced features), so only --profile runs.

Writes results/e11_ablation_<host>.json (after every cell) and .md. Python standard library
only. Exit status 0 if every gate and hash check passed.
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VARIANTS = ["full", "find_start", "distinct_first", "degree_order", "count_only", "sweep"]
DEPS = {"NLOHMANN_JSON": "nlohmann_json-src", "CLI11": "cli11-src", "YAML-CPP": "yaml-cpp-src"}
GRAPH = "data/hi_small.blzg"
T0 = time.monotonic()


def now():
    return round(time.monotonic() - T0, 1)


def run(cmd, env=None, quiet=False):
    if not quiet:
        print("+", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                       env={**os.environ, **(env or {})})
    if p.returncode != 0 or not quiet:
        sys.stdout.write(p.stdout[-3000:])
    return p.returncode, p.stdout


# --- load ------------------------------------------------------------------------------------
def cpu_ticks():
    """pid -> (comm, utime + stime in clock ticks, ppid) for every readable process."""
    out = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/stat") as f:
                s = f.read()
        except OSError:
            continue
        a, b = s.find("("), s.rfind(")")
        fields = s[b + 2:].split()
        out[int(pid)] = (s[a + 1:b], int(fields[11]) + int(fields[12]), int(fields[1]))
    return out


def top_cpu(before, after, seconds, k=5):
    """The k busiest processes; "own" marks this script and the blz runs it started."""
    hz, me = os.sysconf("SC_CLK_TCK"), os.getpid()
    rows = []
    for pid, (comm, t, ppid) in after.items():
        if pid in before:
            pct = 100.0 * (t - before[pid][1]) / (hz * seconds)
            if pct > 0.5:
                rows.append({"pid": pid, "comm": comm, "cpu_pct": round(pct, 1), "own": me in (pid, ppid)})
    rows.sort(key=lambda r: -r["cpu_pct"])
    return rows[:k]


class LoadLog(threading.Thread):
    """The top CPU consumers every `period` seconds (SPEC v2 §2.5)."""

    def __init__(self, period=10.0):
        super().__init__(daemon=True)
        self.period, self.entries, self.stop = period, [], threading.Event()

    def run(self):
        prev, tprev = cpu_ticks(), time.monotonic()
        while not self.stop.wait(self.period):
            cur, t = cpu_ticks(), time.monotonic()
            self.entries.append({"t": now(), "top": top_cpu(prev, cur, t - tprev)})
            prev, tprev = cur, t


def foreign_pct(rows):
    """CPU % of the processes other than this script and the blz runs it times."""
    return sum(r["cpu_pct"] for r in rows if not r["own"])


def wait_idle(limit_pct, max_wait_min):
    deadline = time.monotonic() + 60 * max_wait_min
    checks = []
    while True:
        a = cpu_ticks()
        time.sleep(10)
        rows = top_cpu(a, cpu_ticks(), 10, k=8)
        load = foreign_pct(rows)
        checks.append({"t": now(), "foreign_cpu_pct": round(load, 1), "top": rows})
        print(f"idle check: other processes {load:.0f} % CPU (limit {limit_pct} %)", flush=True)
        if load < limit_pct:
            return True, checks
        if time.monotonic() > deadline:
            return False, checks
        time.sleep(20)


# --- builds ----------------------------------------------------------------------------------
def build(variant, deps_from, jobs):
    d = os.path.join("build", "e11", variant)
    args = ["cmake", "-S", ".", "-B", d, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DBLZ_WITH_XGBOOST=OFF",
            "-DBLZ_ENABLE_CUDA=OFF", "-DBUILD_TESTING=OFF",
            f"-DBLZ_DISABLE_OPTS={'' if variant == 'full' else variant}"]
    for name, src in DEPS.items():
        path = os.path.join(ROOT, deps_from, "_deps", src)
        if os.path.isdir(path):
            args.append(f"-DFETCHCONTENT_SOURCE_DIR_{name}={path}")
    rc, _ = run(args, quiet=True)
    if rc == 0:
        rc, _ = run(["cmake", "--build", d, "--target", "blz", f"-j{jobs}"], quiet=True)
    return rc, os.path.join(d, "blz")


def code_of(path):
    """A generated file without its full-line comments (the plan notes)."""
    with open(path) as f:
        return [l for l in f if not l.lstrip().startswith("//")]


def same_code(variant, stems):
    """Per pattern stem: whether the variant's generated CPU code equals the full build's."""
    out = {}
    for s in stems:
        files = [f"{s}.hpp", f"{s}_cpu.cpp"]
        out[s] = all(code_of(os.path.join(ROOT, "build/e11", variant, "generated", f)) ==
                     code_of(os.path.join(ROOT, "build/e11/full/generated", f)) for f in files)
    return out


# --- timing ----------------------------------------------------------------------------------
def preflight(blz, profile, block, threads, tmp):
    out = os.path.join(tmp, "preflight")
    shutil.rmtree(out, ignore_errors=True)
    rc, _ = run([blz, "mine", "--graph", GRAPH, "--profile", profile, "--backend", "cpu", "--threads", str(threads),
                 "--blocks", block, "--triggers", "sample:1:11", "--out", out], quiet=True)
    if rc != 0:
        return None
    meta = json.load(open(os.path.join(out, "features.json")))
    return meta["blocks"][block]["extrapolated_mine_seconds"]


def bench(blz, profile, block, threads, reps, tmp):
    path = os.path.join(tmp, "bench.json")
    if os.path.exists(path):
        os.remove(path)
    rc, _ = run([blz, "bench", "--graph", GRAPH, "--profile", profile, "--backend", "cpu", "--threads", str(threads),
                 "--reps", str(reps), "--schedule", "dynamic", "--blocks", block, "--out", path, "--name", "run"],
                env={"OMP_PROC_BIND": "close", "OMP_PLACES": "threads"}, quiet=True)
    if rc != 0:
        return None, None
    doc = json.load(open(path))["run"]
    jb = doc["blocks"][block]
    cell = dict(jb["dynamic"][str(threads)])
    cell.update({"output_fnv1a": jb["output_fnv1a"], "cpu_plans": jb["cpu_plans"],
                 "disabled_optimizations": doc["disabled_optimizations"], "peak_rss_mb": doc["peak_rss_mb"]})
    return cell, doc


# --- report ----------------------------------------------------------------------------------
def fmt(x):
    return f"{x:.3f}" if x < 10 else f"{x:.1f}"


def foreign_max(log, t0, t1):
    vals = [foreign_pct(e["top"]) for e in log if t0 <= e["t"] <= t1 + 10]
    return max(vals) if vals else 0.0


def report(doc, md_path):
    variants, blocks, threads = doc["variants_order"], doc["blocks_order"], [str(t) for t in doc["threads"]]
    cells = doc["cells"]
    env = doc.get("env") or {}
    git = env.get("git") or {}
    model = next((l.split(":", 1)[1].strip() for l in env.get("lscpu", "").splitlines() if l.startswith("Model name")), "?")
    out = [f"# E11: optimization ablation, CPU backend ({doc['host']})", "",
           f"- CPU: {model}; OMP_PROC_BIND=close, OMP_PLACES=threads; dynamic schedule",
           f"- Build: {env.get('build', {}).get('type', '?')}, `{env.get('build', {}).get('cxx_flags', '?')}`, "
           f"{env.get('compiler', '?')}; one build per variant in build/e11/<variant>, without XGBoost",
           f"- Git: {git.get('sha', '?')[:12]}{' (dirty)' if git.get('dirty') else ''}; profile "
           f"{doc['profile']['name']} ({doc['profile']['id']}); graph {GRAPH}",
           f"- Each cell: median [min–max] of {doc['reps']} timed runs after 1 warm-up, over every trigger "
           "(SPEC v2 §7.2). The index build is not timed.",
           "- A variant is the full build with one optimization switched off (`blzc --disable`, SPEC v2 §7.2): "
           "find_start = scans start at the list head; distinct_first = a per-thread hash set; degree_order = "
           "intersections always scan their src operand; count_only = len()-only sets are materialized; "
           "sweep = fan and degree keep v1's per-trigger plan.",
           f"- Not run: {doc['skipped_profiles']}", ""]
    bad = [f"{b}/{v}" for b in blocks for v in variants if doc["hash_check"].get(b, {}).get(v) is False]
    gates = {v: doc["variants"][v].get("fuzz_diff", {}).get("mismatched_rows") for v in variants}
    out += [f"Output check: every timed variant's output_fnv1a {'equals' if not bad else 'DIFFERS from'} the full "
            f"build's, block by block{'' if not bad else ': ' + ', '.join(bad)}. Fuzz gate "
            f"(`blz fuzz-diff --seeds {doc['fuzz_seeds']}`, every block against the oracle): mismatching rows "
            + ", ".join(f"{v} {gates[v]}" for v in variants) + ".", ""]
    for t in threads:
        out += [f"## {t} thread{'s' if t != '1' else ''}: seconds, and slowdown against the full build", "",
                "| block | " + " | ".join(variants) + " |", "|---|" + "---:|" * len(variants)]
        for b in blocks:
            base = cells.get(b, {}).get("full", {}).get(t, {}).get("median")
            row = []
            for v in variants:
                c = cells.get(b, {}).get(v, {}).get(t)
                if not c:
                    row.append("–")
                elif "skipped" in c:
                    row.append(f"skipped ({c['skipped']})")
                else:
                    s = f"{fmt(c['median'])} [{fmt(c['min'])}–{fmt(c['max'])}]"
                    if v != "full" and base:
                        s += f" {c['median'] / base:.2f}×"
                    if v != "full" and doc["same_code"].get(v, {}).get(b):
                        s += " ="
                    row.append(s)
            out.append(f"| {b} | " + " | ".join(row) + " |")
        out.append("")
    out += ["`=`: the variant's generated code for this block equals the full build's (comments aside), so the "
            "difference is noise.", ""]
    sweep_blocks = [b for b in blocks if "sweep" in cells.get(b, {}).get("full", {}).get(threads[0], {}).get(
        "cpu_plans", {}).values()]
    if sweep_blocks and "sweep" in variants:
        out += ["## The sweep lowering (SPEC v2 §7.1)", "",
                "Speedup = the per-trigger plan (`--disable sweep`) / the sweep (the full build).", "",
                "| block | " + " | ".join(f"{t} thr: per-trigger → sweep" for t in threads) + " |",
                "|---|" + "---:|" * len(threads)]
        for b in sweep_blocks:
            row = []
            for t in threads:
                f, s = cells[b]["full"].get(t, {}), cells[b].get("sweep", {}).get(t, {})
                row.append(f"{fmt(s['median'])} → {fmt(f['median'])} s: {s['median'] / f['median']:.1f}×"
                           if "median" in f and "median" in s else "–")
            out.append(f"| {b} | " + " | ".join(row) + " |")
        out.append("")
    noisy = []
    for b in blocks:
        for v in variants:
            for t in threads:
                c = cells.get(b, {}).get(v, {}).get(t, {})
                if "t0" in c:
                    fm = foreign_max(doc["load_log"], c["t0"], c["t1"])
                    if fm >= 50:
                        noisy.append(f"{b}/{v}/{t} thr ({fm:.0f} %)")
    out += ["## Measurement notes", "",
            f"- Idle check before timing: {doc.get('idle', {}).get('ok')}; the top CPU consumers were logged every "
            f"10 s ({len(doc['load_log'])} samples, `load_log`).",
            "- Cells with other processes at ≥ 50 % CPU at some sample during their runs: "
            + (", ".join(noisy) if noisy else "none") + "."]
    out += [f"- {n}" for n in doc.get("notes", [])] + [""]
    with open(md_path, "w") as f:
        f.write("\n".join(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="local")
    ap.add_argument("--profile", default="configs/profiles/v1_default.yaml")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--threads", default="1,12")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=6, help="build parallelism")
    ap.add_argument("--deps-from", default="build/rel", help="a build tree whose _deps sources are reused")
    ap.add_argument("--fuzz-seeds", default="0:1000")
    ap.add_argument("--cell-limit-s", type=float, default=900, help="warm-up + timed runs of one cell")
    ap.add_argument("--idle-cpu", type=float, default=100, help="CPU %% of other processes that counts as busy")
    ap.add_argument("--max-wait-min", type=float, default=30)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    json_path = os.path.join(ROOT, f"results/e11_ablation_{args.host}.json")
    md_path = json_path[:-5] + ".md"
    if args.report_only:
        report(json.load(open(json_path)), md_path)
        print("wrote", md_path)
        return 0

    variants = args.variants.split(",")
    assert variants[0] == "full" and all(v in VARIANTS for v in variants), "variants: full first, then " + ",".join(VARIANTS[1:])
    threads = [int(t) for t in args.threads.split(",")]
    doc = {"spec": "SPEC v2 §7.2 (E11)", "host": args.host, "machine": platform.node(), "graph": GRAPH,
           "variants_order": variants, "threads": threads, "reps": args.reps, "fuzz_seeds": args.fuzz_seeds,
           "cell_limit_s": args.cell_limit_s, "skipped_profiles":
               "the frozen DG-2 recipe (configs/dg2_frozen.yaml): its pattern blocks are GFP-produced (G1, DG-3), "
               "so no compiled profile exists", "variants": {}, "cells": {}, "hash_check": {}, "same_code": {},
           "notes": []}
    ok = True

    def save():
        with open(json_path, "w") as f:
            json.dump(doc, f, indent=1)
            f.write("\n")

    # 1-2. Builds and the fuzz gate.
    blz = {}
    for v in variants:
        t = time.monotonic()
        rc, path = build(v, args.deps_from, args.jobs)
        rec = {"build_dir": os.path.dirname(path), "build_seconds": round(time.monotonic() - t, 1),
               "disabled": [] if v == "full" else [v]}
        doc["variants"][v] = rec
        if rc != 0:
            rec["error"] = "build failed"
            ok = False
            continue
        rc, txt = run([path, "fuzz-diff", "--seeds", args.fuzz_seeds], quiet=True)
        line = next((l for l in txt.splitlines() if l.startswith("fuzz-diff")), txt[-200:])
        rows = int(line.split(" mismatching rows")[0].rsplit(" ", 1)[1]) if " mismatching rows" in line else None
        rec["fuzz_diff"] = {"exit": rc, "summary": line, "mismatched_rows": rows}
        print(f"{v}: built in {rec['build_seconds']} s; {line}", flush=True)
        if rc == 0:
            blz[v] = os.path.join(ROOT, path)
        else:
            ok = False
    save()
    if "full" not in blz:
        print("the full build failed its gate; nothing timed")
        return 1

    # 3. An idle machine.
    log = LoadLog()
    log.start()
    idle, checks = wait_idle(args.idle_cpu, args.max_wait_min)
    doc["idle"] = {"ok": idle, "limit_pct": args.idle_cpu, "checks": checks}
    if not idle:
        doc["notes"].append(f"stopped: the machine was not idle within {args.max_wait_min} min; nothing timed")
        doc["load_log"] = log.entries
        save()
        print("not idle; nothing timed (see idle.checks)")
        return 1

    # 4-5. Cells.
    with open(os.path.join(ROOT, args.profile)) as f:
        blocks = [l.split(":")[0].strip() for l in f.read().split("blocks:", 1)[1].splitlines()
                  if l.strip() and not l.strip().startswith("#") and ":" in l]
    doc["blocks_order"] = blocks
    tmp = tempfile.mkdtemp(prefix="e11_", dir=os.path.join(ROOT, "features"))
    try:
        for b in blocks:
            for t in threads:
                for v in variants:
                    if v not in blz:
                        continue
                    cell_doc = doc["cells"].setdefault(b, {}).setdefault(v, {})
                    est = preflight(blz[v], args.profile, b, t, tmp)
                    budget = None if est is None else est * (1 + args.reps)
                    if est is None or budget > args.cell_limit_s:
                        why = "preflight failed" if est is None else (
                            f"preflight: {est:.0f} s per run, {budget:.0f} s > {args.cell_limit_s:.0f} s")
                        cell_doc[str(t)] = {"skipped": why, "preflight_seconds_per_run": est}
                        print(f"{b} {v} {t} thr: skipped, {why}", flush=True)
                        save()
                        continue
                    t0 = now()
                    cell, bdoc = bench(blz[v], args.profile, b, t, args.reps, tmp)
                    if cell is None:
                        cell_doc[str(t)] = {"skipped": "bench failed"}
                        ok = False
                        save()
                        continue
                    cell.update({"preflight_seconds_per_run": round(est, 3), "t0": t0, "t1": now()})
                    cell_doc[str(t)] = cell
                    if v == "full" and "env" not in doc:
                        doc["env"] = bdoc["env"]
                        doc["profile"] = bdoc["profile"]
                    print(f"{b:8s} {v:15s} {t:2d} thr: median {cell['median']:.3f} s "
                          f"[{cell['min']:.3f}–{cell['max']:.3f}]", flush=True)
                    doc["load_log"] = log.entries
                    save()
        stems = {}
        with open(os.path.join(ROOT, args.profile)) as f:
            for l in f:
                if "patterns:" in l:
                    name = l.split(":")[0].strip()
                    stems[name] = [s.strip() for s in l.split("[", 1)[1].split("]", 1)[0].split(",")]
        for v in variants:
            if v in blz and v != "full":
                per = same_code(v, sorted({s for b in blocks for s in stems[b]}))
                doc["same_code"][v] = {b: all(per[s] for s in stems[b]) for b in blocks}
        for b in blocks:
            ref = {c.get("output_fnv1a") for c in doc["cells"][b].get("full", {}).values() if "output_fnv1a" in c}
            if len(ref) != 1:
                ok = False
                continue
            ref = ref.pop()
            doc["hash_check"][b] = {}
            for v in variants:
                got = [c["output_fnv1a"] for c in doc["cells"][b].get(v, {}).values() if "output_fnv1a" in c]
                if got:
                    doc["hash_check"][b][v] = all(g == ref for g in got)
                    ok &= doc["hash_check"][b][v]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        log.stop.set()
        doc["load_log"] = log.entries
        save()
    doc["pass"] = ok
    save()
    report(doc, md_path)
    print("E11:", "pass" if ok else "FAIL", "->", json_path, md_path)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

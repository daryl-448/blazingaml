#!/usr/bin/env python3
"""The code mutants (SPEC v3 §4.4 item 5, V-mut; AC-V3-3): one-line changes to the compiled C++ that
an executable check must catch. The proofs cannot see these; their mutants are L1's
(verify/lean/mutants.py).

    verify/mutants.py [--list] [--escape] [--keep] [--rev REV] [--jobs N] [--report PATH] [NAME ...]

Each mutant is applied in a scratch git worktree of REV (default HEAD) at build/mutants/src, never in
the main tree, and built in build/mutants/cpu or build/mutants/cuda: Release, XGBoost off, CUDA for
sm_75 only, the FetchContent sources taken from an existing preset build (build/rel/_deps). Only the
target the next check needs is built: blz for T4 and L2, blz_unit_tests for L3.

Each build has a ladder of checks, cheapest first (LADDERS). The unmutated tree must pass every rung
that a selected mutant's record reaches (the baseline). Then each mutant runs up the ladder until a
check fails: that check caught it. A mutant is

  CAUGHT   by the check its record names (`caught_by`),
  MOVED    by another check: the record is stale; update `caught_by`,
  ESCAPED  by none: every rung passed, so no check sees the bug,
  BAD      when its `old` text is not on exactly one line, the mutant does not build, or a check
           fails without a mismatch: an exit other than blz's 2 with a counterexample or violation
           in its report, or an L3 failure that is not Catch2's exit 42 on a falsified property (a
           property that threw, a crash, a timeout).

--escape then runs each mutant's `escapes`, the full-bound L2 and T4 checks that run its code path,
and requires that they pass: such a mutant is out of L2's reach, and only L3 sees it. A check that
catches it there is NOT ESCAPED: drop it from `escapes`.

Anything but CAUGHT (and, with --escape, ESCAPES) fails the run. The report (default
build/mutants/report.json) has per mutant the catching check and layer, the counterexample's size
in edges and its text, the seconds, and the load; build/mutants/logs/ has every command's output.
Without --keep the worktree and its builds are removed at the end; with it, the next run rebuilds
only what changed.
"""
import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(ROOT, "build", "mutants")
SRC = os.path.join(WORK, "src")
LOGS = os.path.join(WORK, "logs")
NVCC = "/opt/cuda/bin/nvcc"
DEPS = ("catch2", "nlohmann_json", "cli11", "yaml-cpp", "rapidcheck")
PROFILES = ("configs/profiles/clean_all.yaml", "tests/profiles/l2_legacy_stream.yaml",
            "tests/profiles/l2_legacy_tsvis.yaml")
CHECK_TIMEOUT = 3600  # s; a check that hangs is an error (BAD), not a catch

# L2's bounds, read from verify/codegen.sh so the two never disagree: A, B, T4, DEV, LIMIT_GPU, SMOKE.
with open(os.path.join(ROOT, "verify", "codegen.sh"), encoding="utf-8") as _f:
    BOUND = dict(re.findall(r"^(\w+)=(m=\d+,n=\d+,t=\d+)", _f.read(), re.M))


@dataclass(frozen=True)
class Check:
    name: str
    layer: str           # T4: blz index-contracts; L2: blz fuzz-diff --bound; L3: the [l3] properties
    args: tuple = ()     # blz's arguments; L3 has none
    per_profile: bool = False  # run once per L2 profile, with --profile


def fd(bound, *extra):
    return ("fuzz-diff", "--bound", bound, *extra)


LIMIT1 = ("--batch-edges", "1", "--work-limit", "1")
LADDERS = {  # cheapest first; the smoke rungs take about a second each, L3 a few
    "cpu": [
        Check("t4-smoke", "T4", ("index-contracts", "--bound", BOUND["SMOKE"])),
        Check("l2-smoke-cpu", "L2", fd(BOUND["SMOKE"]), True),
        Check("l2-smoke-cpu-limit1", "L2", fd(BOUND["SMOKE"], *LIMIT1), True),
        Check("l3", "L3"),
        Check("l2-a-cpu", "L2", fd(BOUND["A"]), True),
        Check("l2-a-cpu-limit1", "L2", fd(BOUND["A"], *LIMIT1), True),
        Check("t4-a", "T4", ("index-contracts", "--bound", BOUND["T4"])),
        Check("l2-b-cpu", "L2", fd(BOUND["B"]), True),
    ],
    "cuda": [
        Check("l2-smoke-cuda", "L2", fd(BOUND["SMOKE"], "--backend", "cuda"), True),
        Check("l2-smoke-cuda-fuse-pack", "L2", fd(BOUND["SMOKE"], "--backend", "cuda", "--fuse", "--pack"), True),
        Check("l2-smoke-hybrid", "L2", fd(BOUND["SMOKE"], "--backend", "hybrid"), True),
        Check("l3", "L3"),
        Check("l2-a-cuda", "L2", fd(BOUND["A"], "--backend", "cuda"), True),
        Check("l2-a-cuda-fuse-pack", "L2", fd(BOUND["A"], "--backend", "cuda", "--fuse", "--pack"), True),
        Check("l2-limit1-cuda", "L2", fd(BOUND["LIMIT_GPU"], "--backend", "cuda", *LIMIT1), True),
        Check("l2-a-hybrid", "L2", fd(BOUND["A"], "--backend", "hybrid"), True),
    ],
}


@dataclass(frozen=True)
class Mutant:
    name: str
    file: str
    old: str              # must occur on exactly one line of `file`
    new: str | None       # replaces `old` on that line; None deletes the line
    build: str            # cpu or cuda
    caught_by: str        # the first rung of its build's ladder that fails
    what: str             # the bug, in the code's terms
    origin: str           # who planted it first
    escapes: tuple = ()   # --escape: full-bound L2/T4 checks that run its path and must pass


CPU_A = ("l2-a-cpu", "l2-a-cpu-limit1", "l2-b-cpu")
MUTANTS = [
    Mutant("pair_latest_lo", "include/blz/primitives.hpp",
           "if (b == a || std::int64_t(run_ts[b - 1]) < lo_ts) return kNoTs;",
           "if (b == a || std::int64_t(run_ts[b - 1]) <= lo_ts) return kNoTs;",
           "cpu", "t4-smoke", "pair_latest drops a latest entry whose ts equals lo", "L2 M1"),
    Mutant("order_keep_le", "src/codegen/cpu_emitter.cpp",
           "keep = dsl::cmp_text(r.keep);",
           "keep = dsl::cmp_text(r.keep == dsl::Expr::Cmp::lt ? dsl::Expr::Cmp::le : r.keep);",
           "cpu", "l2-smoke-cpu", "the emitter writes sg_ord's order keep `<` as `<=` (legacy only; clean is `<=`)",
           "L2 M2"),
    Mutant("sweep_evict_le", "include/blz/sweep.hpp",
           "for (; lo < hi && std::int64_t(l.ts[lo]) < lo_ts; ++lo)",
           "for (; lo < hi && std::int64_t(l.ts[lo]) <= lo_ts; ++lo)",
           "cpu", "l2-smoke-cpu", "the sweep evicts the entry at the window's lower end", "L2 M3"),
    Mutant("reach_range_lo", "include/blz/reach.hpp",
           "const Range rg = in.range(y, lo, lim, max_eid);",
           "const Range rg = in.range(y, lo + 1, lim, max_eid);",
           "cpu", "t4-smoke", "the pruning search skips in-edges at the window's lower end", "L2 M4"),
    Mutant("fused_window_m0", "src/codegen/cpu_emitter.cpp",
           'em[k].scan_test(list, "j", "y", "m" + std::to_string(k) + "_lo")',
           'em[k].scan_test(list, "j", "y", "m0_lo")',
           "cuda", "l2-smoke-cuda-fuse-pack", "a fused scan tests every member against member 0's window start",
           "L2 M5"),
    Mutant("work_limit_lt", "include/blz/registry.hpp",
           "if (++work <= limit) return false;",
           "if (++work < limit) return false;",
           "cpu", "l2-smoke-cpu-limit1", "RowSink::over stops one binding before the work limit (R11)", "L2 M6"),
    Mutant("reach_rounds_from_1", "include/blz/reach.hpp",
           "for (int r = 0; r < hops && !frontier_.empty(); ++r) {",
           "for (int r = 1; r < hops && !frontier_.empty(); ++r) {",
           "cpu", "l3", "the pruning search runs 8 rounds, not 9: a K_MAX cycle is cut", "L3 mutant 1",
           CPU_A + ("t4-a",)),
    Mutant("count_clamp_ge", "include/blz/registry.hpp",
           "if (sum > UINT32_MAX) {",
           "if (sum >= UINT32_MAX) {",
           "cpu", "l3", "a count row that reaches UINT32_MAX exactly counts as saturated (R10)", "L3 mutant 2",
           CPU_A),
    Mutant("pack_width_2byte", "src/runtime/cuda_backend.cu",
           "width = max_cell <= 0xffu ? 1 : max_cell <= 0xffffu ? 2 : 4;",
           "width = max_cell <= 0xffu ? 1 : max_cell <= 0x1ffffu ? 2 : 4;",
           "cuda", "l3", "packed D2H keeps 2 bytes for cells up to 131,071", "L3 mutant 3",
           ("l2-a-cuda-fuse-pack",)),
    Mutant("first_ts_int32", "include/blz/primitives.hpp",
           "BLZ_HD std::uint32_t first_ts_at_least(std::uint32_t a, std::uint32_t b, std::int64_t v) const {",
           "BLZ_HD std::uint32_t first_ts_at_least(std::uint32_t a, std::uint32_t b, std::int32_t v) const {",
           "cpu", "l3", "range's time search narrows its int64 bound to int32", "L3 mutant 4",
           CPU_A + ("t4-a",)),
    Mutant("pack_tail_max", "src/runtime/cuda_backend.cu",
           "m = max(m, in[i]);", None,
           "cuda", "l3", "pack_kernel's tail loop leaves its cells out of the width's max", "L3 mutant 5",
           ("l2-a-cuda-fuse-pack",)),
    Mutant("search_stale_mark", "include/blz/reach.hpp",
           "if (mark_[x] == epoch_ && ld_[x] >= ts) continue;",
           "if (ld_[x] >= ts) continue;",
           "cpu", "t4-smoke", "the pruning search trusts an ld left by an earlier trigger (or the initial 0)",
           "P14 Search check"),
    Mutant("keep_eq", "include/blz/reach.hpp",
           "std::int64_t(ld_[x]) > h; }",
           "std::int64_t(ld_[x]) == h + 1; }",
           "cpu", "t4-smoke", "keep holds only at the threshold ld - 1, not below it (drops 0->1@0, 1->0@2)",
           "sol-loop P14 S2"),
]


def sh(cmd, log, cwd=None, timeout=None, env=None):
    """Runs cmd, appending its output to log; returns (rc, seconds, output). rc is -9 on a timeout."""
    t0 = time.monotonic()
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env)
        rc, out = p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        rc, out = -9, f"{e.stdout or ''}{e.stderr or ''}\n(timed out after {timeout} s)"
    dt = time.monotonic() - t0
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"$ {' '.join(cmd)}\n{out}\n(exit {rc}, {dt:.1f} s)\n")
    return rc, dt, out


def load():
    with open("/proc/meminfo") as f:
        avail = int(re.search(r"MemAvailable:\s+(\d+)", f.read()).group(1))
    return {"loadavg": [round(x, 2) for x in os.getloadavg()], "mem_available_gb": round(avail / 2**20, 1)}


def foreign_load():
    p = subprocess.run(["ps", "--sort=-pcpu", "-eo", "pid,pcpu,rss,etime,comm"], capture_output=True, text=True)
    return p.stdout.splitlines()[:8]


def git(*args, cwd=ROOT):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


class Build:
    """One scratch build directory of the worktree: configured on first use, then incremental."""

    def __init__(self, kind, jobs):
        self.kind, self.jobs = kind, jobs
        self.dir = os.path.join(WORK, kind)
        self.built = set()  # targets built since the source last changed
        self.build_seconds = 0.0

    def configure(self, log):
        if os.path.exists(os.path.join(self.dir, "CMakeCache.txt")):
            return True
        args = ["cmake", "-S", SRC, "-B", self.dir, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
                "-DBLZ_WITH_XGBOOST=OFF"]
        for preset in ("rel", "rel-cuda", "rel-noopt", "dev"):  # share the sources already downloaded
            deps = os.path.join(ROOT, "build", preset, "_deps")
            if all(os.path.isdir(os.path.join(deps, d + "-src")) for d in DEPS):
                args.append("-DFETCHCONTENT_FULLY_DISCONNECTED=ON")
                args += [f"-DFETCHCONTENT_SOURCE_DIR_{d.upper()}={deps}/{d}-src" for d in DEPS]
                break
        if self.kind == "cuda":
            args += ["-DBLZ_ENABLE_CUDA=ON", f"-DCMAKE_CUDA_COMPILER={NVCC}", "-DCMAKE_CUDA_ARCHITECTURES=75"]
        return sh(args, log)[0] == 0

    def ensure(self, target, log):
        """Builds target if the source changed since it was last built; False if the build fails."""
        if target in self.built:
            return True
        if not self.configure(log):
            return False
        rc, dt, _ = sh(["cmake", "--build", self.dir, "--target", target, "-j", str(self.jobs)], log)
        self.build_seconds += dt
        if rc == 0:
            self.built.add(target)
        return rc == 0


def l3_tests(build, log):
    exe = os.path.join(build.dir, "tests", "blz_unit_tests")
    rc, _, out = sh([exe, "[l3]", "--list-tests", "--verbosity", "quiet", "--order", "decl"], log)
    return [l for l in out.splitlines() if l.startswith("L3 ")] if rc == 0 else []


def run_check(check, build, log, report):
    """Runs one rung; returns a dict with caught (bool), seconds, and on a catch what and counterexample."""
    target = "blz_unit_tests" if check.layer == "L3" else "blz"
    if not build.ensure(target, log):
        return {"error": f"build of {target} failed"}
    res = {"check": check.name, "layer": check.layer, "seconds": 0.0, "caught": False}
    if check.layer == "L3":
        exe = os.path.join(build.dir, "tests", "blz_unit_tests")
        tests = l3_tests(build, log)
        if not tests:
            return {"error": "no [l3] tests listed"}
        for name in tests:
            rc, dt, out = sh([exe, name], log, timeout=CHECK_TIMEOUT)
            res["seconds"] += dt
            # A catch is Catch2's test-failure exit (42) on a falsified property; a property that threw
            # (a backend error), a crash, a timeout or a setup error is not.
            falsified = rc == 42 and "Falsifiable after" in out and not any(
                x in out for x in ("Exception thrown with message", "Unknown object thrown", "Unknown exception"))
            if rc != 0 and not falsified:
                return {**res, "error": f"{name}: exit {rc} without a falsified property: {tail(out)}"}
            if rc != 0:
                res.update(caught=True, rc=rc, part=name, counterexample=l3_counterexample(out))
                return res
        return res
    exe = os.path.join(build.dir, "blz")
    for prof in (PROFILES if check.per_profile else (None,)):
        args = [exe, *check.args] + (["--profile", os.path.join(SRC, prof)] if prof else [])
        if os.path.exists(report):
            os.remove(report)
        rc, dt, out = sh(args + ["--report", report, "--name", "x"], log, timeout=CHECK_TIMEOUT)
        res["seconds"] += dt
        if rc != 0:
            part = os.path.basename(prof)[:-5] if prof else check.args[0]
            cx = blz_counterexample(report) if rc == 2 else None
            if cx is None:  # blz exits 2 on a mismatch, with its counterexample in the report
                return {**res, "error": f"{part}: exit {rc} without a mismatch in the report: {tail(out)}"}
            res.update(caught=True, rc=rc, part=part, counterexample=cx)
            return res
    return res


def tail(out):
    return " / ".join([l for l in out.splitlines() if l.strip()][-3:])


def blz_counterexample(report):
    try:
        with open(report, encoding="utf-8") as f:
            r = json.load(f)["x"]
    except (OSError, ValueError, KeyError):
        r = {}
    if "counterexample" in r:  # fuzz-diff
        cx = r["counterexample"]
        detail = {k: cx[k] for k in ("block", "window", "path", "trigger_eid", "got", "want", "got_stop", "want_stop")
                  if k in cx}
        return {"edges": len(cx["edges"]), "graph": cx["edges"], **detail}
    if "first_violation" in r:  # index-contracts
        v = r["first_violation"]
        return {"edges": v.get("edges"), "text": v["text"]}
    return None


def l3_counterexample(out):
    m = re.search(r"Falsifiable after (\d+) tests?(?: and (\d+) shrinks?)?", out)
    g = re.search(r"\bn (\d+), m (\d+), windows", out)
    msg = re.search(r"Falsifiable after[^\n]*\n\s*\n(.*?)(?:\n\s*\n|$)", out, re.S)
    return {"edges": int(g.group(2)) if g else None,
            "tests": int(m.group(1)) if m else None, "shrinks": int(m.group(2) or 0) if m else None,
            "text": " ".join(msg.group(1).split())[:400] if msg else " / ".join(out.splitlines()[-3:])}


def apply(mutant):
    """Applies the mutant in the worktree; returns (line number, None) or (None, why not)."""
    path = os.path.join(SRC, mutant.file)
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    hits = [i for i, l in enumerate(lines) if mutant.old in l]
    if len(hits) != 1:
        return None, f"`old` is on {len(hits)} lines of {mutant.file}"
    i = hits[0]
    if mutant.new is None:
        del lines[i]
    else:
        lines[i] = lines[i].replace(mutant.old, mutant.new)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return i + 1, None


def revert(mutant):
    git("checkout", "--", mutant.file, cwd=SRC)


def prepare_worktree(rev):
    if os.path.isdir(SRC):
        git("checkout", "-q", "--detach", "--force", rev, cwd=SRC)
    else:
        git("worktree", "add", "-q", "--detach", SRC, rev)
    if git("status", "--porcelain", cwd=SRC):
        sys.exit(f"the worktree {SRC} is not clean")


def remove_worktree():
    if os.path.isdir(SRC):
        git("worktree", "remove", "--force", SRC)
    git("worktree", "prune")
    for kind in LADDERS:
        shutil.rmtree(os.path.join(WORK, kind), ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="*", help="mutants to run (default: all)")
    ap.add_argument("--list", action="store_true", help="print the mutants and the ladders, run nothing")
    ap.add_argument("--escape", action="store_true", help="also require each mutant to pass its `escapes`")
    ap.add_argument("--keep", action="store_true", help="keep the worktree and its builds for the next run")
    ap.add_argument("--rev", default="HEAD", help="the commit to mutate (default HEAD)")
    ap.add_argument("--jobs", type=int, default=6, help="build jobs (CUDA uses at most 4)")
    ap.add_argument("--report", default=os.path.join(WORK, "report.json"))
    a = ap.parse_args()

    by_name = {m.name: m for m in MUTANTS}
    unknown = [n for n in a.names if n not in by_name]
    if unknown:
        sys.exit(f"unknown mutants: {', '.join(unknown)}; see --list")
    selected = [by_name[n] for n in a.names] if a.names else MUTANTS
    for m in selected:  # the record must name rungs that exist
        rungs = {c.name for c in LADDERS[m.build]}
        assert m.caught_by in rungs and set(m.escapes) <= rungs, m.name
    if a.list:
        for kind, ladder in LADDERS.items():
            print(f"{kind} ladder: " + ", ".join(f"{c.name} ({c.layer})" for c in ladder))
        for m in MUTANTS:
            print(f"{m.name:20} {m.build:4} {m.caught_by:24} {m.file}: {m.what}"
                  + (f"; escapes {', '.join(m.escapes)}" if m.escapes else ""))
        return 0

    os.makedirs(LOGS, exist_ok=True)
    lock = open(os.path.join(WORK, ".lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit("another verify/mutants.py run holds build/mutants/.lock")
    rev = git("rev-parse", a.rev)
    if git("status", "--porcelain", "--untracked-files=no") and a.rev == "HEAD":
        print("note: the main tree has uncommitted changes; the mutants run on HEAD without them")
    t_run = time.monotonic()
    out = {"rev": rev, "started": time.strftime("%Y-%m-%d %H:%M:%S %z"), "bounds": BOUND,
           "load_start": load(), "foreign_load_start": foreign_load(),
           "ladders": {k: [{"check": c.name, "layer": c.layer, "args": list(c.args), "per_profile": c.per_profile}
                           for c in v] for k, v in LADDERS.items()},
           "baseline": {}, "mutants": {}}
    report_tmp = os.path.join(WORK, "check.json")
    prepare_worktree(rev)
    builds = {k: Build(k, a.jobs if k == "cpu" else min(a.jobs, 4)) for k in LADDERS}
    bad = 0
    try:
        for kind, ladder in LADDERS.items():
            mine = [m for m in selected if m.build == kind]
            if not mine:
                continue
            build = builds[kind]
            # The baseline: the unmutated tree passes every rung up to the deepest recorded catch.
            depth = max(next(i for i, c in enumerate(ladder) if c.name == m.caught_by) for m in mine)
            log = os.path.join(LOGS, f"baseline-{kind}.log")
            open(log, "w").close()
            base = []
            for check in ladder[:depth + 1]:
                r = run_check(check, build, log, report_tmp)
                base.append(r)
                print(f"baseline {kind:4} {check.name:24} "
                      + (r["error"] if "error" in r else "FAIL" if r["caught"] else "pass") + f" ({r.get('seconds', 0):.1f} s)",
                      flush=True)
                if "error" in r or r["caught"]:
                    out["baseline"][kind] = base
                    raise SystemExit(f"the unmutated {kind} build fails {check.name}; see {log}")
            out["baseline"][kind] = base
            for m in mine:
                log = os.path.join(LOGS, f"{m.name}.log")
                open(log, "w").close()
                t0, b0 = time.monotonic(), build.build_seconds
                rec = {"file": m.file, "old": m.old, "new": m.new, "build": kind, "what": m.what, "origin": m.origin,
                       "expected": m.caught_by, "load_before": load()}
                line, why = apply(m)
                build.built.clear()
                if line is None:
                    rec.update(status="BAD", why=why)
                else:
                    rec["line"] = line
                    rungs = []
                    for check in ladder:
                        r = run_check(check, build, log, report_tmp)
                        rungs.append(r)
                        if "error" in r or r["caught"]:
                            break
                    rec["rungs"] = [{k: r[k] for k in ("check", "caught", "seconds", "error") if k in r} for r in rungs]
                    last = rungs[-1]
                    if "error" in last:
                        rec.update(status="BAD", why=last["error"])
                    elif not last["caught"]:
                        rec["status"] = "ESCAPED"
                    else:
                        rec.update(status="CAUGHT" if last["check"] == m.caught_by else "MOVED",
                                   caught_by=last["check"], layer=last["layer"], part=last["part"], rc=last["rc"],
                                   check_seconds=round(last["seconds"], 1), counterexample=last["counterexample"])
                    if a.escape and m.escapes and rec["status"] == "CAUGHT":
                        esc = {}
                        for name in m.escapes:
                            r = run_check(next(c for c in ladder if c.name == name), build, log, report_tmp)
                            esc[name] = {k: r[k] for k in ("caught", "seconds", "error", "part") if k in r}
                            if "error" in r or r["caught"]:
                                rec["status"] = "NOT ESCAPED"
                        rec["escapes"] = esc
                    revert(m)
                rec.update(seconds=round(time.monotonic() - t0, 1), build_seconds=round(build.build_seconds - b0, 1),
                           load_after=load())
                out["mutants"][m.name] = rec
                bad += rec["status"] != "CAUGHT"
                cx = rec.get("counterexample") or {}
                n = cx.get("edges")
                size = "" if n is None else f", {n} edge" + ("" if n == 1 else "s")
                tail = (f" by {rec['caught_by']} ({rec['layer']}, {rec['part']}{size}) in {rec['check_seconds']} s"
                        if "caught_by" in rec else f": {rec.get('why', 'no rung failed')}")
                esc = rec.get("escapes")
                if esc:
                    tail += "; escapes " + ", ".join(f"{k} {'FAILS' if v.get('caught') else 'passes'}" for k, v in esc.items())
                print(f"{rec['status']:11} {m.name:20}{tail} [{rec['seconds']} s with builds]", flush=True)
    finally:
        out.update(seconds=round(time.monotonic() - t_run, 1), load_end=load(),
                   finished=time.strftime("%Y-%m-%d %H:%M:%S %z"))
        out["build_seconds"] = {k: round(b.build_seconds, 1) for k, b in builds.items()}
        with open(a.report, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
            f.write("\n")
        if os.path.exists(report_tmp):
            os.remove(report_tmp)
        if not a.keep:
            remove_worktree()
    caught = sum(r["status"] == "CAUGHT" for r in out["mutants"].values())
    print(f"{caught} of {len(selected)} mutants caught where recorded ({out['seconds']:.0f} s); report {a.report}")
    return 1 if bad or caught != len(selected) else 0


if __name__ == "__main__":
    sys.exit(main())

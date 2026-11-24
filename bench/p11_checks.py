#!/usr/bin/env python3
"""P11 checks on the local machine (SPEC v2 §12 P11): AC-V2-5's sweep parity, AC-V2-0 and AC-V2-1.

  python3 bench/p11_checks.py [--blz build/rel/blz] [--cuda build/rel-cuda/blz] [--threads 12]

- AC-V2-5, HI-Small: for every profile that uses the sweep (configs/profiles/*.yaml and
  tests/profiles/tsvis.yaml), `blz mine` of its sweep blocks (fan_in, fan_out, deg_in, deg_out)
  with the oracle and the cpu backend gives the same output_fnv1a, and the cpu manifest names
  the sweep plan for each of them.
- AC-V2-5, fuzz graphs: `blz fuzz-diff` (cpu, both schedules) on 10,000 graphs, every registered
  block under stream visibility, and the sg_ordered_tsvis profile under timestamp visibility.
- AC-V2-1: the metamorphic tests on 2,000 graphs per frozen profile on the cpu backend, whose
  fan and degree code is now the sweep.
- AC-V2-0: v1_default on the oracle, cpu and (if built) CUDA keeps the pinned hashes.

Merges into results/p11_checks.json; blz merges its own fuzz-diff and metamorphic reports. Needs
data/hi_small.blzg. Python standard library only. Exit status 0 if every check passes.
"""
import argparse
import glob
import json
import os
import platform
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = "results/p11_checks.json"
GRAPH = "data/hi_small.blzg"
SWEEP = ["fan_in", "fan_out", "deg_in", "deg_out"]
V1 = SWEEP + ["sg", "tcyc"]
FROZEN = ["v1_default", "v1_parity", "sg_ordered_v1w", "sg_ordered_tsvis"]
# SPEC v2 §2.4 AC-V2-0, from results/e4_cpu_local.json.
EXPECTED = {"fan_in": 10768036772042710147, "fan_out": 904878896487262018, "deg_in": 3321375241280117763,
            "deg_out": 3341707662053112211, "sg": 18124683238158106329, "tcyc": 7737672036140136160}


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    t0 = time.monotonic()
    p = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    sys.stdout.write(p.stdout[-3000:])
    return p.returncode, time.monotonic() - t0


def load_report():
    path = os.path.join(ROOT, REPORT)
    return json.load(open(path)) if os.path.exists(path) else {}


def save_report(doc):
    with open(os.path.join(ROOT, REPORT), "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")


def mine(blz, profile, backend, blocks, out, threads):
    rc, secs = run([blz, "mine", "--graph", GRAPH, "--profile", profile, "--backend", backend, "--blocks",
                    ",".join(blocks), "--threads", str(threads), "--out", out])
    if rc != 0:
        return {"exit": rc}
    meta = json.load(open(os.path.join(ROOT, out, "features.json")))
    res = {"profile_id": meta["run"]["profile_id"],
           "output_fnv1a": {b: v["output_fnv1a"] for b, v in meta["blocks"].items()},
           "mine_seconds": {b: round(v["mine_seconds"], 3) for b, v in meta["blocks"].items()},
           "wall_seconds": round(secs, 1), "peak_rss_mb": meta["peak_rss_mb"]}
    if backend == "cpu":
        res["cpu_plans"] = {b: v["cpu_plans"] for b, v in meta["blocks"].items()}
        res["disabled_optimizations"] = meta["backend"]["disabled_optimizations"]
    shutil.rmtree(os.path.join(ROOT, out))  # the hashes are what is checked; the rows take GBs
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blz", default="build/rel/blz")
    ap.add_argument("--cuda", default="build/rel-cuda/blz", help="'' to skip CUDA")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--fuzz-seeds", default="0:10000")
    ap.add_argument("--meta-seeds", default="0:2000")
    ap.add_argument("--skip-fuzz", action="store_true", help="HI-Small only")
    args = ap.parse_args()
    cuda = args.cuda if args.cuda and os.path.exists(os.path.join(ROOT, args.cuda)) else ""
    ok = True
    load = os.getloadavg()
    doc = load_report()
    checks = {}

    # AC-V2-5 on HI-Small: every profile that uses the sweep.
    profiles = sorted(glob.glob("configs/profiles/*.yaml", root_dir=ROOT)) + ["tests/profiles/tsvis.yaml"]
    hi = {}
    for path in profiles:
        name = os.path.splitext(os.path.basename(path))[0]
        hi[name] = {}
        for be in ["oracle", "cpu"]:
            hi[name][be] = mine(args.blz, path, be, SWEEP, f"features/p11_{name}_{be}", args.threads)
        o, c = hi[name]["oracle"].get("output_fnv1a"), hi[name]["cpu"].get("output_fnv1a")
        checks[f"{name}: sweep blocks, oracle = cpu"] = o is not None and o == c
        plans = hi[name]["cpu"].get("cpu_plans", {})
        # Per block, {pattern: plan}: fan_in's pattern is fan_in, or clean_fan_in in a clean profile.
        checks[f"{name}: cpu plan is the sweep"] = all(set(plans.get(b, {}).values()) == {"sweep"} for b in SWEEP)

    # AC-V2-0: v1_default, every block, every backend.
    v1 = {}
    for be in ["oracle", "cpu"] + (["cuda"] if cuda else []):
        v1[be] = mine(cuda if be == "cuda" else args.blz, "configs/profiles/v1_default.yaml", be, V1,
                      f"features/p11_v1_default_all_{be}", args.threads)
        checks[f"v1_default: {be} keeps the AC-V2-0 hashes"] = v1[be].get("output_fnv1a") == EXPECTED
    checks["v1_default: the cpu build disables nothing"] = v1["cpu"].get("disabled_optimizations") == []

    if not args.skip_fuzz:
        for name, prof in [("registry", None), ("sg_ordered_tsvis", "configs/profiles/sg_ordered_tsvis.yaml")]:
            cmd = [args.blz, "fuzz-diff", "--seeds", args.fuzz_seeds, "--report", REPORT,
                   "--name", f"ac_v2_5/fuzz_diff/{name}/cpu"] + (["--profile", prof] if prof else [])
            rc, _ = run(cmd)
            checks[f"fuzz-diff cpu {args.fuzz_seeds}: {name}"] = rc == 0
            doc = load_report()  # blz merged into it
        for name in FROZEN:
            rc, _ = run([args.blz, "metamorphic", "--seeds", args.meta_seeds, "--backend", "cpu", "--profile",
                         f"configs/profiles/{name}.yaml", "--report", REPORT, "--name", f"ac_v2_1/metamorphic/{name}/cpu"])
            checks[f"metamorphic cpu {args.meta_seeds}: {name}"] = rc == 0
            doc = load_report()

    for k, v in checks.items():
        print(f"{'ok  ' if v else 'FAIL'} {k}")
        ok &= v
    doc["ac_v2_5"] = {**doc.get("ac_v2_5", {}), "hi_small": hi}
    doc["ac_v2_0"] = {"mine_v1_default": v1}
    doc["checks"] = checks
    doc["expected_output_fnv1a"] = EXPECTED
    doc["machine"] = {"host": platform.node(), "loadavg_at_start": [round(x, 2) for x in load],
                      "cuda_binary": cuda or None}
    doc["pass"] = ok
    save_report(doc)
    print("P11 checks:", "pass" if ok else "FAIL", "->", REPORT)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

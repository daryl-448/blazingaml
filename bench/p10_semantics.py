#!/usr/bin/env python3
"""P10 exit checks (SPEC v2 §12): AC-V2-4 on the local machine, plus the AC-V2-0 recheck.

  python3 bench/p10_semantics.py [--blz build/rel/blz] [--cuda build/rel-cuda/blz] [--threads 12]

- AC-V2-4 on fuzz graphs: `blz fuzz-diff --profile` on 10,000 graphs for sg_ordered_v1w (sg_ord
  under stream visibility) and sg_ordered_tsvis (the six v1 blocks and sg_ord under timestamp
  visibility), cpu and CUDA (H = 0, 4, all); the metamorphic tests on 2,000 graphs for both
  profiles on the oracle, cpu and CUDA.
- AC-V2-4 on HI-Small: `blz mine` of both profiles with the oracle, cpu and CUDA gives the same
  output_fnv1a for every block; the tcyc rows of sg_ordered_tsvis equal v1_default's (byte compare
  of tcyc.npy); the six v1 blocks of sg_ordered_v1w keep the AC-V2-0 hashes.
- AC-V2-0: v1_default on the oracle, cpu and CUDA keeps the pinned hashes.
- The invalid-spec suite (fast test 'invalid-spec suite: …') passes.

Merges into results/p10_semantics.json; blz merges its own fuzz-diff and metamorphic reports.
Needs data/hi_small.blzg (.blzg version 2). Python standard library only. Exit status 0 if every
check passes.
"""
import argparse
import filecmp
import json
import os
import platform
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = "results/p10_semantics.json"
GRAPH = "data/hi_small.blzg"
NEW = ["sg_ordered_v1w", "sg_ordered_tsvis"]
V1 = ["fan_in", "fan_out", "deg_in", "deg_out", "sg", "tcyc"]
# SPEC v2 §2.4 AC-V2-0, from results/e4_cpu_local.json.
EXPECTED = {"fan_in": 10768036772042710147, "fan_out": 904878896487262018, "deg_in": 3321375241280117763,
            "deg_out": 3341707662053112211, "sg": 18124683238158106329, "tcyc": 7737672036140136160}


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    t0 = time.monotonic()
    p = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    sys.stdout.write(p.stdout[-4000:])
    return p.returncode, p.stdout, time.monotonic() - t0


def load_report():
    path = os.path.join(ROOT, REPORT)
    return json.load(open(path)) if os.path.exists(path) else {}


def save_report(doc):
    with open(os.path.join(ROOT, REPORT), "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blz", default="build/rel/blz")
    ap.add_argument("--cuda", default="build/rel-cuda/blz", help="'' to skip CUDA")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--fuzz-seeds", default="0:10000")
    ap.add_argument("--meta-seeds", default="0:2000")
    ap.add_argument("--skip-fuzz", action="store_true", help="HI-Small and the suite only")
    args = ap.parse_args()
    cuda = args.cuda if args.cuda and os.path.exists(os.path.join(ROOT, args.cuda)) else ""
    backends = ["oracle", "cpu"] + (["cuda"] if cuda else [])
    ok = True
    load = os.getloadavg()

    if not args.skip_fuzz:
        for prof in NEW:
            path = f"configs/profiles/{prof}.yaml"
            for be in ["cpu"] + (["cuda"] if cuda else []):
                rc, _, _ = run([cuda if be == "cuda" else args.blz, "fuzz-diff", "--seeds", args.fuzz_seeds,
                                "--backend", be, "--profile", path, "--report", REPORT,
                                "--name", f"ac_v2_4/fuzz_diff/{prof}/{be}"])
                ok &= rc == 0
            for be in backends:
                rc, _, _ = run([cuda if be == "cuda" else args.blz, "metamorphic", "--seeds", args.meta_seeds,
                                "--backend", be, "--profile", path, "--report", REPORT,
                                "--name", f"ac_v2_4/metamorphic/{prof}/{be}"])
                ok &= rc == 0

    rc, _, _ = run(["ctest", "--test-dir", os.path.dirname(args.blz), "-R", "invalid-spec suite", "--output-on-failure"])
    suite_ok = rc == 0
    ok &= suite_ok

    doc = load_report()
    hi = {}
    outs = {}
    for prof in ["v1_default"] + NEW:
        hi[prof] = {}
        for be in backends:
            out = f"features/p10_{prof}_{be}"
            cmd = [cuda if be == "cuda" else args.blz, "mine", "--graph", GRAPH, "--profile",
                   f"configs/profiles/{prof}.yaml", "--backend", be, "--out", out, "--threads", str(args.threads)]
            rc, _, secs = run(cmd)
            if rc != 0:
                hi[prof][be] = {"exit": rc}
                ok = False
                continue
            meta = json.load(open(os.path.join(ROOT, out, "features.json")))
            outs[(prof, be)] = out
            hi[prof][be] = {"profile_id": meta["run"]["profile_id"],
                            "output_fnv1a": {b: v["output_fnv1a"] for b, v in meta["blocks"].items()},
                            "mine_seconds": {b: round(v["mine_seconds"], 3) for b, v in meta["blocks"].items()},
                            "wall_seconds": round(secs, 1), "peak_rss_mb": meta["peak_rss_mb"]}

    checks = {}
    for prof in ["v1_default"] + NEW:
        got = [hi[prof][be].get("output_fnv1a") for be in backends]
        checks[f"{prof}: oracle = cpu = cuda"] = all(g is not None and g == got[0] for g in got)
    for prof in ["v1_default", "sg_ordered_v1w"]:
        checks[f"{prof}: the six v1 blocks keep the AC-V2-0 hashes"] = all(
            {b: hi[prof][be].get("output_fnv1a", {}).get(b) for b in V1} == EXPECTED for be in backends)
    checks["sg_ordered_tsvis: tcyc keeps the AC-V2-0 hash"] = all(
        hi["sg_ordered_tsvis"][be].get("output_fnv1a", {}).get("tcyc") == EXPECTED["tcyc"] for be in backends)
    a, b = outs.get(("sg_ordered_tsvis", "cpu")), outs.get(("v1_default", "cpu"))
    checks["tcyc rows equal under both visibilities (tcyc.npy, cpu)"] = bool(a and b) and filecmp.cmp(
        os.path.join(ROOT, a, "tcyc.npy"), os.path.join(ROOT, b, "tcyc.npy"), shallow=False)
    checks["invalid-spec suite"] = suite_ok
    for k, v in checks.items():
        print(f"{'ok  ' if v else 'FAIL'} {k}")
        ok &= v

    doc.setdefault("ac_v2_4", {})["hi_small"] = hi
    doc["ac_v2_0"] = {"mine_v1_default": hi["v1_default"]}
    doc["checks"] = checks
    doc["expected_output_fnv1a"] = EXPECTED
    doc["machine"] = {"host": platform.node(), "loadavg_at_start": [round(x, 2) for x in load],
                      "cuda_binary": cuda or None}
    doc["pass"] = ok
    save_report(doc)
    print("P10 semantics:", "pass" if ok else "FAIL", "->", REPORT)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

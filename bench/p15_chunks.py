#!/usr/bin/env python3
"""Trigger-range mining equals whole-graph mining on HI-Small, by row hash (SPEC v3 §7 P15, H7).

    bench/p15_chunks.py [--blz build/rel-cuda/blz] [--graph data/hi_small.blzg]
                        [--out results/p15_chunks_local.json] [--quick]

Runs `blz mine` on HI-Small for clean_default and clean_all on cpu, cuda, cuda --fuse --pack and
hybrid, at chunk sizes {the whole stream, the default, 1,000,003, 65,537}, and checks every block's
output_fnv1a (vstats and lcyc included) against the same backend's whole-stream run, and
clean_default's and clean_all's against results/p13_a30.json (hi_small/<profile>/<backend>). Then
v1_parity on cpu at 65,537 against AC-V3-0's six hashes (SPEC §2.3), and the oracle on
clean_default at 1,000,003 against p13's. Prints PASS or FAIL per block and run, writes the table
to --out, and exits 1 on any FAIL. Mined rows go to features/p15_chunks/ and are deleted after each
run (features.json is kept). --quick runs clean_default on cpu only.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AC_V3_0_PARITY = {"fan_in": 231040277472725923, "fan_out": 6330076044337963426, "deg_in": 6366489652548864147,
                  "deg_out": 10227089942834718195, "sg": 2140027548842065401, "tcyc": 3161311333401399936}
BACKENDS = {"cpu": ["--backend", "cpu"], "cuda": ["--backend", "cuda"],
            "cuda_fuse_pack": ["--backend", "cuda", "--fuse", "--pack"], "hybrid": ["--backend", "hybrid"]}


def mine(blz, graph, profile, args, chunk, out):
    cmd = [blz, "mine", "--graph", graph, "--profile", f"configs/profiles/{profile}.yaml", *args, "--out", out]
    if chunk is not None:
        cmd += ["--chunk-rows", str(chunk)]
    shutil.rmtree(os.path.join(ROOT, out), ignore_errors=True)  # a fresh manifest: no block inherits an old hash
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"{' '.join(cmd)} failed:\n{r.stderr}")
    meta = json.load(open(os.path.join(ROOT, out, "features.json")))
    missing = [b for b, v in meta["blocks"].items() if not os.path.exists(os.path.join(ROOT, out, v["file"]))]
    if missing:
        sys.exit(f"{' '.join(cmd)}: no file for blocks {missing}")
    for f in os.listdir(os.path.join(ROOT, out)):
        if f.endswith(".npy"):
            os.remove(os.path.join(ROOT, out, f))
    return meta, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blz", default="build/rel-cuda/blz")
    ap.add_argument("--graph", default="data/hi_small.blzg")
    ap.add_argument("--out", default="results/p15_chunks_local.json")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    p13 = json.load(open(os.path.join(ROOT, "results/p13_a30.json")))["hi_small"]
    rows, fails = [], 0

    def check(run, block, got, want, against):
        nonlocal fails
        ok = got == want
        fails += not ok
        rows.append({"run": run, "block": block, "fnv1a": got, "against": against, "pass": ok})
        print(f"{'PASS' if ok else 'FAIL'}  {run:44s} {block:15s} {got:>20d}  vs {against}"
              + ("" if ok else f" (want {want})"), flush=True)

    profiles = ["clean_default"] if a.quick else ["clean_default", "clean_all"]
    backends = ["cpu"] if a.quick else list(BACKENDS)
    runs = {}
    # The whole stream as one chunk needs m: read it from a default run's manifest first.
    for prof in profiles:
        for be in backends:
            base, secs = mine(a.blz, a.graph, prof, BACKENDS[be], None, f"features/p15_chunks/{prof}_{be}_default")
            m = base["triggers"]["rows"]
            got = {"default": (base, secs)}
            for chunk in [m, 1_000_003, 65_537]:
                got[chunk] = mine(a.blz, a.graph, prof, BACKENDS[be], chunk, f"features/p15_chunks/{prof}_{be}_{chunk}")
            whole = {b: v["output_fnv1a"] for b, v in got[m][0]["blocks"].items()}
            for chunk, (meta, secs) in got.items():
                label = "whole" if chunk == m else chunk
                run = f"{prof} {be} chunk={label} ({meta['chunk_rows']})"
                runs[run] = {"seconds": secs, "chunk_rows": meta["chunk_rows"],
                             "chunks": {b: v["chunks"] for b, v in meta["blocks"].items()},
                             "output_fnv1a": {b: v["output_fnv1a"] for b, v in meta["blocks"].items()}}
                if set(meta["blocks"]) != set(whole):
                    check(run, "(blocks)", 0, 1, "the whole run's blocks")
                for b, v in meta["blocks"].items():
                    if chunk != m:
                        check(run, b, v["output_fnv1a"], whole.get(b), "whole")
                    rec = p13.get(prof, {}).get("cuda" if be == "cuda_fuse_pack" else be, {}).get("output_fnv1a", {})
                    if b in rec:
                        check(run, b, v["output_fnv1a"], rec[b], f"p13_a30 {prof}")
    if not a.quick:
        meta, secs = mine(a.blz, a.graph, "v1_parity", BACKENDS["cpu"], 65_537, "features/p15_chunks/v1_parity_cpu_65537")
        run = "v1_parity cpu chunk=65537"
        runs[run] = {"seconds": secs, "output_fnv1a": {b: v["output_fnv1a"] for b, v in meta["blocks"].items()}}
        for b, want in AC_V3_0_PARITY.items():
            check(run, b, meta["blocks"][b]["output_fnv1a"], want, "AC-V3-0 v1_parity")
        meta, secs = mine(a.blz, a.graph, "clean_default", ["--backend", "oracle"], 1_000_003,
                          "features/p15_chunks/clean_default_oracle_1000003")
        run = "clean_default oracle chunk=1000003"
        runs[run] = {"seconds": secs, "output_fnv1a": {b: v["output_fnv1a"] for b, v in meta["blocks"].items()}}
        for b, v in meta["blocks"].items():
            check(run, b, v["output_fnv1a"], p13["clean_default"]["oracle"]["output_fnv1a"][b], "p13_a30 clean_default")
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    doc = {"git": git, "graph": a.graph, "checks": len(rows), "fails": fails, "runs": runs, "rows": rows}
    with open(os.path.join(ROOT, a.out), "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    print(f"{len(rows) - fails} of {len(rows)} checks pass; wrote {a.out}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

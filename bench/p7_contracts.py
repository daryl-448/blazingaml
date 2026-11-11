#!/usr/bin/env python3
"""P7 exit checks (SPEC v2 §12): AC-V2-0 and AC-V2-1 on HI-Small and on fuzz graphs.

  python3 bench/p7_contracts.py [--cuda build/rel-cuda/blz] [--threads 12]

- AC-V2-0: `blz mine --profile configs/profiles/v1_default.yaml` with the oracle, the CPU backend
  and (when --cuda names a binary) the CUDA backend; every block's output_fnv1a must equal the
  value recorded in results/e4_cpu_local.json.
- AC-V2-1: every features.json written here carries the manifest of SPEC v2 §3.2 with the
  profile id; the metamorphic tests (SPEC v2 §3.4) pass on 2,000 fuzz graphs per frozen profile,
  for the oracle and the CPU backend (and CUDA when built); `blz fuzz-diff --profile` (SPEC v2
  §10) passes on 2,000 graphs per frozen profile.
- `--triggers sample:1:7` on HI-Small: the oracle's and the CPU backend's sampled rows equal the
  full runs' rows at the sampled eids; CUDA rejects a sample.

Merges into results/p7_contracts.json. Needs data/hi_small.blzg in .blzg version 2. Python
standard library only. Exit status: 0 if every check passes.
"""
import argparse
import ast
import json
import mmap
import os
import platform
import struct
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = "results/p7_contracts.json"
GRAPH = "data/hi_small.blzg"
PROFILES = ["v1_default", "v1_parity"]
BLOCKS = ["fan_in", "fan_out", "deg_in", "deg_out", "sg", "tcyc"]
# SPEC v2 §2.4 AC-V2-0, from results/e4_cpu_local.json.
EXPECTED = {"fan_in": 10768036772042710147, "fan_out": 904878896487262018, "deg_in": 3321375241280117763,
            "deg_out": 3341707662053112211, "sg": 18124683238158106329, "tcyc": 7737672036140136160}
MANIFEST_KEYS = ["manifest_version", "run", "profile", "patterns", "backend", "triggers", "blzg_sha256",
                 "csv_sha256", "blocks", "peak_rss_mb"]


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    t0 = time.monotonic()
    p = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    sys.stdout.write(p.stdout)
    return p.returncode, p.stdout, time.monotonic() - t0


def load_report():
    path = os.path.join(ROOT, REPORT)
    return json.load(open(path)) if os.path.exists(path) else {}


def save_report(doc):
    with open(os.path.join(ROOT, REPORT), "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")


class Npy:
    """A uint32 (rows, cols) .npy file, rows read on demand."""

    def __init__(self, path):
        self.f = open(path, "rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        assert self.mm[:6] == b"\x93NUMPY", path
        major = self.mm[6]
        hlen, start = (struct.unpack("<H", self.mm[8:10])[0], 10) if major == 1 else (struct.unpack("<I", self.mm[8:12])[0], 12)
        header = ast.literal_eval(self.mm[start:start + hlen].decode("latin1"))
        assert header["descr"] == "<u4" and not header["fortran_order"], path
        self.rows, self.cols = header["shape"]
        self.data = start + hlen

    def row(self, r):
        o = self.data + 4 * self.cols * r
        return self.mm[o:o + 4 * self.cols]


def mine(blz, backend, out, threads, triggers="all"):
    cmd = [blz, "mine", "--graph", GRAPH, "--profile", "configs/profiles/v1_default.yaml", "--backend", backend,
           "--triggers", triggers, "--out", out]
    if threads:
        cmd += ["--threads", str(threads)]
    rc, _, secs = run(cmd)
    meta = json.load(open(os.path.join(ROOT, out, "features.json"))) if rc == 0 else None
    return rc, meta, secs


def manifest_check(meta):
    missing = [k for k in MANIFEST_KEYS if k not in meta]
    if "run" in meta and "profile_id" not in meta["run"]:
        missing.append("run.profile_id")
    if "profile" in meta and "id" not in meta["profile"]:
        missing.append("profile.id")
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blz", default="build/rel/blz")
    ap.add_argument("--cuda", default="build/rel-cuda/blz", help="'' to skip CUDA")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--seeds", default="0:2000")
    args = ap.parse_args()
    cuda = args.cuda if args.cuda and os.path.exists(os.path.join(ROOT, args.cuda)) else ""
    ok = True
    load = os.getloadavg()

    # Metamorphic tests and fuzz-diff per profile: blz merges them into the report itself.
    for prof in PROFILES:
        path = f"configs/profiles/{prof}.yaml"
        for be in ["oracle", "cpu"] + (["cuda"] if cuda else []):
            blz = cuda if be == "cuda" else args.blz
            rc, _, _ = run([blz, "metamorphic", "--seeds", args.seeds, "--backend", be, "--profile", path,
                            "--report", REPORT, "--name", f"ac_v2_1/metamorphic/{prof}/{be}"])
            ok &= rc == 0
        for be in ["cpu"] + (["cuda"] if cuda else []):
            blz = cuda if be == "cuda" else args.blz
            rc, _, _ = run([blz, "fuzz-diff", "--seeds", args.seeds, "--backend", be, "--profile", path,
                            "--report", REPORT, "--name", f"ac_v2_1/fuzz_diff/{prof}/{be}"])
            ok &= rc == 0

    doc = load_report()
    ac0 = {}
    manifests = {}
    full = {}
    for be in ["oracle", "cpu"] + (["cuda"] if cuda else []):
        blz = cuda if be == "cuda" else args.blz
        out = f"features/p7_{be}_default"
        rc, meta, secs = mine(blz, be, out, args.threads)
        if rc != 0:
            ac0[be] = {"exit": rc}
            ok = False
            continue
        full[be] = out
        got = {b: meta["blocks"][b]["output_fnv1a"] for b in BLOCKS}
        match = got == EXPECTED
        ok &= match
        ac0[be] = {"profile": meta["profile"]["name"], "profile_id": meta["run"]["profile_id"],
                   "output_fnv1a": got, "match": match,
                   "mine_seconds": {b: round(meta["blocks"][b]["mine_seconds"], 3) for b in BLOCKS},
                   "wall_seconds": round(secs, 1), "peak_rss_mb": meta["peak_rss_mb"],
                   "backend": meta["backend"]}
        missing = manifest_check(meta)
        manifests[out] = {"missing": missing, "profile_id": meta["run"].get("profile_id")}
        ok &= not missing
        print(f"AC-V2-0 {be}: {'match' if match else 'MISMATCH'}; manifest {'complete' if not missing else missing}")

    # --triggers sample on HI-Small: sampled rows == the full run's rows at the sampled eids.
    sample = {}
    for be in ["oracle", "cpu"]:
        if be not in full:
            continue
        out = f"features/p7_{be}_sample"
        rc, meta, secs = mine(args.blz, be, out, args.threads, "sample:1:7")
        if rc != 0:
            sample[be] = {"exit": rc}
            ok = False
            continue
        trig = Npy(os.path.join(ROOT, out, "triggers.npy"))
        eids = [struct.unpack("<I", trig.row(r))[0] for r in range(trig.rows)]
        bad = {}
        for b in BLOCKS:
            s, f = Npy(os.path.join(ROOT, out, f"{b}.npy")), Npy(os.path.join(ROOT, full[be], f"{b}.npy"))
            bad[b] = sum(s.row(r) != f.row(e) for r, e in enumerate(eids)) if s.rows == len(eids) else -1
        missing = manifest_check(meta)
        manifests[out] = {"missing": missing, "profile_id": meta["run"].get("profile_id")}
        good = all(v == 0 for v in bad.values()) and not missing
        ok &= good
        sample[be] = {"triggers": meta["triggers"], "rows": len(eids), "mismatched_rows": bad, "match": good,
                      "mine_seconds": {b: round(meta["blocks"][b]["mine_seconds"], 3) for b in BLOCKS},
                      "extrapolated_mine_seconds": {b: round(meta["blocks"][b]["extrapolated_mine_seconds"], 1)
                                                    for b in BLOCKS}}
        print(f"sample {be}: {len(eids)} rows, mismatches {bad}")
    if cuda:
        rc, out_text, _ = run([cuda, "mine", "--graph", GRAPH, "--profile", "configs/profiles/v1_default.yaml",
                               "--backend", "cuda", "--triggers", "sample:1:7", "--out", "features/p7_cuda_sample"])
        rejected = rc != 0 and "--triggers sample:" in out_text
        sample["cuda"] = {"rejected": rejected, "exit": rc}
        ok &= rejected

    doc.setdefault("ac_v2_0", {})["mine_v1_default"] = ac0
    doc.setdefault("ac_v2_1", {})["manifests"] = manifests
    doc["sample_hi_small"] = sample
    doc["expected_output_fnv1a"] = EXPECTED
    doc["machine"] = {"host": platform.node(), "loadavg_at_start": [round(x, 2) for x in load],
                      "cuda_binary": cuda or None}
    doc["pass"] = ok
    save_report(doc)
    print("P7 contracts:", "pass" if ok else "FAIL", "->", REPORT)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

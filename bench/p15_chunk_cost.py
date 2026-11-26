#!/usr/bin/env python3
"""What chunked delivery costs M (SPEC v3 §5.1, H7): `blz measure` at several chunk sizes.

    bench/p15_chunk_cost.py [--blz build/rel-cuda/blz] [--graph data/hi_small.blzg]
        [--profile configs/profiles/clean_default.yaml] [--backends cpu,cuda] [--threads 12]
        [--chunks whole,4194304,1048576,262144] [--rounds 2] [--reps 5]
        [--out results/p15_chunk_cost_local.json]

Logs `ps --sort=-pcpu | head` and `free -m`, then waits (up to 30 min) for an idle window: CPU busy
below 10 % of the machine over 3 s. Each round runs `blz measure --reps <reps>` once per backend and
chunk size, the sizes in order and then reversed on the next round (A B C, C B A), so drift favours
none. `whole` is one chunk (--chunk-rows 2^64 - 1). Per backend and size the runs' reps are pooled:
median, min and max; the overhead is the median less one chunk's median, and a size passes when that
is no more than one chunk's min-to-max spread. The measure results and the summary go to --out.
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cpu_busy(seconds=3.0):
    def read():
        with open("/proc/stat") as f:
            v = list(map(int, f.readline().split()[1:]))
        return sum(v), v[3] + v[4]  # total, idle + iowait
    t0, i0 = read()
    time.sleep(seconds)
    t1, i1 = read()
    return 1.0 - (i1 - i0) / max(1, t1 - t0)


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blz", default="build/rel-cuda/blz")
    ap.add_argument("--graph", default="data/hi_small.blzg")
    ap.add_argument("--profile", default="configs/profiles/clean_default.yaml")
    ap.add_argument("--backends", default="cpu,cuda")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--chunks", default="whole,4194304,1048576,262144")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default="results/p15_chunk_cost_local.json")
    a = ap.parse_args()
    out = os.path.join(ROOT, a.out)
    log = {"ps_before": sh("ps --sort=-pcpu -eo pid,pcpu,rss,etime,comm | head -12"), "free_m_before": sh("free -m")}
    print(log["ps_before"], log["free_m_before"], sep="\n", flush=True)
    for waited in range(600):
        busy = cpu_busy()
        if busy < 0.10:
            break
        print(f"busy {busy:.0%}, waiting for an idle window", flush=True)
    log["cpu_busy_at_start"] = busy
    if os.path.exists(out):
        os.remove(out)
    chunks = a.chunks.split(",")
    for rnd in range(a.rounds):
        order = chunks if rnd % 2 == 0 else chunks[::-1]
        for be in a.backends.split(","):
            for c in order:
                name = f"{be}/{c}/round{rnd}"
                cmd = [a.blz, "measure", "--graph", a.graph, "--profile", a.profile, "--backend", be,
                       "--reps", str(a.reps), "--out", a.out, "--name", name]
                if be in ("cpu", "hybrid"):
                    cmd += ["--threads", str(a.threads)]
                cmd += ["--chunk-rows", str(2**64 - 1) if c == "whole" else c]
                r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
                print(r.stderr.splitlines()[-2] if r.stderr else "", f"[{name}]", flush=True)
                if r.returncode != 0:
                    sys.exit(f"{' '.join(cmd)} failed ({r.returncode}):\n{r.stderr}")
    doc = json.load(open(out))
    summary = {}
    for be in a.backends.split(","):
        pooled = {c: [run["seconds"] for rnd in doc[be][c].values() for run in rnd["runs"] if "seconds" in run]
                  for c in chunks}
        stats = {}
        for c, v in pooled.items():
            s = sorted(v)
            med = s[len(s) // 2] if len(s) % 2 else (s[len(s) // 2 - 1] + s[len(s) // 2]) / 2
            stats[c] = {"median": med, "min": s[0], "max": s[-1], "n": len(s)}
        one = stats["whole"]
        spread = one["max"] - one["min"]
        for c, st in stats.items():
            st["overhead"] = st["median"] - one["median"]
            st["passes"] = st["overhead"] <= spread
            print(f"{be:5s} chunk={c:>8s}  median {st['median']:.3f} s  [{st['min']:.3f}, {st['max']:.3f}]  n={st['n']}"
                  f"  overhead {st['overhead'] * 1e3:+.1f} ms  (one chunk's spread {spread * 1e3:.1f} ms)"
                  f"  {'pass' if st['passes'] else 'FAIL'}", flush=True)
        summary[be] = {"one_chunk_spread": spread, "sizes": stats}
    log["ps_after"] = sh("ps --sort=-pcpu -eo pid,pcpu,rss,etime,comm | head -12")
    log["free_m_after"] = sh("free -m")
    doc["summary"] = summary
    doc["log"] = log
    doc["git"] = sh(f"git -C {ROOT} rev-parse --short HEAD").strip()
    with open(out, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

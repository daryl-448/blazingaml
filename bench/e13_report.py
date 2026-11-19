#!/usr/bin/env python3
"""E13 report (SPEC v2 §8 S6): results/e13_hybrid_<host>.json -> results/e13_hybrid_<host>.md.

Reads what bench/e13.sh wrote ("runs"/<profile>/H<h>: `blz bench --backend hybrid` for block tcyc)
and reports, per profile:
- GPU-only T (graph upload + kernels + D2H, E5's form) at each GPU threshold H;
- CPU-only T (12 threads, sorted schedule) without and with reachability pruning;
- the hybrid's T (upload + the call's wall clock: the GPU's split, both sides at once, the merge)
  per CPU threshold, at every H, without and with pruning; T_formula = upload + max(GPU side,
  split + CPU side) + merge is recorded next to it;
- the best of each, and the speed-ups.

Checks: every run of a profile, at every H, gave the same rows (blz bench refuses otherwise), and
their hash equals the recorded one (v1_default: AC-V2-0; dg2_wc and dg2_wb: E12's on the A30).

The default CPU threshold of `blz mine --backend hybrid` (pre-registered rule): the threshold with
the smallest sum, over the three profiles, of the median T with pruning on at the default GPU
threshold (H = 32). The report writes "summary" and "checks" into the JSON.

Usage: bench/e13_report.py results/e13_hybrid_local.json
"""

import argparse
import json
from pathlib import Path

PROFILES = [("v1_default", "24 h"), ("dg2_wc", "89 h, the frozen recipe's W-C"), ("dg2_wb", "20 days")]
WANT = {  # tcyc output_fnv1a: v1_default from AC-V2-0, the others from results/e12_gpu_a30.json
    "v1_default": 7737672036140136160,
    "dg2_wc": 15263752808203538370,
    "dg2_wb": 11834804438481637798,
}
DEFAULT_H = "32"


def ms(x):
    return f"{x * 1e3:.0f} ms" if x < 1 else f"{x:.2f} s"


def h_key(h):
    return int(h[1:]) if h[1:].isdigit() else 1 << 40


def thr_key(t):
    return 1 << 40 if t == "all" else int(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("e13")
    args = ap.parse_args()
    path = Path(args.e13)
    host = path.stem.removeprefix("e13_hybrid_")
    doc = json.loads(path.read_text())
    runs = doc["runs"]
    checks, summary = {}, {}
    md = [f"# E13: tcyc hybrid CPU/GPU and reachability pruning ({host})", ""]

    first = next(iter(next(iter(runs.values())).values()))
    env, cuda = first["env"], first["cuda"]
    cpu_model = next((l.split(":", 1)[1].strip() for l in env.get("lscpu", "").splitlines() if l.startswith("Model name")), "?")
    md += [
        f"- GPU: {cuda.get('device', '?')} (compute capability {cuda.get('compute_capability', '?')}, "
        f"{cuda.get('sm_count', '?')} SMs); CUDA build: {cuda.get('build', '?')}",
        f"- CPU: {cpu_model}, {first['threads']} OpenMP threads, schedule {first['schedule']}; "
        f"build {env.get('build', {}).get('type', '?')}, `{env.get('build', {}).get('cxx_flags', '?')}`",
        f"- Git: {str(env.get('git', {}).get('sha', '?'))[:12]}{' (dirty)' if env.get('git', {}).get('dirty') else ''}",
        f"- Each cell: median of {first['reps']} timed runs after 1 warm-up, every trigger of HI-Small, block tcyc "
        "(K_MAX = 10).",
        "- GPU-only T = graph upload + kernels + D2H at the GPU light/heavy threshold H (E5's form). CPU-only T = the "
        "block's mining time. Hybrid T = graph upload + the call's wall clock: the GPU finds the CPU's triggers "
        "(work estimate >= the CPU threshold H_cpu) and copies them to the host, then the CPU mines them while the GPU "
        "mines the rest, then the CPU rows are merged. Every transfer is inside T.",
        "- Pruning is the CPU's reachability pruning (D-43); the GPU never prunes.",
        "",
    ]

    for prof, label in PROFILES:
        if prof not in runs:
            continue
        hs = sorted(runs[prof], key=h_key)
        blk = {h: runs[prof][h]["blocks"]["tcyc"] for h in hs}
        any_blk = blk[hs[0]]
        hashes = {blk[h]["output_fnv1a"] for h in hs}
        ok = hashes == {WANT[prof]}
        checks[prof] = {"rows_equal_every_run": True, "hash": sorted(hashes), "hash_expected": WANT[prof], "ok": ok}
        thrs = sorted(any_blk["hybrid"]["prune_on"], key=thr_key)

        gpu = {h: blk[h]["gpu"]["total"]["median"] for h in hs}
        cpu = {pk: min(blk[h]["cpu"][pk]["total"]["median"] for h in hs) for pk in ("prune_off", "prune_on")}
        hyb = {
            (pk, h, t): blk[h]["hybrid"][pk][t]["total"]["median"]
            for pk in ("prune_off", "prune_on")
            for h in hs
            for t in thrs
        }
        best_gpu_h = min(gpu, key=gpu.get)
        best = {pk: min(((k, v) for k, v in hyb.items() if k[0] == pk), key=lambda kv: kv[1]) for pk in cpu}
        summary[prof] = {
            "window": any_blk["window"],
            "gpu_only": {"best_H": best_gpu_h[1:], "T": gpu[best_gpu_h], "by_H": {h[1:]: v for h, v in gpu.items()}},
            "cpu_only": cpu,
            "hybrid_best": {
                pk: {"H": k[1][1:], "cpu_threshold": k[2], "T": v,
                     "cpu_triggers": blk[k[1]]["hybrid"][pk][k[2]]["cpu_triggers"]}
                for pk, (k, v) in best.items()
            },
            "speedup_vs_gpu_only": gpu[best_gpu_h] / best["prune_on"][1],
            "speedup_vs_cpu_only": min(cpu.values()) / best["prune_on"][1],
        }

        md += [f"## {prof} (window {any_blk['window']:,} s, {label})", ""]
        md += ["GPU-only and CPU-only:", "", "| | " + " | ".join(f"H={h[1:]}" for h in hs) + " |",
               "|---|" + "---:|" * len(hs)]
        md.append("| GPU-only T | " + " | ".join(("**" + ms(gpu[h]) + "**") if h == best_gpu_h else ms(gpu[h]) for h in hs) + " |")
        md.append("| GPU heavy triggers | " + " | ".join(f"{blk[h]['gpu']['heavy_triggers']:,}" for h in hs) + " |")
        md += ["", f"CPU-only T: {ms(cpu['prune_off'])} without pruning, {ms(cpu['prune_on'])} with it "
               f"({cpu['prune_off'] / cpu['prune_on']:.1f}×).", ""]
        md += ["Hybrid T by CPU threshold (rows) and GPU threshold, without / with pruning:", "",
               "| H_cpu | CPU triggers | " + " | ".join(f"H={h[1:]} off | H={h[1:]} on" for h in hs) + " |",
               "|---:|---:|" + "---:|---:|" * len(hs)]
        best_on = best["prune_on"][0]
        for t in thrs:
            cells = []
            for h in hs:
                for pk in ("prune_off", "prune_on"):
                    v = ms(hyb[(pk, h, t)])
                    cells.append(f"**{v}**" if (pk, h, t) == best_on else v)
            ntrig = blk[hs[0]]["hybrid"]["prune_on"][t]["cpu_triggers"]
            md.append(f"| {t} | {ntrig:,} | " + " | ".join(cells) + " |")
        bo = blk[best_on[1]]["hybrid"]["prune_on"][best_on[2]]
        md += ["", f"Best hybrid (pruning on): H = {best_on[1][1:]}, H_cpu = {best_on[2]}: T = {ms(best['prune_on'][1])} "
               f"(T_formula {ms(bo['total_formula']['median'])}): upload {ms(blk[best_on[1]]['gpu']['total']['median'] - blk[best_on[1]]['gpu']['kernel']['median'] - blk[best_on[1]]['gpu']['d2h']['median'])}, "
               f"split {ms(bo['split']['median'])} (on the GPU: {ms(bo['gpu_split']['median'])}), "
               f"CPU side {ms(bo['cpu']['median'])} on {bo['cpu_triggers']:,} triggers, "
               f"GPU side {ms(bo['gpu']['median'])} (kernels {ms(bo['kernel']['median'])}, D2H {ms(bo['d2h']['median'])}), "
               f"merge {ms(bo['merge']['median'])}.",
               f"Without pruning the best hybrid is {ms(best['prune_off'][1])} (H = {best['prune_off'][0][1][1:]}, "
               f"H_cpu = {best['prune_off'][0][2]}).", "",
               f"Rows: every run equal; hash {'= ' if ok else '!= '}the recorded one ({WANT[prof]}).", ""]

    # The default CPU threshold (pre-registered rule above).
    sums = {}
    for prof, _ in PROFILES:
        h = "H" + DEFAULT_H
        if prof not in runs or h not in runs[prof]:
            continue
        for t, r in runs[prof][h]["blocks"]["tcyc"]["hybrid"]["prune_on"].items():
            sums[t] = sums.get(t, 0) + r["total"]["median"]
    default_t = min(sums, key=sums.get) if sums else None
    summary["default_cpu_threshold"] = {"rule": f"smallest summed median T, pruning on, H = {DEFAULT_H}",
                                        "sums": sums, "chosen": default_t}

    md += ["## Summary", "",
           "| profile | GPU-only (best H) | CPU-only, pruned | hybrid, pruned (H, H_cpu) | vs GPU-only | vs CPU-only | hybrid, not pruned |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for prof, _ in PROFILES:
        if prof not in summary:
            continue
        s = summary[prof]
        b, bo = s["hybrid_best"]["prune_on"], s["hybrid_best"]["prune_off"]
        md.append(f"| {prof} | {ms(s['gpu_only']['T'])} ({s['gpu_only']['best_H']}) | {ms(s['cpu_only']['prune_on'])} | "
                  f"{ms(b['T'])} ({b['H']}, {b['cpu_threshold']}) | {s['speedup_vs_gpu_only']:.1f}× | "
                  f"{s['speedup_vs_cpu_only']:.1f}× | {ms(bo['T'])} |")
    if default_t:
        md += ["", f"Default CPU threshold of `blz mine --backend hybrid`: **{default_t}** (the smallest summed median T "
               f"over the three profiles with pruning on at H = {DEFAULT_H}: " +
               ", ".join(f"{t}: {ms(v)}" for t, v in sorted(sums.items(), key=lambda kv: thr_key(kv[0]))) + ")."]
    md.append("")

    doc["summary"] = summary
    doc["checks"] = checks
    path.write_text(json.dumps(doc, indent=1) + "\n")
    path.with_suffix(".md").write_text("\n".join(md))
    bad = [p for p, c in checks.items() if not c["ok"]]
    print(f"wrote {path.with_suffix('.md')}; checks {'FAILED: ' + ', '.join(bad) if bad else 'ok'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

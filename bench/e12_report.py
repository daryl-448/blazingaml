#!/usr/bin/env python3
"""E12 report (SPEC v2 §7.3, §12 P11): results/e12_gpu_<host>.json -> results/e12_gpu_<host>.md.

Reads what bench/a30_remote.sh wrote on one GPU instance:
- results/e12_gpu_<host>.json: "cuda"/<profile> (blz bench --backend cuda, H swept as in v1's E5)
  and "cpu"/<profile> (blz bench --backend cpu at the instance's thread count);
- results/p11_<host>.json: AC-V2-4 on that GPU, fuzz-diff summaries ("ac_v2_4"/"fuzz_diff") and
  the HI-Small CUDA hashes ("ac_v2_4"/"hi_small");
and, from `local`, the oracle hashes of P10 (results/p10_semantics.json) and v1's E5 on the same
GPU model (results/e5_gpu_<host>.json).

It reports AC-V2-4 (fuzz rows, CUDA hashes = the oracle's), per profile the kernel time by H and
the tuned H (smallest total, as E5), v1_default against v1's E5, and GPU vs CPU on the instance
with the cuda = cpu hash check. It writes "tuned", "checks" and "gpu_vs_cpu" into the E12 JSON.

Usage: bench/e12_report.py results/e12_gpu_a30.json [--p11 ...] [--oracle ...] [--e5 ...]
"""

import argparse
import json
from pathlib import Path

ORDER = ["fan_in", "fan_out", "deg_in", "deg_out", "sg", "sg_ord", "tcyc"]


def ms(x):
    return f"{x * 1e3:.1f}"


def secs(x):
    return f"{x * 1e3:.1f} ms" if x < 1 else f"{x:.2f} s"


def load(path):
    return json.loads(path.read_text()) if path and path.exists() else {}


def order(blocks):
    return sorted(blocks, key=lambda b: ORDER.index(b) if b in ORDER else len(ORDER))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("e12")
    ap.add_argument("--p11", help="default: results/p11_<host>.json")
    ap.add_argument("--oracle", default="results/p10_semantics.json", help="P10's oracle hashes (local)")
    ap.add_argument("--e5", help="v1's E5 on this GPU (default: results/e5_gpu_<host>.json)")
    args = ap.parse_args()
    path = Path(args.e12)
    host = path.stem.removeprefix("e12_gpu_")
    doc = json.loads(path.read_text())
    p11 = load(Path(args.p11) if args.p11 else path.with_name(f"p11_{host}.json"))
    oracle = load(Path(args.oracle)).get("ac_v2_4", {}).get("hi_small", {})
    e5_doc = load(Path(args.e5) if args.e5 else path.with_name(f"e5_gpu_{host}.json"))
    e5 = e5_doc.get("cuda")
    cuda, cpu = doc.get("cuda", {}), doc.get("cpu", {})
    if not cuda:
        raise SystemExit(f"{path}: no cuda results")
    checks = {}

    first = next(iter(cuda.values()))
    env, dev = first["env"], first["cuda"]
    git = env.get("git") or {}
    model = next((l.split(":", 1)[1].strip() for l in env["lscpu"].splitlines() if l.startswith("Model name")), "?")
    out = [f"# E12: GPU mining time by profile, and GPU vs CPU ({host})", ""]
    out += [f"- GPU: {dev['device']} (sm_{dev['compute_capability'].replace('.', '')}, {dev['sm_count']} SMs, "
            f"{dev['memory_mb']['total']} MB); driver {dev['driver_version']}, runtime {dev['runtime_version']}",
            f"- Host CPU: {model}",
            f"- Build: {env['build']['type']}, `{env['build']['cxx_flags']}`, {env['compiler']}; CUDA: {dev['build']}",
            f"- Git: {git.get('sha', '?')[:12]}{' (dirty)' if git.get('dirty') else ''}",
            f"- Each cell: median of {first['reps']} timed runs after {first['warmup']} warm-up, over every "
            "trigger of HI-Small. Every run at every H produced the same rows (SPEC §8).", ""]

    # AC-V2-4 on this GPU.
    fuzz = p11.get("ac_v2_4", {}).get("fuzz_diff", {})
    hi = p11.get("ac_v2_4", {}).get("hi_small", {})
    if fuzz or hi:
        out += [f"## AC-V2-4 on sm_{dev['compute_capability'].replace('.', '')}", "",
                "| profile | fuzz graphs | H | mismatching rows | seconds | HI-Small: CUDA hashes = oracle |",
                "|---|---:|---|---:|---:|---|"]
        for prof in sorted(set(fuzz) | set(hi)):
            f = fuzz.get(prof, {}).get("cuda")
            if f:
                bad = sum(f["mismatched_rows_by_block"].values())
                checks[f"{prof}: fuzz-diff cuda, no mismatching rows"] = bad == 0 and f["mismatched_seeds"] == 0
            got = hi.get(prof, {}).get("cuda", {}).get("output_fnv1a")
            want = oracle.get(prof, {}).get("oracle", {}).get("output_fnv1a")
            verdict = "no CUDA run" if got is None else "no oracle hashes" if want is None else \
                "yes" if got == want else "DIFFER: " + ", ".join(b for b in order(want) if got.get(b) != want[b])
            if got is not None and want is not None:
                checks[f"{prof}: HI-Small cuda = oracle"] = got == want
            out.append(f"| {prof} | {f['graphs']:,} | {', '.join(f['heavy_thresholds'])} | {bad} | "
                       f"{f['seconds']:.0f} | {verdict} |" if f else f"| {prof} | – | – | – | – | {verdict} |")
        out += ["", "The oracle hashes are P10's, mined on `local` (`results/p10_semantics.json`).", ""]

    # Kernel time by H, per profile.
    tuned, gpu_vs_cpu = {}, {}
    for prof, cu in cuda.items():
        blocks = order(cu["blocks"])
        hs = cu["heavy_thresholds"]
        kern = {b: {h: cu["blocks"][b]["by_threshold"][h]["kernel"]["median"] for h in hs} for b in blocks}
        total = {h: sum(kern[b][h] for b in blocks) for h in hs}
        best_h = min(hs, key=lambda h: total[h])
        per_block = {b: min(hs, key=lambda h: kern[b][h]) for b in blocks}
        upload = cu["graph_upload"]["median"]
        tuned[prof] = {"heavy_threshold": best_h, "kernel_seconds_by_threshold": total,
                       "per_block_best": per_block, "profile_id": cu.get("profile", {}).get("id")}
        out += [f"## {prof}: kernel time (ms) by H", "",
                "| block | " + " | ".join(f"H={h}" for h in hs) + " |", "|---|" + "---:|" * len(hs)]
        for b in blocks:
            out.append(f"| {b} | " + " | ".join(f"**{ms(kern[b][h])}**" if h == per_block[b] else ms(kern[b][h])
                                                for h in hs) + " |")
        if len(blocks) > 1:
            out.append("| **all** | " + " | ".join(f"**{ms(total[h])}**" if h == best_h else ms(total[h])
                                                  for h in hs) + " |")
        out += ["", f"Tuned H (smallest total): **{best_h}**. Graph upload {ms(upload)} ms.", ""]

        # GPU vs CPU on this instance, at the tuned H; the rows must hash the same.
        c = cpu.get(prof)
        if not c:
            continue
        thr = str(max(int(t) for t in c["threads"]))
        rows, t_gpu_all, t_cpu_all = {}, upload, 0.0
        for b in blocks:
            r = cu["blocks"][b]["by_threshold"][best_h]
            t_gpu = r["kernel"]["median"] + r["d2h"]["median"]
            t_cpu = c["blocks"][b]["dynamic"][thr]["median"]
            same = cu["blocks"][b]["output_fnv1a"] == c["blocks"][b]["output_fnv1a"]
            checks[f"{prof}: {b} cuda = cpu"] = same
            rows[b] = {"gpu_seconds": t_gpu, "cpu_seconds": t_cpu, "ratio": t_cpu / t_gpu, "same_rows": same,
                       "cpu_plan": c["blocks"][b].get("cpu_plans")}
            t_gpu_all += t_gpu
            t_cpu_all += t_cpu
        gpu_vs_cpu[prof] = {"heavy_threshold": best_h, "cpu_threads": int(thr), "blocks": rows,
                            "gpu_seconds_with_upload": t_gpu_all, "cpu_seconds": t_cpu_all,
                            "ratio": t_cpu_all / t_gpu_all}
        out += [f"GPU vs CPU ({thr} threads, dynamic schedule) on this instance. GPU = kernels + D2H at H = "
                f"{best_h}; CPU = the block's mining time (`blz bench --backend cpu`).", "",
                "| block | CPU plan | GPU | CPU | CPU / GPU | rows equal |", "|---|---|---:|---:|---:|---|"]
        for b in blocks:
            r = rows[b]
            plan = ", ".join(sorted(set((r["cpu_plan"] or {}).values()))) or "?"
            out.append(f"| {b} | {plan} | {secs(r['gpu_seconds'])} | {secs(r['cpu_seconds'])} | "
                       f"{r['ratio']:.1f}× | {'yes' if r['same_rows'] else 'NO'} |")
        out += [f"| **all** | | {secs(t_gpu_all)} (with upload) | {secs(t_cpu_all)} | "
                f"{t_cpu_all / t_gpu_all:.1f}× | |", ""]

    # v1_default against v1's E5 on the same GPU model: the CUDA code of v1's blocks is unchanged.
    v1 = cuda.get("v1_default")
    if v1 and e5:
        h = tuned["v1_default"]["heavy_threshold"]
        h1 = e5_doc.get("tuned", {}).get("heavy_threshold")
        out += [f"## v1_default against v1's E5 (`e5_gpu_{host}.json`)", "",
                f"Kernel time (ms) at each run's tuned H (v2: {h}, v1: {h1}); a different instance of the same "
                "GPU model.", "", "| block | v1 | v2 | v2 / v1 | rows equal |", "|---|---:|---:|---:|---|"]
        for b in order(v1["blocks"]):
            if b not in e5["blocks"] or h1 is None:
                continue
            k1 = e5["blocks"][b]["by_threshold"][h1]["kernel"]["median"]
            k2 = v1["blocks"][b]["by_threshold"][h]["kernel"]["median"]
            same = e5["blocks"][b]["output_fnv1a"] == v1["blocks"][b]["output_fnv1a"]
            checks[f"v1_default: {b} rows = v1's E5"] = same
            out.append(f"| {b} | {ms(k1)} | {ms(k2)} | {k2 / k1:.2f} | {'yes' if same else 'NO'} |")
        out.append("")

    failed = [k for k, v in checks.items() if not v]
    out += ["## Checks", "", f"{len(checks) - len(failed)} of {len(checks)} pass."]
    out += [f"- FAIL: {k}" for k in failed] + [""]
    doc["tuned"], doc["gpu_vs_cpu"], doc["checks"] = tuned, gpu_vs_cpu, checks
    path.write_text(json.dumps(doc, indent=1) + "\n")
    md = path.with_suffix(".md")
    md.write_text("\n".join(out))
    print(f"wrote {md}; {len(checks) - len(failed)}/{len(checks)} checks pass")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

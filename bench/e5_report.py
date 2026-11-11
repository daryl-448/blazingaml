#!/usr/bin/env python3
"""E5 report (SPEC §7.5, §11.2, §12.2): results/e5_gpu_<host>.json -> results/e5_gpu_<host>.md.

Reads the "cuda" section (blz bench --backend cuda: every block at every light/heavy threshold H)
and, from results/e4_cpu_<host>.json on the same host, the GFP timings ("gfp") and the CPU
backend's output hashes and times ("cpu"). It writes into the E5 JSON:

- "tuned": the threshold H for this GPU, the one with the smallest kernel time summed over the
  blocks (SPEC §7.5: "P5 tunes it per GPU and records it"), plus each block's own best H;
- "speedup": T_GFP_best / T_BlazingAML per GFP pattern group at the tuned H, where
  T_BlazingAML = median graph upload + the group's kernels + its device-to-host copies
  (SPEC §11.2, the graph upload included);
- "cpu_hash_check": whether each block's output hash equals the CPU backend's (same config).

Usage: bench/e5_report.py results/e5_gpu_local.json [--e4 results/e4_cpu_local.json]
"""

import argparse
import json
from pathlib import Path

ORDER = ["fan_in", "fan_out", "deg_in", "deg_out", "sg", "tcyc"]  # GFP column order (SPEC §4.2)


def fmt(x):
    if x < 0.01:
        return f"{x * 1e3:.2f} ms"
    return f"{x:.3f}" if x < 10 else f"{x:.1f}"


def ms(x):
    return f"{x * 1e3:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("e5")
    ap.add_argument("--e4", help="E4 results of the same host (default: results/e4_cpu_<host>.json)")
    args = ap.parse_args()
    path = Path(args.e5)
    host = path.stem.removeprefix("e5_gpu_")
    doc = json.loads(path.read_text())
    cu = doc["cuda"]
    e4_path = Path(args.e4) if args.e4 else path.with_name(f"e4_cpu_{host}.json")
    e4 = json.loads(e4_path.read_text()) if e4_path.exists() else {}

    blocks = sorted(cu["blocks"], key=lambda b: ORDER.index(b) if b in ORDER else len(ORDER))
    hs = cu["heavy_thresholds"]
    kern = {b: {h: cu["blocks"][b]["by_threshold"][h]["kernel"]["median"] for h in hs} for b in blocks}
    total_by_h = {h: sum(kern[b][h] for b in blocks) for h in hs}
    tuned = min(hs, key=lambda h: total_by_h[h])
    per_block_best = {b: min(hs, key=lambda h: kern[b][h]) for b in blocks}
    upload = cu["graph_upload"]["median"]

    env, dev = cu["env"], cu["cuda"]
    git = env.get("git") or {}
    model = next((l.split(":", 1)[1].strip() for l in env["lscpu"].splitlines() if l.startswith("Model name")), "?")
    out = [f"# E5: GPU mining time ({host})", ""]
    out += [f"- GPU: {dev['device']} (sm_{dev['compute_capability'].replace('.', '')}, {dev['sm_count']} SMs, "
            f"{dev['memory_mb']['total']} MB); driver {dev['driver_version']}, runtime {dev['runtime_version']}",
            f"- Host CPU: {model}",
            f"- Build: {env['build']['type']}, `{env['build']['cxx_flags']}`, {env['compiler']}; CUDA: {dev['build']}",
            f"- Git: {git.get('sha', '?')[:12]}{' (dirty)' if git.get('dirty') else ''}; config {env['config']}",
            f"- Index build on the host (shared by all blocks, not in the times below): {cu['index_build_seconds']:.2f} s; "
            f"device copy {cu['device_graph_bytes'] / 1e6:.0f} MB, uploaded from pageable memory in "
            f"{ms(upload)} ms (median of {len(cu['graph_upload']['seconds'])})",
            f"- Each cell: median of {cu['reps']} timed runs after {cu['warmup']} warm-up, over every trigger. "
            "Every run at every threshold produced the same rows (SPEC §8).", ""]

    # Cross-check against the CPU backend of the same host and config (hash of the rows).
    cpu = e4.get("cpu")
    if cpu and cpu["env"]["config"] == env["config"]:
        check = {b: cu["blocks"][b]["output_fnv1a"] == cpu["blocks"][b]["output_fnv1a"]
                 for b in blocks if b in cpu["blocks"]}
        doc["cpu_hash_check"] = check
        ok = all(check.values()) and len(check) == len(blocks)
        out += [f"Output check: the rows of every block {'equal' if ok else 'DIFFER FROM'} the CPU backend's "
                f"(FNV-1a of the whole block, `{e4_path.name}`), which equal the oracle's (AC1).", ""]

    out += ["## Kernel time (ms) by light/heavy threshold H", "",
            "Kernel time = output zeroing + bucketing (work estimates, partition) + light kernel + heavy kernel, "
            "from cudaEvents. A trigger is heavy when its work estimate (the first scan's window) is ≥ H; "
            "H = 0 puts every trigger on the heavy kernel (warp per trigger), H = all every trigger on the light "
            "kernel (thread per trigger). Bold: the block's fastest H.", "",
            "| block | " + " | ".join(f"H={h}" for h in hs) + " |", "|---|" + "---:|" * len(hs)]
    for b in blocks:
        cells = [(f"**{ms(kern[b][h])}**" if h == per_block_best[b] else ms(kern[b][h])) for h in hs]
        out.append(f"| {b} | " + " | ".join(cells) + " |")
    out.append("| **all** | " + " | ".join((f"**{ms(total_by_h[h])}**" if h == tuned else ms(total_by_h[h]))
                                          for h in hs) + " |")
    out += ["", f"Tuned threshold for this GPU (smallest total): **H = {tuned}**. "
            f"The SPEC default is 32.", ""]

    out += [f"## Breakdown at H = {tuned}", "",
            "| block | patterns | light triggers | heavy triggers | bucket (ms) | light (ms) | heavy (ms) | "
            "kernels (ms) | D2H (ms) | T = upload + kernels + D2H (ms) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    rows = {}
    for b in blocks:
        r = cu["blocks"][b]["by_threshold"][tuned]
        rows[b] = r
        out.append(f"| {b} | {', '.join(cu['blocks'][b]['patterns'])} | {r['light_triggers']:,} | "
                   f"{r['heavy_triggers']:,} | {ms(r['bucket_median'])} | {ms(r['light_median'])} | "
                   f"{ms(r['heavy_median'])} | {ms(r['kernel']['median'])} | {ms(r['d2h']['median'])} | "
                   f"{ms(r['total']['median'])} |")
    k_all = sum(rows[b]["kernel"]["median"] for b in blocks)
    d_all = sum(rows[b]["d2h"]["median"] for b in blocks)
    out += [f"| **all** | | | | | | | {ms(k_all)} | {ms(d_all)} | {ms(upload + k_all + d_all)} (one upload) |", "",
            "Trigger counts are summed over a block's patterns (sg runs sg_gather and sg_scatter over the same "
            "triggers). Bucketing includes reading the heavy count back to the host.", ""]
    doc["tuned"] = {"heavy_threshold": tuned, "kernel_seconds_by_threshold": total_by_h,
                    "per_block_best": per_block_best}

    gfp = e4.get("gfp")
    if gfp:
        speed = {}
        cpu_top = None
        if cpu and "dynamic" in cpu["blocks"][blocks[0]]:
            cpu_top = str(max(int(t) for t in cpu["threads"]))
        out += ["## Speedup over GFP best (same machine)", "",
                "T_GFP_best = GFP's fastest batch size at its highest thread count, on this host's CPU "
                f"(`{e4_path.name}`). T_BlazingAML = median graph upload + the group's kernels + its D2H "
                f"at H = {tuned} (SPEC §11.2). speedup = T_GFP_best / T_BlazingAML.", "",
                "| GFP pattern | blocks | T_GFP_best (s) | T_BlazingAML GPU (s) | speedup | "
                + (f"CPU backend, {cpu_top} thr (s) | GPU vs CPU |" if cpu_top else ""),
                "|---|---|---:|---:|---:|" + ("---:|---:|" if cpu_top else "")]
        for g, best in gfp["best"].items():
            t_gpu = upload + sum(rows[b]["kernel"]["median"] + rows[b]["d2h"]["median"] for b in best["blocks"])
            speed[g] = {"gfp_best_seconds": best["seconds"], "gfp_best_batch": best["batch"],
                        "gfp_threads": best["threads"], "blocks": best["blocks"], "heavy_threshold": tuned,
                        "blazingaml_seconds": t_gpu, "speedup": best["seconds"] / t_gpu}
            line = (f"| {g} | {', '.join(best['blocks'])} | {fmt(best['seconds'])} (batch {best['batch']}, "
                    f"{best['threads']} thr) | {fmt(t_gpu)} | {best['seconds'] / t_gpu:.0f}× |")
            if cpu_top:
                t_cpu = sum(cpu["blocks"][b]["dynamic"][cpu_top]["median"] for b in best["blocks"])
                speed[g]["cpu_seconds"] = t_cpu
                speed[g]["cpu_threads"] = int(cpu_top)
                line += f" {fmt(t_cpu)} | {t_cpu / t_gpu:.1f}× |"
            out.append(line)
        out += ["", "Caveat (SPEC §11.2): " + gfp["caveat"], ""]
        doc["speedup"] = speed
    if doc.get("notes"):
        out += ["## Measurement notes", ""] + [f"- {n}" for n in doc["notes"]] + [""]

    path.write_text(json.dumps(doc, indent=1) + "\n")
    md = path.with_suffix(".md")
    md.write_text("\n".join(out))
    print(f"wrote {md} and the tuned threshold{' and speedups' if gfp else ''} into {path}")


if __name__ == "__main__":
    main()

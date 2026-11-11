#!/usr/bin/env python3
"""E4 report (SPEC §11.2, §12.2): results/e4_cpu_<host>.json -> results/e4_cpu_<host>.md.

Reads the "cpu" section (blz bench) and, if present, the "gfp" section (tools/gfp/bench_gfp.py),
writes the speedups into the JSON under "speedup" and renders the tables as markdown.

    speedup = T_GFP_best / T_BlazingAML

T_GFP_best is the minimum over batch sizes at GFP's highest thread count. T_BlazingAML is the
median mining time of the GFP group's blocks (fan = fan_in + fan_out, degree = deg_in + deg_out,
scatter-gather = sg, temp-cycle = tcyc) at each thread count, without the shared index build.

Usage: bench/e4_report.py results/e4_cpu_local.json
"""

import json
import sys
from pathlib import Path


def fmt(x):
    return f"{x:.3f}" if x < 10 else f"{x:.1f}"


def main():
    path = Path(sys.argv[1])
    doc = json.loads(path.read_text())
    cpu = doc["cpu"]
    order = ["fan_in", "fan_out", "deg_in", "deg_out", "sg", "tcyc"]  # GFP column order (SPEC §4.2)
    blocks = sorted(cpu["blocks"], key=lambda b: order.index(b) if b in order else len(order))
    threads = [str(t) for t in cpu["threads"]]
    schedules = [s for s in ("dynamic", "sorted") if s in cpu["blocks"][blocks[0]]]
    env = cpu["env"]
    model = next((l.split(":", 1)[1].strip() for l in env["lscpu"].splitlines() if l.startswith("Model name")), "?")

    out = [f"# E4: CPU mining time ({path.stem.removeprefix('e4_cpu_')})", ""]
    git = env.get("git") or {}
    out += [f"- CPU: {model}; OMP_PROC_BIND={env['omp']['OMP_PROC_BIND']}, OMP_PLACES={env['omp']['OMP_PLACES']}",
            f"- Build: {env['build']['type']}, `{env['build']['cxx_flags']}`, {env['compiler']}",
            f"- Git: {git.get('sha', '?')[:12]}{' (dirty)' if git.get('dirty') else ''}; config {env['config']}",
            f"- Index build (shared by all blocks, not in the times below): {cpu['index_build_seconds']:.2f} s, "
            f"{cpu['index_bytes'] / 1e6:.0f} MB; peak RSS {cpu['peak_rss_mb']:.0f} MB",
            f"- Each cell: median of {cpu['reps']} timed runs after {cpu['warmup']} warm-up, over every "
            "trigger; the output is identical in every run (SPEC §8)", ""]

    for s in schedules:
        out += [f"## Mining time (s), schedule `{s}`", "",
                "| block | " + " | ".join(f"{t} thr" for t in threads) + " | 1 → max |",
                "|---|" + "---:|" * (len(threads) + 1)]
        for b in blocks:
            row = cpu["blocks"][b][s]
            med = [row[t]["median"] for t in threads]
            out.append(f"| {b} | " + " | ".join(fmt(x) for x in med) + f" | {med[0] / med[-1]:.1f}× |")
        tot = [sum(cpu["blocks"][b][s][t]["median"] for b in blocks) for t in threads]
        out.append("| **all** | " + " | ".join(fmt(x) for x in tot) + f" | {tot[0] / tot[-1]:.1f}× |")
        out.append("")
    if "sorted" in schedules:
        out += ["`sorted` includes its schedule cost (work estimates plus a sort of every trigger); "
                "loop-only medians are in the JSON (`loop_median`).", ""]

    # The speedups use the default `dynamic` schedule; without it, only our own tables are written.
    gfp = doc.get("gfp") if "dynamic" in schedules else None
    if gfp:
        groups = gfp["best"]
        speed = {}
        out += ["## GFP (snapml " + gfp["snapml"] + "), transform() only, fresh instance per run", "",
                "| pattern | threads | batch | median (s) | runs |", "|---|---:|---:|---:|---:|"]
        for p in gfp["points"]:
            out.append(f"| {p['pattern']} | {p['threads']} | {p['batch']} | {fmt(p['median'])} | {len(p['seconds'])} |")
        out += ["", "## Speedup over GFP best (`dynamic` schedule)", "",
                "T_GFP_best = the fastest batch size at GFP's highest thread count. "
                "speedup = T_GFP_best / T_BlazingAML.", "",
                "| GFP pattern | blocks | T_GFP_best (s) | " + " | ".join(f"{t} thr" for t in threads) + " |",
                "|---|---|---:|" + "---:|" * len(threads)]
        for g, best in groups.items():
            ours = {t: sum(cpu["blocks"][b]["dynamic"][t]["median"] for b in best["blocks"]) for t in threads}
            speed[g] = {"gfp_best_seconds": best["seconds"], "gfp_best_batch": best["batch"],
                        "gfp_threads": best["threads"], "blocks": best["blocks"],
                        "blazingaml_seconds": ours,
                        "speedup": {t: best["seconds"] / ours[t] for t in threads}}
            out.append(f"| {g} | {', '.join(best['blocks'])} | {fmt(best['seconds'])} (batch {best['batch']}) | " +
                       " | ".join(f"{speed[g]['speedup'][t]:.0f}×" for t in threads) + " |")
        out += ["", "Caveat (SPEC §11.2): " + gfp["caveat"], ""]
        doc["speedup"] = speed
        path.write_text(json.dumps(doc, indent=1) + "\n")
    if doc.get("notes"):
        out += ["## Measurement notes", ""] + [f"- {n}" for n in doc["notes"]] + [""]

    md = path.with_suffix(".md")
    md.write_text("\n".join(out))
    print(f"wrote {md}" + (" and the speedups into " + str(path) if gfp else ""))


if __name__ == "__main__":
    main()

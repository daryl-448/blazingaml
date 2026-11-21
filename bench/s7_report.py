#!/usr/bin/env python3
"""E12 delta for S7 (SPEC v2 §8 S7): results/e12_s7_<host>.json -> results/e12_s7_<host>.md.

Reads what bench/s7.sh wrote ("runs"/<profile>/<variant>: `blz bench --backend cuda` with
variant base, pack, fuse or fuse_pack) and reports, per profile:
- the tuned H per variant (smallest summed kernel time, as E12) and, at it, the summed kernel,
  pack and D2H times, the D2H bytes, and T = graph upload (once) + kernels + pack + D2H, with the
  change against base;
- per fusion (fan_in+deg_in, fan_out+deg_out): the fused unit against its blocks run one by one,
  at each variant's tuned H.
The units' own `total` adds the graph upload to each unit (E5's per-block form); T counts it once,
so fused and unfused profiles compare fairly.

Checks: every block's hash is the same in every variant and equals the reference (P10's oracle
hashes for v1_default, sg_ordered_v1w and sg_ordered_tsvis; E12's A30 CUDA hashes, equal to the
CPU's there, for the DG-2 profiles); with packing, the width of every block's copy is recorded.
The report writes "summary" and "checks" into the JSON.

Usage: bench/s7_report.py results/e12_s7_local.json [--oracle ...] [--e12 ...]
"""

import argparse
import json
import statistics
from pathlib import Path

PROFILES = ["v1_default", "sg_ordered_v1w", "sg_ordered_tsvis", "dg2_v1", "dg2_wc", "dg2_wb"]
VARIANTS = ["base", "pack", "fuse", "fuse_pack"]
LABEL = {"base": "base", "pack": "pack", "fuse": "fuse", "fuse_pack": "fuse + pack"}


def ms(x):
    return f"{x * 1e3:.1f}"


def h_key(h):
    return 1 << 40 if h == "all" else int(h)


def unit_outputs(u):
    """{block: output_fnv1a} of one unit (fused units list theirs under "outputs")."""
    return {b: o["output_fnv1a"] for b, o in u["outputs"].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("s7")
    ap.add_argument("--oracle", default="results/p10_semantics.json", help="P10's oracle hashes")
    ap.add_argument("--e12", default="results/e12_gpu_a30.json", help="E12's A30 run (DG-2 hashes)")
    args = ap.parse_args()
    path = Path(args.s7)
    host = path.stem.removeprefix("e12_s7_")
    doc = json.loads(path.read_text())
    runs = doc["runs"]
    oracle = json.loads(Path(args.oracle).read_text())["ac_v2_4"]["hi_small"]
    e12 = json.loads(Path(args.e12).read_text())["cuda"]
    want = {}
    for prof in PROFILES:
        if prof in oracle:
            want[prof] = oracle[prof]["oracle"]["output_fnv1a"]
        elif prof in e12:
            want[prof] = {b: v["output_fnv1a"] for b, v in e12[prof]["blocks"].items()}
    checks, summary = {}, {}

    first = runs[next(iter(runs))]["base"]
    env, cuda = first["env"], first["cuda"]
    cpu_model = next((l.split(":", 1)[1].strip() for l in env.get("lscpu", "").splitlines() if l.startswith("Model name")), "?")
    md = [
        f"# E12 delta for S7: packed D2H and multi-pattern fusion ({host})",
        "",
        f"- GPU: {cuda.get('device', '?')} (compute capability {cuda.get('compute_capability', '?')}, "
        f"{cuda.get('sm_count', '?')} SMs); CUDA build: {cuda.get('build', '?')}",
        f"- CPU: {cpu_model} (the host widening of packed rows runs on its OpenMP team, "
        "`OMP_WAIT_POLICY=passive`)",
        f"- Git: {str(env.get('git', {}).get('sha', '?'))[:12]}{' (dirty)' if env.get('git', {}).get('dirty') else ''}",
        f"- Each cell: median of {first['reps']} timed runs after 1 warm-up, every trigger of HI-Small, "
        "H swept over 0, 8, 16, 32, 64, 128, 256, 1024, all.",
        "- Variants: **base** (S6's backend), **pack** (`--pack`: each block crosses PCIe as uint8 or uint16 when "
        "its largest cell fits, and the host widens it back), **fuse** (`--fuse`: each compiled fusion's blocks "
        "in one kernel), **fuse + pack**.",
        "- D2H is the copy (with packing: the pipelined copy of the narrow rows and their widening, host clock); "
        "pack is the pack kernels and the read-back of the block's largest cell.",
        "- T = graph upload (once) + kernels + pack + D2H over the profile's blocks, at the variant's tuned H "
        "(smallest summed kernel time, as E12).",
        "",
    ]

    for prof in PROFILES:
        checks[f"{prof}: measured"] = prof in runs
        if prof not in runs:
            continue
        pr = runs[prof]
        md += [f"## {prof}", ""]
        # Hashes: equal across variants, and to the reference.
        hashes = {}
        for v in VARIANTS:
            if v not in pr:
                continue
            got = {}
            for u in pr[v]["blocks"].values():
                got.update(unit_outputs(u))
            hashes[v] = got
        same = all(h == hashes["base"] for h in hashes.values())
        ref = want.get(prof)
        ref_ok = ref is not None and hashes["base"] == ref
        checks[f"{prof}: every variant ran"] = sorted(hashes) == sorted(VARIANTS)
        checks[f"{prof}: every variant's hashes equal"] = same
        checks[f"{prof}: hashes = reference"] = ref_ok
        md += [
            f"Hashes: every variant's equal: **{'yes' if same else 'NO'}**; equal to the reference "
            f"({'P10 oracle' if prof in oracle else 'E12 A30'}): **{'yes' if ref_ok else 'NO'}**.",
            "",
        ]

        rows, tuned = [], {}
        for v in VARIANTS:
            if v not in pr:
                continue
            r = pr[v]
            units = r["blocks"]
            hs = sorted(next(iter(units.values()))["by_threshold"], key=h_key)
            kern = {h: sum(u["by_threshold"][h]["kernel"]["median"] for u in units.values()) for h in hs}
            best = min(hs, key=lambda h: kern[h])
            at = {n: u["by_threshold"][best] for n, u in units.items()}
            pack = sum(b["pack"]["median"] for b in at.values() if "pack" in b)
            d2h = sum(b["d2h"]["median"] for b in at.values())
            nbytes = sum(b.get("d2h_bytes", 0) for b in at.values()) if v.endswith("pack") else None
            widths = {}
            for b in at.values():
                widths.update(b.get("d2h_width", {}))
            upload = r["graph_upload"]["median"]
            t = upload + kern[best] + pack + d2h
            tuned[v] = {"heavy_threshold": best, "kernel_seconds": kern[best], "pack_seconds": pack,
                        "d2h_seconds": d2h, "graph_upload_seconds": upload, "t_seconds": t,
                        "d2h_bytes": nbytes, "d2h_width": widths, "at": at}
        base_t = tuned["base"]["t_seconds"]
        md += [
            "| variant | tuned H | kernels (ms) | pack (ms) | D2H (ms) | D2H bytes | T (ms) | vs base |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for v, x in tuned.items():
            nb = f"{x['d2h_bytes'] / 1e6:.0f} MB" if x["d2h_bytes"] is not None else "4 B/cell"
            md.append(f"| {LABEL[v]} | {x['heavy_threshold']} | {ms(x['kernel_seconds'])} | {ms(x['pack_seconds'])} | "
                      f"{ms(x['d2h_seconds'])} | {nb} | {ms(x['t_seconds'])} | "
                      f"{(base_t - x['t_seconds']) / base_t * 100:+.1f} % |")
        widths = tuned.get("pack", {}).get("d2h_width", {})
        if widths:
            md += ["", "Packed widths (bytes per cell): " + ", ".join(f"{b} {w}" for b, w in sorted(widths.items())) + "."]
        md.append("")

        # Per fusion: the fused unit against its blocks one by one.
        fused_units = {n: u for n, u in pr.get("fuse", {}).get("blocks", {}).items() if u.get("fusion")}
        if fused_units:
            md += [
                "| fusion | base: kernels / D2H (ms) | fuse: kernels / D2H | fuse + pack: kernels / pack + D2H | "
                "kernels saved by fusion |",
                "|---|---:|---:|---:|---:|",
            ]
            groups = {}
            for name, u in fused_units.items():
                blocks = u["blocks"]
                b_at = tuned["base"]["at"]
                bk = sum(b_at[b]["kernel"]["median"] for b in blocks)
                bd = sum(b_at[b]["d2h"]["median"] for b in blocks)
                f_at = tuned["fuse"]["at"][name]
                fk, fd = f_at["kernel"]["median"], f_at["d2h"]["median"]
                cell = "—"
                if "fuse_pack" in tuned:
                    p_at = tuned["fuse_pack"]["at"][name]
                    cell = f"{ms(p_at['kernel']['median'])} / {ms(p_at['pack']['median'] + p_at['d2h']['median'])}"
                md.append(f"| {name} | {ms(bk)} / {ms(bd)} | {ms(fk)} / {ms(fd)} | {cell} | "
                          f"{(bk - fk) / bk * 100:+.1f} % |")
                groups[name] = {"blocks": blocks, "base_kernel_seconds": bk,
                                "fuse_kernel_seconds": fk, "base_d2h_seconds": bd, "fuse_d2h_seconds": fd}
            md.append("")
        else:
            groups = {}
        for x in tuned.values():
            del x["at"]
        summary[prof] = {"tuned": tuned, "fusions": groups}

    dw = doc.get("default_wait_policy", {}).get("v1_default", {}).get("pack")
    if dw:
        md += ["## libgomp's default wait policy", "",
               "The same `--pack` run of v1_default at H = 32 without `OMP_WAIT_POLICY` (10 timed runs): D2H per run.", "",
               "| block | D2H min / median / max (ms) | widening median (ms) |", "|---|---:|---:|"]
        spread = {}
        for n, u in dw["blocks"].items():
            b = u["by_threshold"]["32"]
            s = b["d2h"]["seconds"]
            spread[n] = {"min": min(s), "median": statistics.median(s), "max": max(s)}
            md.append(f"| {n} | {ms(min(s))} / {ms(statistics.median(s))} / {ms(max(s))} | {ms(b['widen_median'])} |")
        summary["default_wait_policy"] = spread
        md.append("")

    md += ["## Checks", ""] + [f"- {k}: **{'pass' if v else 'FAIL'}**" for k, v in checks.items()] + [""]
    doc["summary"], doc["checks"] = summary, checks
    path.write_text(json.dumps(doc, indent=1) + "\n")
    path.with_suffix(".md").write_text("\n".join(md))
    print("\n".join(md))
    if not all(checks.values()):
        raise SystemExit("s7_report: a check failed")


if __name__ == "__main__":
    main()

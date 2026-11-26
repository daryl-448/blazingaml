# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pyyaml"]
# ///
"""E17 (SPEC v3 §8, AC-V3-5): per dataset, its prepare invariants, M (§5.1) at 5 reps and peak RSS
per profile and backend, and the §6.2 budgets; `freeze` writes the clean baseline once (§5.2, P15).

  uv run --python 3.12 bench/e17.py run [CELL ...] [--smoke]   # measure the cells, into build/e17/
  uv run --python 3.12 bench/e17.py report [--smoke]           # results/e17_datasets_local.{json,md}
  uv run --python 3.12 bench/e17.py freeze [--smoke]           # results/p15_baseline_local.{json,md}, once

A cell is <dataset>/<profile>/<backend>: HI-Small, TransXion, LI-Small and HI-Medium; clean_default
and the dataset's W-C profile (TransXion takes HI-Small's windows, as AC-V3-1 does); cpu and hybrid
at 12 threads, and cuda; every backend at its default options (no --fuse, --pack or --reach-prune)
and the harness chunk, 2^20 triggers (D-81). Each cell is one `blz measure --reps 5` over every
trigger. Before it: `ps` and `free -m` go into its record; the run waits (up to 30 min) for CPU busy
under 10 % of the machine over 3 s; and the cell's §6.2 estimates (bench/p15_preflight.py's models,
or the preflight's HI-Medium measurement when larger) must fit MemAvailable and, on cuda and hybrid,
the GPU's free memory. A cell that does not fit is recorded as a capacity limit and not run. e20's
sampler records the foreign load every 2 s and nvidia-smi the device peak every 0.2 s. A cell with
a record is skipped. Nothing here stops another process.

`report` adds, per dataset, the invariants `blz prepare` gated (configs/datasets/<id>.yaml against
the .blzg's descriptor hash, and those countable without a test label recounted from the .blzg),
D-68's prepare peak, and the training estimate from the preflight's fit model. It reads no test label.

`freeze` takes the §5.2 workloads from E17's records and measures nothing: HI-Medium both profiles on
cpu (the primary workloads), HI-Small both and TransXion clean_default on cpu, cuda on HI-Small both
and HI-Medium both (in chunks, as every run is), with their capacity limits, the harness commit
and the chunk. It refuses a missing workload, records from different or dirty commits, a run with
errors, and an existing file. `--smoke` measures TransXion clean_default on cpu, cuda and hybrid over
range:0:200000 at 2 reps into build/e17_smoke/, where its report and freeze go too.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import e20 as E20  # noqa: E402
import p15_chunk_cost as CC  # noqa: E402
import p15_preflight as PF  # noqa: E402
from blzg import Blzg  # noqa: E402  (bench/dg2, on the path through p15_preflight)

BLZ = "build/rel-cuda/blz"
DATASETS = ["hi_small", "transxion", "li_small", "hi_medium"]  # smallest first
NAMES = {"hi_small": "HI-Small", "transxion": "TransXion", "li_small": "LI-Small", "hi_medium": "HI-Medium"}
BACKENDS = ["cpu", "cuda", "hybrid"]
THREADS = 12
REPS = 5
IDLE_BUSY, IDLE_TRIES = 0.10, 600  # busy below 10 % over 3 s; 600 tries of 3 s
PREFLIGHT = ROOT / "results" / "p15_preflight_hi_medium.json"
PREPARE_SECONDS = {"hi_small": 5.3, "li_small": 7.5, "hi_medium": 37.7, "transxion": 2.2}  # D-68
FIT_COLUMNS = {"clean_default": 40, "clean_all": 61}  # the preflight's fits: base + vstats + blocks
WORKLOADS = [  # SPEC v3 §5.2
    ("hi_medium/clean_default/cpu", "target (primary): the stop predicate is measured here"),
    ("hi_medium/clean_wc_hi_medium/cpu", "long-window target (primary)"),
    ("hi_small/clean_default/cpu", "guard: development loop"),
    ("hi_small/clean_wc_hi_small/cpu", "guard: development loop"),
    ("transxion/clean_default/cpu", "guard: a different shape"),
    ("hi_small/clean_default/cuda", "guard: 1650S"),
    ("hi_small/clean_wc_hi_small/cuda", "guard: 1650S"),
    ("hi_medium/clean_default/cuda", "guard: 1650S, HI-Medium in trigger-range chunks"),
    ("hi_medium/clean_wc_hi_medium/cuda", "guard: 1650S, HI-Medium in trigger-range chunks"),
]
SMOKE = {"triggers": "range:0:200000", "reps": 2,
         "cells": ["transxion/clean_default/cpu", "transxion/clean_default/cuda", "transxion/clean_default/hybrid"]}


class Mode:
    def __init__(self, smoke):
        self.smoke = smoke
        self.work = ROOT / "build" / ("e17_smoke" if smoke else "e17")
        self.cells_json, self.measure_json = self.work / "cells.json", self.work / "measure.json"
        self.out = self.work / "result.json" if smoke else ROOT / "results" / "e17_datasets_local.json"
        self.frozen = self.work / "baseline.json" if smoke else ROOT / "results" / "p15_baseline_local.json"
        self.triggers = SMOKE["triggers"] if smoke else "all"
        self.reps = SMOKE["reps"] if smoke else REPS
        self.cells = SMOKE["cells"] if smoke else [f"{ds}/{p}/{be}" for ds in DATASETS
                                                   for p in ("clean_default", PF.WC[ds]) for be in BACKENDS]
        self.workloads = [(c, "smoke") for c in SMOKE["cells"]] if smoke else WORKLOADS


def rel(p):
    return str(Path(p).relative_to(ROOT))


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


# ---- §6.2 budgets --------------------------------------------------------------------------------

def budgets(pf, ds, prof, be):
    """A cell's estimates from the preflight's models (host split, device split on cuda and hybrid),
    and the value its gate uses: the estimate, or the preflight's HI-Medium measurement when larger."""
    f = pf["facts"][ds]
    out = {"host_estimate_mib": PF.estimate_host(be, prof, f, pf["host_calibration"]["models"][be], None)}
    host_gate, dev_gate = out["host_estimate_mib"]["peak"], None
    if be in ("cuda", "hybrid"):
        d = pf["device_calibration"][be]
        g = PF.device_graph_bytes(f) / PF.MIB
        out["device_estimate_mib"] = {"context": d["context"], "graph": round(g), "scratch_and_output": d["scratch_and_output"],
                                      "peak": round(d["context"] + g + d["scratch_and_output"])}
        dev_gate = out["device_estimate_mib"]["peak"]
    if ds == "hi_medium":
        row = next((r for r in pf["hi_medium"] if r["backend"] == be and r["profile"] == prof), {})
        m = row.get("measured", {})
        seen = m.get("host_whole_peak_inferred_mib", m.get("host_mib", {}).get("peak"))
        if seen:
            host_gate = max(host_gate, seen)
        # Device: the largest HI-Medium peak of this backend over both profiles. A range run of
        # clean_wc does not show the scratch a whole-selection sweep allocates, and clean_default's
        # whole run has the same sweep blocks.
        peaks = [r["measured"]["device_mib"]["peak"] for r in pf["hi_medium"]
                 if r["backend"] == be and r.get("measured", {}).get("device_mib")]
        if dev_gate is not None and peaks:
            dev_gate = max(dev_gate, *peaks)
    out["gate_mib"] = {"host": host_gate, "device": dev_gate}
    return out


def gate(b, be):
    free, why = PF.mem_available_mib(), None
    g = PF.gpu_free_mib() if be in ("cuda", "hybrid") else None
    if b["gate_mib"]["host"] > free:
        why = f"host {b['gate_mib']['host']} MiB > {free:.0f} MiB available"
    elif g is not None and b["gate_mib"]["device"] > g["free"]:
        why = f"device {b['gate_mib']['device']} MiB > {g['free']} MiB free on the GPU"
    return why, {"host_available_mib": round(free), "gpu": g}


# ---- running a cell ------------------------------------------------------------------------------

def wait_idle():
    t0 = time.time()
    for _ in range(IDLE_TRIES):
        busy = CC.cpu_busy()
        if busy < IDLE_BUSY:
            break
        print(f"busy {busy:.0%}, waiting for an idle window", flush=True)
    return {"cpu_busy_at_start": round(busy, 3), "idle": busy < IDLE_BUSY, "waited_seconds": round(time.time() - t0, 1)}


def run_cell(mode, cell, pf):
    ds, prof, be = cell.split("/")
    rec = {"cell": cell, "ps_before": sh("ps --sort=-pcpu -eo pid,pcpu,rss,etime,comm | head -9"),
           "free_m_before": sh("free -m")}
    print(rec["ps_before"] + rec["free_m_before"], end="", flush=True)
    rec["budget"] = budgets(pf, ds, prof, be)
    rec.update(wait_idle())
    if not rec["idle"]:  # §2.5: timing runs wait for an idle window; this cell stays to be run
        rec["rc"] = None
        print(f"{cell}: no idle window in {rec['waited_seconds']} s, not measured", flush=True)
        return rec
    why, rec["memory_at_start"] = gate(rec["budget"], be)  # just before the launch
    if why:
        rec.update({"rc": None, "capacity": why})
        print(f"{cell}: capacity limit, {why}", flush=True)
        return rec
    cmd = [BLZ, "measure", "--graph", f"data/{ds}.blzg", "--profile", f"configs/profiles/{prof}.yaml", "--backend", be,
           "--reps", str(mode.reps), "--triggers", mode.triggers, "--out", rel(mode.measure_json), "--name", cell]
    if be in ("cpu", "hybrid"):
        cmd += ["--threads", str(THREADS)]
    log = mode.work / (cell.replace("/", "__") + ".log")
    print("+", " ".join(cmd), flush=True)
    sampler = E20.Sampler(every=2)
    sampler.start()
    t0 = time.time()
    with open(log, "w") as fh:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
        poll = PF.GpuPoll(p.pid) if be in ("cuda", "hybrid") else None
        if poll:
            poll.start()
        rc = p.wait()
        if poll:
            poll.stop.set()
            poll.join()
    sampler.stop.set()
    sampler.join()
    rec.update({"cmd": " ".join(cmd), "rc": rc, "wall_seconds": round(time.time() - t0, 1), "log": rel(log),
                "foreign_load": E20.summary(sampler.samples), "load_samples": sampler.samples})
    if poll and poll.samples:
        rec["device_mib"] = {"peak": max(poll.samples), "context": min(s for s in poll.samples if s > 0),
                             "samples": len(poll.samples)}
    print(f"{cell}: rc {rc}, {rec['wall_seconds']} s, foreign load {rec['foreign_load'].get('foreign_pct_mean')} % "
          f"(max {rec['foreign_load'].get('foreign_pct_max')} %)", flush=True)
    return rec


def run(mode, names):
    if names and set(names) - set(mode.cells):
        sys.exit(f"unknown cells {sorted(set(names) - set(mode.cells))}; cells: {', '.join(mode.cells)}")
    mode.work.mkdir(parents=True, exist_ok=True)
    pf = PF.load(PREFLIGHT, None) or sys.exit(f"{rel(PREFLIGHT)} is missing: run bench/p15_preflight.py report first")
    for cell in names or mode.cells:
        cells = PF.load(mode.cells_json, {})
        c = cells.get(cell, {})
        if (c.get("rc") == 0 and c.get("idle")) or "capacity" in c:  # a non-idle record is measured again
            print(f"{cell}: recorded", flush=True)
            continue
        cells[cell] = run_cell(mode, cell, pf)
        PF.save(mode.cells_json, cells)
        if cells[cell]["rc"] not in (0, None):
            sys.exit(f"{cell} failed (rc {cells[cell]['rc']}), see {cells[cell]['log']}")
    return 0


# ---- the report ----------------------------------------------------------------------------------

def invariants(ds):
    """The descriptor's pinned invariants: blz prepare's gate (the .blzg's descriptor hash is the
    descriptor's), and a recount of those that need no test label."""
    import yaml
    desc = ROOT / "configs" / "datasets" / f"{ds}.yaml"
    pinned = yaml.safe_load(desc.read_text())["invariants"]
    g = Blzg(str(ROOT / "data" / f"{ds}.blzg"))
    src, dst, ts = np.asarray(g.src), np.asarray(g.dst), np.asarray(g.ts)
    lab = np.asarray(g.label[:g.b_test])  # fit and validation rows only
    count = {"edges": g.m, "nodes": g.n, "self_loops": int(np.count_nonzero(src == dst)),
             "distinct_timestamps": int(1 + np.count_nonzero(ts[1:] != ts[:-1])) if g.m else 0,
             "b_fv": g.b_fv, "b_test": g.b_test, "fit_positives": int(lab[:g.b_fv].sum()),
             "val_positives": int(lab[g.b_fv:].sum())}
    sha = hashlib.sha256(desc.read_bytes()).hexdigest()
    return {"descriptor": rel(desc), "descriptor_sha256": sha, "blzg_descriptor_sha256": g.descriptor_sha256,
            "prepared_under_descriptor": sha == g.descriptor_sha256, "pinned": pinned,
            "recounted": {k: {"pinned": pinned.get(k), "blzg": v, "equal": pinned.get(k) == v} for k, v in count.items()},
            "gated_by_prepare_only": sorted(set(pinned) - set(count)),
            "prepare_peak_rss_mib": PF.PREPARE_PEAK_MIB[ds], "prepare_seconds": PREPARE_SECONDS[ds],
            "prepare_source": "D-68 (recorded, not rerun)"}


def cell_result(mode, cell, rec, measure, pf):
    ds, prof, be = cell.split("/")
    out = {"dataset": ds, "profile": prof, "backend": be, "budget": rec["budget"]}
    if "capacity" in rec:
        return {**out, "capacity": rec["capacity"]}
    res = measure
    for k in cell.split("/"):
        res = res.get(k, {}) if isinstance(res, dict) else {}
    # M is over error-free reps only, and a run that died (rc not 0 or 2) may have left no record or an
    # older run's: report() fails the cell either way.
    if rec["rc"] != 0 or res.get("errors") or "seconds" not in res:
        return {**out, "failed": True, "rc": rec["rc"], "errors": res.get("errors") if rec["rc"] in (0, 2) else None,
                "log": rec.get("log")}
    f = pf["facts"][ds]
    s = res["seconds"]
    blocks = res["runs"][0]["blocks"]
    out.update({
        "rc": rec["rc"], "errors": res["errors"], "reps": res["reps"], "triggers": res["triggers"],
        "chunk_rows": res["chunk_rows"], "backend_options": res["backend"], "git": res["env"]["git"],
        "m_seconds": {"median": s["median"], "min": s["min"], "max": s["max"], "all": s["seconds"],
                      "spread_pct": round(100 * (s["max"] - s["min"]) / s["median"], 1)},
        "peak_rss_mib": round(res["peak_rss_mb"]), "host_mib": PF.split_rss(res, f, prof, be),
        "block_hash": {b: v["output_fnv1a"] for b, v in blocks.items()},
        "block_median_seconds": {b: v["median"] for b, v in res["block_seconds"].items()},
        "index_build_seconds": res["runs"][0]["index_build_seconds"], "wall_seconds": rec["wall_seconds"],
        "idle": {k: rec[k] for k in ("cpu_busy_at_start", "idle", "waited_seconds")},
        "foreign_load": rec["foreign_load"], "memory_at_start": rec["memory_at_start"]})
    if "device_mib" in rec:
        gm = res["runs"][0].get("device_graph_bytes", 0) / PF.MIB
        d = rec["device_mib"]
        caught = d["peak"] >= d["context"] + gm  # a poll saw the graph resident: short runs end between polls
        out["device_mib"] = {**d, "graph": round(gm), "caught": caught,
                             "scratch_and_output": round(d["peak"] - d["context"] - gm) if caught else None}
    return out


def report(mode):
    cells, measure = PF.load(mode.cells_json, {}), PF.load(mode.measure_json, {})
    # A cell that found no idle window was not measured (rc None, no capacity limit): still missing.
    cells = {c: r for c, r in cells.items() if r.get("rc") is not None or "capacity" in r}
    pf = PF.load(PREFLIGHT, None)
    model = pf["training"]["model"]["fitted"]
    doc = {"experiment": "E17", "spec": "SPEC v3 §8, AC-V3-5, §5.1, §6.2", "smoke": mode.smoke, "machine": "local",
           "blz": BLZ, "reps": mode.reps, "triggers": mode.triggers, "threads": THREADS, "preflight": rel(PREFLIGHT),
           "datasets": {}, "cells": {}, "missing": [c for c in mode.cells if c not in cells]}
    for ds in sorted({c.split("/")[0] for c in mode.cells}, key=DATASETS.index):
        f = pf["facts"][ds]
        doc["datasets"][ds] = {"name": NAMES[ds], "facts": f, "invariants": invariants(ds),
                               "training_estimate": {p: PF.training_estimate(model, f, c) for p, c in FIT_COLUMNS.items()}}
    for c in mode.cells:
        if c in cells:
            doc["cells"][c] = cell_result(mode, c, cells[c], measure, pf)
    agree = {}
    for c, r in doc["cells"].items():
        if "block_hash" in r:
            agree.setdefault("/".join(c.split("/")[:2]), {})[r["backend"]] = r["block_hash"]
    doc["hashes_agree_across_backends"] = {k: len({json.dumps(h, sort_keys=True) for h in v.values()}) == 1
                                           for k, v in agree.items() if len(v) > 1}
    gits = {json.dumps(r["git"], sort_keys=True) for r in doc["cells"].values() if "git" in r}
    doc["git"] = [json.loads(g) for g in sorted(gits)]
    PF.save(mode.out, doc)
    mode.out.with_suffix(".md").write_text(markdown(doc))
    print(f"-> {rel(mode.out)}, {rel(mode.out.with_suffix('.md'))}")
    bad = [c for c, r in doc["cells"].items() if r.get("failed") or r.get("errors") or r.get("rc") not in (0, None)]
    bad += [k for k, v in doc["hashes_agree_across_backends"].items() if not v]
    bad += [ds for ds, d in doc["datasets"].items() if not d["invariants"]["prepared_under_descriptor"]
            or not all(v["equal"] for v in d["invariants"]["recounted"].values())]
    return 1 if bad or doc["missing"] else 0


def device_cell(r, split):
    d = r.get("device_mib")
    if not d:
        return "—"
    if not d["caught"]:
        return f"missed ({d['samples']} polls; the run ends before one sees the graph)"
    return f"{d['peak']} MiB" + (f" ({d['graph']}, {d['scratch_and_output']})" if split else "")


def markdown(doc):
    L = [f"# E17: datasets, M and memory per backend (SPEC v3 §8){' — smoke' if doc['smoke'] else ''}", "",
         f"`{doc['blz']} measure --reps {doc['reps']} --triggers {doc['triggers']}`, cpu and hybrid at {doc['threads']} "
         f"threads, default options, chunk 2^20 (D-81); commits {', '.join(g['sha'][:7] + (' (dirty)' if g['dirty'] else '') for g in doc['git']) or '—'}. "
         "M is §5.1's: median of the reps, min and max kept. Estimates are bench/p15_preflight.py's models.", "",
         "## Prepare invariants", "",
         "| Dataset | Descriptor hash in .blzg | Recounted (no test label) | Gated by prepare only | Prepare peak | Prepare time |",
         "|---|---|---|---|---:|---:|"]
    for ds, d in doc["datasets"].items():
        iv = d["invariants"]
        rc = iv["recounted"]
        ok = sum(v["equal"] for v in rc.values())
        L.append(f"| {d['name']} | {'equal' if iv['prepared_under_descriptor'] else 'DIFFERS'} | {ok} of {len(rc)} equal | "
                 f"{', '.join(iv['gated_by_prepare_only'])} | {iv['prepare_peak_rss_mib']} MiB | {iv['prepare_seconds']} s |")
    L += ["", "Prepare peaks and times are D-68's, recorded and not rerun.", "",
          "## M per cell", "",
          "| Dataset | Profile | Backend | M median | min | max | spread | Errors | Peak RSS | Index / rows / rest | "
          "Device peak (graph, scratch+out) | Foreign load mean / max | Idle at start |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|"]
    for c, r in doc["cells"].items():
        head = f"| {NAMES[r['dataset']]} | {r['profile']} | {r['backend']} |"
        if "capacity" in r:
            L.append(f"{head} capacity limit: {r['capacity']} | | | | | | | | | |")
            continue
        if r.get("failed"):
            L.append(f"{head} failed: rc {r['rc']}, errors {r['errors']}, see {r['log']} | | | | | | | | | |")
            continue
        m, h = r["m_seconds"], r["host_mib"]
        dev = device_cell(r, True)
        fl = r["foreign_load"]
        L.append(f"{head} {m['median']:.3f} s | {m['min']:.3f} | {m['max']:.3f} | {m['spread_pct']} % | {r['errors']} | "
                 f"{r['peak_rss_mib']} MiB | {h['index']} / {h['row_buffers']} / {h['aux_and_rest']} | {dev} | "
                 f"{fl.get('foreign_pct_mean', '—')} / {fl.get('foreign_pct_max', '—')} % | "
                 f"{'yes' if r['idle']['idle'] else 'no'} ({100 * r['idle']['cpu_busy_at_start']:.0f} % busy) |")
    for c in doc["missing"]:
        L.append(f"| {c} | not run | | | | | | | | | | | |")
    L += ["", "Row hashes per block agree across backends: " + (", ".join(
        f"{k} {'yes' if v else 'NO'}" for k, v in doc["hashes_agree_across_backends"].items()) or "no pair measured") + ".", "",
        "## §6.2 budgets: estimate against measured", "",
        "| Dataset | Profile | Backend | Host estimate (index / rows / rest / peak) | Host measured peak | "
        "Device estimate | Device measured |", "|---|---|---|---|---:|---:|---:|"]
    for c, r in doc["cells"].items():
        e = r["budget"]["host_estimate_mib"]
        de = r["budget"].get("device_estimate_mib", {}).get("peak", "—")
        L.append(f"| {NAMES[r['dataset']]} | {r['profile']} | {r['backend']} | {e['index']} / {e['row_buffers']} / "
                 f"{e['aux_and_rest']} / {e['peak']} MiB | {r.get('peak_rss_mib', '—')} | {de} | "
                 f"{device_cell(r, False)} |")
    L += ["", "## Training (estimate, never run here)", "",
          "| Dataset | Recipe | Columns | Fit rows | Validation rows | Peak RSS | Fit time |", "|---|---|---:|---:|---:|---:|---:|"]
    for ds, d in doc["datasets"].items():
        for p, e in d["training_estimate"].items():
            L.append(f"| {d['name']} | {p} | {e['columns']} | {e['fit_rows']:,} | {e['val_rows']:,} | "
                     f"{e['peak_rss_mib']:,} MiB | {e['fit_seconds']} s |")
    L += ["", "Inferred from the preflight's HI-Small fits (bytes per cell, seconds per fit cell).", ""]
    return "\n".join(L)


# ---- the freeze ----------------------------------------------------------------------------------

def freeze(mode):
    for p in (mode.frozen, mode.frozen.with_suffix(".md")):
        if p.exists():
            sys.exit(f"{rel(p)} exists: the baseline is frozen once (SPEC v3 §5.2, P15) and is never overwritten")
    if report(mode) != 0:
        sys.exit(f"E17's report does not pass (see {rel(mode.out)}): nothing frozen")
    doc = PF.load(mode.out, {})
    wl, excluded = {}, {}
    for cell, role in mode.workloads:
        r = doc["cells"].get(cell)
        if r is None:
            sys.exit(f"{cell}: no E17 record; run `bench/e17.py run {cell}` first")
        if "capacity" in r:
            excluded[cell] = {"role": role, "capacity": r["capacity"]}
            continue
        if not r.get("idle", {}).get("idle"):
            sys.exit(f"{cell}: measured without an idle window (SPEC v3 §2.5); rerun it before freezing")
        wl[cell] = {"role": role, "m_seconds": r["m_seconds"], "triggers": r["triggers"], "errors": r["errors"],
                    "block_hash": r["block_hash"], "peak_rss_mib": r["peak_rss_mib"], "backend_options": r["backend_options"],
                    "foreign_load": r["foreign_load"], "idle": r["idle"]}
    gits = {json.dumps(doc["cells"][c]["git"], sort_keys=True) for c in wl}
    if len(gits) != 1:
        sys.exit(f"the workloads were measured at different commits: {sorted(gits)}")
    git = json.loads(gits.pop())
    if git["dirty"] and not mode.smoke:
        sys.exit(f"the workloads were measured on a dirty tree at {git['sha']}: no commit describes the harness")
    chunks = {doc["cells"][c]["chunk_rows"] for c in wl}
    if len(chunks) != 1:
        sys.exit(f"the workloads were measured at different chunk sizes: {sorted(chunks)}")
    out = {"frozen": "SPEC v3 §5.2 clean baseline (P15): P16 starts here", "harness_commit": git["sha"],
           "harness_dirty": git["dirty"], "chunk_rows": chunks.pop(),
           "blz": BLZ, "reps": mode.reps, "threads": THREADS, "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "source": rel(mode.out), "workloads": wl, "capacity_exclusions": excluded,
           "rules": "M = median of the reps (§5.1). A change is kept only when its median beats the incumbent's by more "
                    "than the incumbent's min-to-max spread; a workload regresses when its median exceeds this "
                    "baseline's max by more than 5 %; a candidate that cannot complete a workload this baseline "
                    "completed has regressed on it."}
    PF.save(mode.frozen, out)
    L = [f"# Clean baseline M, frozen (SPEC v3 §5.2, P15){' — smoke' if mode.smoke else ''}", "",
         f"Harness commit {git['sha'][:7]}, chunk {out['chunk_rows']:,} triggers, `blz measure --reps {mode.reps}`, "
         f"{THREADS} threads on cpu. Frozen {out['frozen_at']} from `{out['source']}`; never overwritten.", "",
         "| Workload | Role | M median | min | max | Triggers | Peak RSS |", "|---|---|---:|---:|---:|---:|---:|"]
    for c, w in wl.items():
        m = w["m_seconds"]
        L.append(f"| {c} | {w['role']} | {m['median']:.3f} s | {m['min']:.3f} | {m['max']:.3f} | {w['triggers']['rows']:,} | "
                 f"{w['peak_rss_mib']} MiB |")
    for c, x in excluded.items():
        L.append(f"| {c} | {x['role']} | capacity limit: {x['capacity']} | | | | |")
    L += ["", out["rules"], ""]
    mode.frozen.with_suffix(".md").write_text("\n".join(L))
    print(f"-> {rel(mode.frozen)} (harness {git['sha'][:7]}, {len(wl)} workloads, {len(excluded)} capacity limits)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["run", "report", "freeze"])
    ap.add_argument("cells", nargs="*")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    mode = Mode(a.smoke)
    if a.phase == "run":
        return run(mode, a.cells)
    return report(mode) if a.phase == "report" else freeze(mode)


if __name__ == "__main__":
    sys.exit(main())

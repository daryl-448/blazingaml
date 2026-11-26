# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "xgboost>=3.4,<3.5", "pyyaml"]
# ///
"""The HI-Medium preflight (SPEC v3 §6.2, §2.5): per backend and profile, an estimate and then a
measurement of peak host RSS for index construction, auxiliary structures, the row chunks in flight
and training, and of peak device memory on the 1650S for the graph upload, scratch and output chunks.

  uv run --python 3.12 bench/p15_preflight.py run [STEP ...]   # the steps (default all), skipping done ones
  uv run --python 3.12 bench/p15_preflight.py report           # results/p15_preflight_hi_medium.{json,md}

Estimates come from exact size formulas over the .blzg (index and device graph bytes from m, n, the
self-loops and the distinct pairs, the row buffers from the chunk), and from HI-Small, LI-Small and
TransXion runs scaled by their counts. Recorded runs are reused, not run again: D-82's HI-Medium cpu
measure, U3a's LI-Small and TransXion cpu measures, D-81's HI-Small cpu and cuda measures, D-68's
prepare peaks. Each step is one `blz measure --reps 1` (nvidia-smi polled every 0.2 s for the device
peak of the cuda and hybrid runs) or one XGBoost fit in a child process. Before a HI-Medium step its
host and device estimates are compared with MemAvailable and the GPU's free memory; a step over
what is free is aborted and recorded, never truncated. A sampled step (`sample:<pct>`) times a
uniform sample of the triggers and extrapolates: a sweep block (fan, degree) walks every list entry
whatever the selection, so its time is kept as measured, every other block's is divided by the
sampled share. The oracle's samples also plan AC-V3-1's oracle runs (bench/p15_ac_v3_1_datasets.py).
Training is never run on HI-Medium: fits on HI-Small's fit rows, scored on validation, at two row
counts and two column counts, give bytes per cell, extrapolated to HI-Medium's fit and validation rows.
Runs go to build/p15_preflight/ (measure.json, steps.json, logs).
"""
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench" / "dg2"))
from blzg import Blzg  # noqa: E402

BLZ = "build/rel-cuda/blz"
WORK = ROOT / "build" / "p15_preflight"
STEPS_JSON = WORK / "steps.json"
MEASURE_JSON = WORK / "measure.json"
OUT = ROOT / "results" / "p15_preflight_hi_medium.json"
CHUNK = 1 << 20  # the harness chunk (D-81)
MIB = 2 ** 20
DATASETS = ["hi_small", "li_small", "transxion", "hi_medium"]
WC = {"hi_small": "clean_wc_hi_small", "li_small": "clean_wc_li_small", "transxion": "clean_wc_hi_small",
      "hi_medium": "clean_wc_hi_medium"}
# Recorded runs, reused (file, key path): blz measure results of the same harness.
RECORDED = {
    "hi_small/cpu/clean_default": ("results/p15_chunk_cost_local.json", ["cpu", "1048576", "round0"], "D-81"),
    "hi_small/cuda/clean_default": ("results/p15_chunk_cost_local.json", ["cuda", "1048576", "round0"], "D-81"),
    "li_small/cpu/clean_default": ("features/p15_smoke_measure.json", ["li_small"], "U3a smoke"),
    "transxion/cpu/clean_default": ("features/p15_smoke_measure.json", ["transxion"], "U3a smoke"),
    "hi_medium/cpu/clean_default": ("results/p15_measure_hi_medium_local.json", ["hi_medium", "clean_default", "cpu"],
                                    "D-82"),
}
PREPARE_PEAK_MIB = {"hi_small": 303, "li_small": 403, "hi_medium": 1643, "transxion": 145}  # D-68
MID = 15_000_000  # a HI-Medium range for cuda and hybrid under clean_wc: [MID, MID + 2^21)
# Steps: key -> (dataset, backend, profile, triggers). Sampled oracle runs plan AC-V3-1 and estimate
# the whole; HI-Small's oracle sample is checked against its recorded whole run (106.1 s, p15_chunks).
STEPS = {}
for ds, pct in (("hi_small", "1"), ("li_small", "1"), ("transxion", "2"), ("hi_medium", "0.2")):
    for prof in (["clean_default"] if ds == "hi_small" else ["clean_default", WC[ds]] +
                 (["clean_all"] if ds in ("li_small", "transxion") else [])):
        STEPS[f"{ds}/oracle/{prof}/sample"] = (ds, "oracle", prof, f"sample:{pct}:1")
STEPS["transxion/oracle/clean_e16/sample"] = ("transxion", "oracle", "clean_e16", "sample:2:1")  # E16's cost
STEPS["hi_small/cuda/clean_default/gpu"] = ("hi_small", "cuda", "clean_default", "all")  # device calibration
STEPS["hi_small/hybrid/clean_default/gpu"] = ("hi_small", "hybrid", "clean_default", "all")
STEPS["hi_medium/cpu/clean_wc_hi_medium/sample"] = ("hi_medium", "cpu", "clean_wc_hi_medium", "sample:1:1")
STEPS["hi_medium/cuda/clean_default"] = ("hi_medium", "cuda", "clean_default", "all")
STEPS["hi_medium/hybrid/clean_default"] = ("hi_medium", "hybrid", "clean_default", "all")
STEPS["hi_medium/cuda/clean_wc_hi_medium/range"] = ("hi_medium", "cuda", "clean_wc_hi_medium",
                                                    f"range:{MID}:{MID + 2 * CHUNK}")
STEPS["hi_medium/hybrid/clean_wc_hi_medium/range"] = ("hi_medium", "hybrid", "clean_wc_hi_medium",
                                                      f"range:{MID}:{MID + 2 * CHUNK}")
# Training fits on HI-Small (fit rows, scored on validation): clean_default's 33 columns and
# clean_all's, with the ladder's base (ids and S2's raw columns), at half and all of the fit rows.
FITS = {"fit/clean_default/half": ("features/p13/hi_small/clean_default_cpu", 0.5),
        "fit/clean_default/all": ("features/p13/hi_small/clean_default_cpu", 1.0),
        "fit/clean_all/all": ("features/p13/hi_small/clean_all_cpu", 1.0)}


def load(path, default):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def save(path, doc):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(doc, indent=1) + "\n")


def get(doc, keys):
    for k in keys:
        doc = doc[k]
    return doc


def mem_available_mib():
    return int(re.search(r"MemAvailable:\s+(\d+)", Path("/proc/meminfo").read_text()).group(1)) / 1024


def gpu_free_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.split(",")
    return {"total": int(out[0]), "used": int(out[1]), "free": int(out[2])}


# ---- exact sizes ---------------------------------------------------------------------------------

def facts(ds, cache):
    """m, n, self-loops, the index's list length L = m - self-loops (clean drops them, R1) and the
    distinct non-loop (u, v) pairs P (each direction's P-CSR has P entries)."""
    if ds in cache:
        return cache[ds]
    g = Blzg(str(ROOT / "data" / f"{ds}.blzg"))
    src, dst = np.asarray(g.src), np.asarray(g.dst)
    keep = src != dst
    key = (src[keep].astype(np.uint64) << np.uint64(32)) | dst[keep].astype(np.uint64)
    key.sort()
    pairs = int(1 + np.count_nonzero(key[1:] != key[:-1])) if key.size else 0
    f = {"m": g.m, "n": g.n, "self_loops": int((~keep).sum()), "L": int(keep.sum()), "P": pairs,
         "b_fv": g.b_fv, "b_test": g.b_test}
    cache[ds] = f
    return f


def index_bytes(f):
    """GraphIndex::bytes(): per direction a T-CSR (off n+1, nbr, ts, eid, prev_same over L) and a P-CSR
    (off n+1, nbr P, run P+1, run_ts and run_eid over L), 4 bytes each."""
    return 2 * 4 * ((f["n"] + 1) + 4 * f["L"] + (f["n"] + 1) + f["P"] + (f["P"] + 1) + 2 * f["L"])


def device_graph_bytes(f):
    """DeviceGraph: src, dst, ts and both directions' arrays, each rounded up to 256 bytes."""
    r = lambda c: (4 * c + 255) // 256 * 256  # noqa: E731
    m, n, L, P = f["m"], f["n"], f["L"], f["P"]
    return 3 * r(m) + 2 * (r(n + 1) + 4 * r(L) + r(n + 1) + r(P) + r(P + 1) + 2 * r(L))


def profile_cols(prof):
    import yaml
    p = yaml.safe_load((ROOT / "configs" / "profiles" / f"{prof}.yaml").read_text())
    cols = {b: (len(v["bins"]) if v.get("bins") else 1) for b, v in p["blocks"].items()}
    return cols, "vstats" in p


SWEEP = {"fan_in", "fan_out", "deg_in", "deg_out"}  # CPU sweep plans (D-79): mined once per selection


def row_buffer_bytes(prof, backend, rows):
    """The host row buffers (D-77, D-79): one slot of chunk x the widest block's cells, reused; on cpu
    and the hybrid's CPU side a sweep block's whole selection; vstats' 52 float64 per chunk row."""
    cols, vstats = profile_cols(prof)
    chunked = 4 * min(CHUNK, rows) * max(cols.values())
    sweep = 4 * rows * max([c for b, c in cols.items() if b in SWEEP] or [0]) if backend in ("cpu", "hybrid") else 0
    return max(chunked, sweep) + (8 * 52 * min(CHUNK, rows) if vstats else 0)


def used_column_bytes(f, prof):
    """The .blzg columns a run faults in (D-80): src, dst, ts, and Amount Received for vstats."""
    return f["m"] * (12 + (8 if profile_cols(prof)[1] else 0))


# ---- running a step ------------------------------------------------------------------------------

class GpuPoll(threading.Thread):
    """Every 0.2 s: this pid's device memory (nvidia-smi, MiB); the smallest nonzero sample is the
    CUDA context before the graph upload."""

    def __init__(self, pid):
        super().__init__(daemon=True)
        self.pid, self.samples, self.stop = pid, [], threading.Event()

    def run(self):
        while not self.stop.wait(0.2):
            out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True).stdout
            for ln in out.splitlines():
                p, used = (x.strip() for x in ln.split(","))
                if p == str(self.pid):
                    self.samples.append(int(used))


def run_measure(key, ds, backend, prof, triggers):
    WORK.mkdir(parents=True, exist_ok=True)
    cmd = [BLZ, "measure", "--graph", f"data/{ds}.blzg", "--profile", f"configs/profiles/{prof}.yaml", "--backend",
           backend, "--reps", "1", "--triggers", triggers, "--out", str(MEASURE_JSON.relative_to(ROOT)), "--name", key]
    log = WORK / (key.replace("/", "__") + ".log")
    t0 = time.time()
    with open(log, "w") as fh:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
        poll = GpuPoll(p.pid) if backend in ("cuda", "hybrid") else None
        if poll:
            poll.start()
        rc = p.wait()
        if poll:
            poll.stop.set()
            poll.join()
    rec = {"cmd": " ".join(cmd), "rc": rc, "wall_seconds": round(time.time() - t0, 1), "log": str(log.relative_to(ROOT))}
    if poll and poll.samples:
        rec["device_mib"] = {"peak": max(poll.samples), "context": min(s for s in poll.samples if s > 0),
                             "samples": len(poll.samples)}
    return rec


def run_fit(key):
    """One fit in a child process, so its VmHWM is its own."""
    cmd = ["uv", "run", "--quiet", "--python", "3.12", "bench/p15_preflight.py", "fit-child", key]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        return {"rc": r.returncode, "error": r.stderr[-2000:]}
    rec = json.loads(r.stdout.strip().splitlines()[-1])
    rec.update({"rc": 0, "wall_seconds": round(time.time() - t0, 1)})
    return rec


def fit_child(key):
    import xgboost as xgb
    import ladder as Lad
    d, share = FITS[key]
    data = Lad.Data(False)
    (fa, fb), (va, vb) = data.splits["fit"], data.splits["val"]
    fb = fa + int((fb - fa) * share)
    data.splits["fit"] = (fa, fb)
    feats = Lad.Features(data, d)
    cols = Lad.recipe_columns(data, {"ids": "v1", "raw": "S2", "family": None}, "xgb_only") + \
        [c for b in feats.blocks for c in feats.columns(b)]
    params, rounds, threshold, batch_rows = Lad.xgb_setup({"xgb": {}})
    Lad.reset_peak_rss()
    rss0 = Lad.own_peak_rss_mb()
    t0 = time.time()
    yf, yv = data.labels("fit")[: fb - fa], data.labels("val")
    dfit = xgb.QuantileDMatrix(Lad.Batches(cols, fa, fb, yf, batch_rows), max_bin=params["max_bin"],
                               nthread=params["nthread"])
    dval = xgb.QuantileDMatrix(Lad.Batches(cols, va, vb, yv, batch_rows), ref=dfit, nthread=params["nthread"])
    t_mat = time.time() - t0
    bst = xgb.train(params, dfit, num_boost_round=rounds)
    met = Lad.metrics(bst.predict(dval), yv, threshold)
    print(json.dumps({"features": d, "fit_rows": fb - fa, "val_rows": vb - va, "columns": len(cols),
                      "cells": (fb - fa + vb - va) * len(cols), "rss_before_mib": round(rss0),
                      "peak_rss_mib": round(Lad.own_peak_rss_mb()), "matrices_seconds": round(t_mat, 1),
                      "fit_seconds": round(time.time() - t0, 1), "val_f1_x100": round(met["f1_x100"], 2),
                      "batch_rows": batch_rows, "xgb": xgb.__version__}))


# ---- estimates -----------------------------------------------------------------------------------

def recorded_or_step(key, steps, measure):
    """A measure result for key: a recorded one (RECORDED) or this preflight's."""
    if key in RECORDED:
        f, path, why = RECORDED[key]
        return get(load(ROOT / f, {}), path), why
    if steps.get(key, {}).get("rc") == 0:  # blz measure --name is a JSON pointer path
        return get(measure, key.split("/")), "this preflight"
    return None, None


def split_rss(res, f, prof, backend):
    """Peak host RSS of a measure run, split: file-backed (used columns and the binary), index,
    row buffers, and the rest (auxiliary structures, thread stacks, the allocator, CUDA's host side)."""
    run = res["runs"][0]
    peak = res["peak_rss_mb"]
    file_ = run["rss"]["RssFile_mb"] if "rss" in run else used_column_bytes(f, prof) / MIB
    idx = run.get("index_bytes", index_bytes(f) if backend != "oracle" else 0) / MIB
    rows = run.get("row_buffer_bytes", row_buffer_bytes(prof, backend, res["triggers"]["rows"])) / MIB
    return {"peak": round(peak), "file_backed": round(file_), "index": round(idx), "row_buffers": round(rows),
            "aux_and_rest": round(peak - file_ - idx - rows)}


def whole_seconds(res):
    """A sampled run's time extrapolated to every trigger: sweep blocks as measured, the rest scaled."""
    run, t = res["runs"][0], res["triggers"]
    share = t["rows"] / max(1, t.get("m", 0) or 1)
    blocks = run["blocks"]
    mined = sum(b["seconds"] for b in blocks.values())
    sweep = sum(b["seconds"] for b in blocks.values() if b.get("whole"))
    other = mined - sweep
    build = run["index_build_seconds"] + run.get("upload_seconds", 0)
    return build + sweep + other / share, build, share


def estimate_host(backend, prof, f, calib, f_cal):
    """Peak host RSS on dataset f: file-backed columns + binary, the index (exact), the row buffers
    (exact), and the rest scaled from the calibration run by n (per-thread n-sized scratch: the
    distinct sweep's seen arrays, 12 x n x 4 bytes, D-82) with its fixed part kept."""
    file_ = used_column_bytes(f, prof) / MIB + calib["binary_mib"]
    idx = (index_bytes(f) if backend != "oracle" else 0) / MIB
    rows = row_buffer_bytes(prof, backend, f["m"]) / MIB
    rest = calib["fixed_mib"] + calib["per_node_bytes"] * f["n"] / MIB + calib["per_edge_bytes"] * f["m"] / MIB
    return {"file_backed": round(file_), "index": round(idx), "row_buffers": round(rows), "aux_and_rest": round(rest),
            "peak": round(file_ + idx + rows + rest)}


def training_base_mib(model, f):
    """The fitting process before the fit: Python, numpy and xgboost, and the per-row arrays the base
    columns hold (Multi-GNN's amounts, 2 x 8 bytes, and the account codes, 4 bytes a node), scaled
    from HI-Small's fits to dataset f."""
    hs = model["at"]
    return model["rss_before_mib"] + (16 * (f["m"] - hs["m"]) + 4 * (f["n"] - hs["n"])) / MIB


def training_estimate(model, f, cols):
    """One fit on dataset f's fit rows, scored on its validation rows, with `cols` columns: peak RSS
    (the base, then the batched QuantileDMatrix's bytes per cell) and fit time per fit cell."""
    fr, vr = f["b_fv"], f["b_test"] - f["b_fv"]
    cells = (fr + vr) * cols
    return {"columns": cols, "fit_rows": fr, "val_rows": vr, "cells": cells,
            "peak_rss_mib": round(training_base_mib(model, f) + model["fixed_mib"] + model["mib_per_million_cells"] * cells / 1e6),
            "fit_seconds": round(model["seconds_per_fit_cell"] * fr * cols)}


def fit_rest(points, per):
    """Least squares rest = fixed + a x over calibration points (n, m, rest MiB), x = n (per node:
    cpu's per-thread n-sized scratch) or m (per edge: the oracle's per-node edge lists)."""
    A = np.array([[1.0, (n if per == "node" else m) / MIB] for n, m, _ in points])
    y = np.array([r for _, _, r in points])
    x, *_ = np.linalg.lstsq(A, y, rcond=None)
    x = [x[0], x[1], 0.0] if per == "node" else [x[0], 0.0, x[1]]
    return {"fixed_mib": float(x[0]), "per_node_bytes": float(x[1]), "per_edge_bytes": float(x[2]),
            "points": [{"n": n, "m": m, "rest_mib": round(r)} for n, m, r in points],
            "residuals_mib": [round(float(r - (x[0] + x[1] * n / MIB + x[2] * m / MIB))) for n, m, r in points]}


def report():
    steps, measure = load(STEPS_JSON, {}), load(MEASURE_JSON, {})
    cache = steps.setdefault("_facts", {})
    F = {ds: facts(ds, cache) for ds in DATASETS}
    save(STEPS_JSON, steps)
    hm = F["hi_medium"]
    doc = {"experiment": "HI-Medium preflight", "spec": "SPEC v3 §6.2, §2.5", "git": git(), "chunk_rows": CHUNK,
           "facts": F, "memory_now": {"host_available_mib": round(mem_available_mib()), "gpu": gpu_free_mib()},
           "prepare_peak_mib": {"recorded": PREPARE_PEAK_MIB, "source": "D-68"}}
    # Exact sizes against the recorded index bytes.
    checks = {}
    for ds in DATASETS:
        res, why = recorded_or_step(f"{ds}/cpu/clean_default", steps, measure)
        if res and "index_bytes" in res["runs"][0]:
            checks[ds] = {"formula": index_bytes(F[ds]), "measured": res["runs"][0]["index_bytes"], "source": why}
    doc["index_bytes_check"] = checks
    # Host calibration per backend: the rest (peak - file - index - rows) against n and m.
    calib, splits = {}, {}
    for backend, keys in (("cpu", [f"{d}/cpu/clean_default" for d in ("hi_small", "li_small", "transxion")]),
                          ("oracle", [f"{d}/oracle/clean_default/sample" for d in ("hi_small", "li_small", "transxion")]),
                          ("cuda", ["hi_small/cuda/clean_default/gpu"]), ("hybrid", ["hi_small/hybrid/clean_default/gpu"])):
        pts, bins = [], []
        for k in keys:
            res, why = recorded_or_step(k, steps, measure)
            if res is None:
                continue
            ds = k.split("/")[0]
            s = split_rss(res, F[ds], "clean_default", backend)
            if backend == "oracle":  # a sample's row buffers are its chunk; the rest is the lists
                s["row_buffers"] = round(row_buffer_bytes("clean_default", "oracle", res["triggers"]["rows"]) / MIB)
                s["aux_and_rest"] = s["peak"] - s["file_backed"] - s["row_buffers"]
            splits[k] = {**s, "source": why}
            pts.append((F[ds]["n"], F[ds]["m"], s["aux_and_rest"]))
            bins.append(s["file_backed"] - used_column_bytes(F[ds], "clean_default") / MIB)
        if pts:
            calib[backend] = fit_rest(pts, "node" if backend == "cpu" else "edge") if backend in ("cpu", "oracle") else None
            if backend in ("cuda", "hybrid"):  # one point: cpu's model plus this backend's extra (CUDA's host side)
                extra = pts[0][2] - (calib["cpu"]["fixed_mib"] + calib["cpu"]["per_node_bytes"] * pts[0][0] / MIB)
                calib[backend] = {**{k: v for k, v in calib["cpu"].items() if k != "points"}, "fixed_mib":
                                  calib["cpu"]["fixed_mib"] + extra, "cuda_host_extra_mib": round(extra),
                                  "points": [{"n": pts[0][0], "m": pts[0][1], "rest_mib": round(pts[0][2])}]}
            calib[backend]["binary_mib"] = round(max(0.0, float(np.median(bins))))
    doc["host_calibration"] = {"runs": splits, "models": calib}
    # Device calibration: HI-Small on cuda and hybrid.
    dev_cal = {}
    for backend in ("cuda", "hybrid"):
        k = f"hi_small/{backend}/clean_default/gpu"
        st = steps.get(k, {})
        if st.get("rc") == 0 and "device_mib" in st:
            g = get(measure, k.split("/"))["runs"][0]["device_graph_bytes"] / MIB
            dev_cal[backend] = {"peak": st["device_mib"]["peak"], "context": st["device_mib"]["context"],
                                "graph": round(g), "graph_formula": round(device_graph_bytes(F["hi_small"]) / MIB),
                                "scratch_and_output": round(st["device_mib"]["peak"] - st["device_mib"]["context"] - g)}
    doc["device_calibration"] = dev_cal
    # HI-Medium: per backend and profile, the estimate and the measurement.
    rows = []
    for backend in ("cpu", "oracle", "cuda", "hybrid"):
        for prof in ("clean_default", "clean_wc_hi_medium"):
            row = {"backend": backend, "profile": prof}
            if backend in calib:
                row["host_estimate_mib"] = estimate_host(backend, prof, hm, calib[backend], None)
            if backend in dev_cal:
                d = dev_cal[backend]
                gm = device_graph_bytes(hm) / MIB
                row["device_estimate_mib"] = {"context": d["context"], "graph": round(gm),
                                              "scratch_and_output": d["scratch_and_output"],
                                              "peak": round(d["context"] + gm + d["scratch_and_output"])}
            cands = [f"hi_medium/{backend}/{prof}{x}" for x in ("", "/sample", "/range")]
            key = next((k for k in cands if recorded_or_step(k, steps, measure)[0]), None)
            st = steps.get(key or "", {})
            aborted = [steps[k]["aborted"] for k in cands if steps.get(k, {}).get("aborted")]
            if key is None and aborted:
                row["measured"] = {"aborted": aborted[0]}
            elif key:
                res, why = recorded_or_step(key, steps, measure)
                s = split_rss(res, hm, prof, backend)
                t = res["triggers"]
                m = {"step": key, "source": why, "triggers": t["spec"], "trigger_rows": t["rows"], "host_mib": s,
                     "errors": res["errors"], "seconds": round(res["runs"][0]["seconds"], 2)}
                if t["rows"] < hm["m"]:
                    if t.get("sampled") or backend in ("cpu", "oracle"):
                        res2 = dict(res)
                        res2["triggers"] = {**t, "m": hm["m"]}
                        w, b, share = whole_seconds(res2)
                        m["whole_seconds_inferred"] = round(w, 1)
                        m["build_seconds"] = round(b, 2)
                    else:
                        run = res["runs"][0]
                        build = run["index_build_seconds"] + run.get("upload_seconds", 0)
                        m["whole_seconds_inferred"] = round(build + (run["seconds"] - build) * hm["m"] / t["rows"], 1)
                        m["build_seconds"] = round(build, 2)
                    # A whole run's row buffers: a sweep block holds the whole selection.
                    m["host_whole_peak_inferred_mib"] = round(
                        s["peak"] - s["row_buffers"] + row_buffer_bytes(prof, backend, hm["m"]) / MIB)
                if "device_mib" in st:
                    m["device_mib"] = {**st["device_mib"], "graph": round(res["runs"][0]["device_graph_bytes"] / MIB),
                                       "graph_formula": round(device_graph_bytes(hm) / MIB)}
                    m["device_mib"]["scratch_and_output"] = m["device_mib"]["peak"] - m["device_mib"]["context"] - \
                        m["device_mib"]["graph"]
                row["measured"] = m
            rows.append(row)
    doc["hi_medium"] = rows
    # Oracle timing for AC-V3-1's plan: per dataset and profile, the sample and the whole estimate.
    orc = {}
    for k, (ds, be, prof, trig) in STEPS.items():
        if be != "oracle":
            continue
        res, why = recorded_or_step(k, steps, measure)
        if res is None:
            continue
        r2 = dict(res)
        r2["triggers"] = {**res["triggers"], "m": F[ds]["m"]}
        w, b, share = whole_seconds(r2)
        orc[f"{ds}/{prof}"] = {"sample": trig, "sample_rows": res["triggers"]["rows"], "seconds": round(res["runs"][0]["seconds"], 2),
                               "build_seconds": round(b, 2), "per_trigger_us": round((res["runs"][0]["seconds"] - b) /
                                                                                    res["triggers"]["rows"] * 1e6, 2),
                               "whole_seconds_inferred": round(w, 1), "peak_rss_mib": round(res["peak_rss_mb"]),
                               "errors": res["errors"], "error": res["runs"][0].get("error")}
    if "hi_small/clean_default" in orc:
        orc["hi_small/clean_default"]["whole_seconds_recorded"] = {"seconds": 106.1, "source":
                                                                   "results/p15_chunks_local.json (blz mine, chunk 1,000,003)"}
    doc["oracle_timing"] = orc
    # Training: bytes per cell and seconds per fit row from HI-Small's fits, to HI-Medium.
    fits = {k: steps[k] for k in FITS if steps.get(k, {}).get("rc") == 0}
    train = {"fits": fits}
    if len(fits) >= 2:
        pts = [(v["cells"], v["peak_rss_mib"] - v["rss_before_mib"]) for v in fits.values()]
        A = np.array([[1.0, c / 1e6] for c, _ in pts])
        (a, b), *_ = np.linalg.lstsq(A, np.array([r for _, r in pts]), rcond=None)
        sec = [(v["fit_rows"] * v["columns"], v["fit_seconds"]) for v in fits.values()]
        spc = float(np.median([s / c for c, s in sec]))
        model = {"rss_before_mib": float(np.median([v["rss_before_mib"] for v in fits.values()])), "fixed_mib": float(a),
                 "mib_per_million_cells": float(b), "seconds_per_fit_cell": spc, "at": F["hi_small"]}
        est = {}
        for name, d in (("clean_default", FITS["fit/clean_default/all"][0]), ("clean_all", FITS["fit/clean_all/all"][0])):
            est[name] = training_estimate(model, hm, next(v["columns"] for k, v in fits.items() if v["features"] == d))
        base = training_base_mib(model, hm)
        train["model"] = {"rss_mib": f"{round(base)} (Python, numpy, xgboost and the base columns' arrays at HI-Medium's m) "
                                     f"+ {a:.0f} + {b:.3f} per million cells",
                          "bytes_per_cell": round(b * MIB / 1e6, 2), "residuals_mib": [round(float(r - a - b * c / 1e6)) for c, r in pts],
                          "seconds_per_fit_cell_ns": round(spc * 1e9, 2), "fitted": model}
        train["hi_medium_estimate"] = est
    doc["training"] = train
    doc["verdict"] = verdict(doc)
    save(OUT, doc)
    OUT.with_suffix(".md").write_text(markdown(doc))
    print(f"wrote {OUT.relative_to(ROOT)} and .md: {doc['verdict']['summary']}")


def verdict(doc):
    free, out = doc["memory_now"]["host_available_mib"], {}
    for r in doc["hi_medium"]:
        k = f"{r['backend']}/{r['profile']}"
        m = r.get("measured", {})
        host = m.get("host_whole_peak_inferred_mib", m.get("host_mib", {}).get("peak"))
        dev = m.get("device_mib", {}).get("peak")
        if "aborted" in m:
            out[k] = f"aborted: {m['aborted']}"
        elif host is None:
            out[k] = "not measured"
        else:
            ok = host < min(free, 3 * 1024) and (dev is None or dev < 4096)
            out[k] = (f"fits: host {host} MiB" + (f", device {dev} MiB" if dev else "") +
                      ("" if ok else " (over the 3 GiB per-process rule or the device)"))
    t = doc["training"].get("hi_medium_estimate", {})
    for name, e in t.items():
        out[f"training/{name}"] = (f"estimate {e['peak_rss_mib']} MiB, {e['fit_seconds']} s (inferred): " +
                                   ("fits under 3 GiB" if e["peak_rss_mib"] < 3 * 1024 else
                                    f"over §2.5's 3 GiB but under the {free} MiB free now, so it fits local while that "
                                    "much is free; the user chooses at P17 (§6.2)" if e["peak_rss_mib"] < free else
                                    "over what is free: needs the user's choice (§6.2)"))
    out["summary"] = "; ".join(f"{k} {v.split(':')[0]}" for k, v in out.items())
    return out


def markdown(doc):
    L = ["# HI-Medium preflight (SPEC v3 §6.2)", "",
         f"git {doc['git']}, chunk {doc['chunk_rows']:,} triggers. HI-Medium: m = {doc['facts']['hi_medium']['m']:,}, "
         f"n = {doc['facts']['hi_medium']['n']:,}, {doc['facts']['hi_medium']['P']:,} distinct pairs. Free now: "
         f"{doc['memory_now']['host_available_mib']:,} MiB host, {doc['memory_now']['gpu']['free']:,} MiB on the 1650S. "
         "MiB throughout; `inferred` marks extrapolations. Generated by `bench/p15_preflight.py report`.", "",
         f"Recorded, not rerun: `blz prepare` peaks (D-68) {', '.join(f'{k} {v:,}' for k, v in doc['prepare_peak_mib']['recorded'].items())} "
         "MiB; `blz gt-join` on HI-Medium 594 MiB (D-73). The device context is the smallest nvidia-smi sample of the "
         "process (0.2 s apart), so it is approximate, and so is the scratch, which is the peak less it and the graph.", "",
         "## Estimate against measured, per backend and profile", "",
         "| Backend | Profile | Step | Estimate (MiB) | Measured (MiB) | Run | Seconds |", "|---|---|---|---:|---:|---|---|"]
    for r in doc["hi_medium"]:
        he, de, m = r.get("host_estimate_mib", {}), r.get("device_estimate_mib", {}), r.get("measured", {})
        hm, dm = m.get("host_mib", {}), m.get("device_mib", {})
        run = m.get("aborted") and f"aborted: {m['aborted']}" or (f"{m.get('triggers', '-')} ({m.get('source', '-')})" if m else "-")
        secs = "-" if not m or "seconds" not in m else (f"{m['seconds']}" + (f" (whole {m['whole_seconds_inferred']}, inferred)"
                                                                               if "whole_seconds_inferred" in m else ""))
        for step, e, v in (("host: index", he.get("index"), hm.get("index")),
                           ("host: auxiliary and rest", he.get("aux_and_rest"), hm.get("aux_and_rest")),
                           ("host: row buffers", he.get("row_buffers"), hm.get("row_buffers")),
                           ("host: file-backed", he.get("file_backed"), hm.get("file_backed")),
                           ("host: peak", he.get("peak"), hm.get("peak")),
                           ("host: peak, whole run", he.get("peak"), m.get("host_whole_peak_inferred_mib")),
                           ("device: context", de.get("context"), dm.get("context")),
                           ("device: graph upload", de.get("graph"), dm.get("graph")),
                           ("device: scratch and output chunks", de.get("scratch_and_output"), dm.get("scratch_and_output")),
                           ("device: peak", de.get("peak"), dm.get("peak"))):
            if e is None and v is None:
                continue
            L.append(f"| {r['backend']} | {r['profile']} | {step} | {e if e is not None else '-'} | "
                     f"{v if v is not None else '-'} | {run} | {secs} |")
    t = doc["training"]
    L += ["", "## Training (batched QuantileDMatrix, fit rows, scored on validation)", ""]
    if "model" in t:
        L.append(f"Model from HI-Small's fits: RSS = {t['model']['rss_mib']} ({t['model']['bytes_per_cell']} B per cell; "
                 f"residuals {t['model']['residuals_mib']} MiB). Not run on HI-Medium.")
        L += ["", "| Recipe | Columns | Fit rows | Validation rows | Peak RSS (MiB, inferred) | Fit seconds (inferred) |",
              "|---|---:|---:|---:|---:|---:|"]
        for n, e in t["hi_medium_estimate"].items():
            L.append(f"| {n} | {e['columns']} | {e['fit_rows']:,} | {e['val_rows']:,} | {e['peak_rss_mib']:,} | {e['fit_seconds']:,} |")
        L += ["", "| HI-Small fit | Columns | Fit rows | Peak RSS | Fit seconds | Validation F1 x100 |", "|---|---:|---:|---:|---:|---:|"]
        for k, v in t["fits"].items():
            L.append(f"| {k} | {v['columns']} | {v['fit_rows']:,} | {v['peak_rss_mib']:,} | {v['fit_seconds']} | {v['val_f1_x100']} |")
    L += ["", "## Oracle timing (12 threads, sampled triggers)", "",
          "| Dataset / profile | Sample | Sampled rows | Seconds | Build s | µs per trigger | Whole (inferred) | Peak RSS |",
          "|---|---|---:|---:|---:|---:|---:|---:|"]
    for k, v in doc["oracle_timing"].items():
        rec = v.get("whole_seconds_recorded")
        L.append(f"| {k} | {v['sample']} | {v['sample_rows']:,} | {v['seconds']} | {v['build_seconds']} | {v['per_trigger_us']} | "
                 f"{v['whole_seconds_inferred']}" + (f" (recorded {rec['seconds']})" if rec else "") + f" | {v['peak_rss_mib']} |")
    L += ["", "## Verdict", ""] + [f"- {k}: {v}" for k, v in doc["verdict"].items() if k != "summary"]
    return "\n".join(L) + "\n"


def git():
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True,
                           text=True).stdout.strip()
    return sha + ("-dirty" if dirty else "")


def gate(key, ds, backend, prof, steps, measure):
    """A HI-Medium step runs only when its estimates fit what is free now."""
    if ds != "hi_medium":
        return None
    report_doc = None
    try:
        report()
        report_doc = load(OUT, {})
    except Exception as e:  # a missing calibration: say so and do not run blind
        return f"no estimate ({e})"
    row = next((r for r in report_doc["hi_medium"] if r["backend"] == backend and r["profile"] == prof), {})
    host = row.get("host_estimate_mib", {}).get("peak")
    dev = row.get("device_estimate_mib", {}).get("peak")
    free, gfree = mem_available_mib(), gpu_free_mib()["free"]
    if host is None:
        return "no host estimate"
    if host > free:
        return f"host estimate {host} MiB > {free:.0f} MiB available"
    if backend in ("cuda", "hybrid") and (dev is None or dev > gfree):
        return f"device estimate {dev} MiB > {gfree} MiB free"
    return None


def run(names):
    steps, measure = load(STEPS_JSON, {}), load(MEASURE_JSON, {})
    todo = names or list(STEPS) + list(FITS)
    for key in todo:
        if steps.get(key, {}).get("rc") == 0:
            print(f"{key}: done", flush=True)
            continue
        if key in FITS:
            print(f"{key}: fitting", flush=True)
            steps[key] = run_fit(key)
        else:
            ds, backend, prof, trig = STEPS[key]
            why = gate(key, ds, backend, prof, steps, measure)
            steps = load(STEPS_JSON, {})
            if why:
                steps[key] = {"rc": None, "aborted": why}
                print(f"{key}: aborted, {why}", flush=True)
                save(STEPS_JSON, steps)
                continue
            print(f"{key}: blz measure {backend} {prof} {trig} (free {mem_available_mib():.0f} MiB)", flush=True)
            steps[key] = run_measure(key, ds, backend, prof, trig)
            measure = load(MEASURE_JSON, {})
        save(STEPS_JSON, steps)
        r = steps[key]
        print(f"{key}: rc {r.get('rc')}, {r.get('wall_seconds')} s" +
              (f", device peak {r['device_mib']['peak']} MiB" if "device_mib" in r else ""), flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("run", "report", "fit-child"):
        sys.exit(__doc__)
    if sys.argv[1] == "fit-child":
        fit_child(sys.argv[2])
    elif sys.argv[1] == "run":
        unknown = [s for s in sys.argv[2:] if s not in STEPS and s not in FITS]
        if unknown:
            sys.exit(f"unknown steps {unknown}; steps: {', '.join(list(STEPS) + list(FITS))}")
        run(sys.argv[2:])
    else:
        report()

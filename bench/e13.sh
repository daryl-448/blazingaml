#!/usr/bin/env bash
# E13 (SPEC v2 §8 S6): tcyc on HI-Small, hybrid CPU/GPU against GPU-only and CPU-only, with and
# without reachability pruning, at three windows: v1_default (24 h), dg2_wc (89 h, the frozen
# recipe's W-C) and dg2_wb (20 days). -> results/e13_hybrid_<host>.{json,md}
#
# Per profile and GPU light/heavy threshold H: `blz bench --backend hybrid` (5 timed runs after a
# warm-up, every run's rows equal) over the CPU thresholds below, pruning off and on; the CPU runs
# the sorted schedule (heaviest first, chunk 16), as the hybrid's few heavy triggers need. Run it on a
# quiet machine: other jobs on the shared desktop perturb the timings.
set -euo pipefail
cd "$(dirname "$0")/.."
BLZ=${BLZ:-build/rel-cuda/blz}
HOST=${HOST:-local}
OUT=results/e13_hybrid_$HOST.json
CPU_THRESHOLDS=4,8,16,32,64,128,256,all

uptime
for prof in v1_default dg2_wc dg2_wb; do
  for h in 32 128 1024; do
    "$BLZ" bench --backend hybrid --graph data/hi_small.blzg --profile "configs/profiles/$prof.yaml" --blocks tcyc \
      --reps 5 --schedule sorted --heavy "$h" --cpu-threshold "$CPU_THRESHOLDS" --prune off,on --out "$OUT" --name "runs/$prof/H$h"
    uptime
  done
done
python3 bench/e13_report.py "$OUT"

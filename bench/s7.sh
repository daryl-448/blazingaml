#!/usr/bin/env bash
# E12 delta for S7 (SPEC v2 §8 S7): packed D2H and multi-pattern fusion on HI-Small, against the
# plain CUDA backend, for every block of the six E12 profiles. -> results/e12_s7_<host>.{json,md}
#
# Per profile and variant (base, pack, fuse, fuse+pack): `blz bench --backend cuda`, H swept as in
# E12 (0, 8, 16, 32, 64, 128, 256, 1024, all), 5 timed runs after a warm-up, every run's rows equal.
# OMP_WAIT_POLICY=passive: under libgomp's default policy the host widening of packed rows shows
# run-to-run outliers (sg: 10-44 ms for 11 ms of work); one v1_default pack run keeps the default
# to record them. Run it on a quiet machine: other jobs on the shared desktop perturb the timings.
set -euo pipefail
cd "$(dirname "$0")/.."
BLZ=${BLZ:-build/rel-cuda/blz}
HOST=${HOST:-local}
OUT=results/e12_s7_$HOST.json
HEAVY=0,8,16,32,64,128,256,1024,all

uptime
for prof in v1_default sg_ordered_v1w sg_ordered_tsvis dg2_v1 dg2_wc dg2_wb; do
  for variant in base pack fuse fuse_pack; do
    flags=()
    [[ $variant == *pack* ]] && flags+=(--pack)
    [[ $variant == fuse* ]] && flags+=(--fuse)
    OMP_WAIT_POLICY=passive "$BLZ" bench --backend cuda --graph data/hi_small.blzg --profile "configs/profiles/$prof.yaml" \
      --heavy "$HEAVY" --reps 5 "${flags[@]}" --out "$OUT" --name "runs/$prof/$variant"
  done
  uptime
done
"$BLZ" bench --backend cuda --graph data/hi_small.blzg --profile configs/profiles/v1_default.yaml \
  --heavy 32 --reps 10 --pack --out "$OUT" --name "default_wait_policy/v1_default/pack"
python3 bench/s7_report.py "$OUT"

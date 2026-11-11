#!/bin/bash
# AC1 for the generated CUDA backend (SPEC §1.4, §12.2 E1): HI-Small and 10,000 fuzz graphs, bit
# for bit against the oracle. Merges into results/e1_parity.json:
#   hi_small_cuda_vs_oracle/<run>/<block>   blz compare reports
#   fuzz_cuda_vs_oracle                     blz fuzz-diff summary (thresholds 0, 4 and all)
# Runs: the default config at the default light/heavy threshold (mixed kernels), and the parity
# config twice, with every trigger on the heavy kernel (H = 0) and on the light one (H = all).
# Needs the oracle features of P1/P2 (features/oracle_default, features/oracle_parity) and a
# rel-cuda build. Exit status: 0 if everything matches.
set -u
cd "$(dirname "$0")/.." || exit 1
BLZ=${BLZ:-build/rel-cuda/blz}
G=data/hi_small.blzg
R=${R:-results/e1_parity.json}
rc=0
run() {  # run name, config name, heavy threshold
  local run=$1 name=$2 heavy=$3
  "$BLZ" mine --graph "$G" --config "configs/features_$name.yaml" --backend cuda --heavy-threshold "$heavy" \
    --out "features/cuda_$run" || return 1
  for b in fan_in fan_out deg_in deg_out sg tcyc; do
    "$BLZ" compare "features/oracle_$name/$b.npy" "features/cuda_$run/$b.npy" \
      --report "$R" --name "hi_small_cuda_vs_oracle/$run/$b" || rc=1
  done
}
run default default 32 || rc=1
run parity_h0 parity 0 || rc=1
run parity_light parity 4294967295 || rc=1
"$BLZ" fuzz-diff --backend cuda --seeds 0:10000 --report "$R" --name fuzz_cuda_vs_oracle || rc=1
exit $rc

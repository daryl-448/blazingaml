#!/bin/bash
# AC1 for the generated CPU backend (SPEC §1.4, §12.2 E1): HI-Small and 10,000 fuzz graphs, bit
# for bit against the oracle. Merges into results/e1_parity.json:
#   hi_small_cpu_vs_oracle/<config>/<block>   blz compare reports
#   fuzz_cpu_vs_oracle                        blz fuzz-diff summary (both schedules)
# The default config runs at 12 threads with the dynamic schedule, the parity config at 1 thread
# with the sorted one, so the comparison also covers both schedules and both thread extremes.
# Needs the oracle features of P1/P2 (features/oracle_default, features/oracle_parity).
# Exit status: 0 if everything matches.
set -u
cd "$(dirname "$0")/.." || exit 1
BLZ=${BLZ:-build/rel/blz}
G=data/hi_small.blzg
R=results/e1_parity.json
rc=0
run() {  # config name, threads, schedule
  local name=$1 threads=$2 sched=$3
  "$BLZ" mine --graph "$G" --config "configs/features_$name.yaml" --backend cpu --threads "$threads" \
    --schedule "$sched" --out "features/cpu_$name" || return 1
  for b in fan_in fan_out deg_in deg_out sg tcyc; do
    "$BLZ" compare "features/oracle_$name/$b.npy" "features/cpu_$name/$b.npy" \
      --report "$R" --name "hi_small_cpu_vs_oracle/$name/$b" || rc=1
  done
}
run default 12 dynamic || rc=1
run parity 1 sorted || rc=1
"$BLZ" fuzz-diff --seeds 0:10000 --report "$R" --name fuzz_cpu_vs_oracle || rc=1
exit $rc

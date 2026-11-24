#!/bin/bash
# AC-V3-1's fuzz part (SPEC v3 AC-V3-1, R11, §4.4): clean_all against the clean oracle on the cpu,
# cuda and hybrid backends, dense and sparse settings, then the metamorphic suite on every backend.
# One run at a time, from the repo root. R11: a gate counts only the graphs that every compared
# side completed, so each block needs completed_graphs_by_block >= 10,000. Dense lcyc reaches the
# 1e6 work limit on about 4.7 % of graphs (14 of 300 in U2b's smoke), hence 11,000 dense seeds.
# Writes $OUT.json (every run's summary) and $OUT/<run>.log; prints and returns the verdict.
# Usage: bench/ac_v3_1.sh [fd|mt|all]   (env: BLZ, OUT, N_DENSE, N_SPARSE, WL)
cd "$(dirname "$0")/.." || exit 1
B=${BLZ:-build/rel-cuda/blz}
P=configs/profiles/clean_all.yaml
N_DENSE=${N_DENSE:-11000}
N_SPARSE=${N_SPARSE:-10000}
WL=${WL:-1000000}
D=${OUT:-results/ac_v3_1}
R=$D.json
what=${1:-all}
mkdir -p "$D"
NAMES=()
run() {  # name, command...
  local name=$1 t0=$SECONDS rc
  shift
  NAMES+=("$name")
  "$@" --profile $P --report "$R" --name "$name" > "$D/$name.log" 2>&1
  rc=$?
  echo "$name rc=$rc $((SECONDS - t0)) s: $(head -1 "$D/$name.log")"
}
if [[ $what == fd || $what == all ]]; then
  for be in cpu cuda hybrid; do
    run fd_${be}_dense  "$B" fuzz-diff --backend $be --seeds 0:$N_DENSE --work-limit $WL
    run fd_${be}_sparse "$B" fuzz-diff --backend $be --seeds 0:$N_SPARSE --setting sparse
  done
fi
if [[ $what == mt || $what == all ]]; then
  for be in oracle cpu cuda hybrid; do
    run mt_${be}_dense  "$B" metamorphic --backend $be --seeds 0:$N_DENSE --work-limit $WL
    run mt_${be}_sparse "$B" metamorphic --backend $be --seeds 0:$N_SPARSE --setting sparse
  done
fi
# The verdict: every run reported, 0 mismatching rows, and 10,000 completed graphs in every block.
python3 - "$R" "${NAMES[@]}" <<'EOF'
import json, os, sys
doc = json.load(open(sys.argv[1])) if os.path.exists(sys.argv[1]) else {}
ok = True
for name in sys.argv[2:]:
    if name not in doc:
        ok = False
        print(f"{name}: FAIL  no summary (see its log)")
        continue
    r = doc[name]
    mm = sum(r['mismatched_rows_by_block'].values()) if 'mismatched_rows_by_block' in r else \
        sum(t['mismatched_rows'] for t in r['tests'].values())
    low = min(r['completed_graphs_by_block'].values())
    good = mm == 0 and low >= 10000
    ok &= good
    print(f"{name}: {'PASS' if good else 'FAIL'}  mismatched rows {mm}, fewest completed graphs {low}")
sys.exit(0 if ok else 1)
EOF

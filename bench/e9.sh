#!/usr/bin/env bash
# E9 (SPEC v2 §5): ground-truth coverage of v1_default and the frozen DG-2 recipe ->
# results/e9_coverage.{json,md}. Intermediate files go to features/e9/.
#
# - bench/e9/predict.py refits the validation-selected and E8 models (about 10 min, <= 2.4 GB) and
#   writes their per-edge predictions; it is skipped when they exist.
# - tools/gfp/coverage_gfp.py: GFP G1 eligibility of the frozen recipe's blocks (D-33).
# - blz coverage, once per profile; bench/e9/report.py merges and renders.
#
# The frozen recipe (configs/dg2_frozen.yaml): GFP G1 blocks in features/dg2_R6_G1, at the
# windows and bins of configs/profiles/dg2_wc.yaml (W-C); report.py checks the blocks' sha256.
set -euo pipefail
cd "$(dirname "$0")/.."
BLZ=${BLZ:-build/rel/blz}
GRAPH=data/hi_small.blzg
GT=results/gt_join.json
D=features/e9

if [[ ! -f $D/pred_v1_default.json || ! -f $D/pred_frozen.json ]]; then
  uv run --python 3.12 bench/e9/predict.py --out "$D"
fi
(cd tools/gfp && .venv/bin/python coverage_gfp.py --graph "../../$GRAPH" --gt "../../$GT" \
   --features ../../features/dg2_R6_G1 --out "../../$D/gfp_eligibility_G1.json")
"$BLZ" coverage --graph "$GRAPH" --gt "$GT" --profile configs/profiles/v1_default.yaml \
  --features features/p7_cpu_default --pred "$D/pred_v1_default.npy" --out "$D/coverage_v1_default.json"
"$BLZ" coverage --graph "$GRAPH" --gt "$GT" --profile configs/profiles/dg2_wc.yaml \
  --features features/dg2_R6_G1 --pred "$D/pred_frozen.npy" --eligibility "$D/gfp_eligibility_G1.json" \
  --out "$D/coverage_frozen.json"
python3 bench/e9/report.py --dir "$D"

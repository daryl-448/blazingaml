#!/usr/bin/env bash
# E2 + E3 (SPEC §12.2): mine the default features with the oracle, train the five Table 2 sets
# with configs/xgb_default.yaml, and render results/e2_table2.md.
set -euo pipefail
cd "$(dirname "$0")/.."
BLZ=${BLZ:-build/rel/blz}
GRAPH=data/hi_small.blzg
FEATS=features/oracle_default
OUT=results/e2_table2.json

"$BLZ" mine --graph "$GRAPH" --config configs/features_default.yaml --out "$FEATS"
rm -f "$OUT"
for s in xgb_only fan fan_deg fan_deg_cyc all; do
  "$BLZ" train --graph "$GRAPH" --features "$FEATS" --set "$s" --score-test --out "$OUT"
done
python3 bench/e2_table2.py "$OUT" > results/e2_table2.md

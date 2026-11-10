#!/bin/bash
# E1 / AC2 on HI-Small (SPEC §12.2): oracle (configs/features_parity.yaml) vs GFP stream mode
# (run_parity_hi_small.sh). Merges one entry per block into results/e1_parity.json.
# Exit status: 0 if every block matches.
set -u
cd "$(dirname "$0")/../.." || exit 1
BLZ=${BLZ:-build/rel/blz}
ORACLE=${ORACLE:-features/oracle_parity}
GFP=${GFP:-data/gfp}
rc=0
for b in fan_in:86400 fan_out:86400 deg_in:86400 deg_out:86400 sg:21600 tcyc:86400; do
  blk=${b%%:*}; tw=${b##*:}
  "$BLZ" compare "$ORACLE/$blk.npy" "$GFP/$blk.npy" --graph data/hi_small.blzg --tw "$tw" \
    --report results/e1_parity.json --name "hi_small_oracle_vs_gfp/$blk" || rc=1
done
exit $rc

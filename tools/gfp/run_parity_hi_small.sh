#!/bin/bash
# HI-Small GFP parity runs (SPEC §11.1), one block at a time, into data/gfp/.
set -u
cd "$(dirname "$0")" || exit 1
mkdir -p ../../data/gfp/logs
rc=0
for b in fan:86400 degree:86400 sg:21600 tcyc:86400; do
  blk=${b%%:*}; tw=${b##*:}
  /usr/bin/time -v uv run parity.py --edges ../../data/edges.npy --block "$blk" --tw "$tw" \
    --bins 2..33 --out ../../data/gfp/ > "../../data/gfp/logs/$blk.log" 2>&1 \
    || { echo "FAILED $blk" >> ../../data/gfp/logs/status; rc=1; }
done
echo done >> ../../data/gfp/logs/status
exit $rc

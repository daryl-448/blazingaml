#!/bin/bash
# P5 runbook for the JarvisLabs A30 (SPEC §16 P5). From the repo root on `local`:
#   1. creates an on-demand A30 container (or uses --id);
#   2. uploads the working tree (tracked and untracked non-ignored files, without .git), the
#      binary cache and edges.npy, and the commit it was taken from (BLZ_GIT_SHA, BLZ_GIT_DIRTY);
#   3. runs bench/a30_remote.sh there with `jl run` and waits for it, checking every minute;
#   4. downloads results/{e5_gpu_a30,e4_cpu_a30}.json and the A30 fuzz check (merged into
#      results/e1_parity.json as fuzz_cuda_vs_oracle_a30), plus a30_env.txt and the run log;
#   5. stops the instance: pause (default) or destroy (--destroy). On a failure it pauses too,
#      unless --keep, so a broken run does not keep billing the GPU.
# Then render the reports: bench/e5_report.py results/e5_gpu_a30.json (and e4_report.py).
#
# Usage: bench/a30.sh [--id MACHINE_ID] [--keep] [--destroy]
# Needs: jl (logged in, an SSH key added), data/hi_small.blzg and data/edges.npy.
set -euo pipefail
cd "$(dirname "$0")/.."
ID="" KEEP=0 DESTROY=0
while [ $# -gt 0 ]; do
  case $1 in
    --id) ID=$2; shift 2 ;;
    --keep) KEEP=1; shift ;;
    --destroy) DESTROY=1; shift ;;
    *) echo "usage: $0 [--id MACHINE_ID] [--keep] [--destroy]" >&2; exit 2 ;;
  esac
done
for f in data/hi_small.blzg data/edges.npy; do
  [ -f "$f" ] || { echo "missing $f (blz prepare / blz export-edges)" >&2; exit 1; }
done
W=/home/blz
TMP=$(mktemp -d)
LOG=results/a30_run.log
json() { python3 -c "import json,sys; d=json.load(sys.stdin); print(d$1)"; }

stop_instance() {
  [ -n "$ID" ] || return 0
  if [ "$KEEP" = 1 ]; then
    echo "instance $ID left running (--keep): jl pause $ID --yes" >&2
  elif [ "$DESTROY" = 1 ] && [ "${1:-}" = ok ]; then
    jl destroy "$ID" --yes --json
  else
    jl pause "$ID" --yes --json
  fi
}
trap 'rc=$?; if [ $rc -ne 0 ]; then echo "a30.sh failed ($rc)" >&2; stop_instance; fi; rm -rf "$TMP"' EXIT

if [ -z "$ID" ]; then
  ID=$(jl create --gpu A30 --storage 50 --name blz-e5 --yes --json | json '["machine_id"]')
  echo "created A30 instance $ID"
fi
jl exec "$ID" -- mkdir -p "$W"

# The working tree and the commit it came from.
git ls-files -co --exclude-standard -z | tar --null -czf "$TMP/repo.tgz" -T -
dirty=0
[ -n "$(git status --porcelain)" ] && dirty=1
printf 'BLZ_GIT_SHA=%s\nBLZ_GIT_DIRTY=%s\n' "$(git rev-parse HEAD)" "$dirty" > "$TMP/source.env"
jl upload "$ID" "$TMP/repo.tgz" "$W/repo.tgz"
jl upload "$ID" "$TMP/source.env" "$W/source.env"
jl upload "$ID" data/hi_small.blzg "$W/hi_small.blzg"
jl upload "$ID" data/edges.npy "$W/edges.npy"

RUN=$(jl run bench/a30_remote.sh --on "$ID" --no-follow --yes --json | json '["run_id"]')
echo "run $RUN started on $ID"
sleep 20
jl run logs "$RUN" --tail 20 || true
# `jl run status --json`: exit_code is null while the run goes on; state "instance-paused" (or
# another instance-*) means the instance stopped under the run. A failed status call is retried.
while :; do
  if st=$(jl run status "$RUN" --json); then
    code=$(json '.get("exit_code")' <<< "$st")
    [ "$code" != None ] && break
    state=$(json '.get("state")' <<< "$st")
    case $state in
      running|unknown) ;;
      *) echo "run $RUN: state $state" >&2; code=$state; break ;;
    esac
  fi
  sleep 60
  jl run logs "$RUN" --tail 3 2>/dev/null | tail -3 || true
done
jl run logs "$RUN" --tail 400 > "$LOG" 2>&1 || true
echo "run $RUN exited with $code (log: $LOG)"

mkdir -p "$TMP/out"
for f in e5_gpu_a30.json e4_cpu_a30.json e1_parity_a30.json a30_env.txt; do
  jl download "$ID" "$W/repo/results/$f" "$TMP/out/$f" 2>/dev/null || echo "not produced: $f"
done
[ -f "$TMP/out/e5_gpu_a30.json" ] && cp "$TMP/out/e5_gpu_a30.json" results/
[ -f "$TMP/out/e4_cpu_a30.json" ] && cp "$TMP/out/e4_cpu_a30.json" results/
[ -f "$TMP/out/a30_env.txt" ] && cp "$TMP/out/a30_env.txt" results/
if [ -f "$TMP/out/e1_parity_a30.json" ]; then
  python3 - "$TMP/out/e1_parity_a30.json" <<'EOF'
import json, sys
src = json.load(open(sys.argv[1]))["fuzz_cuda_vs_oracle"]
doc = json.load(open("results/e1_parity.json"))
doc["fuzz_cuda_vs_oracle_a30"] = src
open("results/e1_parity.json", "w").write(json.dumps(doc, indent=1) + "\n")
EOF
fi
[ "$code" = 0 ] || exit 1
stop_instance ok
trap 'rm -rf "$TMP"' EXIT
echo "done; render with bench/e5_report.py results/e5_gpu_a30.json"

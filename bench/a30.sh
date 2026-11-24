#!/bin/bash
# The P11 A30 campaign (SPEC v2 §7.3; v1's P5 runbook is in git history). From the repo root on
# `local`:
#   1. creates an on-demand A30 container (or uses --id);
#   2. uploads the working tree (tracked and untracked non-ignored files, without .git), the
#      binary cache, and the commit it was taken from (BLZ_GIT_SHA, BLZ_GIT_DIRTY);
#   3. runs bench/a30_remote.sh there with `jl run` and waits for it, checking every minute;
#   4. downloads results/{p11_a30,e12_gpu_a30,e11_ablation_a30}.json, the E11 report (or why E11
#      was skipped), a30_p11_env.txt, and the run log (results/a30_p11_run.log);
#   5. stops the instance: pause (default) or destroy (--destroy). On a failure it pauses too,
#      unless --keep, so a broken run does not keep billing the GPU.
# Then render the report: bench/e12_report.py results/e12_gpu_a30.json.
#
# --campaign s7 runs S7's campaign instead (SPEC v2 §8 S7, see bench/a30_remote.sh) and downloads
# results/{s7_a30.json,e12_s7_a30.{json,md}}, a30_s7_env.txt and results/a30_s7_run.log.
#
# --campaign p13 runs P13's AC-V3-1 campaign (SPEC v3 P13, see bench/a30_remote.sh) and downloads
# results/{p13_a30.json,ac_v3_1_a30.json,ac_v3_1_a30_logs.tgz}, a30_p13_env.txt and
# results/a30_p13_run.log. It runs for hours: set MAX_WAIT_MIN.
#
# Usage: bench/a30.sh [--campaign p11|s7|p13] [--id MACHINE_ID] [--keep] [--destroy]
# Needs: jl (logged in, an SSH key added) and data/hi_small.blzg. MAX_WAIT_MIN (default 120)
# bounds the wait for the remote run; past it the run counts as failed and the instance is paused.
set -euo pipefail
cd "$(dirname "$0")/.."
ID="" KEEP=0 DESTROY=0 CAMPAIGN=p11
while [ $# -gt 0 ]; do
  case $1 in
    --campaign) CAMPAIGN=$2; shift 2 ;;
    --id) ID=$2; shift 2 ;;
    --keep) KEEP=1; shift ;;
    --destroy) DESTROY=1; shift ;;
    *) echo "usage: $0 [--campaign p11|s7|p13] [--id MACHINE_ID] [--keep] [--destroy]" >&2; exit 2 ;;
  esac
done
case $CAMPAIGN in
  p11) OUTPUTS="p11_a30.json e12_gpu_a30.json e11_ablation_a30.json e11_ablation_a30.md e11_a30_skipped.txt a30_p11_env.txt" ;;
  s7) OUTPUTS="s7_a30.json e12_s7_a30.json e12_s7_a30.md a30_s7_env.txt" ;;
  p13) OUTPUTS="p13_a30.json ac_v3_1_a30.json ac_v3_1_a30_logs.tgz a30_p13_env.txt" ;;
  *) echo "unknown campaign $CAMPAIGN" >&2; exit 2 ;;
esac
[ -f data/hi_small.blzg ] || { echo "missing data/hi_small.blzg (blz prepare)" >&2; exit 1; }
W=/home/blz
TMP=$(mktemp -d)
LOG=results/a30_${CAMPAIGN}_run.log
MAX_WAIT_MIN=${MAX_WAIT_MIN:-120}
json() { python3 -c "import json,sys; d=json.load(sys.stdin); print(d$1)"; }

# Whether `jl run` $1 has ended (AGENTS.md): `jl run status --json` has exit_code null while it
# runs, and any state other than "running" (succeeded, failed, instance-paused, instance-missing,
# unknown) also ends the wait. Sets code: the exit code, or that state. A failed status call
# counts as not ended.
run_ended() {
  local st state
  st=$(jl run status "$1" --json 2>/dev/null) || return 1
  code=$(json '.get("exit_code")' <<< "$st")
  [ "$code" != None ] && return 0
  state=$(json '.get("state")' <<< "$st")
  [ "$state" = running ] && return 1
  echo "run $1: state $state" >&2
  code=$state
}

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
  ID=$(jl create --gpu A30 --storage 50 --name "blz-$CAMPAIGN" --yes --json | json '["machine_id"]')
  echo "created A30 instance $ID"
fi
jl exec "$ID" -- mkdir -p "$W"

# The working tree and the commit it came from.
git ls-files -co --exclude-standard -z | tar --null -czf "$TMP/repo.tgz" -T -
dirty=0
[ -n "$(git status --porcelain)" ] && dirty=1
printf 'BLZ_GIT_SHA=%s\nBLZ_GIT_DIRTY=%s\nBLZ_CAMPAIGN=%s\n' "$(git rev-parse HEAD)" "$dirty" "$CAMPAIGN" > "$TMP/source.env"
jl upload "$ID" "$TMP/repo.tgz" "$W/repo.tgz"
jl upload "$ID" "$TMP/source.env" "$W/source.env"
jl upload "$ID" data/hi_small.blzg "$W/hi_small.blzg"

RUN=$(jl run bench/a30_remote.sh --on "$ID" --no-follow --yes --json | json '["run_id"]')
echo "run $RUN started on $ID"
sleep 20
jl run logs "$RUN" --tail 20 || true
deadline=$(( $(date +%s) + MAX_WAIT_MIN * 60 ))
until run_ended "$RUN"; do
  if [ "$(date +%s)" -gt "$deadline" ]; then
    echo "run $RUN: not ended after $MAX_WAIT_MIN min" >&2
    code=timeout
    break
  fi
  sleep 60
  jl run logs "$RUN" --tail 3 2>/dev/null | tail -3 || true
done
jl run logs "$RUN" --tail 400 > "$LOG" 2>&1 || true
echo "run $RUN exited with $code (log: $LOG)"

mkdir -p "$TMP/out"
for f in $OUTPUTS; do
  if jl download "$ID" "$W/repo/results/$f" "$TMP/out/$f" 2>/dev/null; then
    cp "$TMP/out/$f" results/
  else
    echo "not produced: $f"
  fi
done
[ "$code" = 0 ] || exit 1
stop_instance ok
trap 'rm -rf "$TMP"' EXIT
if [ "$CAMPAIGN" = p11 ]; then echo "done; render with bench/e12_report.py results/e12_gpu_a30.json"; else echo done; fi

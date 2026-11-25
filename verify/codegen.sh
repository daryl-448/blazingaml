#!/usr/bin/env bash
# L2, the bounded code check (SPEC v3 §4.3): the real generated code and every optimised path
# against the oracle on every graph within a bound, every trigger, every window (T1, T2), and the
# index primitives' contracts (T4), for the builds present. One summary table at the end.
#
#   verify/codegen.sh [--smoke] [rel] [rel-cuda] [dev] [rel-noopt]     (default: every build present)
#
# The checks (profiles: clean_all, which enables every clean pattern and vstats, and
# tests/profiles/l2_legacy_{stream,tsvis}.yaml, every legacy pattern under each visibility):
#   rel       cpu (both schedules, reachability pruning both ways) at bounds A and B; cpu with each
#             graph alone and a work limit of 1 binding (R11: lcyc's stop eid, SPEC v3 §3.2) at bound
#             A; T4 at bound T4
#   rel-cuda  cuda at H = 0, 4, all; cuda --fuse --pack at H = 0, 4, all (each fusion again with its
#             members at different windows); the hybrid at every pair of H and C in 0, 4, all; each at
#             bound A, against the oracle; cuda with each graph alone and a work limit of 1 binding at
#             bound LIMIT_GPU (a graph alone costs cuda about 6 ms, so not at A; the hybrid, at about
#             170 ms a graph alone, has no work-limit check here)
#   dev       ASan + UBSan: cpu with each graph alone (no batch), and T4, at bound DEV
#   rel-noopt the generated code with every blzc --disable optimisation (the rel-noopt preset,
#             BLZ_DISABLE_OPTS): cpu and cpu with each graph alone at a work limit of 1, as rel, at
#             bound A
# A bound m=M,n=N,t=T is every edge list of 1..M edges over at most N nodes with ts in [0, T), once
# per node relabelling, at the windows 1..T (blz fuzz-diff --bound). --smoke runs every check at
# bound SMOKE instead, in seconds. Reports go to build/verify/codegen.json (--report), logs to
# build/verify/codegen.log. A check passes when blz exits 0 and its report shows the comparisons
# ran: rows or stop eids compared in every block, a graph at the work limit in each limit-1 run, and
# nonzero counts on the paths the check asks for. Exit status 0 if every check passed.
set -uo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)

A=m=5,n=5,t=4      # every graph of <= 5 edges: the smallest sg, sg_ord and gs instance (4 edges) plus one, cycles and chains up to 5
B=m=6,n=4,t=3      # every graph of 6 edges on <= 4 nodes
T4=m=5,n=5,t=4     # index contracts: lists of every length 0..5
DEV=m=4,n=4,t=3    # sanitizers, each graph alone
LIMIT_GPU=m=4,n=4,t=3  # cuda at a work limit of 1, each graph alone: the smallest bound where lcyc stops
SMOKE=m=4,n=4,t=3  # every block has nonzero rows (sg, sg_ord and gs from 4 edges)

smoke=0
builds=()
for a in "$@"; do
  case $a in
    --smoke) smoke=1 ;;
    rel | rel-cuda | dev | rel-noopt) builds+=("$a") ;;
    *) echo "usage: $0 [--smoke] [rel] [rel-cuda] [dev] [rel-noopt]" >&2; exit 64 ;;
  esac
done
if [ ${#builds[@]} -eq 0 ]; then
  for b in rel rel-cuda dev rel-noopt; do [ -x "$root/build/$b/blz" ] && builds+=("$b"); done
fi
# The smoke LIMIT_GPU: 3 edges on up to 4 nodes still give lcyc 2 bindings at a trigger, so it stops.
if [ $smoke -eq 1 ]; then A=$SMOKE B= T4=$SMOKE DEV=$SMOKE LIMIT_GPU=m=3,n=4,t=2; fi
# A UBSan report stops the run (exit 1) instead of printing and going on with matching rows.
export UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1

out=$root/build/verify
mkdir -p "$out"
report=$out/codegen.json
status=$out/codegen.status.tsv
log=$out/codegen.log
rm -f "$report" "$status"
: > "$log"

clean=$root/configs/profiles/clean_all.yaml
stream=$root/tests/profiles/l2_legacy_stream.yaml
tsvis=$root/tests/profiles/l2_legacy_tsvis.yaml
profiles=("$clean" "$stream" "$tsvis")

# run <name> <build> <blz args...>: one check; its report goes under <name>, its exit code to $status.
run() {
  local name=$1 build=$2
  shift 2
  echo "== $name: blz $*" | tee -a "$log"
  "$root/build/$build/blz" "$@" --report "$report" --name "$name" 2>&1 | tee -a "$log"
  printf '%s\t%s\t%s\n' "$name" "$build" "${PIPESTATUS[0]}" >> "$status"
}

for build in "${builds[@]}"; do
  [ -x "$root/build/$build/blz" ] || { echo "no build/$build/blz" | tee -a "$log"; printf 'missing\t%s\t127\n' "$build" >> "$status"; continue; }
  case $build in
    rel)
      for p in "${profiles[@]}"; do
        pn=$(basename "$p" .yaml)
        for bound in $A $B; do run "$build/cpu/$pn/$bound" "$build" fuzz-diff --bound "$bound" --profile "$p"; done
        run "$build/cpu-limit1/$pn/$A" "$build" fuzz-diff --bound "$A" --batch-edges 1 --work-limit 1 --profile "$p"
      done
      run "$build/index-contracts/$T4" "$build" index-contracts --bound "$T4"
      ;;
    rel-cuda)
      for p in "${profiles[@]}"; do
        pn=$(basename "$p" .yaml)
        run "$build/cuda/$pn/$A" "$build" fuzz-diff --backend cuda --bound "$A" --profile "$p"
        run "$build/cuda-fuse-pack/$pn/$A" "$build" fuzz-diff --backend cuda --fuse --pack --bound "$A" --profile "$p"
        run "$build/hybrid/$pn/$A" "$build" fuzz-diff --backend hybrid --bound "$A" --profile "$p"
        run "$build/cuda-limit1/$pn/$LIMIT_GPU" "$build" fuzz-diff --backend cuda --bound "$LIMIT_GPU" --batch-edges 1 \
          --work-limit 1 --profile "$p"
      done
      ;;
    dev)
      for p in "${profiles[@]}"; do
        pn=$(basename "$p" .yaml)
        run "$build/cpu-alone/$pn/$DEV" "$build" fuzz-diff --bound "$DEV" --batch-edges 1 --profile "$p"
      done
      run "$build/index-contracts/$DEV" "$build" index-contracts --bound "$DEV"
      ;;
    rel-noopt)
      for p in "${profiles[@]}"; do
        pn=$(basename "$p" .yaml)
        run "$build/cpu/$pn/$A" "$build" fuzz-diff --bound "$A" --profile "$p"
        run "$build/cpu-limit1/$pn/$A" "$build" fuzz-diff --bound "$A" --batch-edges 1 --work-limit 1 --profile "$p"
      done
      ;;
  esac
done

python3 -I - "$report" "$status" <<'EOF' | tee -a "$log"
import json, os, sys
report = json.load(open(sys.argv[1])) if os.path.exists(sys.argv[1]) else {}
def at(path):
    node = report
    for k in path.split("/"):
        node = node.get(k) if isinstance(node, dict) else None
        if node is None: return None
    return node
def missing(name, build, r):
    """What a passing check's report does not show ran (SPEC v3 §4.4 item 6), or None."""
    if not r:
        return "no report"
    if "queries" in r:  # index-contracts
        return None if r["queries"] else "no queries"
    parts = name.split("/")
    check, legacy = parts[1], parts[2].startswith("l2_legacy")
    blocks, rows, stops = r["mismatched_rows_by_block"], r["compared_rows_by_block"], r["compared_stops_by_block"]
    idle = [b for b in blocks if b != "vstats" and not rows[b] and not stops[b]]
    if "vstats" in blocks and not r["completed_batches_by_block"]["vstats"]:
        idle.append("vstats")
    if idle:
        return "nothing compared in " + ", ".join(idle)
    if check.endswith("limit1") and not r["limited_batches"]:
        return "no graph reached the work limit"
    op = r["path_hits"]["optimised_paths"]
    need = {"cuda": ["light_kernel_triggers", "heavy_kernel_triggers"],
            "cuda-fuse-pack": ["light_kernel_triggers", "heavy_kernel_triggers", "fused_rows"],
            "hybrid": ["hybrid_cpu_triggers", "hybrid_gpu_triggers"] + ["reach_pruned_rows"] * legacy}
    if build == "rel":  # rel-noopt has no sweep (blzc --disable)
        need["cpu"] = ["sweep_rows"] + ["reach_pruned_rows"] * legacy
    zero = [k for k in need.get(check, []) if not op[k]]
    if check == "cuda-fuse-pack" and not any(op["packed_rows_by_width"].values()):
        zero.append("packed_rows")
    return "no " + ", ".join(zero) if zero else None

rows, bad = [], 0
for line in open(sys.argv[2]):
    name, build, rc = line.rstrip("\n").split("\t")
    r = at(name) or {}
    verdict = {"0": "pass", "2": "FAIL"}.get(rc, f"error {rc}")
    if rc == "0" and (why := missing(name, build, r)):
        verdict, rc = f"FAIL ({why})", "evidence"
    bad += rc != "0"
    parts = name.split("/")
    check = parts[1] if len(parts) > 1 else name
    profile = parts[2] if len(parts) > 3 else "-"
    if "queries" in r:  # index-contracts
        checks, found = f"{r['queries']:,} queries", r["violations"]
    else:
        checks = f"{r['trigger_checks']:,}" if "trigger_checks" in r else "-"
        found = sum(r.get("mismatched_rows_by_block", {}).values()) if r else "-"
    rows.append([build, check, profile, parts[-1] if len(parts) > 2 else "-", f"{r.get('graphs', '-'):,}" if r else "-",
                 checks, str(found), f"{r['seconds']:.1f}" if "seconds" in r else "-", verdict])
head = ["build", "check", "profile", "bound", "graphs", "trigger checks", "mismatches", "seconds", "result"]
w = [max(len(str(x[i])) for x in rows + [head]) for i in range(len(head))]
print()
print("L2 bounded code check (SPEC v3 §4.3)")
for x in [head] + rows:
    print("  ".join(str(c).ljust(w[i]) if i < 3 else str(c).rjust(w[i]) for i, c in enumerate(x)))
print(f"{len(rows) - bad} of {len(rows)} checks passed")
sys.exit(1 if bad else 0)
EOF

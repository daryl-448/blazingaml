#!/usr/bin/env bash
# L1, the algorithm proofs (SPEC v3 §4.3): the Lean 4 project in verify/lean proves the generated
# code's shortcuts against SPEC v3 §3.2's rules for all sizes (T3). This builds it from scratch,
# prints each main theorem's axioms and fails if any proof depends on sorryAx or on an axiom
# beyond Lean's own three; with --mutants it then checks that every proof mutant breaks the build
# (verify/lean/mutants.py, AC-V3-3). verify/lean/COVERAGE.md maps each theorem to the C++ it
# justifies and each contract to its executable check.
#
#   verify/algorithms.sh [--mutants]
#
# Lake comes from $ELAN_HOME/bin (default ~/.cache/blz-fv/lean/elan), the toolchain from
# verify/lean/lean-toolchain. The build runs in a fresh copy of verify/lean under a temporary
# directory, so it starts from an empty .lake and leaves the source tree alone. Exit status 0 if
# everything passed.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
src=$root/verify/lean

mutants=0
for a in "$@"; do
  case $a in
    --mutants) mutants=1 ;;
    *) echo "usage: verify/algorithms.sh [--mutants]" >&2; exit 2 ;;
  esac
done

export ELAN_HOME=${ELAN_HOME:-$HOME/.cache/blz-fv/lean/elan}
lake=$ELAN_HOME/bin/lake
toolchain=$(<"$src/lean-toolchain")
if [[ ! -x $lake ]] || ! "$ELAN_HOME/bin/elan" toolchain list 2>/dev/null | grep -qF "$toolchain"; then
  cat >&2 <<EOF
L1 needs lake and the pinned toolchain $toolchain under ELAN_HOME=$ELAN_HOME. Install them with:
  export ELAN_HOME=$ELAN_HOME
  curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \\
    | sh -s -- -y --no-modify-path --default-toolchain none
  \$ELAN_HOME/bin/elan toolchain install $toolchain
EOF
  exit 1
fi

work=$(mktemp -d -t blz-l1.XXXXXX)
trap 'rm -rf "$work"' EXIT
cp -r "$src/lakefile.toml" "$src/lean-toolchain" "$src/lake-manifest.json" "$src/Blz.lean" \
  "$src/Blz" "$src/Axioms.lean" "$work/"
cd "$work"

t0=$SECONDS
echo "== lake build ($toolchain, from scratch)"
if ! "$lake" build 2>&1 | tee build.log; then
  echo "L1 FAILED: the build failed" >&2; exit 1
fi
if grep -q "declaration uses .sorry." build.log; then
  echo "L1 FAILED: a declaration uses sorry" >&2; exit 1
fi

echo "== axioms of each main theorem (Axioms.lean)"
"$lake" env lean Axioms.lean | tee axioms.log
want=$(grep -c '^#print axioms' Axioms.lean)
ok_re="^'[^']+' (depends on axioms: \[((propext|Classical\.choice|Quot\.sound)(, )?)+\]|does not depend on any axioms)$"
if grep -q sorryAx axioms.log; then
  echo "L1 FAILED: a main theorem depends on sorryAx" >&2; exit 1
fi
if [[ $(grep -cE "$ok_re" axioms.log) -ne $want || $(wc -l <axioms.log) -ne $want ]]; then
  echo "L1 FAILED: expected $want theorems on propext, Classical.choice and Quot.sound only" >&2; exit 1
fi
echo "L1 proofs: ok, $want main theorems, no sorry ($((SECONDS - t0)) s)"

if ((mutants)); then
  t1=$SECONDS
  echo "== proof mutants (verify/lean/mutants.py)"
  LAKE=$lake python3 -I "$src/mutants.py" "$work" || { echo "L1 FAILED: a mutant was not killed where recorded" >&2; exit 1; }
  echo "L1 mutants: ok ($((SECONDS - t1)) s)"
fi

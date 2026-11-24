#!/bin/bash
# The A30 side of the P11 campaign (SPEC v2 §7.3); bench/a30.sh uploads the files and starts this
# with `jl run`. In $BLZ_WORK (default /home/blz) it expects repo.tgz (the working tree, without
# .git) and hi_small.blzg, and the environment BLZ_GIT_SHA / BLZ_GIT_DIRTY (in source.env). It:
#   1. checks the toolchain (SPEC §2.2: GCC >= 12, CMake >= 3.28 + Ninja, CUDA >= 12.4) and
#      installs what is missing;
#   2. builds rel-cuda for sm_80 only and without XGBoost (nothing trains here, D-15);
#   3. AC-V2-4 on sm_80: the fast tests; blz fuzz-diff --backend cuda on 10,000 graphs at H = 0, 4
#      and all for v1_default, sg_ordered_v1w and sg_ordered_tsvis; HI-Small mined on CUDA under
#      the same profiles (bench/e12_report.py compares the hashes with the oracle's);
#   4. E12: blz bench --backend cuda, H swept as in v1's E5, for v1_default and every compiled v2
#      profile, and blz bench --backend cpu at nproc threads (GPU vs CPU on this instance);
#   5. with more than 12 vCPUs and at most E11_START_MIN minutes spent so far, E11 at 1 and nproc
#      threads (bench/e11.py --host a30). GFP is not timed (SPEC v2 §7.3).
# With BLZ_CAMPAIGN=s7 (SPEC v2 §8 S7), steps 3-5 are instead: the fast tests; blz fuzz-diff
# --backend cuda --fuse --pack on 10,000 graphs at H = 0, 4 and all, for every registered block
# and for sg_ordered_tsvis; HI-Small mined with --fuse --pack under v1_default (its hashes and
# copy widths into results/s7_a30.json); and bench/s7.sh (results/e12_s7_a30.{json,md}).
# With BLZ_CAMPAIGN=p13 (SPEC v3 P13, AC-V3-1 on sm_80), steps 3-5 are instead: the fast tests;
# HI-Small mined under clean_default, clean_wc_hi_small and clean_all on the oracle, cpu, cuda and
# hybrid, each block's hash equal to the oracle's (results/p13_a30.json); and bench/ac_v3_1.sh
# (results/ac_v3_1_a30.json, its logs in results/ac_v3_1_a30_logs.tgz).
# Results go to $BLZ_WORK/repo/results/{p11_a30,e12_gpu_a30,e11_ablation_a30}.json (and the E11
# .md) and a30_p11_env.txt. Exit status: non-zero on the first failure.
set -euxo pipefail
START=$(date +%s)
W=${BLZ_WORK:-/home/blz}
E11_START_MIN=${E11_START_MIN:-30}
cd "$W"
set -a
. "$W/source.env"
set +a
CAMPAIGN=${BLZ_CAMPAIGN:-p11}
rm -rf repo
mkdir -p repo
tar -xzf repo.tgz -C repo
mkdir -p repo/data repo/results repo/features
ln -sf "$W/hi_small.blzg" repo/data/hi_small.blzg
cd repo
NP=$(nproc)
{
  echo "nproc: $NP"
  lscpu
  free -m
  nvidia-smi
  cat /etc/os-release
} > "results/a30_${CAMPAIGN}_env.txt" 2>&1 || true

# --- toolchain ---------------------------------------------------------------------------------
# CMake and Ninja in a venv of their own (`jl run` may already run inside ~/.venv).
python3 -m venv "$W/venv" 2> /dev/null || { apt-get install -y -qq python3-venv && python3 -m venv "$W/venv"; }
"$W/venv/bin/pip" install --quiet 'cmake>=3.28' ninja
export PATH="$W/venv/bin:/usr/local/cuda/bin:$PATH"
CXX=g++
if [ "$(g++ -dumpversion | cut -d. -f1)" -lt 12 ]; then
  for v in 14 13 12; do
    if command -v "g++-$v" > /dev/null; then CXX=g++-$v; break; fi
  done
  if [ "$CXX" = g++ ]; then
    apt-get update -qq
    apt-get install -y -qq g++-12
    CXX=g++-12
  fi
fi
export CXX
NVCC=$(command -v nvcc || true)
cuda_ok() {  # nvcc at least 12.4
  [ -n "$NVCC" ] || return 1
  local v
  v=$("$NVCC" --version | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1 \2/p')
  set -- $v
  [ "$1" -gt 12 ] || { [ "$1" -eq 12 ] && [ "$2" -ge 4 ]; }
}
if ! cuda_ok; then
  # The NVIDIA apt repository of the image's Ubuntu release; a toolkit that the driver supports.
  . /etc/os-release
  repo="ubuntu${VERSION_ID//./}"
  wget -q "https://developer.download.nvidia.com/compute/cuda/repos/$repo/x86_64/cuda-keyring_1.1-1_all.deb"
  dpkg -i cuda-keyring_1.1-1_all.deb
  apt-get update -qq
  drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d. -f1)
  pkg=cuda-toolkit-12-4
  [ "$drv" -ge 560 ] && pkg=cuda-toolkit-12-6
  [ "$drv" -ge 570 ] && pkg=cuda-toolkit-12-8
  apt-get install -y -qq "$pkg"
  NVCC=$(ls -d /usr/local/cuda-12.*/bin/nvcc | sort -V | tail -1)
  cuda_ok
fi
$CXX --version | head -1
"$NVCC" --version | tail -2
cmake --version | head -1

# --- build and AC-V2-4 on sm_80 ------------------------------------------------------------------
cmake --preset rel-cuda -DCMAKE_CUDA_COMPILER="$NVCC" -DCMAKE_CUDA_ARCHITECTURES=80 -DBLZ_WITH_XGBOOST=OFF \
  -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_CUDA_HOST_COMPILER="$CXX"
cmake --build build/rel-cuda -j"$NP"
ctest --preset rel-cuda -j4
BLZ=build/rel-cuda/blz

if [ "$CAMPAIGN" = s7 ]; then
  R=results/s7_a30.json
  "$BLZ" fuzz-diff --backend cuda --fuse --pack --seeds 0:10000 --report "$R" --name fuzz_diff/registry
  "$BLZ" fuzz-diff --backend cuda --fuse --pack --seeds 0:10000 --profile configs/profiles/sg_ordered_tsvis.yaml \
    --report "$R" --name fuzz_diff/sg_ordered_tsvis
  "$BLZ" mine --graph data/hi_small.blzg --profile configs/profiles/v1_default.yaml --backend cuda --fuse --pack \
    --out features/a30_s7
  python3 - "$R" features/a30_s7/features.json <<'PY'
import json, os, sys
path, meta = sys.argv[1], json.load(open(sys.argv[2]))
doc = json.load(open(path)) if os.path.exists(path) else {}
doc["hi_small"] = {"v1_default": {"backend": meta["backend"], "cuda": meta["cuda"]["env"],
                                  "output_fnv1a": {b: v["output_fnv1a"] for b, v in meta["blocks"].items()},
                                  "pack": {b: v["cuda"].get("pack") for b, v in meta["blocks"].items()},
                                  "fused": {b: v["cuda"].get("fused") for b, v in meta["blocks"].items()}}}
doc["git"] = meta["git"]
open(path, "w").write(json.dumps(doc, indent=1) + "\n")
PY
  rm -rf features/a30_s7
  HOST=a30 bench/s7.sh
  echo "a30_remote (s7): done in $(( ($(date +%s) - START) / 60 )) min"
  exit 0
fi

if [ "$CAMPAIGN" = p13 ]; then
  # AC-V3-1 on sm_80: HI-Small under the clean profiles on the oracle, cpu, cuda and hybrid
  # (output_fnv1a per block, vstats included, into results/p13_a30.json), then bench/ac_v3_1.sh.
  R=results/p13_a30.json
  for p in clean_default clean_wc_hi_small clean_all; do
    for be in oracle cpu cuda hybrid; do
      t0=$(date +%s)
      "$BLZ" mine --graph data/hi_small.blzg --profile "configs/profiles/$p.yaml" --backend $be --out "features/a30_$p"
      python3 - "$R" "$p" "$be" "$(( $(date +%s) - t0 ))" "features/a30_$p/features.json" <<'PY'
import json, os, sys
path, prof, be, secs, meta = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), json.load(open(sys.argv[5]))
doc = json.load(open(path)) if os.path.exists(path) else {}
doc.setdefault("hi_small", {}).setdefault(prof, {})[be] = {
    "profile_id": meta["run"]["profile_id"], "wall_seconds": secs, "peak_rss_mb": meta.get("peak_rss_mb"),
    "cuda": (meta.get("cuda") or {}).get("env"),
    "output_fnv1a": {b: v["output_fnv1a"] for b, v in meta["blocks"].items()}}
doc["git"] = meta["git"]
open(path, "w").write(json.dumps(doc, indent=1) + "\n")
PY
      rm -rf "features/a30_$p"
    done
  done
  python3 - "$R" <<'PY'
import json, sys
path = sys.argv[1]
doc = json.load(open(path))
ok = True
for prof, by in doc["hi_small"].items():
    ref = by["oracle"]["output_fnv1a"]
    differ = {be: sorted(b for b in set(ref) | set(r["output_fnv1a"]) if ref.get(b) != r["output_fnv1a"].get(b))
              for be, r in by.items()}
    differ = {be: d for be, d in differ.items() if d}
    ok &= not differ
    print(f"hi_small {prof}: {'equal on ' + ', '.join(by) if not differ else 'DIFFER ' + str(differ)}")
doc["hi_small_equal"] = ok
open(path, "w").write(json.dumps(doc, indent=1) + "\n")
sys.exit(0 if ok else 1)
PY
  gates=0
  BLZ=$BLZ OUT=results/ac_v3_1_a30 bench/ac_v3_1.sh all || gates=$?
  tar -czf results/ac_v3_1_a30_logs.tgz -C results ac_v3_1_a30
  echo "a30_remote (p13): done in $(( ($(date +%s) - START) / 60 )) min, gates rc $gates"
  exit $gates
fi

R=results/p11_a30.json
for p in v1_default sg_ordered_v1w sg_ordered_tsvis; do
  "$BLZ" fuzz-diff --backend cuda --seeds 0:10000 --profile "configs/profiles/$p.yaml" \
    --report "$R" --name "ac_v2_4/fuzz_diff/$p/cuda"
  "$BLZ" mine --graph data/hi_small.blzg --profile "configs/profiles/$p.yaml" --backend cuda --out "features/a30_$p"
  python3 - "$R" "$p" "features/a30_$p/features.json" <<'EOF'
import json, os, sys
path, prof, meta = sys.argv[1], sys.argv[2], json.load(open(sys.argv[3]))
doc = json.load(open(path)) if os.path.exists(path) else {}
doc.setdefault("ac_v2_4", {}).setdefault("hi_small", {})[prof] = {
    "cuda": {"profile_id": meta["run"]["profile_id"], "cuda": meta["cuda"]["env"],
             "output_fnv1a": {b: v["output_fnv1a"] for b, v in meta["blocks"].items()}}}
doc["git"] = meta["git"]
open(path, "w").write(json.dumps(doc, indent=1) + "\n")
EOF
  rm -rf "features/a30_$p"
done

# --- E12 ---------------------------------------------------------------------------------------
E=results/e12_gpu_a30.json
for p in v1_default sg_ordered_v1w sg_ordered_tsvis dg2_wb dg2_wc dg2_v1; do
  "$BLZ" bench --graph data/hi_small.blzg --profile "configs/profiles/$p.yaml" --backend cuda \
    --heavy 0,8,16,32,64,128,256,1024,all --reps 5 --out "$E" --name "cuda/$p"
  OMP_PROC_BIND=close OMP_PLACES=threads "$BLZ" bench --graph data/hi_small.blzg \
    --profile "configs/profiles/$p.yaml" --backend cpu --threads "$NP" --schedule dynamic --reps 5 \
    --out "$E" --name "cpu/$p"
done

# --- E11 on the instance CPU -------------------------------------------------------------------
spent=$(( ($(date +%s) - START) / 60 ))
if [ "$NP" -gt 12 ] && [ "$spent" -le "$E11_START_MIN" ]; then
  python3 bench/e11.py --host a30 --threads "1,$NP" --jobs "$NP" --deps-from build/rel-cuda
else
  echo "E11 skipped: nproc $NP, $spent min spent (start limit $E11_START_MIN min)" | tee results/e11_a30_skipped.txt
fi
echo "a30_remote: done in $(( ($(date +%s) - START) / 60 )) min"

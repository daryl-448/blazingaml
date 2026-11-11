#!/bin/bash
# The A30 side of the P5 runbook (SPEC §16 P5); bench/a30.sh uploads the files and starts this
# with `jl run`. In $BLZ_WORK (default /home/blz) it expects repo.tgz (the working tree, without
# .git), hi_small.blzg and edges.npy, and the environment BLZ_GIT_SHA / BLZ_GIT_DIRTY (in
# source.env). It:
#   1. checks the toolchain (SPEC §2.2: GCC >= 12, CMake >= 3.28 + Ninja, CUDA >= 12.4) and
#      installs what is missing;
#   2. builds rel-cuda for sm_80 only and without XGBoost (nothing trains here, D-15);
#   3. checks the CUDA backend on sm_80: the fast tests (golden probes, 500 fuzz graphs at three
#      thresholds) and 10,000 fuzz graphs against the oracle;
#   4. runs E5 (blz bench --backend cuda, every threshold of the sweep);
#   5. with more than 12 vCPUs, runs E4 (blz bench --backend cpu) and the GFP timing at nproc
#      threads, which defines T_GFP_best on this machine (D-16).
# Results go to $BLZ_WORK/repo/results/{e5_gpu_a30,e4_cpu_a30,e1_parity_a30}.json and
# a30_env.txt. Exit status: non-zero on the first failure.
set -euxo pipefail
W=${BLZ_WORK:-/home/blz}
cd "$W"
set -a
. "$W/source.env"
set +a
rm -rf repo
mkdir -p repo
tar -xzf repo.tgz -C repo
mkdir -p repo/data repo/results
ln -sf "$W/hi_small.blzg" repo/data/hi_small.blzg
ln -sf "$W/edges.npy" repo/data/edges.npy
cd repo
NP=$(nproc)
{
  echo "nproc: $NP"
  lscpu
  free -m
  nvidia-smi
  cat /etc/os-release
} > results/a30_env.txt 2>&1 || true

# --- toolchain ---------------------------------------------------------------------------------
# CMake, Ninja and uv in a venv of their own (`jl run` may already run inside ~/.venv).
python3 -m venv "$W/venv" 2> /dev/null || { apt-get install -y -qq python3-venv && python3 -m venv "$W/venv"; }
"$W/venv/bin/pip" install --quiet 'cmake>=3.28' ninja uv
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

# --- build and check ---------------------------------------------------------------------------
cmake --preset rel-cuda -DCMAKE_CUDA_COMPILER="$NVCC" -DCMAKE_CUDA_ARCHITECTURES=80 -DBLZ_WITH_XGBOOST=OFF \
  -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_CUDA_HOST_COMPILER="$CXX"
cmake --build build/rel-cuda -j"$NP"
ctest --preset rel-cuda -j4
BLZ=build/rel-cuda/blz
"$BLZ" fuzz-diff --backend cuda --seeds 0:10000 --report results/e1_parity_a30.json --name fuzz_cuda_vs_oracle

# --- E5 ----------------------------------------------------------------------------------------
"$BLZ" bench --graph data/hi_small.blzg --config configs/features_default.yaml --backend cuda \
  --heavy 0,8,16,32,64,128,256,1024,all --reps 5 --out results/e5_gpu_a30.json --name cuda

# --- E4 and GFP, with more than 12 vCPUs (SPEC §16 P5) ---------------------------------------------
if [ "$NP" -gt 12 ]; then
  threads=""
  for t in 1 2 4 8 12 16 24 32 48 64; do
    [ "$t" -lt "$NP" ] && threads="$threads$t,"
  done
  threads="$threads$NP"
  OMP_PROC_BIND=close OMP_PLACES=threads "$BLZ" bench --graph data/hi_small.blzg \
    --config configs/features_default.yaml --backend cpu --threads "$threads" --schedule dynamic,sorted \
    --reps 5 --out results/e4_cpu_a30.json --name cpu
  (cd tools/gfp && uv run bench_gfp.py --edges ../../data/edges.npy --threads "$NP" \
    --out ../../results/e4_cpu_a30.json)
fi
echo "a30_remote: done"

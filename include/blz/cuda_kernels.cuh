#pragma once
// The kernel scaffolding that generated CUDA code instantiates (SPEC §7.5). A generated
// <stem>_cuda.cu defines a struct K with three static device functions and wraps the launchers
// below as <pattern>_cuda_bucket and <pattern>_cuda:
//
//   K::trigger(g, p, e, sink)         <pattern>_trigger from <stem>.hpp, shared with the CPU
//   K::warp(g, p, e, lane, sink)      the heavy kernel's body, derived from the same plan
//   K::work(g, p, e)                  <pattern>_work from <stem>.hpp
//
// Light kernel: one thread per trigger, emitting straight into the trigger's row (RowSink).
// Heavy kernel: one warp per trigger. The lanes emit into the warp's 64-bit histogram in shared
// memory; after the body the lanes add it into the row, saturating at UINT32_MAX exactly as
// RowSink does one emit at a time. Each trigger's row is written by one thread or one warp, so
// no global atomics are needed except for the saturation counter.
#include <cstdint>

#include <cuda_runtime.h>

#include "blz/registry.hpp"

namespace blz::cuda {

inline constexpr unsigned kFullMask = 0xffffffffu;
inline constexpr std::uint32_t kWarpSize = 32;
inline constexpr int kLightThreads = 128;     // threads per block, light kernel
inline constexpr int kHeavyWarps = 8;         // warps per block, heavy kernel
inline constexpr int kBucketThreads = 256;

// The sum of x over the warp, in every lane. All 32 lanes must call it.
__device__ inline std::int64_t warp_sum(std::int64_t x) {
  for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(kFullMask, x, o);
  return x;
}

// The heavy kernel's sink: one warp's histogram. Same binning as RowSink (SPEC §4.2).
struct WarpSink {
  unsigned long long* hist = nullptr;
  const std::uint32_t* bins = nullptr;
  std::uint32_t nbins = 0;

  __device__ void emit(std::int64_t size) {
    if (size < 2 || size < std::int64_t(bins[0])) return;
    std::uint32_t i = nbins - 1;
    while (size < std::int64_t(bins[i])) --i;
    atomicAdd(&hist[i], 1ULL);
  }
};

template <class K>
__global__ void bucket_kernel(GraphView g, Params p, std::uint64_t first, std::uint32_t count,
                              std::uint32_t threshold, std::uint8_t* flags) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) flags[i] = K::work(g, p, std::uint32_t(first + i)) >= threshold;
}

template <class K>
__global__ void light_kernel(GraphView g, Params p, CudaRun r) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= r.count) return;
  const std::uint32_t e = r.triggers[i];
  RowSink sink{r.out + (e - r.first) * r.nbins, r.bins, r.nbins, 0};
  K::trigger(g, p, e, sink);
  if (sink.saturated) atomicAdd(r.saturated, (unsigned long long)sink.saturated);
}

template <class K>
__global__ void heavy_kernel(GraphView g, Params p, CudaRun r) {
  extern __shared__ unsigned long long hist_all[];
  const std::uint32_t lane = threadIdx.x % kWarpSize, warp = threadIdx.x / kWarpSize;
  const std::uint64_t w = std::uint64_t(blockIdx.x) * (blockDim.x / kWarpSize) + warp;
  if (w >= r.count) return;  // the same for every lane of the warp
  const std::uint32_t e = r.triggers[w];
  unsigned long long* hist = hist_all + std::size_t(warp) * r.nbins;
  for (std::uint32_t i = lane; i < r.nbins; i += kWarpSize) hist[i] = 0;
  __syncwarp();
  WarpSink sink{hist, r.bins, r.nbins};
  K::warp(g, p, e, lane, sink);
  __syncwarp();
  std::uint32_t* row = r.out + (e - r.first) * r.nbins;
  unsigned long long lost = 0;
  for (std::uint32_t i = lane; i < r.nbins; i += kWarpSize) {
    const unsigned long long v = row[i] + hist[i];
    if (v > UINT32_MAX) {
      lost += v - UINT32_MAX;
      row[i] = UINT32_MAX;
    } else {
      row[i] = std::uint32_t(v);
    }
  }
  if (lost) atomicAdd(r.saturated, lost);
}

template <class K>
void launch_bucket(const GraphView& g, const Params& p, std::uint64_t first, std::uint32_t count,
                   std::uint32_t threshold, std::uint8_t* flags, void* stream) {
  if (count == 0) return;
  const unsigned blocks = (count + kBucketThreads - 1) / kBucketThreads;
  bucket_kernel<K><<<blocks, kBucketThreads, 0, cudaStream_t(stream)>>>(g, p, first, count, threshold, flags);
}

// Shared memory of one heavy block: one histogram per warp.
inline std::size_t heavy_shared_bytes(std::uint32_t nbins) {
  return std::size_t(kHeavyWarps) * nbins * sizeof(unsigned long long);
}

template <class K>
void launch(const GraphView& g, const Params& p, const CudaRun& r) {
  if (r.count == 0) return;
  const cudaStream_t s = cudaStream_t(r.stream);
  if (r.heavy) {
    const unsigned blocks = (r.count + kHeavyWarps - 1) / kHeavyWarps;
    heavy_kernel<K><<<blocks, kHeavyWarps * kWarpSize, heavy_shared_bytes(r.nbins), s>>>(g, p, r);
  } else {
    const unsigned blocks = (r.count + kLightThreads - 1) / kLightThreads;
    light_kernel<K><<<blocks, kLightThreads, 0, s>>>(g, p, r);
  }
}

}  // namespace blz::cuda

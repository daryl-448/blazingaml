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
// RowSink does one emit at a time. A count block's histogram is one cell holding the emitted size;
// its pattern emits at most once per trigger (E403), so the clamp of the row loses one emit. Each trigger's row is written by one thread or one warp, so
// no global atomics are needed except for the saturation counter and the work-limit report.
//
// Work limit (SPEC v2 §8 S3): a `work_limit` pattern calls sink.over(limit) once per binding. The
// light kernel counts in its RowSink; the heavy kernel's lanes count into one counter per warp
// in shared memory, so a trigger aborts exactly when its bindings exceed the limit, as on the
// CPU. An aborted trigger's eid goes to run.aborted (atomicMin); the host turns it into an error.
// A trigger with a larger eid than run.aborted is skipped, so the report is the smallest aborted
// eid, as on the CPU and in the oracle.
//
// Fusions (blzc --fuse, SPEC v2 §8 S7): a generated fused_<id>_cuda.cu defines a struct F whose
// trigger, warp and work count every member in one scan of their shared trigger list (D-46), member
// k with p.member[k], emitting into its slot's sink; F::kSlots is the number of slots. Both fused
// kernels give each slot a RowSink on the trigger's row of that slot. In the heavy kernel the lanes
// split the scan and lane 0 alone emits (each member's count is summed over the warp first), so a
// member emits exactly as it does alone, and members that share a slot add into the same row. Counts are non-negative and
// saturate at UINT32_MAX, so the row is min(sum of emits, UINT32_MAX) whatever the order or the
// kernel, as in the unfused run. No member counts work (blzc refuses work_limit patterns), so the
// fused kernels have no abort.
#include <cstdint>

#include <cuda/atomic>
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

// The smallest aborted eid so far: a relaxed atomic load, since other triggers atomicMin it.
__device__ inline unsigned long long abort_bound(unsigned long long* a) {
  return ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>(*a).load(::cuda::std::memory_order_relaxed);
}

// The heavy kernel's sink: one warp's histogram. Same binning as RowSink (SPEC §4.2); a count
// row's one cell sums the sizes (SPEC v3 R7).
struct WarpSink {
  unsigned long long* hist = nullptr;
  const std::uint32_t* bins = nullptr;
  std::uint32_t nbins = 0;
  bool count = false;
  unsigned long long* work = nullptr;  // the warp's binding count (shared memory)
  bool aborted = false;                // this lane saw the count pass the limit

  __device__ void emit(std::int64_t size, std::int64_t min_size) {
    if (count) {
      atomicAdd(&hist[0], (unsigned long long)size);
      return;
    }
    if (size < min_size || size < std::int64_t(bins[0])) return;
    std::uint32_t i = nbins - 1;
    while (size < std::int64_t(bins[i])) --i;
    atomicAdd(&hist[i], 1ULL);
  }
  __device__ bool over(std::uint64_t limit) {
    if (atomicAdd(work, 1ULL) + 1 <= limit) return false;
    aborted = true;
    return true;
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
  if (e > abort_bound(r.aborted)) return;  // past an abort already seen
  RowSink sink{r.out + (e - r.first) * r.nbins, r.bins, r.nbins, r.count_block};
  K::trigger(g, p, e, sink);
  if (sink.saturated) atomicAdd(r.saturated, (unsigned long long)sink.saturated);
  if (sink.aborted) atomicMin(r.aborted, (unsigned long long)e);
}

template <class K>
__global__ void heavy_kernel(GraphView g, Params p, CudaRun r) {
  extern __shared__ unsigned long long hist_all[];
  const std::uint32_t lane = threadIdx.x % kWarpSize, warp = threadIdx.x / kWarpSize;
  const std::uint64_t w = std::uint64_t(blockIdx.x) * (blockDim.x / kWarpSize) + warp;
  if (w >= r.count) return;  // the same for every lane of the warp
  const std::uint32_t e = r.triggers[w];
  unsigned long long bound = 0;  // one read per warp, so the whole warp returns or none of it
  if (lane == 0) bound = abort_bound(r.aborted);
  if (e > __shfl_sync(kFullMask, bound, 0)) return;
  unsigned long long* hist = hist_all + std::size_t(warp) * r.nbins;
  unsigned long long* work = hist_all + std::size_t(blockDim.x / kWarpSize) * r.nbins + warp;
  for (std::uint32_t i = lane; i < r.nbins; i += kWarpSize) hist[i] = 0;
  if (lane == 0) *work = 0;
  __syncwarp();
  WarpSink sink{hist, r.bins, r.nbins, r.count_block, work};
  K::warp(g, p, e, lane, sink);
  __syncwarp();
  if (sink.aborted) atomicMin(r.aborted, (unsigned long long)e);
  std::uint32_t* row = r.out + (e - r.first) * r.nbins;
  unsigned long long lost = 0;
  for (std::uint32_t i = lane; i < r.nbins; i += kWarpSize) {
    const unsigned long long v = row[i] + hist[i];
    if (v > UINT32_MAX) {
      lost += r.count_block ? 1 : v - UINT32_MAX;
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

// Shared memory of one heavy block: one histogram per warp, then one work counter per warp.
inline std::size_t heavy_shared_bytes(std::uint32_t nbins) {
  return std::size_t(kHeavyWarps) * (nbins + 1) * sizeof(unsigned long long);
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

// --- fusions (SPEC v2 §8 S7) -------------------------------------------------------------------

template <class F>
__global__ void fused_bucket_kernel(GraphView g, FusedParams p, std::uint64_t first, std::uint32_t count,
                                    std::uint32_t threshold, std::uint8_t* flags) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) flags[i] = F::work(g, p, std::uint32_t(first + i)) >= threshold;
}

template <class F>
__global__ void fused_light_kernel(GraphView g, FusedParams p, FusedRun r) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= r.count) return;
  const std::uint32_t e = r.triggers[i];
  RowSink sink[F::kSlots];
#pragma unroll
  for (std::uint32_t s = 0; s < F::kSlots; ++s)
    sink[s] = RowSink{r.out[s] + (e - r.first) * r.nbins[s], r.bins[s], r.nbins[s], r.count_block[s]};
  F::trigger(g, p, e, sink);
#pragma unroll
  for (std::uint32_t s = 0; s < F::kSlots; ++s)
    if (sink[s].saturated) atomicAdd(r.saturated + s, (unsigned long long)sink[s].saturated);
}

template <class F>
__global__ void fused_heavy_kernel(GraphView g, FusedParams p, FusedRun r) {
  const std::uint32_t lane = threadIdx.x % kWarpSize, warp = threadIdx.x / kWarpSize;
  const std::uint64_t w = std::uint64_t(blockIdx.x) * (blockDim.x / kWarpSize) + warp;
  if (w >= r.count) return;  // the same for every lane of the warp
  const std::uint32_t e = r.triggers[w];
  RowSink sink[F::kSlots];
#pragma unroll
  for (std::uint32_t s = 0; s < F::kSlots; ++s)
    sink[s] = RowSink{r.out[s] + (e - r.first) * r.nbins[s], r.bins[s], r.nbins[s], r.count_block[s]};
  F::warp(g, p, e, lane, sink);  // every lane scans; only lane 0 emits
#pragma unroll
  for (std::uint32_t s = 0; s < F::kSlots; ++s)
    if (sink[s].saturated) atomicAdd(r.saturated + s, (unsigned long long)sink[s].saturated);
}

template <class F>
void launch_fused_bucket(const GraphView& g, const FusedParams& p, std::uint64_t first, std::uint32_t count,
                         std::uint32_t threshold, std::uint8_t* flags, void* stream) {
  if (count == 0) return;
  const unsigned blocks = (count + kBucketThreads - 1) / kBucketThreads;
  fused_bucket_kernel<F><<<blocks, kBucketThreads, 0, cudaStream_t(stream)>>>(g, p, first, count, threshold, flags);
}

template <class F>
void launch_fused(const GraphView& g, const FusedParams& p, const FusedRun& r) {
  if (r.count == 0) return;
  const cudaStream_t s = cudaStream_t(r.stream);
  if (r.heavy) {
    const unsigned blocks = (r.count + kHeavyWarps - 1) / kHeavyWarps;
    fused_heavy_kernel<F><<<blocks, kHeavyWarps * kWarpSize, 0, s>>>(g, p, r);
  } else {
    const unsigned blocks = (r.count + kLightThreads - 1) / kLightThreads;
    fused_light_kernel<F><<<blocks, kLightThreads, 0, s>>>(g, p, r);
  }
}

}  // namespace blz::cuda

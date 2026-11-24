#pragma once
// The hybrid CPU/GPU driver (SPEC v2 §8 S6): a block's heavy triggers on the CPU backend while
// the GPU mines the light ones, concurrently, then a deterministic merge.
#include <cstdint>
#include <span>

#include "blz/registry.hpp"
#include "runtime/cpu_backend.hpp"
#include "runtime/cuda_backend.hpp"
#include "runtime/features.hpp"

namespace blz {

// Every trigger whose work estimate is >= cpu_threshold goes to the CPU. kAllLight sends every
// trigger to the GPU, 0 every trigger to the CPU.
inline constexpr std::uint32_t kAllGpu = kAllLight;
// The default CPU threshold of `blz mine --backend hybrid`: E13's pre-registered rule on the 1650S
// (results/e13_hybrid_local.md: the smallest summed median T, pruning on, H = 32; D-44).
inline constexpr std::uint32_t kDefaultCpuThreshold = 32;

struct HybridTiming {
  double split_seconds = 0;    // from the call to the CPU's trigger list being known (the GPU's split)
  double cpu_seconds = 0;      // mine_cpu on the CPU's triggers (wall clock, this thread's OpenMP team)
  double gpu_seconds = 0;      // the GPU side's wall clock, split included: kernels, D2H
  double overlap_seconds = 0;  // from the list being known to both sides done
  double merge_seconds = 0;    // the CPU rows into the output
  double total_seconds = 0;    // split + overlap + merge, wall clock
  std::uint64_t cpu_triggers = 0, gpu_triggers = 0;
  std::uint64_t saturated = 0;
  CpuBlockTiming cpu;
  CudaBlockTiming gpu;
};

// Mines block rows [first, last) with the patterns `pats` under visibility `vis`. The triggers
// whose first pattern's work estimate (the estimate of the CPU's sorted schedule and the CUDA
// bucketing) is >= cpu_threshold run on the CPU (mine_cpu, schedule `schedule`, the current
// OpenMP thread count, Params::reach_prune from p), the others on the GPU (`miner`, bucketed at
// gpu_threshold), at the same time. The GPU computes the split (CudaMiner::mine_split) and hands
// the CPU its list first. host_out gets (last - first) rows of block_cols(p) columns,
// overwritten: the GPU writes every row (zero for the CPU's triggers), then the CPU's rows are
// copied over theirs. Each row is computed by one side, so the result does not depend on the
// split or on timing. A work-limit abort on either side is reported with the smallest eid.
// `miner` should be built with blocking_sync, so its host thread sleeps while the GPU runs.
HybridTiming mine_hybrid(const GraphView& host, const DeviceGraph& dev, CudaMiner& miner,
                         std::span<const PatternInfo* const> pats, const BlockParams& p, Visibility vis,
                         std::uint64_t first, std::uint64_t last, std::uint32_t cpu_threshold,
                         std::uint32_t gpu_threshold, Schedule schedule, std::uint32_t* host_out);

}  // namespace blz

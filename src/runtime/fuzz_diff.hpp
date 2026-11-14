#pragma once
#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "runtime/cpu_backend.hpp"
#include "runtime/profile.hpp"

namespace blz {

// AC1 on fuzz graphs (SPEC §13.3): the generated CPU backend against the oracle, every row of
// every block, on the graphs of seeds [s0, s1). Each graph uses its own window for every block,
// `bins`, and the compiled K_MAX for the oracle's tcyc cap. Both schedules are checked.
// The profile forms (SPEC v2 §10) take the blocks, their patterns and k_max from `prof`; the
// windows still come from each graph, and the bins from `bins`, or from each profile block when
// `bins` is empty. The forms without a profile use registry_profile(): every registered block (v1's
// six, and sg_ord from P10).
//
// Work limit (SPEC v2 §8 S3): every run has `work_limit`. Where the oracle stops a block (lcyc),
// the graph is limited: that block's rows are not compared, but the backend must stop at the same
// eid, the smallest over the limit; a backend that stops elsewhere, or not at all, or stops where
// the oracle does not, counts as one mismatched row.
struct FuzzDiff {
  std::uint64_t graphs = 0, edges = 0;
  std::array<std::uint64_t, kAllBlocks.size()> mismatched_rows{};  // per block, both schedules
  std::vector<std::uint64_t> bad_seeds;                             // ascending
  std::vector<std::uint64_t> limited_seeds;                         // ascending: the oracle hit the limit
};

FuzzDiff fuzz_diff_cpu(std::uint64_t s0, std::uint64_t s1, std::span<const std::uint32_t> bins);
FuzzDiff fuzz_diff_cpu(const Profile& prof, std::uint64_t s0, std::uint64_t s1, std::span<const std::uint32_t> bins,
                       std::uint64_t work_limit = kDefaultWorkLimit);

// AC1 for the generated CUDA backend, the same way: every row of every block against the oracle,
// once per light/heavy threshold in `thresholds` (0 sends every trigger to the heavy kernel,
// kAllLight every trigger to the light kernel). mismatched_rows sums over the thresholds.
FuzzDiff fuzz_diff_cuda(std::uint64_t s0, std::uint64_t s1, std::span<const std::uint32_t> bins,
                        std::span<const std::uint32_t> thresholds);
FuzzDiff fuzz_diff_cuda(const Profile& prof, std::uint64_t s0, std::uint64_t s1, std::span<const std::uint32_t> bins,
                        std::span<const std::uint32_t> thresholds, std::uint64_t work_limit = kDefaultWorkLimit);

}  // namespace blz

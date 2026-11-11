#pragma once
#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "runtime/cpu_backend.hpp"

namespace blz {

// AC1 on fuzz graphs (SPEC §13.3): the generated CPU backend against the oracle, every row of
// every block, on the graphs of seeds [s0, s1). Each graph uses its own window for every block,
// `bins`, and the compiled K_MAX for the oracle's tcyc cap. Both schedules are checked.
struct FuzzDiff {
  std::uint64_t graphs = 0, edges = 0;
  std::array<std::uint64_t, kAllBlocks.size()> mismatched_rows{};  // per block, both schedules
  std::vector<std::uint64_t> bad_seeds;                             // ascending
};

FuzzDiff fuzz_diff_cpu(std::uint64_t s0, std::uint64_t s1, std::span<const std::uint32_t> bins);

// AC1 for the generated CUDA backend, the same way: every row of every block against the oracle,
// once per light/heavy threshold in `thresholds` (0 sends every trigger to the heavy kernel,
// kAllLight every trigger to the light kernel). mismatched_rows sums over the thresholds.
FuzzDiff fuzz_diff_cuda(std::uint64_t s0, std::uint64_t s1, std::span<const std::uint32_t> bins,
                        std::span<const std::uint32_t> thresholds);

}  // namespace blz

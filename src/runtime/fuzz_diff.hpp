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
// `bins`, and the compiled K_MAX for the oracle's tcyc cap. Both schedules are checked, and for a
// block with a pattern planned for reachability pruning (tcyc, SPEC v2 §8 S6) both again with it.
// The profile forms (SPEC v2 §10) take the blocks, their patterns and k_max from `prof`; the
// windows still come from each graph, and the bins from fuzz_bins(): `bins`, or each profile
// block's own when `bins` is empty, and none for a count block (SPEC v3 R7). The graphs are indexed
// as the profile says (Profile::self_loops). The forms without a profile use registry_profile():
// every registered legacy block (v1's six, and sg_ord from P10).
//
// Work limit (SPEC v2 §8 S3): every run has `work_limit`. Where the oracle stops a block (lcyc),
// the graph is limited: that block's rows are not compared, but the backend must stop at the same
// eid, the smallest over the limit; a backend that stops elsewhere, or not at all, or stops where
// the oracle does not, counts as one mismatched row.
// What the rows are compared against: the oracle, or the CPU backend with the dynamic schedule
// (a backend-vs-backend check for a profile the oracle cannot run, SPEC v3 clean profiles). With
// Reference::cpu, fuzz_diff_cpu checks the sorted schedule (and reachability pruning) against it.
enum class Reference { oracle, cpu };
const char* reference_name(Reference r);

struct FuzzDiff {
  std::uint64_t graphs = 0, edges = 0;
  std::array<std::uint64_t, kAllBlocks.size()> mismatched_rows{};  // per block, both schedules
  std::vector<std::uint64_t> bad_seeds;                             // ascending
  std::vector<std::uint64_t> limited_seeds;                         // ascending: the oracle hit the limit
};

FuzzDiff fuzz_diff_cpu(std::uint64_t s0, std::uint64_t s1, std::span<const std::uint32_t> bins);
FuzzDiff fuzz_diff_cpu(const Profile& prof, std::uint64_t s0, std::uint64_t s1, std::span<const std::uint32_t> bins,
                       std::uint64_t work_limit = kDefaultWorkLimit, Reference ref = Reference::oracle);

// AC1 for the generated CUDA backend, the same way: every row of every block against the oracle,
// once per light/heavy threshold in `thresholds` (0 sends every trigger to the heavy kernel,
// kAllLight every trigger to the light kernel). mismatched_rows sums over the thresholds.
FuzzDiff fuzz_diff_cuda(std::uint64_t s0, std::uint64_t s1, std::span<const std::uint32_t> bins,
                        std::span<const std::uint32_t> thresholds);
// SPEC v2 §8 S7: `opts.fuse` mines each compiled fusion's blocks in one kernel (cuda_units), and
// `opts.pack` copies the rows packed; the rows must not change.
struct CudaOptions {
  bool fuse = false;
  bool pack = false;
};
FuzzDiff fuzz_diff_cuda(const Profile& prof, std::uint64_t s0, std::uint64_t s1, std::span<const std::uint32_t> bins,
                        std::span<const std::uint32_t> thresholds, std::uint64_t work_limit = kDefaultWorkLimit,
                        CudaOptions opts = {}, Reference ref = Reference::oracle);

// AC1 for the hybrid (SPEC v2 §8 S6): mine_hybrid with reachability pruning on its CPU side and
// the sorted schedule, once per pair of a GPU light/heavy threshold in `thresholds` and a CPU
// threshold in `cpu_thresholds` (0: every trigger on the CPU, kAllGpu: every trigger on the GPU).
// With no CPU threshold it is fuzz_diff_cuda.
// `opts` is for the plain CUDA runs only (no CPU threshold).
FuzzDiff fuzz_diff_hybrid(const Profile& prof, std::uint64_t s0, std::uint64_t s1, std::span<const std::uint32_t> bins,
                          std::span<const std::uint32_t> thresholds, std::span<const std::uint32_t> cpu_thresholds,
                          std::uint64_t work_limit = kDefaultWorkLimit, CudaOptions opts = {},
                          Reference ref = Reference::oracle);

}  // namespace blz

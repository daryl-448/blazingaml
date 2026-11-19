// SPEC v2 §8 S6: the hybrid CPU/GPU driver and the CUDA subset run it uses. The hybrid's rows
// equal the oracle's on fuzz graphs for every split, and the subset run writes exactly the listed
// triggers' rows. CUDA builds only.
#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "graph/fuzz_gen.hpp"
#include "graph/index.hpp"
#include "runtime/cpu_backend.hpp"
#include "runtime/cuda_backend.hpp"
#include "runtime/fuzz_diff.hpp"
#include "runtime/hybrid.hpp"
#include "runtime/profile.hpp"

using namespace blz;

// The slow share is fuzz_diff_hybrid_10k (tests/CMakeLists.txt).
TEST_CASE("hybrid: rows equal the oracle's on 150 fuzz graphs, every split (SPEC v2 §8 S6)") {
  const std::vector<std::uint32_t> bins = {2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16};
  const std::vector<std::uint32_t> gpu = {0, kAllLight}, cpu = {0, 4, kAllGpu};
  const FuzzDiff d = fuzz_diff_hybrid(registry_profile(), 0, 150, bins, gpu, cpu);
  CHECK(d.graphs == 150);
  for (Block b : kAllBlocks) {
    INFO(block_name(b));
    CHECK(d.mismatched_rows[int(b)] == 0);
  }
  CHECK(d.bad_seeds.empty());
}

TEST_CASE("hybrid: the split follows the work estimate, and every row is written") {
  CudaMiner miner(true);
  std::uint64_t mixed = 0;
  for (std::uint64_t seed = 300; seed < 340; ++seed) {
    const FuzzGraph fg = fuzz_graph(seed);
    const GraphIndex idx(fg.src, fg.dst, fg.ts, fg.n);
    const DeviceGraph dg(idx.view());
    const std::uint64_t m = fg.src.size(), first = m / 4, last = m - m / 5;
    BlockParams p{fg.window, {2, 3, 4, 6, 8}};
    p.reach_prune = true;
    const auto pats = patterns_for(Block::tcyc);
    std::vector<std::uint32_t> want((last - first) * p.bins.size(), 0);
    mine_cpu(idx.view(), Block::tcyc, p, first, last, Schedule::dynamic, want.data());
    const Params params{p.window, false};
    for (std::uint32_t hc : {0u, 3u, 8u, kAllGpu}) {
      INFO("seed " << seed << " cpu threshold " << hc);
      std::vector<std::uint32_t> got(want.size(), 0xdeadbeefu);
      const HybridTiming tm =
          mine_hybrid(idx.view(), dg, miner, pats, p, Visibility::stream, first, last, hc, 4, Schedule::sorted, got.data());
      CHECK(got == want);
      std::uint64_t heavy = 0;
      for (std::uint64_t e = first; e < last; ++e) heavy += pats.front()->work(idx.view(), params, std::uint32_t(e)) >= hc;
      CHECK(tm.cpu_triggers == (hc == kAllGpu ? 0 : heavy));
      CHECK(tm.cpu_triggers + tm.gpu_triggers == last - first);
      mixed += tm.cpu_triggers > 0 && tm.gpu_triggers > 0;
    }
  }
  CHECK(mixed > 20);  // most graphs split at 3 or 8
}

TEST_CASE("cuda: mine_subset writes the listed triggers' rows and zeroes the others") {
  CudaMiner miner;
  const FuzzGraph fg = fuzz_graph(7);
  const GraphIndex idx(fg.src, fg.dst, fg.ts, fg.n);
  const DeviceGraph dg(idx.view());
  const std::uint64_t m = fg.src.size();
  const BlockParams p{fg.window, {2, 3, 4, 5}};
  const std::size_t nb = p.bins.size();
  for (Block b : {Block::fan_out, Block::sg, Block::tcyc}) {
    const auto pats = patterns_for(b);
    std::vector<std::uint32_t> full(m * nb);
    miner.mine(dg, pats, p, Visibility::stream, 0, m, 4, full.data());
    std::vector<std::uint32_t> subset;
    for (std::uint32_t e = 1; e < m; e += 3) subset.push_back(e);
    for (std::uint32_t h : {0u, 4u, kAllLight}) {
      INFO(block_name(b) << " H=" << h);
      std::vector<std::uint32_t> got(m * nb, 0xdeadbeefu);
      const CudaBlockTiming tm = miner.mine_subset(dg, pats, p, Visibility::stream, 0, m, subset, h, got.data());
      CHECK(tm.light + tm.heavy == subset.size() * pats.size());
      for (std::uint64_t e = 0; e < m; ++e) {
        const bool listed = e % 3 == 1;
        for (std::size_t c = 0; c < nb; ++c) CHECK(got[e * nb + c] == (listed ? full[e * nb + c] : 0u));
      }
      std::fill(got.begin(), got.end(), 0xdeadbeefu);
      miner.mine_subset(dg, pats, p, Visibility::stream, 0, m, {}, h, got.data());
      CHECK(std::all_of(got.begin(), got.end(), [](std::uint32_t x) { return x == 0; }));
    }
    const std::vector<std::uint32_t> unsorted = {5, 2};
    std::vector<std::uint32_t> sink(m * nb);
    CHECK_THROWS_AS(miner.mine_subset(dg, pats, p, Visibility::stream, 0, m, unsorted, 4, sink.data()),
                    std::invalid_argument);
    const std::vector<std::uint32_t> outside = {std::uint32_t(m)};
    CHECK_THROWS_AS(miner.mine_subset(dg, pats, p, Visibility::stream, 0, m, outside, 4, sink.data()),
                    std::invalid_argument);
  }
}

// The generated CUDA backend (SPEC §7.5): golden probes (§13.2) and AC1 on fuzz graphs (§13.3,
// the fast share; `blz fuzz-diff --backend cuda --seeds 0:10000` is the slow test), each with
// every trigger on the heavy kernel, mixed, and every trigger on the light kernel; trigger
// ranges and determinism (§8); the metamorphic tests (SPEC v2 §3.4) on 200 graphs. Built only with BLZ_ENABLE_CUDA; needs a GPU.
#include <catch2/catch_test_macros.hpp>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "graph/fuzz_gen.hpp"
#include "graph/index.hpp"
#include "runtime/cpu_backend.hpp"
#include "runtime/cuda_backend.hpp"
#include "runtime/fuzz_diff.hpp"
#include "runtime/metamorphic.hpp"

using namespace blz;

namespace {

constexpr std::uint32_t kThresholds[] = {0, 2, kAllLight};  // all heavy, mixed, all light

// A run's rows, or the eid of the trigger that stopped it (SPEC v2 §8 S3).
struct Outcome {
  std::vector<std::uint32_t> rows;
  std::uint64_t aborted = UINT64_MAX;
  bool operator==(const Outcome&) const = default;
};

template <class F>
Outcome outcome(F&& f) {
  Outcome o;
  try {
    o.rows = f();
  } catch (const WorkLimitExceeded& e) {
    o.aborted = e.eid;
  }
  return o;
}

std::vector<std::uint32_t> mine_gpu(CudaMiner& miner, const DeviceGraph& dg, Block b, const BlockParams& p,
                                    std::uint64_t first, std::uint64_t last, std::uint32_t h) {
  std::vector<std::uint32_t> out((last - first) * p.bins.size(), 0xdeadbeef);
  miner.mine(dg, b, p, first, last, h, out.data());
  return out;
}

}  // namespace

TEST_CASE("cuda: the registry has launchers for every pattern") {
  REQUIRE(cuda_built());
  for (const PatternInfo& p : registered_patterns()) {
    INFO(p.name);
    CHECK(p.cuda != nullptr);
    CHECK(p.cuda_bucket != nullptr);
  }
}

// SPEC §13.2: the golden probes, evaluated by the generated CUDA code.
TEST_CASE("cuda backend matches the GFP golden probes") {
  std::ifstream in(std::string(BLZ_SOURCE_DIR) + "/tests/golden/gfp_probes.json");
  REQUIRE(in);
  const auto doc = nlohmann::json::parse(in);
  const auto bins = doc["params"]["bins"].get<std::vector<std::uint32_t>>();
  REQUIRE(doc["cases"].size() >= 30);
  CudaMiner miner;
  for (const auto& c : doc["cases"]) {
    std::vector<std::uint32_t> src, dst;
    std::vector<std::int32_t> ts;
    std::uint32_t n = 0;
    for (const auto& e : c["edges"]) {
      src.push_back(e[0]);
      dst.push_back(e[1]);
      ts.push_back(e[2]);
      n = std::max(n, std::max(src.back(), dst.back()) + 1);
    }
    const GraphIndex idx(src, dst, ts, n);
    const DeviceGraph dg(idx.view());
    const std::uint64_t m = src.size();
    for (std::uint32_t h : kThresholds)
      for (Block b : kV1Blocks) {  // GFP has no sg_ord
        const auto got = mine_gpu(miner, dg, b, {c["tw"].get<std::int32_t>(), bins}, 0, m, h);
        for (std::uint64_t e = 0; e < m; ++e) {
          const auto want = c["output"][e][std::string(block_name(b))].get<std::vector<std::uint32_t>>();
          const std::vector<std::uint32_t> row(got.begin() + std::ptrdiff_t(e * bins.size()),
                                               got.begin() + std::ptrdiff_t((e + 1) * bins.size()));
          INFO(c["id"].get<std::string>() << " edge " << e << " " << block_name(b) << " H=" << h);
          CHECK(row == want);
        }
      }
  }
}

// AC1, fast share: every row of every block on 500 fuzz graphs, at every threshold.
TEST_CASE("cuda backend equals the oracle on fuzz graphs (seeds 0..499)") {
  const std::vector<std::uint32_t> bins = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
  const FuzzDiff d = fuzz_diff_cuda(0, 500, bins, kThresholds);
  CHECK(d.graphs == 500);
  for (Block b : kAllBlocks) {
    INFO(block_name(b));
    CHECK(d.mismatched_rows[int(b)] == 0);
  }
  INFO("first bad seed " << (d.bad_seeds.empty() ? -1 : std::int64_t(d.bad_seeds[0])));
  CHECK(d.bad_seeds.empty());
}

TEST_CASE("cuda: metamorphic tests on 200 fuzz graphs, every threshold (SPEC v2 §3.4)") {
  const std::vector<std::uint32_t> bins = {2, 3, 4, 5, 6, 8, 10};
  const MetamorphicResult r = metamorphic(registry_profile(), Backend::cuda, 0, 200, bins, kThresholds);
  CHECK(r.graphs == 200);
  for (MetaTest t : kMetaTests) {
    INFO(meta_test_name(t));
    CHECK((r.rows_compared[int(t)] > 0) == meta_test_applies(t, registry_profile(), Backend::cuda));
    for (std::uint64_t n : r.mismatched_rows[int(t)]) CHECK(n == 0);
  }
  CHECK(r.bad_seeds.empty());
}

// SPEC v2 §6.1 (P10): oracle = CUDA and the metamorphic tests under timestamp visibility.
TEST_CASE("cuda backend under timestamp visibility: oracle parity and metamorphic tests, every threshold") {
  Profile ts = registry_profile();
  ts.visibility = Visibility::timestamp;
  const std::vector<std::uint32_t> bins = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
  const FuzzDiff d = fuzz_diff_cuda(ts, 0, 300, bins, kThresholds);
  CHECK(d.graphs == 300);
  for (const ProfileBlock& pb : ts.blocks) {
    INFO(block_name(pb.block));
    CHECK(d.mismatched_rows[int(pb.block)] == 0);
  }
  CHECK(d.bad_seeds.empty());
  const MetamorphicResult r = metamorphic(ts, Backend::cuda, 0, 100, std::vector<std::uint32_t>{2, 3, 4, 5, 6, 8, 10},
                                          kThresholds);
  CHECK(r.graphs == 100);
  for (MetaTest t : kMetaTests) {
    INFO(meta_test_name(t));
    CHECK((r.rows_compared[int(t)] > 0) == meta_test_applies(t, ts, Backend::cuda));
    for (std::uint64_t n : r.mismatched_rows[int(t)]) CHECK(n == 0);
  }
  CHECK(r.bad_seeds.empty());
}

// SPEC §8: CUDA rows equal the CPU's, for any trigger range and threshold, and repeat exactly. On
// an index without self-loops (SPEC v3 R1) too, whose lists are shorter than m.
TEST_CASE("cuda backend rows equal the cpu backend's for any trigger range and threshold") {
  CudaMiner miner;
  for (std::uint64_t seed = 100; seed < 130; ++seed)
    for (const SelfLoops sl : {SelfLoops::keep, SelfLoops::drop}) {
      const FuzzGraph fg = fuzz_graph(seed);
      const GraphIndex idx(fg.src, fg.dst, fg.ts, fg.n, sl);
      const DeviceGraph dg(idx.view());
      const std::uint64_t m = fg.src.size(), mid = m / 3;
      const BlockParams p{fg.window, {2, 3, 4, 5, 6, 8}};
      for (Block b : kAllBlocks) {
        if (b == Block::lcyc) continue;  // needs a work limit: the next test
        std::vector<std::uint32_t> cpu(m * p.bins.size(), 0);
        mine_cpu(idx.view(), b, p, 0, m, Schedule::dynamic, cpu.data());
        for (std::uint32_t h : kThresholds) {
          INFO("seed " << seed << " drop " << (sl == SelfLoops::drop) << " " << block_name(b) << " H=" << h);
          CHECK(mine_gpu(miner, dg, b, p, 0, m, h) == cpu);
          CHECK(mine_gpu(miner, dg, b, p, 0, m, h) == cpu);
          const auto tail = mine_gpu(miner, dg, b, p, mid, m, h);
          CHECK(std::equal(tail.begin(), tail.end(), cpu.begin() + std::ptrdiff_t(mid * p.bins.size())));
          CHECK(mine_gpu(miner, dg, b, p, mid, mid, h).empty());
        }
      }
    }
}

// SPEC v2 §8 S3: lcyc's CUDA rows equal the CPU's at every threshold, and a run over the work
// limit stops with the CPU's eid, the smallest over the limit.
TEST_CASE("lcyc: cuda rows or the smallest eid over the work limit equal the cpu's, every threshold") {
  CudaMiner miner;
  const std::vector<std::uint32_t> bins = {2, 3, 4, 5, 6, 7, 8, 9, 10};
  int stopped = 0;
  for (std::uint64_t seed = 0; seed < 40; ++seed) {
    const FuzzGraph fg = fuzz_graph(seed);
    const GraphIndex idx(fg.src, fg.dst, fg.ts, fg.n);
    const DeviceGraph dg(idx.view());
    const std::uint64_t m = fg.src.size();
    BlockParams p{fg.window, bins};
    p.work_limit = 20'000;
    const Outcome want = outcome([&] {
      std::vector<std::uint32_t> out(m * bins.size(), 0);
      mine_cpu(idx.view(), Block::lcyc, p, 0, m, Schedule::dynamic, out.data());
      return out;
    });
    stopped += want.aborted != UINT64_MAX;
    for (std::uint32_t h : kThresholds) {
      INFO("seed " << seed << " H=" << h);
      CHECK(outcome([&] { return mine_gpu(miner, dg, Block::lcyc, p, 0, m, h); }) == want);
    }
  }
  CHECK(stopped > 0);
  CHECK(stopped < 30);
}

// Bucketing: every trigger goes to exactly one kernel, once per pattern of the block.
TEST_CASE("cuda bucketing sends every trigger to one kernel per pattern") {
  std::vector<std::uint32_t> src(40, 1), dst(40, 2);
  std::vector<std::int32_t> ts(40);
  for (int i = 0; i < 40; ++i) ts[std::size_t(i)] = i;
  const GraphIndex idx(src, dst, ts, 3);  // 40 parallel edges 1 -> 2, one second apart
  const DeviceGraph dg(idx.view());
  CudaMiner miner;
  std::vector<std::uint32_t> out(40, 0);
  for (std::uint32_t h : {0u, 10u, kAllLight}) {
    const CudaBlockTiming deg = miner.mine(dg, Block::deg_in, {1000, {2}}, 0, 40, h, out.data());
    CHECK(deg.light + deg.heavy == 40);
    CHECK(deg.heavy == (h == 0 ? 40 : h == 10 ? 31 : 0));  // trigger e's window holds e + 1 edges
    CHECK(out[0] == 0);
    CHECK(out[39] == 1);
    const CudaBlockTiming sg = miner.mine(dg, Block::sg, {1000, {2}}, 0, 40, h, out.data());
    CHECK(sg.light + sg.heavy == 80);  // sg_gather and sg_scatter
  }
}

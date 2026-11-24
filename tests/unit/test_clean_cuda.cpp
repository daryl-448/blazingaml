// The clean family on the CUDA backend and the hybrid (SPEC v3 §3.2): count rows (R7) on the
// loop-free index (R1) through the light, heavy, fused and packed paths, on a hand-checked graph,
// against the clean oracle on fuzz graphs, and the metamorphic tests.
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "blz/registry.hpp"
#include "graph/index.hpp"
#include "runtime/cuda_backend.hpp"
#include "runtime/fuzz_diff.hpp"
#include "runtime/hybrid.hpp"
#include "runtime/metamorphic.hpp"
#include "runtime/profile.hpp"

using namespace blz;

namespace {

const PatternInfo* pattern(const std::string& name) {
  for (const PatternInfo& p : registered_patterns())
    if (p.name == name) return &p;
  FAIL("no pattern " << name);
  return nullptr;
}

const FusionInfo& fusion(std::string_view name) {
  for (const FusionInfo& f : registered_fusions())
    if (name == f.name) return f;
  FAIL("no fusion " << name);
  throw 0;
}

constexpr std::uint32_t kThresholds[] = {0, 4, kAllLight};

}  // namespace

TEST_CASE("cuda: count rows on a hand-checked graph, light, heavy, fused and packed (SPEC v3 R1, R7)") {
  // test_clean.cpp's graph: e3 is a self-loop, e0 and e1 tie at t = 10.
  const std::vector<std::uint32_t> src = {1, 2, 2, 0, 3, 0}, dst = {0, 0, 0, 0, 4, 2};
  const std::vector<std::int32_t> ts = {10, 10, 11, 12, 13, 14};
  const GraphIndex idx(src, dst, ts, 5, SelfLoops::drop);
  const DeviceGraph dg(idx.view());
  const std::uint64_t m = src.size();
  const std::vector<std::pair<const char*, std::vector<std::uint32_t>>> want = {
      {"clean_fan_in", {2, 2, 2, 0, 1, 1}},  {"clean_deg_in", {2, 2, 3, 0, 1, 1}},
      {"clean_fan_out", {1, 1, 1, 0, 1, 1}}, {"clean_deg_out", {1, 1, 2, 0, 1, 1}},
      {"clean_gs", {0, 0, 0, 0, 0, 1}},      {"clean_fan_total", {2, 2, 2, 0, 1, 1}},
      {"clean_fan_oneway_out", {1, 1, 1, 0, 1, 0}}};
  CudaMiner miner;
  const BlockParams p{100, {}};
  for (const std::uint32_t h : kThresholds)
    for (const bool pack : {false, true}) {
      for (const auto& [name, rows] : want) {
        INFO(name << " H=" << h << " pack " << pack);
        std::vector<std::uint32_t> got(m, 0xdeadbeefu);
        const PatternInfo* pi = pattern(name);
        const CudaBlockTiming tm = miner.mine(dg, std::span(&pi, 1), p, Visibility::timestamp, 0, m, h, got.data(), pack);
        CHECK(got == rows);
        CHECK(tm.saturated == 0);
        if (pack) CHECK(tm.d2h_width == 1);
      }
      for (const auto& [fname, a, b] : {std::tuple{"clean_fan_in+clean_deg_in", 0, 1},
                                        std::tuple{"clean_fan_out+clean_deg_out", 2, 3}}) {
        INFO(fname << " H=" << h << " pack " << pack);
        const FusionInfo& f = fusion(fname);
        std::vector<std::uint32_t> ga(m, 0xdeadbeefu), gb(m, 0xdeadbeefu);
        const BlockParams* ps[] = {&p, &p};
        std::uint32_t* outs[] = {ga.data(), gb.data()};
        miner.mine_fused(dg, f, ps, Visibility::timestamp, 0, m, h, outs, pack);
        // Slot order is the members' blocks in fusion order: fan, then degree.
        CHECK(ga == want[std::size_t(a)].second);
        CHECK(gb == want[std::size_t(b)].second);
      }
    }
  // The hybrid, every trigger on the CPU, then every trigger on the GPU.
  for (const std::uint32_t hc : {0u, kAllGpu}) {
    INFO("hybrid cpu threshold " << hc);
    CudaMiner hm(true);
    std::vector<std::uint32_t> got(m, 0xdeadbeefu);
    const PatternInfo* pi = pattern("clean_fan_in");
    mine_hybrid(idx.view(), dg, hm, std::span(&pi, 1), p, Visibility::timestamp, 0, m, hc, 4, Schedule::sorted,
                got.data());
    CHECK(got == want[0].second);
  }
}

TEST_CASE("cuda, hybrid: clean_all equals the clean oracle on fuzz graphs, every threshold, fused and packed") {
  const Profile all = load_profile(std::string(BLZ_SOURCE_DIR) + "/configs/profiles/clean_all.yaml");
  // lcyc apart: its unordered search on one GPU thread per trigger (H = all) is the slow part, so
  // it runs on fewer graphs with a lower work limit (R11: a limited graph compares stop eids).
  Profile rest = all, lcyc = all;
  std::erase_if(rest.blocks, [](const ProfileBlock& pb) { return pb.block == Block::lcyc; });
  std::erase_if(lcyc.blocks, [](const ProfileBlock& pb) { return pb.block != Block::lcyc; });
  REQUIRE(lcyc.blocks.size() == 1);
  const auto check = [](const Profile& prof, const FuzzDiff& d, std::uint64_t graphs) {
    CHECK(d.graphs == graphs);
    for (const ProfileBlock& pb : prof.blocks) {
      INFO(block_name(pb.block));
      CHECK(d.mismatched_rows[int(pb.block)] == 0);
    }
    CHECK(d.bad_seeds.empty());
  };
  for (const CudaOptions opts : {CudaOptions{false, false}, CudaOptions{true, true}, CudaOptions{false, true}}) {
    INFO("fuse " << opts.fuse << " pack " << opts.pack);
    check(rest, fuzz_diff_cuda(rest, 0, 120, {}, kThresholds, kDefaultWorkLimit, opts), 120);
    check(lcyc, fuzz_diff_cuda(lcyc, 0, 24, {}, kThresholds, 20'000, opts), 24);
  }
  const std::uint32_t cpu_thresholds[] = {0, 4, kAllGpu};
  check(rest, fuzz_diff_hybrid(rest, 0, 40, {}, kThresholds, cpu_thresholds), 40);
  check(lcyc, fuzz_diff_hybrid(lcyc, 0, 12, {}, kThresholds, cpu_thresholds, 20'000), 12);
  // The sparse-time setting (SPEC v3 §4.4), every block at once.
  const auto sparse = [&](const FuzzDiff& d, std::uint64_t graphs) {
    check(all, d, graphs);
    for (const ProfileBlock& pb : all.blocks) CHECK(d.nonzero_rows[int(pb.block)] > 0);
  };
  sparse(fuzz_diff_cuda(all, 0, 300, {}, kThresholds, kDefaultWorkLimit, {true, true}, Reference::oracle,
                        FuzzSetting::sparse),
         300);
  sparse(fuzz_diff_hybrid(all, 0, 40, {}, kThresholds, cpu_thresholds, kDefaultWorkLimit, {}, Reference::oracle,
                          FuzzSetting::sparse),
         40);
}

// The metamorphic suite on clean_all (SPEC v3 §4.4: every clean pattern), lcyc at a low work limit
// (R11: a limited graph is inconclusive for lcyc).
TEST_CASE("cuda, hybrid: the metamorphic tests on clean_all, every threshold") {
  const Profile all = load_profile(std::string(BLZ_SOURCE_DIR) + "/configs/profiles/clean_all.yaml");
  const std::uint32_t cpu_thresholds[] = {0, 4, kAllGpu};
  for (const Backend b : {Backend::cuda, Backend::hybrid}) {
    INFO(backend_name(b));
    const std::uint64_t graphs = b == Backend::cuda ? 30 : 12;
    const MetamorphicResult r = metamorphic(all, b, 0, graphs, {}, kThresholds, cpu_thresholds, 20'000);
    CHECK(r.graphs == graphs);
    for (MetaTest t : kMetaTests) {
      INFO(meta_test_name(t));
      CHECK((r.rows_compared[int(t)] > 0) == meta_test_applies(t, all, b));
      for (std::uint64_t n : r.mismatched_rows[int(t)]) CHECK(n == 0);
    }
    CHECK(r.limited_graphs[int(Block::lcyc)] < graphs);
    CHECK(r.bad_seeds.empty());
    // The sparse-time setting (SPEC v3 §4.4).
    const MetamorphicResult sp = metamorphic(all, b, 0, b == Backend::cuda ? 150 : 24, {}, kThresholds, cpu_thresholds,
                                             kDefaultWorkLimit, FuzzSetting::sparse);
    for (MetaTest t : kMetaTests) {
      INFO(meta_test_name(t));
      CHECK((sp.rows_compared[int(t)] > 0) == meta_test_applies(t, all, b));
      for (std::uint64_t n : sp.mismatched_rows[int(t)]) CHECK(n == 0);
    }
    for (const ProfileBlock& pb : all.blocks) CHECK(sp.nonzero_rows[int(pb.block)] > 0);
  }
}

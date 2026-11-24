// SPEC v2 §8 S7 on the GPU: packed D2H is lossless at every width and size, and fused runs (one
// scan for several patterns, D-46) write the rows the unfused ones do, for any windows, bins, trigger
// range and threshold. The slow share is fuzz_diff_cuda_s7_10k (tests/CMakeLists.txt). CUDA builds only.
#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <string>
#include <vector>

#include "graph/fuzz_gen.hpp"
#include "graph/index.hpp"
#include "runtime/cpu_backend.hpp"
#include "runtime/cuda_backend.hpp"
#include "runtime/fusion.hpp"
#include "runtime/fuzz_diff.hpp"
#include "runtime/profile.hpp"

using namespace blz;

namespace {

constexpr std::uint32_t kThresholds[] = {0, 2, kAllLight};

const FusionInfo& fusion(const char* name) {
  for (const FusionInfo& f : registered_fusions())
    if (std::string(name) == f.name) return f;
  FAIL("no fusion " << name);
  throw 0;
}

}  // namespace

TEST_CASE("pack: the copy is lossless, as narrow as the block's largest cell allows") {
  CudaMiner miner;
  constexpr std::uint64_t kChunk8 = 4u << 20, kChunk16 = 2u << 20;  // cells per 4 MiB staging buffer
  struct Case {
    std::uint64_t cells;
    std::uint32_t max;
    std::uint32_t width;
  };
  const std::vector<Case> cases = {
      {1, 0, 1},           {3, 1, 1},           {5, 255, 1},          {7, 256, 2},
      {1023, 65535, 2},    {1025, 65536, 4},    {9, UINT32_MAX, 4},   {kChunk8, 255, 1},
      {2 * kChunk8 + 7, 200, 1}, {3 * kChunk16 + 5, 40000, 2}, {kChunk16 - 1, 300, 2},
  };
  for (const Case& c : cases) {
    INFO("cells " << c.cells << " max " << c.max);
    std::vector<std::uint32_t> v(c.cells);
    std::uint64_t x = c.cells * 2654435761u;
    for (std::uint32_t& y : v) {
      x = x * 6364136223846793005ull + 1442695040888963407ull;
      y = c.max == 0 ? 0 : std::uint32_t((x >> 33) % (std::uint64_t(c.max) + 1));
    }
    v.back() = c.max;  // the largest cell in the pack kernel's tail (cells % 4 != 0 for most cases)
    // The host rows one cell past a 32-byte boundary: the widening's unaligned head.
    std::vector<std::uint32_t> host(c.cells + 8, 0xdeadbeefu);
    std::uint32_t* out = host.data() + 1;
    const CudaBlockTiming tm = miner.fetch_for_test(v, out, true);
    CHECK(tm.d2h_width == c.width);
    CHECK(tm.d2h_bytes == c.cells * c.width);
    CHECK(std::memcmp(out, v.data(), c.cells * sizeof(std::uint32_t)) == 0);
    CHECK(host[0] == 0xdeadbeefu);
    CHECK(host[c.cells + 1] == 0xdeadbeefu);
    std::vector<std::uint32_t> plain(c.cells, 0xdeadbeefu);
    const CudaBlockTiming tp = miner.fetch_for_test(v, plain.data(), false);
    CHECK(tp.d2h_width == 4);
    CHECK(plain == v);
  }
  std::uint32_t none = 0xdeadbeefu;
  CHECK(miner.fetch_for_test({}, &none, true).d2h_bytes == 0);
  CHECK(none == 0xdeadbeefu);
}

TEST_CASE("fuse, pack: rows equal the oracle's on 300 fuzz graphs, every threshold, both visibilities") {
  REQUIRE(!registered_fusions().empty());
  for (const Visibility vis : {Visibility::stream, Visibility::timestamp}) {
    Profile prof = registry_profile();
    prof.visibility = vis;
    std::vector<const ProfileBlock*> blocks;
    for (const ProfileBlock& b : prof.blocks) blocks.push_back(&b);
    std::size_t fused = 0, legacy = 0;
    for (const CudaUnit& u : cuda_units(blocks, true)) fused += u.fusion != nullptr;
    for (const FusionInfo& f : registered_fusions()) legacy += !is_clean(*f.members[0]);
    CHECK(fused == legacy);  // every legacy fusion's blocks are in the profile
    const std::vector<std::uint32_t> bins = {2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20};
    for (const CudaOptions opts : {CudaOptions{true, true}, CudaOptions{true, false}, CudaOptions{false, true}}) {
      INFO("visibility " << int(vis) << " fuse " << opts.fuse << " pack " << opts.pack);
      const FuzzDiff d = fuzz_diff_cuda(prof, 0, 300, bins, kThresholds, kDefaultWorkLimit, opts);
      CHECK(d.graphs == 300);
      for (const ProfileBlock& pb : prof.blocks) {
        INFO(block_name(pb.block));
        CHECK(d.mismatched_rows[int(pb.block)] == 0);
      }
      CHECK(d.bad_seeds.empty());
    }
  }
}

// The fuzz profiles give every block one window; here each slot has its own window and bins.
TEST_CASE("fuse: each member counts over its own block's window and bins, for any trigger range") {
  CudaMiner miner;
  struct Group {
    const char* fusion;
    std::vector<Block> blocks;
  };
  const std::vector<Group> groups = {{"fan_in+deg_in", {Block::fan_in, Block::deg_in}},
                                     {"fan_out+deg_out", {Block::fan_out, Block::deg_out}}};
  std::uint64_t compared = 0;
  for (std::uint64_t seed = 100; seed < 140; ++seed) {
    const FuzzGraph fg = fuzz_graph(seed);
    const GraphIndex idx(fg.src, fg.dst, fg.ts, fg.n);
    const DeviceGraph dg(idx.view());
    const std::uint64_t m = fg.src.size(), mid = m / 3;
    const std::vector<BlockParams> params = {{fg.window, {2, 3, 4, 6}}, {std::max(1, fg.window / 3), {2, 5, 8, 12, 30}}};
    for (const Group& g : groups) {
      const FusionInfo& f = fusion(g.fusion);
      std::vector<std::vector<std::uint32_t>> want(g.blocks.size());
      for (std::size_t s = 0; s < g.blocks.size(); ++s) {
        want[s].assign(m * params[s].bins.size(), 0);
        mine_cpu(idx.view(), g.blocks[s], params[s], 0, m, Schedule::dynamic, want[s].data());
      }
      std::vector<const BlockParams*> ps;
      for (std::size_t s = 0; s < g.blocks.size(); ++s) ps.push_back(&params[s]);
      for (const std::uint32_t h : kThresholds)
        for (const bool pack : {false, true})
          for (const std::uint64_t first : {std::uint64_t(0), mid}) {
            INFO("seed " << seed << " " << g.fusion << " H=" << h << " pack " << pack << " first " << first);
            std::vector<std::vector<std::uint32_t>> got(g.blocks.size());
            std::vector<std::uint32_t*> outs;
            for (std::size_t s = 0; s < g.blocks.size(); ++s) {
              got[s].assign((m - first) * params[s].bins.size(), 0xdeadbeefu);
              outs.push_back(got[s].data());
            }
            const CudaMiner::FusedTiming tm = miner.mine_fused(dg, f, ps, Visibility::stream, first, m, h, outs, pack);
            CHECK(tm.group.light + tm.group.heavy == m - first);
            for (std::size_t s = 0; s < g.blocks.size(); ++s) {
              const auto at = want[s].begin() + std::ptrdiff_t(first * params[s].bins.size());
              CHECK(std::equal(got[s].begin(), got[s].end(), at, want[s].end()));
              ++compared;
            }
          }
    }
  }
  CHECK(compared == 40 * 4 * 3 * 2 * 2);
}

TEST_CASE("fuse: mine_fused takes one block per slot, each with as many bins as it takes alone") {
  CudaMiner miner;
  const FuzzGraph fg = fuzz_graph(7);
  const GraphIndex idx(fg.src, fg.dst, fg.ts, fg.n);
  const DeviceGraph dg(idx.view());
  const std::uint64_t m = fg.src.size();
  const FusionInfo& f = fusion("fan_in+deg_in");
  const BlockParams p{fg.window, {2, 3}};
  std::vector<std::uint32_t> a(m * 2);
  std::vector<const BlockParams*> one = {&p};
  std::vector<std::uint32_t*> one_out = {a.data()};
  CHECK_THROWS_AS(miner.mine_fused(dg, f, one, Visibility::stream, 0, m, 0, one_out), std::invalid_argument);
  // Two blocks of 400 bins (each within mine()'s limit), every trigger on the heavy kernel.
  BlockParams wide{fg.window, {}};
  for (std::uint32_t b = 2; b < 402; ++b) wide.bins.push_back(b);
  std::vector<std::uint32_t> want_in(m * 400), want_deg(m * 400), got_in(m * 400, 0xdeadbeefu), got_deg(m * 400, 0xdeadbeefu);
  mine_cpu(idx.view(), Block::fan_in, wide, 0, m, Schedule::dynamic, want_in.data());
  mine_cpu(idx.view(), Block::deg_in, wide, 0, m, Schedule::dynamic, want_deg.data());
  std::vector<const BlockParams*> two = {&wide, &wide};
  std::vector<std::uint32_t*> two_out = {got_in.data(), got_deg.data()};
  for (const std::uint32_t h : kThresholds) {
    INFO("H=" << h);
    miner.mine_fused(dg, f, two, Visibility::stream, 0, m, h, two_out);
    CHECK(got_in == want_in);
    CHECK(got_deg == want_deg);
  }
}

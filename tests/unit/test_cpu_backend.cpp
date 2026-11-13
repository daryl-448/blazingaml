// The generated CPU backend (SPEC §7.4, §7.6): golden probes (§13.2), AC1 on fuzz graphs
// (§13.3, the fast share; `blz fuzz-diff --seeds 0:10000` is the slow test), determinism (§8)
// and the start-up checks wired to the generated registry (E402, K_MAX).
#include <catch2/catch_test_macros.hpp>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <omp.h>

#include "graph/fuzz_gen.hpp"
#include "graph/index.hpp"
#include "runtime/cpu_backend.hpp"
#include "runtime/fuzz_diff.hpp"

using namespace blz;

namespace {

std::vector<std::uint32_t> mine(const GraphIndex& idx, Block b, const BlockParams& p, std::uint64_t first,
                                std::uint64_t last, Schedule s = Schedule::dynamic) {
  std::vector<std::uint32_t> out((last - first) * p.bins.size(), 0);
  mine_cpu(idx.view(), b, p, first, last, s, out.data());
  return out;
}

std::string error_of(const FeatureConfig& cfg) {
  try {
    check_backend_config(cfg, "cfg.yaml");
  } catch (const std::runtime_error& e) {
    return e.what();
  }
  return "ok";
}

}  // namespace

TEST_CASE("registry: the library's seven patterns, one block each, sg from two") {
  std::set<std::string> names;
  for (const PatternInfo& p : registered_patterns()) names.insert(p.name);
  CHECK(names == std::set<std::string>{"deg_in", "deg_out", "fan_in", "fan_out", "sg_gather", "sg_scatter", "tcyc"});
  for (Block b : kAllBlocks) CHECK(!patterns_for(b).empty());
  CHECK(patterns_for(Block::sg).size() == 2);
  CHECK(compiled_k_max() == 10);
}

TEST_CASE("registry: start-up checks use the generated E402 locations and the compiled K_MAX") {
  const std::string blocks5 =
      "blocks:\n  fan_in: {window: 10, bins: [2]}\n  fan_out: {window: 10, bins: [2]}\n"
      "  deg_in: {window: 10, bins: [2]}\n  deg_out: {window: 10, bins: [2]}\n  sg: {window: 5, bins: [2]}\n";
  CHECK(error_of(parse_feature_config(blocks5 + "  tcyc: {window: 10, bins: [2]}\n", "cfg.yaml")) == "ok");
  CHECK(error_of(parse_feature_config(blocks5, "cfg.yaml")) ==
        "patterns/tcyc.yaml:4:8: error E402: pattern 'tcyc' emits into block 'tcyc', which the feature "
        "config cfg.yaml does not define");
  CHECK(error_of(parse_feature_config("k_max: 12\n" + blocks5 + "  tcyc: {window: 10, bins: [2]}\n", "cfg.yaml")) ==
        "patterns/tcyc.yaml: pattern 'tcyc' was compiled with K_MAX = 10, but cfg.yaml has k_max: 12");
  // The shipped configs pass.
  for (const char* c : {"configs/features_default.yaml", "configs/features_parity.yaml"})
    CHECK(error_of(load_feature_config(std::string(BLZ_SOURCE_DIR) + "/" + c)) == "ok");
}

// SPEC §13.2: the golden probes, evaluated by the generated code.
TEST_CASE("cpu backend matches the GFP golden probes") {
  std::ifstream in(std::string(BLZ_SOURCE_DIR) + "/tests/golden/gfp_probes.json");
  REQUIRE(in);
  const auto doc = nlohmann::json::parse(in);
  const auto bins = doc["params"]["bins"].get<std::vector<std::uint32_t>>();
  REQUIRE(doc["cases"].size() >= 30);
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
    const std::uint64_t m = src.size();
    for (Block b : kAllBlocks) {
      const auto got = mine(idx, b, {c["tw"].get<std::int32_t>(), bins}, 0, m);
      for (std::uint64_t e = 0; e < m; ++e) {
        const auto want = c["output"][e][std::string(block_name(b))].get<std::vector<std::uint32_t>>();
        const std::vector<std::uint32_t> row(got.begin() + std::ptrdiff_t(e * bins.size()),
                                             got.begin() + std::ptrdiff_t((e + 1) * bins.size()));
        INFO(c["id"].get<std::string>() << " edge " << e << " " << block_name(b));
        CHECK(row == want);
      }
    }
  }
}

// AC1, fast share: every row of every block on 1,000 fuzz graphs, both schedules.
TEST_CASE("cpu backend equals the oracle on fuzz graphs (seeds 0..999)") {
  const std::vector<std::uint32_t> bins = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
  const FuzzDiff d = fuzz_diff_cpu(0, 1000, bins);
  CHECK(d.graphs == 1000);
  for (Block b : kAllBlocks) {
    INFO(block_name(b));
    CHECK(d.mismatched_rows[int(b)] == 0);
  }
  INFO("first bad seed " << (d.bad_seeds.empty() ? -1 : std::int64_t(d.bad_seeds[0])));
  CHECK(d.bad_seeds.empty());
}

// SPEC v2 §6.1 (P10): the same under timestamp visibility, 500 graphs.
TEST_CASE("cpu backend equals the oracle on fuzz graphs under timestamp visibility (seeds 0..499)") {
  Profile ts = registry_profile();
  ts.visibility = Visibility::timestamp;
  const std::vector<std::uint32_t> bins = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
  const FuzzDiff d = fuzz_diff_cpu(ts, 0, 500, bins);
  CHECK(d.graphs == 500);
  for (const ProfileBlock& pb : ts.blocks) {
    INFO(block_name(pb.block));
    CHECK(d.mismatched_rows[int(pb.block)] == 0);
  }
  INFO("first bad seed " << (d.bad_seeds.empty() ? -1 : std::int64_t(d.bad_seeds[0])));
  CHECK(d.bad_seeds.empty());
}

// SPEC §8: rows do not depend on the trigger range, the thread count or the schedule.
TEST_CASE("cpu backend rows do not depend on the trigger range, threads or schedule") {
  for (std::uint64_t seed = 0; seed < 40; ++seed) {
    const FuzzGraph fg = fuzz_graph(seed);
    const GraphIndex idx(fg.src, fg.dst, fg.ts, fg.n);
    const std::uint64_t m = fg.src.size(), mid = m / 3;
    const BlockParams p{fg.window, {2, 3, 4, 5, 6, 8}};
    for (Block b : kAllBlocks) {
      omp_set_num_threads(1);
      const auto one = mine(idx, b, p, 0, m);
      omp_set_num_threads(4);
      CHECK(mine(idx, b, p, 0, m) == one);
      CHECK(mine(idx, b, p, 0, m, Schedule::sorted) == one);
      const auto tail = mine(idx, b, p, mid, m, Schedule::sorted);
      CHECK(std::equal(tail.begin(), tail.end(), one.begin() + std::ptrdiff_t(mid * p.bins.size())));
    }
  }
  omp_set_num_threads(omp_get_num_procs());
}

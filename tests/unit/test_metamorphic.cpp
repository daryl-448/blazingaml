// Metamorphic tests (SPEC v2 §3.4, §10): the fast share, 200 fuzz graphs per backend and
// profile. `blz metamorphic --seeds 0:2000` is the slow test.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <vector>

#include "runtime/fuzz_diff.hpp"
#include "runtime/metamorphic.hpp"

using namespace blz;

namespace {

void check_clean(const MetamorphicResult& r, std::uint64_t graphs, const Profile& prof, Backend b) {
  CHECK(r.graphs == graphs);
  CHECK(r.bad_seeds.empty());
  for (MetaTest t : kMetaTests) {
    INFO(meta_test_name(t));
    CHECK((r.rows_compared[int(t)] > 0) == meta_test_applies(t, prof, b));
    for (std::uint64_t n : r.mismatched_rows[int(t)]) CHECK(n == 0);
  }
}

}  // namespace

TEST_CASE("metamorphic: 200 fuzz graphs, oracle and cpu, the registry and v1_parity profiles") {
  const std::vector<std::uint32_t> bins = {2, 3, 4, 5, 6, 8, 10};
  for (Backend b : {Backend::oracle, Backend::cpu}) {
    INFO(backend_name(b));
    check_clean(metamorphic(registry_profile(), b, 0, 200, bins, {}), 200, registry_profile(), b);
  }
  const Profile parity = load_profile(std::string(BLZ_SOURCE_DIR) + "/configs/profiles/v1_parity.yaml");
  check_clean(metamorphic(parity, Backend::cpu, 200, 260, {}, {}), 60, parity, Backend::cpu);
}

TEST_CASE("metamorphic: 200 fuzz graphs, oracle and cpu, under timestamp visibility (SPEC v2 §6.1)") {
  Profile ts = registry_profile();
  ts.visibility = Visibility::timestamp;
  const std::vector<std::uint32_t> bins = {2, 3, 4, 5, 6, 8, 10};
  for (Backend b : {Backend::oracle, Backend::cpu}) {
    INFO(backend_name(b));
    const MetamorphicResult r = metamorphic(ts, b, 0, 200, bins, {});
    check_clean(r, 200, ts, b);
    // Visibility invariance compares the tcyc and tchain rows only, once per graph.
    CHECK(r.rows_compared[int(MetaTest::visibility_invariance)] == 2 * r.edges);
  }
}

TEST_CASE("metamorphic: every row of every block is compared") {
  const MetamorphicResult r = metamorphic(registry_profile(), Backend::oracle, 0, 20, std::vector<std::uint32_t>{2}, {});
  const std::uint64_t blocks = registry_profile().blocks.size();
  CHECK(r.graphs == 20);
  CHECK(r.rows_compared[int(MetaTest::relabeling)] == r.edges * blocks);
  CHECK(r.rows_compared[int(MetaTest::time_shift)] == r.edges * blocks);
  // Two cuts per graph, each shorter than the graph.
  CHECK(r.rows_compared[int(MetaTest::prefix_closure)] > 0);
  CHECK(r.rows_compared[int(MetaTest::prefix_closure)] < 2 * r.edges * blocks);
}

TEST_CASE("metamorphic: E404 for a profile the backend cannot run") {
  Profile p = registry_profile();
  for (ProfileBlock& pb : p.blocks)
    if (pb.block == Block::sg) pb.patterns.resize(1);
  CHECK_THROWS_WITH(metamorphic(p, Backend::oracle, 0, 1, {}, {}),
                    Catch::Matchers::ContainsSubstring("E404: the oracle computes block sg"));
}

#include <catch2/catch_test_macros.hpp>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <omp.h>

#include "graph/fuzz_gen.hpp"
#include "graph/index.hpp"
#include "oracle/oracle.hpp"
#include "runtime/cpu_backend.hpp"
#include "runtime/profile.hpp"

using namespace blz;

namespace {

struct Small {
  std::vector<std::uint32_t> src, dst;
  std::vector<std::int32_t> ts;
  std::uint64_t n = 0;
  EdgeStream view() const { return {src, dst, ts, n}; }
};

Small from_triples(const nlohmann::json& edges) {
  Small g;
  for (const auto& e : edges) {
    g.src.push_back(e[0]);
    g.dst.push_back(e[1]);
    g.ts.push_back(e[2]);
    g.n = std::max<std::uint64_t>(g.n, std::max(g.src.back(), g.dst.back()) + 1);
  }
  return g;
}

std::vector<std::uint32_t> mine_all(const Oracle& o, Block b, const BlockParams& p, int k_max,
                                    std::uint64_t first, std::uint64_t last) {
  std::vector<std::uint32_t> out((last - first) * p.bins.size(), 0);
  o.mine(b, p, k_max, first, last, out.data());
  return out;
}

}  // namespace

// SPEC §13.2: the oracle reproduces every committed GFP probe exactly.
TEST_CASE("oracle matches the GFP golden probes") {
  std::ifstream in(std::string(BLZ_SOURCE_DIR) + "/tests/golden/gfp_probes.json");
  REQUIRE(in);
  const auto doc = nlohmann::json::parse(in);
  const auto bins = doc["params"]["bins"].get<std::vector<std::uint32_t>>();
  REQUIRE(doc["cases"].size() >= 30);
  for (const auto& c : doc["cases"]) {
    const Small g = from_triples(c["edges"]);
    const Oracle o(g.view());
    const std::uint64_t m = g.src.size();
    for (Block b : kAllBlocks) {
      const auto got = mine_all(o, b, {c["tw"].get<std::int32_t>(), bins}, 10, 0, m);
      for (std::uint64_t e = 0; e < m; ++e) {
        const auto want = c["output"][e][std::string(block_name(b))].get<std::vector<std::uint32_t>>();
        const std::vector<std::uint32_t> row(got.begin() + e * bins.size(),
                                             got.begin() + (e + 1) * bins.size());
        INFO(c["id"].get<std::string>() << " edge " << e << " " << block_name(b));
        CHECK(row == want);
      }
    }
  }
}

TEST_CASE("tcyc respects k_max") {
  // A26: one 6-cycle closed by the last edge.
  const Small g = from_triples(nlohmann::json::parse(
      "[[30,31,1],[31,32,2],[32,33,3],[33,34,4],[34,35,5],[35,30,6]]"));
  const Oracle o(g.view());
  const BlockParams p{100, {2, 3, 4, 5, 6, 7}};
  CHECK(mine_all(o, Block::tcyc, p, 5, 5, 6) == std::vector<std::uint32_t>{0, 0, 0, 0, 0, 0});
  CHECK(mine_all(o, Block::tcyc, p, 6, 5, 6) == std::vector<std::uint32_t>{0, 0, 0, 0, 1, 0});
}

TEST_CASE("tcyc needs some parallel edge after the previous hop, not the earliest") {
  // 0->1 at 5 only; 1->2 at 1 and at 7; 2->0 closes at 9. Only the later 1->2 edge fits.
  const Small g = from_triples(nlohmann::json::parse("[[1,2,1],[0,1,5],[1,2,7],[2,0,9]]"));
  const Oracle o(g.view());
  CHECK(mine_all(o, Block::tcyc, {100, {2, 3}}, 10, 3, 4) == std::vector<std::uint32_t>{0, 1});
}

TEST_CASE("oracle rows do not depend on the trigger range or the thread count") {
  for (std::uint64_t seed = 0; seed < 30; ++seed) {
    const FuzzGraph fg = fuzz_graph(seed);
    const Oracle o({fg.src, fg.dst, fg.ts, fg.n});
    const std::uint64_t m = fg.src.size(), mid = m / 2;
    const BlockParams p{fg.window, {2, 3, 4, 5, 6, 8}};
    for (Block b : kAllBlocks) {
      omp_set_num_threads(1);
      const auto one = mine_all(o, b, p, 64, 0, m);
      omp_set_num_threads(4);
      CHECK(mine_all(o, b, p, 64, 0, m) == one);
      auto tail = mine_all(o, b, p, 64, mid, m);
      CHECK(std::equal(tail.begin(), tail.end(), one.begin() + mid * p.bins.size()));
    }
  }
}

// SPEC v2 §6.1: timestamp visibility, t' <= t, so a trigger also sees the later edges of its own
// timestamp. A4 and A15 of Appendix A under it.
TEST_CASE("timestamp visibility: same-timestamp edges later in the stream are visible") {
  const BlockParams p{100, {2, 3, 4}};
  const Small a4 = from_triples(nlohmann::json::parse("[[1,9,5],[2,9,5],[3,9,5]]"));
  const Oracle st(a4.view()), ts(a4.view(), Visibility::timestamp);
  CHECK(mine_all(st, Block::fan_in, p, 10, 0, 3) == std::vector<std::uint32_t>{0, 0, 0, 1, 0, 0, 0, 1, 0});
  CHECK(mine_all(ts, Block::fan_in, p, 10, 0, 3) == std::vector<std::uint32_t>{0, 1, 0, 0, 1, 0, 0, 1, 0});
  CHECK(mine_all(ts, Block::deg_in, p, 10, 0, 3) == std::vector<std::uint32_t>{0, 1, 0, 0, 1, 0, 0, 1, 0});
  // A15: 0->1, 0->2, 1->5, 2->5, all at ts 5. Under stream visibility only 2->5 completes the
  // instance (s = 0, w = 5, I = {1, 2}); under timestamp visibility every edge does, the first
  // two in the scatter role and the last two in the gather role.
  const Small a15 = from_triples(nlohmann::json::parse("[[0,1,5],[0,2,5],[1,5,5],[2,5,5]]"));
  CHECK(mine_all(Oracle(a15.view()), Block::sg, {100, {2}}, 10, 0, 4) == std::vector<std::uint32_t>{0, 0, 0, 1});
  CHECK(mine_all(Oracle(a15.view(), Visibility::timestamp), Block::sg, {100, {2}}, 10, 0, 4) ==
        std::vector<std::uint32_t>{1, 1, 1, 1});
  // The window's lower bound is unchanged: an open bound still drops t' = t - delta.
  const Small open = from_triples(nlohmann::json::parse("[[1,9,0],[2,9,10],[3,9,10]]"));
  CHECK(mine_all(Oracle(open.view(), Visibility::timestamp), Block::fan_in, {10, {2, 3}}, 10, 1, 2) ==
        std::vector<std::uint32_t>{1, 0});
  // A self-loop trigger still emits nothing.
  const Small loop = from_triples(nlohmann::json::parse("[[1,1,5],[2,1,5]]"));
  CHECK(mine_all(Oracle(loop.view(), Visibility::timestamp), Block::fan_in, p, 10, 0, 1) ==
        std::vector<std::uint32_t>{0, 0, 0});
}

// SPEC v2 §6.1, checked against an independent implementation: a trigger's row under timestamp
// visibility is its stream-visibility row in the stream where it is moved to the end of its own
// timestamp group (then exactly the edges with t' <= t precede it). The reference is the
// generated CPU backend under stream visibility, which shares no code with the oracle.
TEST_CASE("timestamp visibility = stream visibility with the trigger last in its timestamp group") {
  const Profile reg = registry_profile();
  std::uint64_t moved = 0;
  omp_set_num_threads(1);  // one tiny CPU-backend run per trigger: thread start-up would dominate
  for (std::uint64_t seed = 0; seed < 200; ++seed) {
    const FuzzGraph fg = fuzz_graph(seed);
    const std::uint64_t m = fg.src.size();
    const Oracle ts({fg.src, fg.dst, fg.ts, fg.n}, Visibility::timestamp);
    for (const ProfileBlock& pb : reg.blocks) {
      const BlockParams p{fg.window, {2, 3, 4, 5, 6, 8}};
      const std::size_t nb = p.bins.size();
      const auto want = mine_all(ts, pb.block, p, reg.k_max, 0, m);
      for (std::uint64_t a = 0; a < m;) {
        std::uint64_t b = a;
        while (b < m && fg.ts[b] == fg.ts[a]) ++b;  // [a, b) = one timestamp group
        for (std::uint64_t e = a; e < b; ++e) {
          std::vector<std::uint32_t> src(fg.src), dst(fg.dst);
          std::rotate(src.begin() + std::ptrdiff_t(e), src.begin() + std::ptrdiff_t(e) + 1,
                      src.begin() + std::ptrdiff_t(b));
          std::rotate(dst.begin() + std::ptrdiff_t(e), dst.begin() + std::ptrdiff_t(e) + 1,
                      dst.begin() + std::ptrdiff_t(b));
          const GraphIndex index(src, dst, fg.ts, fg.n);
          TriggerSel one;
          one.first = b - 1;
          one.last = b;
          std::vector<std::uint32_t> got(nb, 0);
          mine_cpu(index.view(), pb.patterns, p, Visibility::stream, one, Schedule::dynamic, got.data());
          INFO("seed " << seed << " eid " << e << " " << block_name(pb.block));
          CHECK(std::equal(got.begin(), got.end(), want.begin() + std::ptrdiff_t(e * nb)));
          moved += e + 1 < b;
        }
        a = b;
      }
    }
  }
  omp_set_num_threads(omp_get_num_procs());
  CHECK(moved > 50000);  // fuzz graphs have many ties, so most triggers do move
}

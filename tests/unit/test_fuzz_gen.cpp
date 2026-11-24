#include <catch2/catch_test_macros.hpp>
#include <algorithm>

#include "graph/fuzz_gen.hpp"

using namespace blz;

TEST_CASE("fuzz graphs are reproducible and within the SPEC §13.3 ranges") {
  for (std::uint64_t seed = 0; seed < 200; ++seed) {
    const FuzzGraph g = fuzz_graph(seed), h = fuzz_graph(seed);
    CHECK(g.src == h.src);
    CHECK(g.dst == h.dst);
    CHECK(g.ts == h.ts);
    CHECK(g.window == h.window);
    CHECK(g.n >= 3);
    CHECK(g.n <= 40);
    CHECK(g.src.size() >= 5);
    CHECK(g.src.size() <= 400);
    CHECK(g.window >= 1);
    for (std::size_t e = 0; e < g.src.size(); ++e) {
      CHECK(g.src[e] < g.n);
      CHECK(g.dst[e] < g.n);
      if (e > 0) CHECK(g.ts[e - 1] <= g.ts[e]);
    }
  }
  CHECK(fuzz_graph(1).src != fuzz_graph(2).src);
}

namespace {

// FNV-1a over every field of the graphs of seeds [s0, s1).
std::uint64_t graphs_hash(std::uint64_t s0, std::uint64_t s1) {
  std::uint64_t h = 14695981039346656037ULL;
  const auto add = [&](std::uint64_t x) {
    for (int i = 0; i < 8; ++i, x >>= 8) h = (h ^ (x & 0xff)) * 1099511628211ULL;
  };
  for (std::uint64_t seed = s0; seed < s1; ++seed) {
    const FuzzGraph g = fuzz_graph(seed);
    add(g.n);
    add(std::uint64_t(g.window));
    add(g.src.size());
    for (std::size_t e = 0; e < g.src.size(); ++e) {
      add(g.src[e]);
      add(g.dst[e]);
      add(std::uint64_t(std::int64_t(g.ts[e])));
    }
  }
  return h;
}

}  // namespace

// The dense graphs feed recorded results (results/ac1_*_run.log, the fuzz-diff and metamorphic
// reports), so the generator must not change them.
TEST_CASE("the dense fuzz graphs are the recorded ones") { CHECK(graphs_hash(0, 1000) == 11919179825238948272ULL); }

// SPEC v3 §4.4, R18: windows far below the time span, edges on the window bounds, several components.
TEST_CASE("sparse-time fuzz graphs are reproducible, within their ranges, and exercise what they claim") {
  FuzzHits hits;
  int long_spans = 0;
  for (std::uint64_t seed = 0; seed < 300; ++seed) {
    const FuzzGraph g = fuzz_graph(seed, FuzzSetting::sparse), h = fuzz_graph(seed, FuzzSetting::sparse);
    CHECK(g.src == h.src);
    CHECK(g.dst == h.dst);
    CHECK(g.ts == h.ts);
    CHECK(g.window == h.window);
    CHECK(g.src != fuzz_graph(seed).src);
    CHECK(g.n >= 6);
    CHECK(g.n <= 60);
    CHECK(g.src.size() >= 20);
    CHECK(g.src.size() <= 400);
    CHECK(g.window >= 2);
    CHECK(g.window <= 41);
    for (std::size_t e = 0; e < g.src.size(); ++e) {
      CHECK(g.src[e] < g.n);
      CHECK(g.dst[e] < g.n);
      CHECK(g.ts[e] >= 0);
      if (e > 0) CHECK(g.ts[e - 1] <= g.ts[e]);
    }
    long_spans += g.ts.back() - g.ts.front() >= 10 * g.window;
    hits += fuzz_hits(g);
  }
  CHECK(long_spans >= 290);
  CHECK(hits.multi_component_graphs == 300);
  CHECK(hits.components >= 900);
  CHECK(hits.bound_out * 4 >= hits.triggers);  // a quarter of the triggers, each bound
  CHECK(hits.bound_in * 4 >= hits.triggers);
  CHECK(hits.tied * 2 >= hits.triggers);
  CHECK(hits.duplicated * 10 >= hits.triggers);
  CHECK(parse_fuzz_setting("sparse") == FuzzSetting::sparse);
  CHECK(std::string(fuzz_setting_name(FuzzSetting::dense)) == "dense");
  CHECK_THROWS_AS(parse_fuzz_setting("Sparse"), std::invalid_argument);
}

TEST_CASE("fuzz_hits: components, bound edges at t - delta and t - delta + 1, ties and duplicates") {
  FuzzGraph g;
  g.n = 6;
  g.window = 2;
  g.src = {0, 1, 1, 2, 3, 4};
  g.dst = {1, 2, 2, 0, 3, 5};
  g.ts = {0, 1, 1, 2, 2, 5};
  const FuzzHits h = fuzz_hits(g);
  CHECK(h.graphs == 1);
  CHECK(h.components == 3);  // {0, 1, 2}, {3} (a self-loop), {4, 5}
  CHECK(h.multi_component_graphs == 1);
  CHECK(h.triggers == 5);
  CHECK(h.bound_out == 1);  // e3 (2 -> 0 at 2): e0 at node 0 at 2 - 2
  CHECK(h.bound_in == 3);   // e1, e2: e0 at node 1 at 1 - 2 + 1; e3: e1 at node 2 at 2 - 2 + 1
  CHECK(h.tied == 4);
  CHECK(h.duplicated == 2);
  // delta = 1: t - delta + 1 is the trigger's own time, and the trigger is not its own bound edge.
  g.window = 1;
  g.src = {0, 2, 3};
  g.dst = {1, 3, 4};
  g.ts = {0, 0, 0};
  CHECK(fuzz_hits(g).bound_in == 2);  // e1 and e2 share node 3; e0 is alone
}

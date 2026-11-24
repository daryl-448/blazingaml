#include <catch2/catch_test_macros.hpp>

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

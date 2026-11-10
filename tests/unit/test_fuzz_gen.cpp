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

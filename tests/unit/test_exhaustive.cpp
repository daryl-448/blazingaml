// L2's graph domain (SPEC v3 §4.3): the canonical enumeration lists every edge list of a bound once
// per node relabelling, and a disjoint union keeps each member as it is alone.
#include <catch2/catch_test_macros.hpp>

#include <map>
#include <optional>
#include <set>
#include <tuple>

#include "graph/exhaustive.hpp"
#include "runtime/fuzz_diff.hpp"

using namespace blz;

namespace {

using Key = std::tuple<std::vector<std::uint32_t>, std::vector<std::uint32_t>, std::vector<std::int32_t>>;

// The canonical form of an edge list: ts shifted so ts[0] = 0, node ids renamed in first-appearance
// order over src0, dst0, src1, dst1, ...
Key canonical(const std::vector<std::uint32_t>& src, const std::vector<std::uint32_t>& dst,
              const std::vector<std::int32_t>& ts) {
  std::map<std::uint32_t, std::uint32_t> name;
  const auto rename = [&](std::uint32_t x) { return name.emplace(x, std::uint32_t(name.size())).first->second; };
  Key k;
  for (std::size_t e = 0; e < src.size(); ++e) {
    std::get<0>(k).push_back(rename(src[e]));
    std::get<1>(k).push_back(rename(dst[e]));
    std::get<2>(k).push_back(ts[e] - ts[0]);
  }
  return k;
}

}  // namespace

TEST_CASE("exhaustive: every edge list of the bound once per relabelling, m ascending", "[exhaustive]") {
  for (const ExhaustiveBound b : {ExhaustiveBound{3, 3, 3}, ExhaustiveBound{3, 4, 2}, ExhaustiveBound{2, 4, 4},
                                  ExhaustiveBound{4, 2, 2}, ExhaustiveBound{1, 1, 1}}) {
    INFO(bound_name(b));
    // Brute force: every node string over [0, n_max) and every non-decreasing ts sequence in [0, t).
    std::set<Key> want;
    for (std::uint32_t m = 1; m <= b.m_max; ++m) {
      std::vector<std::uint32_t> id(2 * m, 0);
      std::vector<std::int32_t> ts(m, 0);
      for (bool more_ids = true; more_ids;) {
        for (bool more_ts = true; more_ts;) {
          bool sorted = true;
          for (std::uint32_t k = 1; k < m; ++k) sorted = sorted && ts[k - 1] <= ts[k];
          if (sorted) {
            std::vector<std::uint32_t> src(m), dst(m);
            for (std::uint32_t k = 0; k < m; ++k) src[k] = id[2 * k], dst[k] = id[2 * k + 1];
            want.insert(canonical(src, dst, ts));
          }
          std::uint32_t p = 0;
          while (p < m && ++ts[p] == b.t) ts[p++] = 0;
          more_ts = p < m;
        }
        std::uint32_t p = 0;
        while (p < 2 * m && ++id[p] == b.n_max) id[p++] = 0;
        more_ids = p < 2 * m;
      }
    }
    const ExhaustiveGraphs gs(b);
    std::set<Key> got;
    std::uint32_t prev_m = 1;
    for (std::uint64_t i = 0; i < gs.size(); ++i) {
      const EdgeList g = gs.graph(i);
      const std::uint32_t m = std::uint32_t(g.src.size());
      CHECK(gs.edges(i) == m);
      CHECK(m >= prev_m);
      prev_m = m;
      const Key k = canonical(g.src, g.dst, g.ts);
      CHECK(k == Key{g.src, g.dst, g.ts});  // already canonical
      CHECK(got.insert(k).second);          // listed once
      std::uint32_t n = 0;
      for (std::uint32_t k2 = 0; k2 < m; ++k2) n = std::max({n, g.src[k2] + 1, g.dst[k2] + 1});
      CHECK(g.n == n);
      CHECK(g.n <= b.n_max);
    }
    CHECK(gs.size() == want.size());
    CHECK(got == want);
  }
}

TEST_CASE("exhaustive: graph counts are the restricted-growth strings times the ts sequences", "[exhaustive]") {
  // m = 5, n <= 4: S(10,1..4) = 1 + 511 + 9330 + 34105 node strings; ts: C(4 + 3, 3) = 35 at t = 4.
  const ExhaustiveGraphs gs({5, 4, 4});
  CHECK(gs.first(6) - gs.first(5) == 43947ull * 35);
  // m = 4: S(8,1..4) = 1 + 127 + 966 + 1701; C(3 + 3, 3) = 20.
  CHECK(gs.first(5) - gs.first(4) == 2795ull * 20);
  CHECK(gs.size() == gs.first(6));
  CHECK_THROWS(parse_exhaustive_bound("m=5,n=4"));
  CHECK_THROWS(parse_exhaustive_bound("m=5,n=4,t=0"));
  const ExhaustiveBound b = parse_exhaustive_bound("t=4,m=5,n=4");
  CHECK((b.m_max == 5 && b.n_max == 4 && b.t == 4));
  CHECK(bound_name(b) == "m=5,n=4,t=4");
}

TEST_CASE("exhaustive: a disjoint union keeps each member as it is alone", "[exhaustive]") {
  const ExhaustiveGraphs gs({4, 4, 3});
  for (const auto [first, last] : {std::pair<std::uint64_t, std::uint64_t>{0, gs.size()},
                                   {gs.first(4) + 17, gs.first(4) + 400}, {5, 6}}) {
    const GraphUnion u = disjoint_union(gs, first, last);
    std::vector<std::uint32_t> base(last - first + 1, 0);
    for (std::uint64_t i = first; i < last; ++i) base[i - first + 1] = base[i - first] + gs.graph(i).n;
    CHECK(u.g.n == base.back());
    std::uint64_t m = 0;
    for (std::uint64_t i = first; i < last; ++i) m += gs.edges(i);
    REQUIRE(u.g.src.size() == m);
    std::map<std::uint64_t, std::uint32_t> next;  // member -> its next local eid
    for (std::uint64_t e = 0; e < m; ++e) {
      if (e > 0) {  // merged by (ts, member, local eid)
        CHECK(u.g.ts[e - 1] <= u.g.ts[e]);
        if (u.g.ts[e - 1] == u.g.ts[e]) CHECK(std::pair(u.graph[e - 1], u.local[e - 1]) < std::pair(u.graph[e], u.local[e]));
      }
      const std::uint64_t i = u.graph[e];
      REQUIRE((i >= first && i < last));
      CHECK(u.local[e] == next[i]++);  // the member's eid order, unchanged
      const EdgeList g = gs.graph(i);
      CHECK(u.g.src[e] == base[i - first] + g.src[u.local[e]]);
      CHECK(u.g.dst[e] == base[i - first] + g.dst[u.local[e]]);
      CHECK(u.g.ts[e] == g.ts[u.local[e]]);
    }
  }
}

TEST_CASE("exhaustive fuzz-diff: every graph checked at every window, the first failure alone", "[exhaustive]") {
  const ExhaustiveBound b{3, 3, 2};
  const ExhaustiveGraphs gs(b);
  std::uint64_t edges = 0;
  for (std::uint64_t i = 0; i < gs.size(); ++i) edges += gs.edges(i);
  const Profile prof = registry_profile();
  const std::vector<std::uint32_t> bins = {2, 3, 4};
  for (const std::uint64_t batch : {1, 7, 1 << 14}) {
    INFO("batch edges " << batch);
    const FuzzDiff d = fuzz_diff_cpu(prof, DiffSource::exhaustive(b, batch), bins);
    CHECK(d.graphs == gs.size());
    CHECK(d.edges == edges);
    CHECK(d.trigger_checks == edges * 2);  // windows 1 and 2
    CHECK(d.bad_seeds.empty());
    CHECK(!d.counterexample);
  }
  // A profile whose fan_in block runs the fan_out pattern disagrees with the oracle's fan_in: the
  // counterexample is the first such graph in canonical order, whatever the batch size, re-run alone.
  Profile wrong = prof;
  for (ProfileBlock& pb : wrong.blocks)
    if (pb.block == Block::fan_in) pb.patterns = patterns_for(Block::fan_out);
  std::optional<std::uint64_t> first;
  for (const std::uint64_t batch : {1, 7, 1 << 14}) {
    INFO("batch edges " << batch);
    const FuzzDiff d = fuzz_diff_cpu(wrong, DiffSource::exhaustive(b, batch), bins);
    REQUIRE(d.counterexample);
    const Counterexample& cx = *d.counterexample;
    CHECK(cx.alone);
    CHECK(cx.first.block == "fan_in");
    CHECK(cx.first.got != cx.first.want);
    CHECK(cx.g.src == gs.graph(cx.graph).src);
    CHECK(cx.g.src.size() == 2);  // two in-neighbours of v, one out-neighbour of u
    if (!first) first = cx.graph;
    CHECK(cx.graph == *first);
  }
}

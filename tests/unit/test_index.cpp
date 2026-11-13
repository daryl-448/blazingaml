// T-CSR / P-CSR construction and every §5.3 primitive against naive scans (SPEC §13.1), on fuzz
// graphs: random multigraphs with ties, parallel edges and self-loops.
#include <catch2/catch_test_macros.hpp>
#include <limits>
#include <map>
#include <set>
#include <vector>

#include "graph/fuzz_gen.hpp"
#include "graph/index.hpp"

using namespace blz;

namespace {

struct Lcg {
  std::uint64_t s;
  std::uint32_t below(std::uint32_t k) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return std::uint32_t((s >> 33) % k);
  }
};

// The naive list of x in one direction: (nbr, ts, eid) of its edges, in eid order.
struct Entry {
  std::uint32_t nbr;
  std::int32_t ts;
  std::uint32_t eid;
};
std::vector<Entry> naive_list(const FuzzGraph& g, std::uint32_t x, bool out) {
  std::vector<Entry> l;
  for (std::uint32_t e = 0; e < g.src.size(); ++e)
    if ((out ? g.src[e] : g.dst[e]) == x) l.push_back({out ? g.dst[e] : g.src[e], g.ts[e], e});
  return l;
}

bool in_bounds(const Entry& en, std::int64_t lo, std::int64_t hi, std::uint32_t max_eid) {
  return en.ts >= lo && en.ts <= hi && en.eid <= max_eid;
}

}  // namespace

TEST_CASE("index: T-CSR lists are the eid-ordered incident edges, prev_same links equal neighbours") {
  for (std::uint64_t seed = 0; seed < 200; ++seed) {
    const FuzzGraph fg = fuzz_graph(seed);
    const GraphIndex idx(fg.src, fg.dst, fg.ts, fg.n);
    const GraphView& g = idx.view();
    REQUIRE(g.m == fg.src.size());
    for (bool out : {true, false}) {
      const TCsrView& t = out ? g.out : g.in;
      REQUIRE(t.off[fg.n] == fg.src.size());
      for (std::uint32_t x = 0; x < fg.n; ++x) {
        const auto want = naive_list(fg, x, out);
        REQUIRE(t.off[x + 1] - t.off[x] == want.size());
        for (std::uint32_t k = 0; k < want.size(); ++k) {
          const std::uint32_t j = t.off[x] + k;
          INFO("seed " << seed << " x " << x << " k " << k);
          CHECK(t.nbr[j] == want[k].nbr);
          CHECK(t.ts[j] == want[k].ts);
          CHECK(t.eid[j] == want[k].eid);
          std::int32_t prev = -1;
          for (std::uint32_t q = 0; q < k; ++q)
            if (want[q].nbr == want[k].nbr) prev = std::int32_t(t.off[x] + q);
          CHECK(t.prev_same[j] == prev);
        }
      }
    }
  }
}

TEST_CASE("index: P-CSR holds each node's distinct neighbours by id, each with its eid-ordered run") {
  for (std::uint64_t seed = 0; seed < 200; ++seed) {
    const FuzzGraph fg = fuzz_graph(seed);
    const GraphIndex idx(fg.src, fg.dst, fg.ts, fg.n);
    for (bool out : {true, false}) {
      const PCsrView& p = out ? idx.view().pout : idx.view().pin;
      for (std::uint32_t x = 0; x < fg.n; ++x) {
        std::map<std::uint32_t, std::vector<Entry>> want;
        for (const Entry& en : naive_list(fg, x, out)) want[en.nbr].push_back(en);
        REQUIRE(p.off[x + 1] - p.off[x] == want.size());
        std::uint32_t pair = p.off[x];
        for (const auto& [y, run] : want) {
          INFO("seed " << seed << " x " << x << " y " << y);
          CHECK(p.nbr[pair] == y);
          REQUIRE(p.run[pair + 1] - p.run[pair] == run.size());
          for (std::uint32_t k = 0; k < run.size(); ++k) {
            CHECK(p.run_ts[p.run[pair] + k] == run[k].ts);
            CHECK(p.run_eid[p.run[pair] + k] == run[k].eid);
          }
          ++pair;
        }
      }
    }
  }
}

TEST_CASE("primitives: range, distinct_first, pair_exists, pair_latest and the counts equal naive scans") {
  Lcg r{7};
  for (std::uint64_t seed = 0; seed < 300; ++seed) {
    const FuzzGraph fg = fuzz_graph(seed);
    const GraphIndex idx(fg.src, fg.dst, fg.ts, fg.n);
    const GraphView& g = idx.view();
    const std::int32_t T = fg.ts.back() + 1;
    for (int q = 0; q < 100; ++q) {
      const bool out = r.below(2) == 0;
      const TCsrView& t = out ? g.out : g.in;
      const PCsrView& pc = out ? g.pout : g.pin;
      const std::uint32_t x = r.below(fg.n);
      const std::int64_t lo = std::int64_t(r.below(std::uint32_t(T) + 4)) - 2;
      const std::int64_t hi = r.below(4) == 0 ? kNoUpper : lo + std::int64_t(r.below(std::uint32_t(T) + 2)) - 1;
      // kAnyEid: no eid bound (timestamp visibility, SPEC v2 §6.1).
      const std::uint32_t max_eid = r.below(5) == 0 ? kAnyEid : r.below(std::uint32_t(fg.src.size()) + 1);
      INFO("seed " << seed << " out " << out << " x " << x << " lo " << lo << " hi " << hi << " eid<= " << max_eid);
      const auto list = naive_list(fg, x, out);

      // range: exactly the entries within the bounds, as one contiguous run of x's list.
      const Range rg = t.range(x, lo, hi, max_eid);
      REQUIRE(rg.lo >= t.off[x]);
      REQUIRE(rg.hi <= t.off[x + 1]);
      REQUIRE(rg.lo <= rg.hi);
      std::uint32_t inside = 0;
      for (std::uint32_t k = 0; k < list.size(); ++k) {
        const bool want = in_bounds(list[k], lo, hi, max_eid);
        const std::uint32_t j = t.off[x] + k;
        CHECK(want == (j >= rg.lo && j < rg.hi));
        inside += want;
      }
      CHECK(rg.size() == inside);

      // distinct_first: each neighbour once, at its earliest in-range witness.
      std::map<std::uint32_t, std::int32_t> earliest;
      for (const Entry& en : list)
        if (in_bounds(en, lo, hi, max_eid) && !earliest.count(en.nbr)) earliest[en.nbr] = en.ts;
      std::map<std::uint32_t, std::int32_t> got;
      for (std::uint32_t j = rg.lo; j < rg.hi; ++j)
        if (t.distinct_first(j, rg.lo)) {
          CHECK(!got.count(t.nbr[j]));
          got[t.nbr[j]] = t.ts[j];
        }
      CHECK(got == earliest);

      // count_distinct / count_edges, excluding one node.
      const std::uint32_t skip = r.below(fg.n);
      std::uint32_t edges = 0;
      for (const Entry& en : list) edges += in_bounds(en, lo, hi, max_eid) && en.nbr != skip;
      auto keep = [&](std::uint32_t y) { return y != skip; };
      CHECK(count_edges(t, rg, keep) == edges);
      CHECK(count_distinct(t, rg, keep) == earliest.size() - earliest.count(skip));

      // pair_exists: the earliest in-bounds edge between x and y in this direction.
      for (std::uint32_t y = 0; y < fg.n; ++y) {
        std::int64_t want = kNoTs;
        for (const Entry& en : list)
          if (en.nbr == y && in_bounds(en, lo, hi, max_eid)) {
            want = en.ts;
            break;
          }
        CHECK(pc.pair_exists(x, y, lo, hi, max_eid) == want);
        // pair_latest (SPEC v2 §6.2): the latest in-bounds edge.
        std::int64_t latest = kNoTs;
        for (const Entry& en : list)
          if (en.nbr == y && in_bounds(en, lo, hi, max_eid)) latest = en.ts;
        CHECK(pc.pair_latest(x, y, lo, hi, max_eid) == latest);
      }
    }
  }
}

TEST_CASE("index: rejects an unsorted stream and out-of-range node ids") {
  const std::vector<std::uint32_t> src = {0, 1}, dst = {1, 0};
  const std::vector<std::int32_t> ts = {5, 4};
  CHECK_THROWS_AS(GraphIndex(src, dst, ts, 2), std::invalid_argument);
  const std::vector<std::int32_t> ok = {4, 5};
  CHECK_THROWS_AS(GraphIndex(src, dst, ok, 1), std::invalid_argument);
  CHECK_NOTHROW(GraphIndex(src, dst, ok, 2));
}

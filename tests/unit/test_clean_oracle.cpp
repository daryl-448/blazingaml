// The clean oracle (SPEC v3 §3.2, P13 F6, F7): one test per rule R1-R11 on hand-checked graphs, the
// tie permutation on fuzz graphs, and with every quirk on the legacy oracle's rows and vstats_cpu's
// bits on fuzz graphs.
#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <map>
#include <tuple>
#include <vector>

#include <omp.h>

#include "graph/fuzz_gen.hpp"
#include "graph/index.hpp"
#include "oracle/clean_oracle.hpp"
#include "oracle/oracle.hpp"
#include "runtime/vstats.hpp"

using namespace blz;

namespace {

struct Edge {
  std::uint32_t u, v;
  std::int32_t t;
  double amount = 1;
};

struct Graph {
  std::vector<std::uint32_t> src, dst;
  std::vector<std::int32_t> ts;
  std::vector<double> amount;
  std::uint64_t n = 0;
  Graph(std::initializer_list<Edge> edges) {
    for (const Edge& e : edges) {
      src.push_back(e.u);
      dst.push_back(e.v);
      ts.push_back(e.t);
      amount.push_back(e.amount);
      n = std::max<std::uint64_t>(n, std::max(e.u, e.v) + 1);
    }
  }
  std::uint64_t m() const { return src.size(); }
  EdgeStream view() const { return {src, dst, ts, n}; }
};

using Row = std::vector<std::uint32_t>;

// The blocks with one instance per trigger, which can be count blocks (R7).
constexpr std::array kCountBlocks = {Block::fan_in,         Block::fan_out,   Block::deg_in, Block::deg_out,
                                     Block::fan_oneway_out, Block::fan_total, Block::gs};

BlockParams count(std::int32_t window) { return {window, {}}; }

// Every trigger's row of block b, row after row.
Row rows(const Graph& g, Block b, const BlockParams& p, Quirks q = {}, int k_max = 10) {
  Row out(g.m() * block_cols(p), 0);
  CleanOracle(g.view(), g.amount, q).mine(b, p, k_max, 0, g.m(), out.data());
  return out;
}

Row row(const Graph& g, Block b, const BlockParams& p, std::uint32_t e, Quirks q = {}, int k_max = 10) {
  Row out(block_cols(p), 0);
  const std::uint32_t eids[] = {e};
  CleanOracle(g.view(), g.amount, q).mine(b, p, k_max, eids, out.data());
  return out;
}

std::vector<double> vs_row(const Graph& g, std::int32_t window, std::uint32_t e, Quirks q = {}) {
  std::vector<double> out(kVstatsCols);
  CleanOracle(g.view(), g.amount, q).vstats(window, e, e + 1, out.data());
  return out;
}

// Side k of a vertex-statistics row: 0 u_out, 1 u_in, 2 v_out, 3 v_in.
std::vector<double> side(const std::vector<double>& r, int k) {
  return {r.begin() + k * kVstatsPerSide, r.begin() + (k + 1) * kVstatsPerSide};
}

bool same_bits(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (std::memcmp(&a[i], &b[i], sizeof(double)) != 0) {
      UNSCOPED_INFO("first difference at " << i << ": " << a[i] << " vs " << b[i]);
      return false;
    }
  return true;
}

// Rows, or the smallest eid over the work limit (R11: such a case is inconclusive).
struct Outcome {
  Row rows;
  std::uint64_t aborted = UINT64_MAX;
  bool operator==(const Outcome&) const = default;
};
template <class Mine>
Outcome outcome(Mine mine) {
  try {
    return {mine(), UINT64_MAX};
  } catch (const WorkLimitExceeded& e) {
    return {{}, e.eid};
  }
}

const std::vector<std::uint32_t> kFuzzBins = {2, 3, 4, 5, 6, 8};

// Fuzz graphs are tiny: one thread per oracle run, or thread start-up dominates.
struct OneThread {
  int threads = omp_get_max_threads();
  OneThread() { omp_set_num_threads(1); }
  ~OneThread() { omp_set_num_threads(threads); }
};

}  // namespace

TEST_CASE("R1: a self-loop is never a neighbour, a witness or a value, and a self-loop trigger's pattern rows are zero") {
  const Graph g{
      {0, 0, 1, 1000},  // e0: s's self-loop
      {0, 1, 1, 1},     // e1: s -> a
      {1, 3, 1, 2},     // e2: a -> w
      {0, 3, 1, 3},     // e3: s -> w
      {0, 2, 2, 4},     // e4: s -> x
      {2, 3, 2, 5},     // e5: x -> w
  };
  // e0 is a self-loop trigger: zero in every block, though u = 0 has out-edges e1 and e3.
  for (Block b : kAllBlocks) {
    INFO(block_name(b));
    CHECK(row(g, b, {100, {2, 3}}, 0) == Row{0, 0});
  }
  for (Block b : kCountBlocks) {
    INFO(block_name(b));
    CHECK(row(g, b, count(100), 0) == Row{0});
  }
  // e5 = x -> w, gather source s = 0: I(0, 3) = N_out(0) ∩ N_in(3) = {1, 2}, size 2. The self-loop
  // e0 does not put s in N_out(s); v1 (Q1) did, with s in N_in(w) through e3: size 3.
  CHECK(row(g, Block::sg, {100, {2, 3}}, 5) == Row{1, 0});
  CHECK(row(g, Block::sg, {100, {2, 3}}, 5, parse_quirks("Q1")) == Row{0, 1});
  // sg_ord, the same instance: y = 1 (e1, e2 at 1, 1) and y = 2 (e4, e5 at 2, 2): size 2.
  CHECK(row(g, Block::sg_ord, {100, {2, 3}}, 5) == Row{1, 0});
  // e4: u = 0's window out-edges are e1, e3, e4 (not e0): degree 3, neighbours {1, 3, 2}.
  CHECK(row(g, Block::deg_out, count(100), 4) == Row{3});
  CHECK(row(g, Block::fan_out, count(100), 4) == Row{3});
  // e0's vertex statistics, over u = v = 0's non-loop edges at t = 1: out-edges e1, e3 (ages 0, 0;
  // amounts 1, 3: mean 2, sum 4, var 1, skew 0, kurt 1); no in-edge (e0's 1000 is not a value).
  const std::vector<double> u_out = {2, 2, 1, 0, 0, 0, 0, 0, 2, 4, 1, 0, 1}, none(kVstatsPerSide, 0.0);
  const std::vector<double> r = vs_row(g, 100, 0);
  CHECK(same_bits(side(r, 0), u_out));
  CHECK(same_bits(side(r, 1), none));
  CHECK(same_bits(side(r, 2), u_out));
  CHECK(same_bits(side(r, 3), none));
}

TEST_CASE("R2: every pattern and the vertex statistics see (t - delta, t], so an edge at exactly t - delta is out") {
  // Each graph's last edge is the trigger at t = 10 and its earliest edge is at 0: out with
  // delta = 10, in with delta = 11.
  const Graph fan{{1, 9, 0}, {2, 9, 1}, {3, 9, 10}};
  CHECK(row(fan, Block::fan_in, count(10), 2) == Row{2});  // {2, 3}
  CHECK(row(fan, Block::fan_in, count(11), 2) == Row{3});  // {1, 2, 3}
  CHECK(row(fan, Block::deg_in, count(10), 2) == Row{2});
  CHECK(row(fan, Block::deg_in, count(11), 2) == Row{3});
  // v_in's degree: e1, e2 with W = 10; e0 too with W = 11, or with GFP's window [ts[e-1] - W, t] (Q9).
  CHECK(vs_row(fan, 10, 2)[3 * kVstatsPerSide + 1] == 2);
  CHECK(vs_row(fan, 11, 2)[3 * kVstatsPerSide + 1] == 3);
  CHECK(vs_row(fan, 10, 2, parse_quirks("Q9"))[3 * kVstatsPerSide + 1] == 3);
  // sg and sg_ord for the trigger 2 -> 3 with source 0: I(0, 3) = {2}, plus 1 once s -> a (e0) is in.
  const Graph sg{{0, 1, 0}, {1, 3, 5}, {0, 2, 5}, {2, 3, 10}};
  for (Block b : {Block::sg, Block::sg_ord}) {
    INFO(block_name(b));
    CHECK(row(sg, b, {10, {2}}, 3) == Row{0});
    CHECK(row(sg, b, {11, {2}}, 3) == Row{1});
    CHECK(row(sg, b, {10, {2}}, 3, parse_quirks("Q3")) == Row{1});  // v1's closed bound
  }
  // The trigger 0 -> 1: the cycle 0 -> 1 -> 0 through e0 (tcyc, lcyc) and the chain 2 -> 0 -> 1
  // through e1 (tchain), both at 0.
  const Graph cyc{{1, 0, 0}, {2, 0, 0}, {0, 1, 10}};
  for (Block b : {Block::tcyc, Block::lcyc, Block::tchain}) {
    INFO(block_name(b));
    CHECK(row(cyc, b, {10, {2}}, 2) == Row{0});
    CHECK(row(cyc, b, {11, {2}}, 2) == Row{1});
  }
}

TEST_CASE("R3: a trigger sees every edge with t' <= t, and rows do not depend on the order within a timestamp") {
  // Three edges into 9 at the same time: each trigger sees all three (stream visibility, Q4: 1, 2, 3).
  const Graph tie{{1, 9, 5}, {2, 9, 5}, {3, 9, 5}};
  CHECK(rows(tie, Block::fan_in, count(100)) == Row{3, 3, 3});
  CHECK(rows(tie, Block::deg_in, count(100)) == Row{3, 3, 3});
  CHECK(rows(tie, Block::fan_in, count(100), parse_quirks("Q4")) == Row{1, 2, 3});
  // The cycle 0 -> 1 -> 0 at one time: e0 sees e1, which comes later in the stream.
  const Graph cyc{{0, 1, 5}, {1, 0, 5}};
  CHECK(rows(cyc, Block::tcyc, {100, {2}}) == Row{1, 1});
  CHECK(rows(cyc, Block::tcyc, {100, {2}}, parse_quirks("Q4")) == Row{0, 1});

  // On fuzz graphs: (a) reversing every timestamp group moves each row with its edge; (b) two edges
  // with the same (u, v, t) have the same rows. k_max 6 keeps the cycle and chain searches short.
  const OneThread one;
  std::uint64_t moved = 0, duplicates = 0;
  for (std::uint64_t seed = 0; seed < 100; ++seed) {
    const FuzzGraph fg = fuzz_graph(seed);
    const std::uint64_t m = fg.src.size();
    const std::vector<double> amount = fuzz_amounts(seed, m);
    std::vector<std::uint32_t> perm(m);  // perm[e] = e's eid in the permuted stream
    for (std::uint64_t a = 0, b; a < m; a = b) {
      for (b = a; b < m && fg.ts[b] == fg.ts[a]; ++b) {}
      for (std::uint64_t e = a; e < b; ++e) perm[e] = std::uint32_t(a + b - 1 - e);
    }
    std::vector<std::uint32_t> src(m), dst(m);
    std::vector<double> amt(m);
    for (std::uint64_t e = 0; e < m; ++e) {
      src[perm[e]] = fg.src[e];
      dst[perm[e]] = fg.dst[e];
      amt[perm[e]] = amount[e];
      moved += perm[e] != e;
    }
    const CleanOracle o({fg.src, fg.dst, fg.ts, fg.n}, amount), op({src, dst, fg.ts, fg.n}, amt);
    std::map<std::tuple<std::uint32_t, std::uint32_t, std::int32_t>, std::uint32_t> first;  // (u, v, t) -> eid
    for (std::uint32_t e = 0; e < m; ++e) first.try_emplace({fg.src[e], fg.dst[e], fg.ts[e]}, e);
    const auto check = [&](Block b, const BlockParams& p, const auto& eq_rows) {
      const std::size_t c = block_cols(p);
      const auto mine = [&](const CleanOracle& x) {
        return outcome([&] {
          Row out(m * c, 0);
          x.mine(b, p, 6, 0, m, out.data());
          return out;
        });
      };
      const Outcome a = mine(o), pa = mine(op);
      if (a.aborted != UINT64_MAX || pa.aborted != UINT64_MAX) return;  // inconclusive (R11)
      for (std::uint64_t e = 0; e < m; ++e) eq_rows(a.rows, pa.rows, c, e);
    };
    for (Block b : kAllBlocks) {
      BlockParams p{fg.window, kFuzzBins};
      p.work_limit = 2'000;
      const auto eq = [&](const Row& a, const Row& pa, std::size_t c, std::uint64_t e) {
        INFO("seed " << seed << " " << block_name(b) << " cols " << c << " eid " << e);
        CHECK(std::equal(a.begin() + e * c, a.begin() + (e + 1) * c, pa.begin() + perm[e] * c));
        const std::uint32_t f = first.at({fg.src[e], fg.dst[e], fg.ts[e]});
        CHECK(std::equal(a.begin() + e * c, a.begin() + (e + 1) * c, a.begin() + f * c));
      };
      check(b, p, eq);
      if (std::find(kCountBlocks.begin(), kCountBlocks.end(), b) != kCountBlocks.end()) check(b, count(fg.window), eq);
    }
    std::vector<double> v(m * kVstatsCols), pv(m * kVstatsCols);
    o.vstats(fg.window, 0, m, v.data());
    op.vstats(fg.window, 0, m, pv.data());
    for (std::uint64_t e = 0; e < m; ++e) {
      const auto at = [](const std::vector<double>& x, std::uint64_t r) {
        return std::vector<double>(x.begin() + r * kVstatsCols, x.begin() + (r + 1) * kVstatsCols);
      };
      INFO("seed " << seed << " vstats eid " << e);
      CHECK(same_bits(at(v, e), at(pv, perm[e])));
      const std::uint32_t f = first.at({fg.src[e], fg.dst[e], fg.ts[e]});
      duplicates += f != e;
      CHECK(same_bits(at(v, e), at(v, f)));
    }
  }
  CHECK(moved > 10000);
  CHECK(duplicates > 100);
}

TEST_CASE("R4: tcyc and tchain hops are in non-decreasing time order, the trigger a latest edge") {
  const BlockParams p{100, {2, 3}};
  // The cycle 1 -> 2 -> 0 closed by 0 -> 1, every edge at 3: t_1 = t_2 = t, so each edge, a latest
  // edge, completes it (size 3). Strictly increasing (Q5): nothing.
  const Graph same{{1, 2, 3}, {2, 0, 3}, {0, 1, 3}};
  CHECK(rows(same, Block::tcyc, p) == Row{0, 1, 0, 1, 0, 1});
  CHECK(rows(same, Block::tcyc, p, parse_quirks("Q5")) == Row{0, 0, 0, 0, 0, 0});
  // The same cycle at 2 (1 -> 2), 1 (2 -> 0), 3 (0 -> 1): no rotation is in order.
  const Graph back{{2, 0, 1}, {1, 2, 2}, {0, 1, 3}};
  CHECK(rows(back, Block::tcyc, p) == Row{0, 0, 0, 0, 0, 0});
  // Parallel hops: 2 -> 0 at 9 closes 0 -> 1 (5) -> 2 through the 1 -> 2 edge at 5, at the previous
  // hop's time; the one at 1 is too early. Strictly (Q5) neither fits.
  const Graph par{{1, 2, 1}, {0, 1, 5}, {1, 2, 5}, {2, 0, 9}};
  CHECK(row(par, Block::tcyc, p, 3) == Row{0, 1});
  CHECK(row(par, Block::tcyc, p, 3, parse_quirks("Q5")) == Row{0, 0});
  // The trigger 0 -> 1 at 6 reaches 2 at 1 or 5 (parallel 1 -> 2 edges); 2 -> 0 at 3 follows the
  // arrival at 1 only.
  const Graph arr{{1, 2, 1}, {2, 0, 3}, {1, 2, 5}, {0, 1, 6}};
  CHECK(row(arr, Block::tcyc, p, 3) == Row{0, 1});
  // tchain into the trigger 0 -> 1 at 4: 2 -> 0 (size 2) and 3 -> 2 -> 0 (size 3), every edge at 4.
  // For e1 = 2 -> 0: 3 -> 2 -> 0 (size 2).
  const Graph chain{{3, 2, 4}, {2, 0, 4}, {0, 1, 4}};
  CHECK(rows(chain, Block::tchain, p) == Row{0, 0, 1, 0, 1, 1});
  CHECK(rows(chain, Block::tchain, p, parse_quirks("Q5")) == Row{0, 0, 0, 0, 0, 0});
  // 3 -> 2 at 2 is later than 2 -> 0 at 1: only the chain 2 -> 0 -> 1.
  const Graph late{{2, 0, 1}, {3, 2, 2}, {0, 1, 3}};
  CHECK(row(late, Block::tchain, p, 2) == Row{1, 0});
  // The chain edge 2 -> 0 into the trigger 0 -> 1 at 7 can be at 2 or 6 (parallel); 3 -> 2 at 4 is
  // no later than the one at 6 only: 2 -> 0 -> 1 and 3 -> 2 -> 0 -> 1.
  const Graph dep{{2, 0, 2}, {3, 2, 4}, {2, 0, 6}, {0, 1, 7}};
  CHECK(row(dep, Block::tchain, p, 3) == Row{1, 1});
}

TEST_CASE("R5: lcyc counts the simple cycles through the trigger in any time order, ties included") {
  const BlockParams p{100, {2, 3}};
  // The trigger e2 = 0 -> 1 at 5 closes 0 -> 1 -> 0 through e3, tied with it and later in the
  // stream (size 2), and 0 -> 1 -> 2 -> 0 through e1 at 3 and e0 at 1, decreasing (size 3).
  // e3 = 1 -> 0 closes 1 -> 0 -> 1 through e2 (size 2); e0 and e1 see no edge out of 0.
  const Graph g{{2, 0, 1}, {1, 2, 3}, {0, 1, 5}, {1, 0, 5}};
  CHECK(rows(g, Block::lcyc, p) == Row{0, 0, 0, 0, 1, 1, 1, 0});
  // tcyc keeps the tie but not the decreasing cycle.
  CHECK(row(g, Block::tcyc, p, 2) == Row{1, 0});
  // v2 (Q6) drops every edge at ts = t.
  CHECK(rows(g, Block::lcyc, p, parse_quirks("Q6")) == Row{0, 0, 0, 0, 0, 1, 0, 0});
}

TEST_CASE("R6a: sg is unordered, reported by both roles, with endpoints never intermediates") {
  const BlockParams p{100, {2, 3}};
  // s = 0 pays 1, 2, 3 at 2 after they paid w = 4 at 1. The gather edges (e0-e2) see no source yet;
  // each scatter edge s -> x (e3-e5) reports (0, 4) for the sink 4 of x: I = {1, 2, 3}, size 3.
  const Graph g{{1, 4, 1}, {2, 4, 1}, {3, 4, 1}, {0, 1, 2}, {0, 2, 2}, {0, 3, 2}};
  CHECK(rows(g, Block::sg, p) == Row{0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1});
  Row gather(g.m() * 2, 0);
  CleanOracle(g.view()).mine_sg_gather(p, 0, g.m(), gather.data());
  CHECK(gather == Row(g.m() * 2, 0));
  // A pair needs s != w. For e1 = 3 -> 4 the only source is 4 = w, so no gather instance; (4, 4)
  // would have I = {3, 5}. The scatter instance (3, 5) has I = {4}, size 1. Likewise for every edge.
  const Graph ends{{4, 3, 1}, {3, 4, 1}, {4, 5, 1}, {5, 4, 1}};
  CHECK(rows(ends, Block::sg, {100, {2}}) == Row{0, 0, 0, 0});
}

TEST_CASE("R6b: sg_ord needs s -> y no later than y -> w, and only gather edges report") {
  const BlockParams p{100, {2, 3}};
  const Graph g{
      {4, 3, 1},  // e0: y' -> w, before s -> y'
      {0, 1, 2},  // e1: s -> x
      {0, 2, 2},  // e2: s -> y
      {0, 4, 2},  // e3: s -> y'
      {2, 3, 2},  // e4: y -> w, tied with s -> y
      {1, 3, 2},  // e5: x -> w
  };
  // e4 and e5 have the source 0 and I(0, 3) = {1, 2}: size 2. 4 is out (2 > 1).
  CHECK(rows(g, Block::sg_ord, p) == Row{0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0});
  // Strictly ordered (Q5): no source edge before t = 2.
  CHECK(rows(g, Block::sg_ord, p, parse_quirks("Q5")) == Row(12, 0));
  // sg's gather role for e5 has I = {1, 2, 4}: sg_ord's I is a subset.
  Row gather(g.m() * 2, 0);
  CleanOracle(g.view()).mine_sg_gather(p, 0, g.m(), gather.data());
  CHECK(Row(gather.begin() + 10, gather.end()) == Row{0, 1});
  // The scatter edge e2 is a latest edge of (0, 3) and sg reports it (I = {1, 2, 4}); sg_ord does not.
  CHECK(row(g, Block::sg, p, 2) == Row{0, 1});
  CHECK(row(g, Block::sg_ord, p, 2) == Row{0, 0});
}

TEST_CASE("R7: count blocks emit their count, 0 and 1 included, in one column") {
  const Graph g{
      {2, 1, 1},  // e0
      {2, 1, 1},  // e1: parallel
      {1, 3, 1},  // e2
      {3, 0, 1},  // e3
      {4, 0, 1},  // e4
      {0, 3, 1},  // e5
      {0, 5, 1},  // e6
      {6, 0, 2},  // e7: tied with the trigger
      {0, 1, 2},  // e8: the trigger u = 0 -> v = 1
  };
  // e8: N_in(1) = {2, 0}, 3 edges; N_out(0) = {3, 5, 1}, 3 edges; N_in(1) ∪ N_out(1) = {0, 2, 3};
  // N_out(0) \ N_in(0) = {3, 5, 1} \ {3, 4, 6} = {1, 5}; gs: A = {3, 4, 6}, B = N_out(0): min 3.
  // e0 at t = 1: N_in(1) = {2}, 2 edges (e0, e1); N_out(2) = {1}, 2 edges; {2} ∪ {3}; {1} \ {};
  // gs: A = {} and B = {1}: 0.
  const std::map<Block, std::pair<std::uint32_t, std::uint32_t>> want = {
      {Block::fan_in, {1, 2}},         {Block::deg_in, {2, 3}},    {Block::fan_out, {1, 3}},
      {Block::deg_out, {2, 3}},        {Block::fan_total, {2, 3}}, {Block::fan_oneway_out, {1, 2}},
      {Block::gs, {0, 3}}};
  for (const auto& [b, w] : want) {
    INFO(block_name(b));
    CHECK(row(g, b, count(100), 0) == Row{w.first});
    CHECK(row(g, b, count(100), 8) == Row{w.second});
  }
  // gs's A had ts < t in v2 (Q5): {3, 4}.
  CHECK(row(g, Block::gs, count(100), 8, parse_quirks("Q5")) == Row{2});
  // A histogram drops sizes below 2: fan_out = 1 for e0 is all zeros.
  CHECK(row(g, Block::fan_out, {100, {2, 3}}, 0) == Row{0, 0});
  // The blocks with several instances per trigger have no count form.
  for (Block b : {Block::sg, Block::sg_ord, Block::tcyc, Block::lcyc, Block::tchain}) {
    INFO(block_name(b));
    CHECK_THROWS_AS(rows(g, b, count(100)), std::invalid_argument);
  }
}

TEST_CASE("R8: vertex statistics, small sides and order independence") {
  // n = 1 (u_out, v_in: the trigger alone, age 0, amount 7) and n = 0 (u_in, v_out).
  {
    const Graph g{{0, 1, 10, 7}};
    const std::vector<double> r = vs_row(g, 100, 0), one = {1, 1, 1, 0, 0, 0, 0, 0, 7, 7, 0, 0, 0},
                              none(kVstatsPerSide, 0.0);
    CHECK(same_bits(side(r, 0), one));
    CHECK(same_bits(side(r, 1), none));
    CHECK(same_bits(side(r, 2), none));
    CHECK(same_bits(side(r, 3), one));
  }
  // Equal values: three amounts 0.1 into 9. The computed m2 is 1.9e-34, not 0, and would give
  // skew -1 and kurt 1; min = max makes var, skew and kurt 0.
  {
    const Graph g{{1, 9, 5, 0.1}, {2, 9, 5, 0.1}, {3, 9, 5, 0.1}};
    const double sum = 0.1 + 0.1 + 0.1;
    CHECK(same_bits(side(vs_row(g, 100, 2), 3), {3, 3, 1, 0, 0, 0, 0, 0, sum / 3, sum, 0, 0, 0}));
  }
  // Distinct values whose computed m2 is 0 (the squared deviations underflow): var is that 0, and
  // skew and kurt are 0, not 0/0.
  {
    const Graph g{{1, 9, 5, 1e-200}, {2, 9, 5, 2e-200}};
    const double sum = 1e-200 + 2e-200;
    CHECK(same_bits(side(vs_row(g, 100, 1), 3), {2, 2, 1, 0, 0, 0, 0, 0, sum / 2, sum, 0, 0, 0}));
  }
  // D-58: a positive computed m2 whose m2^1.5 and m2^2 underflow (1e-160 and 3e-160 give m2 = 1e-320):
  // var is that m2, and skew and kurt are 0, not 0/0, whose NaN bits differ between x86 and CUDA.
  {
    const Graph g{{1, 9, 5, 1e-160}, {2, 9, 5, 3e-160}};
    const double sum = 1e-160 + 3e-160, mean = sum / 2, d1 = 1e-160 - mean, d2 = 3e-160 - mean;
    const double m2 = (0.0 + d1 * d1 + d2 * d2) / 2;
    REQUIRE(m2 > 0);
    REQUIRE(m2 * std::sqrt(m2) == 0);
    CHECK(same_bits(side(vs_row(g, 100, 1), 3), {2, 2, 1, 0, 0, 0, 0, 0, mean, sum, m2, 0, 0}));
  }
  // The moments: into 9 at 5, 9, 10, the trigger at t = 10. Ages {5, 1, 0}: mean 2, sum 6, m2 = 14/3,
  // m3 = 6, m4 = 98/3. Amounts {1, 2, 3}: mean 2, sum 6, m2 = 2/3, m3 = 0, m4 = 2/3.
  {
    const Graph g{{1, 9, 5, 3}, {2, 9, 9, 1}, {3, 9, 10, 2}};
    const double a2 = 14.0 / 3, a4 = 98.0 / 3, b2 = 2.0 / 3;
    CHECK(same_bits(side(vs_row(g, 100, 2), 3), {3, 3, 1, 2, 6, a2, 6 / (a2 * std::sqrt(a2)), a4 / (a2 * a2), 2,
                                                 6, b2, 0, b2 / (b2 * b2)}));
  }
  // Order independence: amounts 1, 1e16, 1 into 9 at one time. In ascending order the sum is
  // 1e16 + 2; in eid order (Q10) 1 + 1e16 rounds to 1e16 and so does the sum. Every order of the
  // three edges gives every row the same bits, moved with its edge.
  {
    const Edge e[] = {{1, 9, 5, 1}, {2, 9, 5, 1e16}, {3, 9, 5, 1}};
    const Graph g{e[0], e[1], e[2]};
    CHECK(vs_row(g, 100, 0)[3 * kVstatsPerSide + 9] == 1e16 + 2);
    CHECK(vs_row(g, 100, 0, parse_quirks("Q10"))[3 * kVstatsPerSide + 9] == 1e16);
    std::array<int, 3> order = {0, 1, 2};
    do {
      const Graph h{e[order[0]], e[order[1]], e[order[2]]};
      for (int i = 0; i < 3; ++i) {
        INFO("order " << order[0] << order[1] << order[2] << " row " << i);
        CHECK(same_bits(vs_row(h, 100, std::uint32_t(i)), vs_row(g, 100, std::uint32_t(order[i]))));
      }
    } while (std::next_permutation(order.begin(), order.end()));
  }
  // The four sides in order, each with its fan, degree and ratio.
  {
    const Graph g{{3, 0, 1}, {3, 0, 1}, {4, 0, 1},                          // u_in: 3 edges, 2 neighbours
                  {1, 5, 1}, {1, 5, 1}, {1, 5, 1}, {1, 5, 1},              // v_out: 4 edges, 1 neighbour
                  {6, 1, 1}, {7, 1, 1}, {8, 1, 1}, {9, 1, 1}, {0, 2, 1},  // v_in, u_out, with the trigger:
                  {0, 1, 2}};                                              // 5 and 2 edges
    const std::vector<double> r = vs_row(g, 100, 12);
    const double want[4][3] = {{2, 2, 1}, {2, 3, 1.5}, {1, 4, 4}, {5, 5, 1}};
    for (int k = 0; k < 4; ++k) {
      INFO("side " << k);
      CHECK(same_bits(std::vector<double>(r.begin() + k * kVstatsPerSide, r.begin() + k * kVstatsPerSide + 3),
                      {want[k][0], want[k][1], want[k][2]}));
    }
  }
}

TEST_CASE("R9: each pattern credits the edges that complete an instance") {
  const BlockParams p{100, {2, 3}};
  // tcyc: the cycle 0 -> 1 (1), 1 -> 0 (2) is completed by its latest edge, and again by a parallel
  // one (Q15).
  const Graph cyc{{0, 1, 1}, {1, 0, 2}, {1, 0, 2}};
  CHECK(rows(cyc, Block::tcyc, {100, {2}}) == Row{0, 1, 1});
  // lcyc: the cycle 0 -> 1 (1), 1 -> 2 (3), 2 -> 0 (2), out of order, is completed by its latest
  // edge e2 and its parallel e3; the earlier edges do not see e2. tcyc does not have it.
  const Graph lc{{0, 1, 1}, {2, 0, 2}, {1, 2, 3}, {1, 2, 3}};
  CHECK(rows(lc, Block::lcyc, p) == Row{0, 0, 0, 0, 0, 1, 0, 1});
  CHECK(rows(lc, Block::tcyc, p) == Row(8, 0));
  // sg and sg_ord, every edge at 1: s = 0 pays 1, 2, 3 (e0-e2), which pay w = 4 (e3-e5). sg
  // credits every edge (scatter, then gather role): I = {1, 2, 3}. sg_ord credits the gather edges.
  const Graph sg{{0, 1, 1}, {0, 2, 1}, {0, 3, 1}, {1, 4, 1}, {2, 4, 1}, {3, 4, 1}};
  CHECK(rows(sg, Block::sg, p) == Row{0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1});
  CHECK(rows(sg, Block::sg_ord, p) == Row{0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1});
  // A count block credits every trigger: deg_in of v.
  CHECK(rows(sg, Block::deg_in, count(100)) == Row{1, 1, 1, 3, 3, 3});
}

TEST_CASE("R10: instance sizes, and saturation at UINT32_MAX") {
  // sg's size is |I|: e5 = 2 -> 4 with source 0 has I = {1, 2}, each once despite the parallel edges.
  const Graph sg{{0, 1, 1}, {0, 1, 1}, {1, 4, 1}, {1, 4, 1}, {0, 2, 1}, {2, 4, 1}};
  CHECK(row(sg, Block::sg, {100, {2, 3}}, 5) == Row{1, 0});
  // tcyc and lcyc: a cycle's size is its edge count. The 4-cycle needs k_max >= 4.
  const Graph c4{{0, 1, 1}, {1, 2, 2}, {2, 3, 3}, {3, 0, 4}};
  for (Block b : {Block::tcyc, Block::lcyc}) {
    INFO(block_name(b));
    CHECK(row(c4, b, {100, {2, 3, 4}}, 3, {}, 4) == Row{0, 0, 1});
    CHECK(row(c4, b, {100, {2, 3, 4}}, 3, {}, 3) == Row{0, 0, 0});
  }
  // tchain: one chain per node sequence, sized by its edges: into the trigger 0 -> 1, 2 -> 0 (twice
  // in parallel, one chain, size 2) and 3 -> 2 -> 0 (size 3); k_max 2 keeps the first.
  const Graph ch{{2, 0, 1}, {2, 0, 1}, {3, 2, 1}, {0, 1, 2}};
  CHECK(row(ch, Block::tchain, {100, {2, 3}}, 3) == Row{1, 1});
  CHECK(row(ch, Block::tchain, {100, {2, 3}}, 3, {}, 2) == Row{1, 0});
  // Saturation: rows add into `out`. One instance into a bin at UINT32_MAX is lost and counted; a
  // count that does not fit clamps and counts once.
  const CleanOracle o(ch.view());
  const std::uint32_t e3[] = {3};
  Row out = {UINT32_MAX - 1, UINT32_MAX};
  CHECK(o.mine(Block::tchain, {100, {2, 3}}, 10, e3, out.data()) == 1);
  CHECK(out == Row{UINT32_MAX, UINT32_MAX});
  out = {UINT32_MAX - 1};  // fan_in of e3 is 1: fits
  CHECK(o.mine(Block::fan_in, count(100), 10, e3, out.data()) == 0);
  CHECK(out == Row{UINT32_MAX});
  out = {UINT32_MAX};
  CHECK(o.mine(Block::fan_in, count(100), 10, e3, out.data()) == 1);
  CHECK(out == Row{UINT32_MAX});
}

TEST_CASE("R11: lcyc's work limit aborts the run at the smallest eid over it") {
  // Bindings (extensions of the path to a new node): e0 = 1 -> 2 extends to 0, 3, then 0 again: 3.
  // e1, e2: 1 each. e3, e4: 0. The trigger e5 = 0 -> 1 extends to 2, 3 (from 2), 3 (from 1): 3,
  // closing 0 -> 1 -> 2 -> 0, 0 -> 1 -> 2 -> 3 -> 0 and 0 -> 1 -> 3 -> 0.
  const Graph g{{1, 2, 1}, {1, 3, 1}, {2, 3, 1}, {2, 0, 1}, {3, 0, 1}, {0, 1, 2}};
  BlockParams p{100, {2, 3, 4}};
  p.work_limit = 3;
  CHECK(row(g, Block::lcyc, p, 5) == Row{0, 2, 1});
  const CleanOracle o(g.view());
  p.work_limit = 2;
  Row out(g.m() * 3, 0);
  try {
    o.mine(Block::lcyc, p, 10, 0, g.m(), out.data());
    FAIL("no WorkLimitExceeded");
  } catch (const WorkLimitExceeded& e) {
    CHECK(e.eid == 0);
    CHECK(e.limit == 2);
  }
  try {
    o.mine(Block::lcyc, p, 10, 1, g.m(), out.data());
    FAIL("no WorkLimitExceeded");
  } catch (const WorkLimitExceeded& e) {
    CHECK(e.eid == 5);
  }
  p.work_limit = 0;
  CHECK(rows(Graph{{2, 0, 1}, {3, 0, 1}}, Block::lcyc, p) == Row(6, 0));
  // No other block counts work.
  CHECK(row(g, Block::tcyc, p, 5) == Row{0, 2, 1});
}

TEST_CASE("parse_quirks: ledger ids, all, or nothing") {
  const auto on = [](const Quirks& q) {
    return std::array{q.q1_sg_self_loops,     q.q2_vstats_self_loops, q.q3_closed_windows,    q.q4_stream_visibility,
                      q.q5_strict_order,      q.q6_lcyc_drops_ties,   q.q9_vstats_gfp_window, q.q10_vstats_gfp_arith};
  };
  CHECK(on(parse_quirks("")) == std::array{false, false, false, false, false, false, false, false});
  CHECK(on(parse_quirks("Q1,Q3")) == std::array{true, false, true, false, false, false, false, false});
  CHECK(on(parse_quirks("Q10")) == std::array{false, false, false, false, false, false, false, true});
  CHECK(on(parse_quirks("all")) == std::array{true, true, true, true, true, true, true, true});
  for (const char* bad : {"Q7", "Q8", "Q11", "q1", "Q1,", "Q1;Q3", " Q1", "none"}) {
    INFO(bad);
    CHECK_THROWS_AS(parse_quirks(bad), std::invalid_argument);
  }
}

// F6: with every quirk on, the clean oracle is the legacy oracle at stream visibility, every block,
// histogram bins; a trigger range, a sample of eids and sg's gather role alone.
TEST_CASE("clean oracle, every quirk on: the legacy oracle's rows on fuzz graphs") {
  const Quirks all = parse_quirks("all");
  const OneThread one;
  int stopped = 0, completed = 0;
  for (std::uint64_t seed = 0; seed < 300; ++seed) {
    const FuzzGraph fg = fuzz_graph(seed);
    const EdgeStream s{fg.src, fg.dst, fg.ts, fg.n};
    const std::uint64_t m = fg.src.size(), nb = kFuzzBins.size();
    const CleanOracle clean(s, {}, all);
    const Oracle legacy(s);
    BlockParams p{fg.window, kFuzzBins};
    p.work_limit = 2'000;
    std::vector<std::uint32_t> sample;
    for (std::uint32_t e = 0; e < m; e += 3) sample.push_back(e);
    for (Block b : kAllBlocks) {
      INFO("seed " << seed << " " << block_name(b));
      const auto range = [&](const auto& o) {
        return outcome([&] {
          Row out(m * nb, 0);
          o.mine(b, p, 10, 0, m, out.data());
          return out;
        });
      };
      const Outcome want = range(legacy);
      CHECK(range(clean) == want);
      if (want.aborted != UINT64_MAX) {
        ++stopped;
        continue;
      }
      completed += b == Block::lcyc;
      Row got(sample.size() * nb, 0);
      clean.mine(b, p, 10, sample, got.data());
      for (std::size_t r = 0; r < sample.size(); ++r)
        CHECK(std::equal(got.begin() + r * nb, got.begin() + (r + 1) * nb, want.rows.begin() + sample[r] * nb));
    }
    Row want(m * nb, 0), got(m * nb, 0);
    legacy.mine_sg_gather(p, 0, m, want.data());
    clean.mine_sg_gather(p, 0, m, got.data());
    INFO("seed " << seed << " sg gather");
    CHECK(got == want);
  }
  CHECK(stopped > 50);  // lcyc hits the work limit on some graphs and not on others
  CHECK(completed > 150);
}

// F6: with every quirk on, the clean oracle's vertex statistics are vstats_cpu's on an index that
// keeps self-loops, bit for bit.
TEST_CASE("clean oracle, every quirk on: vstats_cpu's rows bit for bit on fuzz graphs") {
  const Quirks all = parse_quirks("all");
  const OneThread one;
  for (std::uint64_t seed = 0; seed < 300; ++seed) {
    const FuzzGraph fg = fuzz_graph(seed);
    const std::uint64_t m = fg.src.size();
    const std::vector<double> amount = fuzz_amounts(seed, m);
    const GraphIndex idx(fg.src, fg.dst, fg.ts, fg.n);
    TriggerSel sel;
    sel.last = m;
    std::vector<double> want(m * kVstatsCols), got(m * kVstatsCols);
    vstats_cpu(idx.view(), amount, {fg.window}, sel, want.data());
    CleanOracle({fg.src, fg.dst, fg.ts, fg.n}, amount, all).vstats(fg.window, 0, m, got.data());
    INFO("seed " << seed);
    CHECK(same_bits(got, want));
  }
}

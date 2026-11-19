// SPEC v2 §8 S6, D-43: reachability pruning for chain patterns. The search's latest departure
// times on a hand graph, the planner's chain recognition (tcyc only in the library, and the shapes
// it refuses), hand-checked tcyc rows with and without pruning, and how often the cut fires on
// fuzz graphs. The fuzz-graph parity is fuzz_diff_cpu (test_cpu_backend.cpp), which runs every
// block planned for pruning both ways.
#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include "blz/reach.hpp"
#include "graph/fuzz_gen.hpp"
#include "graph/index.hpp"
#include "ir/ir.hpp"
#include "oracle/oracle.hpp"
#include "runtime/cpu_backend.hpp"

using namespace blz;

namespace {

struct Graph {
  std::vector<std::uint32_t> src, dst;
  std::vector<std::int32_t> ts;
  std::uint32_t n = 0;
  explicit Graph(const std::vector<std::array<int, 3>>& edges) {  // src, dst, ts in stream (ts) order
    for (const auto& [s, d, t] : edges) {
      src.push_back(std::uint32_t(s));
      dst.push_back(std::uint32_t(d));
      ts.push_back(t);
      n = std::max(n, std::uint32_t(std::max(s, d)) + 1);
    }
  }
};

std::string src_path(const std::string& rel) { return std::string(BLZ_SOURCE_DIR) + "/" + rel; }

// A 4-cycle chain pattern in tcyc's form, with `hop2` as hop_2's order constraint and `extra` as
// more of hop_2's fields.
std::string chain_spec(const std::string& hop2, const std::string& extra = "") {
  return "version: 1\npattern: c\nblock: tcyc\nwindow: {span: d, lower: closed}\n"
         "trigger: {edge: e0, src: U, dst: V}\nconstraints: [{skip_if: \"U == V\"}]\nstages:\n"
         "  - {id: hop_1, op: for_all, src: V.out, edge: h1, dst_var: C1,\n"
         "     constraints: [{break_if: \"h1.t >= t\"}, {skip_if: \"C1 in [U, V]\"}]}\n"
         "  - {id: close_3, op: exists, src: C1.out, node: U, edge: c3,\n"
         "     constraints: [{skip_if: \"c3.t <= h1.t\"}, {break_if: \"c3.t >= t\"}], emit: {size: 3}}\n"
         "  - {id: hop_2, op: for_all, src: C1.out, edge: h2, dst_var: C2" + extra + ",\n"
         "     constraints: [{skip_if: \"" + hop2 + "\"}, {break_if: \"h2.t >= t\"}, {skip_if: \"C2 in [U, V, C1]\"}]}\n"
         "  - {id: close_4, op: exists, src: C2.out, node: U, edge: c4,\n"
         "     constraints: [{skip_if: \"c4.t <= h2.t\"}, {break_if: \"c4.t >= t\"}], emit: {size: 4}}\n";
}

}  // namespace

TEST_CASE("reach: latest departure times towards the target, hop bound, window, visibility") {
  // Target 0 at t = 100. Stream (ts) order.
  const Graph g({{8, 0, 0},     // eid 0: before the window's lower end (lo = 1)
                 {4, 3, 5},     // eid 1: 4 -> 3 -> 2 -> 1 -> 0, four hops
                 {3, 2, 10},    // eid 2
                 {7, 7, 20},    // eid 3: a self-loop
                 {2, 1, 40},    // eid 4: 2 can leave at 40 (< ld[1] = 50)
                 {3, 2, 45},    // eid 5: too late for 2 (ld[2] = 40), 3 keeps 10
                 {1, 0, 50},    // eid 6: ld[1] = 50
                 {2, 1, 60},    // eid 7: too late for 1
                 {5, 0, 99},    // eid 8: ld[5] = 99
                 {6, 0, 100}});  // eid 9: ts = t is not before the trigger
  const GraphIndex idx(g.src, g.dst, g.ts, g.n);
  ReachScratch r(g.n);
  r.search(idx.view().in, 0, 100, 1, 99, kAnyEid, 9);
  CHECK(r.keep(1, 49));
  CHECK_FALSE(r.keep(1, 50));
  CHECK(r.keep(2, 39));
  CHECK_FALSE(r.keep(2, 40));
  CHECK(r.keep(3, 9));
  CHECK_FALSE(r.keep(3, 10));
  CHECK(r.keep(4, 4));
  CHECK_FALSE(r.keep(4, 5));
  CHECK(r.keep(5, 98));
  CHECK_FALSE(r.keep(6, 0));  // its only edge is at t
  CHECK_FALSE(r.keep(7, 0));  // a self-loop is not a walk
  CHECK_FALSE(r.keep(8, -1));  // below the window

  r.search(idx.view().in, 0, 100, 1, 99, kAnyEid, 3);  // three hops: 4 is out of reach
  CHECK(r.keep(3, 9));
  CHECK_FALSE(r.keep(4, 0));

  r.search(idx.view().in, 0, 100, 1, 99, 7, 9);  // stream visibility at eid 7: 5 -> 0 is not seen
  CHECK_FALSE(r.keep(5, 0));
  CHECK(r.keep(4, 4));

  r.search(idx.view().in, 3, 100, 1, 99, kAnyEid, 9);  // another target: nothing of the last search stays
  CHECK(r.keep(4, 4));
  CHECK_FALSE(r.keep(1, 0));
  CHECK_FALSE(r.keep(2, 0));
}

TEST_CASE("reach: the planner prunes chain patterns only") {
  const ir::Pattern tcyc = ir::compile_file(src_path("patterns/tcyc.yaml"));
  CHECK(tcyc.plan.reach);
  CHECK(tcyc.plan.reach_target == "U");
  CHECK(tcyc.plan.reach_hops == 9);  // K_MAX - 1
  CHECK(dsl::to_string(tcyc.plan.reach_lo) == "t - delta_cyc");
  CHECK(ir::compile_file(src_path("patterns/tcyc.yaml"), {{"K_MAX", 5}}).plan.reach_hops == 4);
  for (const char* other : {"tchain", "lcyc", "sg_gather", "sg_ordered", "fan_out", "gs", "fan_total"}) {
    INFO(other);
    CHECK_FALSE(ir::compile_file(src_path(std::string("patterns/") + other + ".yaml")).plan.reach);
  }

  CHECK(ir::compile_text(chain_spec("h2.t <= h1.t"), "c.yaml").plan.reach);
  // Hops not strictly increasing: a walk could reuse a time, so ld would not bound it.
  CHECK_FALSE(ir::compile_text(chain_spec("h2.t < h1.t"), "c.yaml").plan.reach);
  // An emit on a hop needs no walk to U.
  CHECK_FALSE(ir::compile_text(chain_spec("h2.t <= h1.t", ", emit: {size: 2}"), "c.yaml").plan.reach);
}

TEST_CASE("reach: hand-checked tcyc rows are the same with and without pruning") {
  const std::vector<std::uint32_t> bins = {2, 3, 4, 5, 6};
  struct Case {
    const char* what;
    std::vector<std::array<int, 3>> edges;
    std::uint32_t trigger;
    std::vector<std::uint32_t> want;
  };
  const std::vector<Case> cases = {
      // U = 0, V = 1. 2 is a dead end, 3 reaches U only before the hop into it, 1 -> 4 -> 5 -> 0
      // is the one cycle.
      {"dead ends and a late hop are cut, the 4-cycle stays",
       {{3, 0, 1}, {1, 2, 1}, {1, 3, 2}, {1, 4, 3}, {4, 5, 4}, {5, 0, 5}, {0, 1, 10}}, 6, {0, 0, 1, 0, 0}},
      {"V cannot reach U: nothing below the root", {{1, 2, 1}, {2, 3, 2}, {0, 1, 10}}, 2, {0, 0, 0, 0, 0}},
      {"the 2-cycle is a root check", {{1, 0, 1}, {1, 2, 2}, {0, 1, 10}}, 2, {1, 0, 0, 0, 0}},
      {"a walk through a node twice keeps a binding the simple path rejects",
       {{1, 2, 1}, {2, 1, 2}, {1, 3, 3}, {3, 0, 4}, {0, 1, 10}}, 4, {0, 1, 0, 0, 0}},
      {"a hop tied with the next one", {{1, 2, 3}, {2, 0, 3}, {0, 1, 10}}, 2, {0, 0, 0, 0, 0}},
      {"a closing edge tied with the trigger", {{1, 2, 3}, {0, 1, 10}, {2, 0, 10}}, 1, {0, 0, 0, 0, 0}},
  };
  const auto pats = patterns_for(Block::tcyc);
  for (const Case& c : cases) {
    INFO(c.what);
    const Graph g(c.edges);
    const std::uint64_t m = g.src.size();
    BlockParams p{100, bins};
    std::vector<std::uint32_t> want(m * bins.size(), 0);
    Oracle({g.src, g.dst, g.ts, g.n}).mine(Block::tcyc, p, compiled_k_max(), 0, m, want.data());
    CHECK(std::vector<std::uint32_t>(want.begin() + std::ptrdiff_t(c.trigger * bins.size()),
                                     want.begin() + std::ptrdiff_t((c.trigger + 1) * bins.size())) == c.want);
    const GraphIndex idx(g.src, g.dst, g.ts, g.n);
    TriggerSel all;
    all.last = m;
    for (const bool prune : {false, true})
      for (const Schedule s : {Schedule::dynamic, Schedule::sorted}) {
        p.reach_prune = prune;
        std::vector<std::uint32_t> got(m * bins.size(), 0);
        mine_cpu(idx.view(), pats, p, Visibility::stream, all, s, got.data());
        INFO("prune " << prune << ", " << schedule_name(s));
        CHECK(got == want);
      }
  }
}

// How often the cut fires on fuzz graphs: the share of tcyc's first-hop bindings (distinct C1 in
// V's window, C1 not U or V) that the search rejects. The fuzz parity is only a test of the
// pruning if it cuts often, including bindings whose node has edges to U (too early).
TEST_CASE("reach: the cut fires on fuzz graphs") {
  std::uint64_t bindings = 0, cut = 0, cut_with_edge_to_u = 0, early_out = 0, triggers = 0;
  for (std::uint64_t seed = 0; seed < 300; ++seed) {
    const FuzzGraph fg = fuzz_graph(seed);
    const GraphIndex idx(fg.src, fg.dst, fg.ts, fg.n);
    const GraphView& g = idx.view();
    ReachScratch r(fg.n);
    for (std::uint32_t e = 0; e < g.m; ++e) {
      const std::uint32_t u = g.src[e], v = g.dst[e];
      const std::int64_t t = g.ts[e], lo = t - fg.window;
      if (u == v) continue;
      ++triggers;
      r.search(g.in, u, t, lo, t - 1, e, compiled_k_max() - 1);
      early_out += !r.keep(v, lo - 1);
      const Range h1 = g.out.range(v, lo, t - 1, e);
      for (std::uint32_t j = h1.lo; j < h1.hi; ++j) {
        if (!g.out.distinct_first(j, h1.lo)) continue;
        const std::uint32_t c1 = g.out.nbr[j];
        if (c1 == u || c1 == v) continue;
        ++bindings;
        if (r.keep(c1, g.out.ts[j])) continue;
        ++cut;
        cut_with_edge_to_u += g.pout.pair_exists(c1, u, lo, t - 1, e) != kNoTs;
      }
    }
  }
  INFO(triggers << " triggers (" << early_out << " end at the root), " << bindings << " first-hop bindings, " << cut
                << " cut (" << cut_with_edge_to_u << " of them with an edge to U)");
  CHECK(cut * 10 > bindings);
  CHECK(cut * 10 < bindings * 9);
  CHECK(cut_with_edge_to_u > 0);
  CHECK(early_out * 20 > triggers);
}

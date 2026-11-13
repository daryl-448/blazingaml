// The sweep lowering (SPEC v2 §7.1, D-29) and the ablation helpers (§7.2): the sweep drivers of
// the library's fan and degree patterns against their per-trigger plan (<pattern>_trigger, the
// body CUDA runs), and sweep_count against the §5.3 primitives on lists with ties, repeated
// neighbours and self-loops, under both visibilities, for trigger ranges and samples.
#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include <omp.h>

#include "blz/ablation.hpp"
#include "blz/sweep.hpp"
#include "deg_in.hpp"
#include "deg_out.hpp"
#include "fan_in.hpp"
#include "fan_out.hpp"
#include "graph/fuzz_gen.hpp"
#include "graph/index.hpp"
#include "runtime/triggers.hpp"

using namespace blz;

namespace {

// A dense multigraph in stream order: few nodes and few distinct timestamps, so lists are long,
// with many ties and repeated neighbours, and some self-loops.
struct Graph {
  std::vector<std::uint32_t> src, dst;
  std::vector<std::int32_t> ts;
  std::uint32_t n = 0;
};

Graph dense_graph(std::uint64_t seed) {
  std::uint64_t s = seed * 0x9E3779B97F4A7C15ull + 1;
  auto next = [&] {
    s ^= s >> 12, s ^= s << 25, s ^= s >> 27;
    return s * 2685821657736338717ull;
  };
  Graph g;
  g.n = 2 + std::uint32_t(next() % 6);
  const std::uint32_t m = 1 + std::uint32_t(next() % 300);
  std::int32_t t = 0;
  for (std::uint32_t i = 0; i < m; ++i) {
    t += std::int32_t(next() % 3);  // steps of 0, 1 or 2: many ties
    g.src.push_back(std::uint32_t(next() % g.n));
    g.dst.push_back(std::uint32_t(next() % g.n));
    g.ts.push_back(t);
  }
  return g;
}

struct Pattern {
  const char* name;
  void (*cpu)(const GraphView&, const Params&, std::span<const std::uint32_t>, BlockOut&, int);
  void (*trigger)(const GraphView&, const Params&, std::uint32_t, RowSink&);
};

const Pattern kPatterns[] = {
    {"fan_in", &gen::fan_in_cpu, &gen::fan_in_trigger<RowSink>},
    {"fan_out", &gen::fan_out_cpu, &gen::fan_out_trigger<RowSink>},
    {"deg_in", &gen::deg_in_cpu, &gen::deg_in_trigger<RowSink>},
    {"deg_out", &gen::deg_out_cpu, &gen::deg_out_trigger<RowSink>},
};

const std::vector<std::uint32_t> kBins = {2, 3, 4, 6, 9};

// Rows of the selected triggers: by the sweep driver, and by the per-trigger plan.
std::pair<std::vector<std::uint32_t>, std::vector<std::uint32_t>> both(const Pattern& pat, const GraphView& g,
                                                                       const Params& params, const TriggerSel& sel) {
  const std::uint32_t nb = std::uint32_t(kBins.size());
  std::vector<std::uint32_t> sweep(sel.rows() * nb, 0), each(sel.rows() * nb, 0);
  BlockOut out{sweep.data(), sel.first, kBins.data(), nb, 0, sel.sampled ? sel.eids.data() : nullptr,
               sel.sampled ? sel.eids.size() : 0, sel.last};
  pat.cpu(g, params, {}, out, 1);
  for (std::uint64_t r = 0; r < sel.rows(); ++r) {
    RowSink sink{each.data() + r * nb, kBins.data(), nb, 0};
    pat.trigger(g, params, sel.eid(r), sink);
  }
  return {sweep, each};
}

}  // namespace

TEST_CASE("sweep: the library's fan and degree patterns are planned as sweeps") {
  for (const PatternInfo& p : registered_patterns()) {
    const std::string name = p.name;
    const bool fan_deg = name == "fan_in" || name == "fan_out" || name == "deg_in" || name == "deg_out";
    INFO(name);
    CHECK(p.cpu_sweep == fan_deg);
  }
  CHECK(disabled_optimizations().empty());
}

TEST_CASE("sweep: the drivers equal the per-trigger plan, every window, both visibilities, ranges and samples") {
  // A few threads: the fast suite runs tests in parallel, and the lists are short.
  const int threads = omp_get_max_threads();
  int graphs = 0;
  for (std::uint64_t seed = 0; seed < 300; ++seed) {
    const Graph dg = dense_graph(seed);
    const FuzzGraph fg = fuzz_graph(seed);
    for (int which = 0; which < 2; ++which) {
      const auto& src = which ? fg.src : dg.src;
      const auto& dst = which ? fg.dst : dg.dst;
      const auto& ts = which ? fg.ts : dg.ts;
      const std::uint32_t n = which ? fg.n : dg.n;
      const GraphIndex index(src, dst, ts, n);
      const std::uint64_t m = src.size();
      std::vector<TriggerSel> sels;
      sels.push_back(select_triggers("all", ts));
      sels.push_back(select_triggers("range:" + std::to_string(m / 3) + ":" + std::to_string(m - m / 4), ts));
      sels.push_back(select_triggers("sample:30:" + std::to_string(seed), ts));
      ++graphs;
      for (std::int64_t span : {std::int64_t(0), std::int64_t(1), std::int64_t(3), std::int64_t(which ? fg.window : 25),
                                std::int64_t(1) << 40})
        for (bool tsvis : {false, true})
          for (const Pattern& pat : kPatterns)
            for (const TriggerSel& sel : sels)
              for (int t : {1, 3}) {
                omp_set_num_threads(t);
                const auto [sweep, each] = both(pat, index.view(), Params{span, tsvis}, sel);
                INFO(pat.name << " seed " << seed << (which ? " fuzz" : " dense") << " span " << span << " tsvis "
                              << tsvis << " triggers " << sel.spec << " threads " << t);
                REQUIRE(sweep == each);
              }
    }
  }
  omp_set_num_threads(threads);
  CHECK(graphs == 600);
}

TEST_CASE("sweep: sweep_count equals the primitives with a closed bound and a neighbour filter") {
  const int threads = omp_get_max_threads();
  omp_set_num_threads(2);
  for (std::uint64_t seed = 0; seed < 200; ++seed) {
    const Graph dg = dense_graph(seed + 1000);
    const GraphIndex index(dg.src, dg.dst, dg.ts, dg.n);
    const GraphView& g = index.view();
    const std::uint64_t m = dg.src.size();
    const std::uint32_t nb = std::uint32_t(kBins.size());
    for (bool tsvis : {false, true})
      for (std::int64_t span : {0, 2, 7}) {
        const auto lower = [&](std::int64_t t) { return t - span; };  // closed
        const auto keep = [](std::uint32_t, std::uint32_t y) { return y % 3 != 1; };
        for (bool distinct : {false, true}) {
          std::vector<std::uint32_t> got(m * nb, 0), want(m * nb, 0);
          BlockOut out{got.data(), 0, kBins.data(), nb, 0, nullptr, 0, m};
          const auto emit = [](std::uint32_t, std::int64_t c, RowSink& s) { s.emit(c + 1); };  // a size other than the count
          if (distinct) sweep_count<true>(g, g.in, tsvis, out, lower, keep, emit);
          else sweep_count<false>(g, g.in, tsvis, out, lower, keep, emit);
          for (std::uint32_t e = 0; e < m; ++e) {
            const std::int64_t t = g.ts[e];
            const Range r = g.in.range(g.dst[e], t - span, tsvis ? t : kNoUpper, tsvis ? kAnyEid : e);
            const auto k = [](std::uint32_t y) { return y % 3 != 1; };
            const std::int64_t c = distinct ? count_distinct(g.in, r, k) : count_edges(g.in, r, k);
            RowSink s{want.data() + e * nb, kBins.data(), nb, 0};
            s.emit(c + 1);
          }
          INFO("seed " << seed << " tsvis " << tsvis << " span " << span << " distinct " << distinct);
          REQUIRE(got == want);
        }
      }
  }
  omp_set_num_threads(threads);
}

TEST_CASE("sweep: sweep_order lists the non-empty lists, longest first, ties by node id") {
  const std::vector<std::uint32_t> src = {3, 1, 3, 0, 1, 3}, dst = {0, 0, 1, 2, 2, 2};
  const std::vector<std::int32_t> ts = {0, 1, 2, 3, 4, 5};
  const GraphIndex index(src, dst, ts, 5);
  CHECK(sweep_order(index.view().out, 5) == std::vector<std::uint32_t>{3, 1, 0});
  CHECK(sweep_order(index.view().in, 5) == std::vector<std::uint32_t>{2, 0, 1});
}

TEST_CASE("ablation: DistinctSet inserts each id once and clears in O(size)") {
  DistinctSet ds;
  std::set<std::uint32_t> ref;
  std::uint64_t s = 7;
  for (int round = 0; round < 50; ++round) {
    ds.clear();
    ref.clear();
    const int k = round % 7 == 0 ? 5000 : 40;
    for (int i = 0; i < k; ++i) {
      s = s * 6364136223846793005ull + 1442695040888963407ull;
      std::uint32_t y = std::uint32_t(s >> 33) % (round % 2 ? 100 : 1000000);
      if (i % 11 == 0) y = UINT32_MAX - 1 - y % 100;  // large ids too; UINT32_MAX is never a node id
      CHECK(ds.insert(y) == ref.insert(y).second);
    }
    CHECK(ds.size() == ref.size());
  }
}

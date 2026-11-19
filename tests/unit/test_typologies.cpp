// SPEC v2 §8 S5: the typologies' library features, tchain (temporal chains, `witness: latest`) and
// gs (gather-scatter, `min()`), on hand-checked cases for the oracle, the generated CPU code and
// the CUDA kernels, and relations on fuzz graphs that show the AC1 parity is not vacuous. The
// fuzz-graph parity is in test_cpu_backend.cpp and test_cuda_backend.cpp (every registered block).
#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <array>
#include <span>
#include <string>
#include <vector>

#include "graph/fuzz_gen.hpp"
#include "graph/index.hpp"
#include "oracle/oracle.hpp"
#include "runtime/cpu_backend.hpp"
#include "runtime/cuda_backend.hpp"

using namespace blz;

namespace {

const std::vector<std::uint32_t> kBins = {2, 3, 4, 5, 6};  // unit width; the last bin is open

struct Case {
  const char* what;
  Block block;
  std::int32_t window;
  Visibility vis;
  std::vector<std::array<int, 3>> edges;  // src, dst, ts in stream order
  std::uint32_t trigger;
  std::vector<std::uint32_t> want;  // the trigger's row under kBins
};

// tchain: the chains C_{k-1} -> ... -> C1 -> u -> v, every hop strictly before the next, all nodes
// distinct, in the window (closed: ts >= t - window). gs: min(|A|, |B|), A = the distinct sources
// of x = u before t (u and v excluded), B = the distinct sinks of x in the window (x excluded, v
// included), open lower bound. Node 0 is u (x), node 1 is v (w).
const std::vector<Case> kCases = {
    {"tchain: one 2-hop chain", Block::tchain, 10, Visibility::stream, {{2, 0, 1}, {0, 1, 5}}, 1, {1, 0, 0, 0, 0}},
    {"tchain: a 4-edge chain ends three chains", Block::tchain, 10, Visibility::stream,
     {{4, 3, 1}, {3, 2, 2}, {2, 0, 3}, {0, 1, 5}}, 3, {1, 1, 1, 0, 0}},
    {"tchain: the latest witness (2 -> 0 at 7, not 3, lets 3 -> 2 at 5 chain)", Block::tchain, 10,
     Visibility::stream, {{2, 0, 3}, {3, 2, 5}, {2, 0, 7}, {0, 1, 10}}, 3, {1, 1, 0, 0, 0}},
    {"tchain: parallel edges collapse, distinct sources each chain", Block::tchain, 10, Visibility::stream,
     {{2, 0, 1}, {2, 0, 2}, {3, 0, 3}, {0, 1, 5}}, 3, {2, 0, 0, 0, 0}},
    {"tchain: branching, two 3-edge chains", Block::tchain, 10, Visibility::stream,
     {{3, 2, 1}, {4, 2, 2}, {2, 0, 3}, {0, 1, 5}}, 3, {1, 2, 0, 0, 0}},
    {"tchain: a hop tied with the next one does not chain", Block::tchain, 10, Visibility::stream,
     {{3, 2, 4}, {2, 0, 4}, {0, 1, 5}}, 2, {1, 0, 0, 0, 0}},
    {"tchain: a hop tied with the trigger does not chain", Block::tchain, 10, Visibility::stream,
     {{2, 0, 5}, {0, 1, 5}}, 1, {0, 0, 0, 0, 0}},
    {"tchain: a later edge of the trigger's timestamp, timestamp visibility", Block::tchain, 10,
     Visibility::timestamp, {{0, 1, 5}, {2, 0, 5}}, 0, {0, 0, 0, 0, 0}},
    {"tchain: the closed lower bound keeps ts = t - window", Block::tchain, 5, Visibility::stream,
     {{2, 0, 5}, {0, 1, 10}}, 1, {1, 0, 0, 0, 0}},
    {"tchain: ts below t - window is outside", Block::tchain, 5, Visibility::stream, {{2, 0, 4}, {0, 1, 10}}, 1,
     {0, 0, 0, 0, 0}},
    {"tchain: a chain through v is a cycle, not a chain", Block::tchain, 10, Visibility::stream,
     {{1, 0, 2}, {0, 1, 5}}, 1, {0, 0, 0, 0, 0}},
    {"tchain: a chain may not revisit u", Block::tchain, 10, Visibility::stream, {{0, 2, 1}, {2, 0, 2}, {0, 1, 5}},
     2, {1, 0, 0, 0, 0}},
    {"tchain: a self-loop never chains", Block::tchain, 10, Visibility::stream, {{2, 2, 1}, {2, 0, 2}, {0, 1, 5}},
     2, {1, 0, 0, 0, 0}},
    {"tchain: a self-loop trigger emits nothing", Block::tchain, 10, Visibility::stream, {{2, 0, 1}, {0, 0, 5}},
     1, {0, 0, 0, 0, 0}},
    {"tchain: K_MAX = 10 caps the chains of an 11-edge path", Block::tchain, 100, Visibility::stream,
     {{11, 10, 1}, {10, 9, 2}, {9, 8, 3}, {8, 7, 4}, {7, 6, 5}, {6, 5, 6}, {5, 4, 7}, {4, 3, 8}, {3, 2, 9},
      {2, 0, 10}, {0, 1, 20}},
     10, {1, 1, 1, 1, 5}},
    {"gs: two sources, two sinks", Block::gs, 10, Visibility::stream, {{2, 0, 1}, {3, 0, 2}, {0, 4, 3}, {0, 1, 5}},
     3, {1, 0, 0, 0, 0}},
    {"gs: the smaller side decides", Block::gs, 10, Visibility::stream,
     {{2, 0, 1}, {3, 0, 1}, {4, 0, 2}, {7, 0, 2}, {0, 5, 3}, {0, 6, 3}, {0, 1, 5}}, 6, {0, 1, 0, 0, 0}},
    {"gs: w is never a source, but is a sink", Block::gs, 10, Visibility::stream,
     {{1, 0, 1}, {2, 0, 2}, {0, 3, 3}, {0, 1, 5}}, 3, {0, 0, 0, 0, 0}},
    {"gs: a source edge tied with the trigger does not count", Block::gs, 10, Visibility::stream,
     {{2, 0, 1}, {0, 4, 3}, {3, 0, 5}, {0, 1, 5}}, 3, {0, 0, 0, 0, 0}},
    {"gs: sinks need no order against the sources", Block::gs, 10, Visibility::stream,
     {{0, 4, 1}, {2, 0, 2}, {3, 0, 3}, {0, 1, 5}}, 3, {1, 0, 0, 0, 0}},
    {"gs: a source at the open lower bound is outside", Block::gs, 5, Visibility::stream,
     {{2, 0, 5}, {3, 0, 6}, {0, 4, 7}, {0, 1, 10}}, 3, {0, 0, 0, 0, 0}},
    {"gs: self-loops at x are neither sources nor sinks", Block::gs, 10, Visibility::stream,
     {{0, 0, 1}, {2, 0, 2}, {3, 0, 3}, {0, 0, 4}, {0, 4, 4}, {0, 1, 5}}, 5, {1, 0, 0, 0, 0}},
    {"gs: parallel edges count once", Block::gs, 10, Visibility::stream,
     {{2, 0, 1}, {2, 0, 2}, {3, 0, 3}, {0, 4, 4}, {0, 4, 4}, {0, 1, 5}}, 5, {1, 0, 0, 0, 0}},
    {"gs: a later sink of the trigger's timestamp, stream visibility", Block::gs, 10, Visibility::stream,
     {{2, 0, 1}, {3, 0, 2}, {0, 1, 5}, {0, 6, 5}}, 2, {0, 0, 0, 0, 0}},
    {"gs: the same, timestamp visibility", Block::gs, 10, Visibility::timestamp,
     {{2, 0, 1}, {3, 0, 2}, {0, 1, 5}, {0, 6, 5}}, 2, {1, 0, 0, 0, 0}},
    {"gs: a self-loop trigger emits nothing", Block::gs, 10, Visibility::stream,
     {{2, 0, 1}, {3, 0, 2}, {0, 4, 3}, {0, 0, 5}}, 3, {0, 0, 0, 0, 0}},
};

// The rows of every backend: the oracle, the CPU (both schedules), CUDA (every threshold, if built).
std::vector<std::pair<std::string, std::vector<std::uint32_t>>> all_rows(const Case& c) {
  std::vector<std::uint32_t> src, dst;
  std::vector<std::int32_t> ts;
  std::uint32_t n = 0;
  for (const auto& [s, d, t] : c.edges) {
    src.push_back(std::uint32_t(s));
    dst.push_back(std::uint32_t(d));
    ts.push_back(t);
    n = std::max(n, std::uint32_t(std::max(s, d)) + 1);
  }
  const std::uint64_t m = src.size();
  const BlockParams p{c.window, kBins};
  const auto pats = patterns_for(c.block);
  std::vector<std::pair<std::string, std::vector<std::uint32_t>>> out;
  std::vector<std::uint32_t> rows(m * kBins.size(), 0);
  Oracle({src, dst, ts, n}, c.vis).mine(c.block, p, compiled_k_max(), 0, m, rows.data());
  out.emplace_back("oracle", rows);
  const GraphIndex idx(src, dst, ts, n);
  TriggerSel all;
  all.last = m;
  for (Schedule s : {Schedule::dynamic, Schedule::sorted}) {
    std::fill(rows.begin(), rows.end(), 0);
    mine_cpu(idx.view(), pats, p, c.vis, all, s, rows.data());
    out.emplace_back(std::string("cpu ") + schedule_name(s), rows);
  }
  if (cuda_built()) {
    CudaMiner miner;
    const DeviceGraph dg(idx.view());
    for (std::uint32_t h : {0u, 2u, kAllLight}) {
      miner.mine(dg, pats, p, c.vis, 0, m, h, rows.data());
      out.emplace_back("cuda H=" + std::to_string(h), rows);
    }
  }
  return out;
}

// The size an emitted single-instance row stands for under unit-width bins, or 0 for none.
std::uint64_t size_of(const std::uint32_t* row, std::span<const std::uint32_t> bins) {
  for (std::size_t i = 0; i < bins.size(); ++i)
    if (row[i]) return bins[i];
  return 0;
}

}  // namespace

TEST_CASE("typologies: tchain and gs on hand-checked cases, every backend") {
  for (const Case& c : kCases) {
    const auto rows = all_rows(c);
    for (const auto& [backend, r] : rows) {
      INFO(c.what << " (" << block_name(c.block) << ", " << backend << ")");
      const std::vector<std::uint32_t> got(r.begin() + std::ptrdiff_t(c.trigger * kBins.size()),
                                           r.begin() + std::ptrdiff_t((c.trigger + 1) * kBins.size()));
      CHECK(got == c.want);
      CHECK(r == rows.front().second);  // every trigger's row, not only the case's
    }
  }
}

// B is fan_out's set (the distinct sinks of u but u), so gs <= fan_out on every trigger, and gs is
// strictly smaller where the sources are fewer. For tchain, the share of rows with a chain of 4 or
// more edges. Both show that the AC1 parity on fuzz graphs exercises the two features.
TEST_CASE("typologies: gs <= fan_out and long chains on fuzz graphs") {
  std::vector<std::uint32_t> bins;
  for (std::uint32_t b = 2; b <= 60; ++b) bins.push_back(b);  // fuzz graphs have n <= 40: sizes are exact
  std::uint64_t rows = 0, gs_rows = 0, gs_below = 0, long_chains = 0;
  for (std::uint64_t seed = 0; seed < 300; ++seed) {
    const FuzzGraph fg = fuzz_graph(seed);
    const std::uint64_t m = fg.src.size();
    const Oracle o({fg.src, fg.dst, fg.ts, fg.n});
    const BlockParams p{fg.window, bins};
    const auto mine = [&](Block b) {
      std::vector<std::uint32_t> r(m * bins.size(), 0);
      o.mine(b, p, compiled_k_max(), 0, m, r.data());
      return r;
    };
    const auto gs = mine(Block::gs), fan_out = mine(Block::fan_out), chain = mine(Block::tchain);
    for (std::uint64_t e = 0; e < m; ++e) {
      const std::size_t at = e * bins.size();
      const std::uint64_t g = size_of(&gs[at], bins), fo = size_of(&fan_out[at], bins);
      INFO("seed " << seed << " edge " << e);
      CHECK(g <= fo);
      ++rows;
      gs_rows += g != 0;
      gs_below += g != 0 && g < fo;
      long_chains += std::any_of(chain.data() + at + 2, chain.data() + at + bins.size(), [](std::uint32_t x) { return x != 0; });
    }
  }
  INFO(rows << " rows: gs on " << gs_rows << ", below fan_out on " << gs_below << "; chains of >= 4 edges on "
            << long_chains);
  CHECK(gs_rows * 10 > rows);
  CHECK(gs_below * 20 > rows);
  CHECK(long_chains * 10 > rows);
}

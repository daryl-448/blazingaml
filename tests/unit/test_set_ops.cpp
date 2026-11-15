// SPEC v2 §8 S4: the library features of the set ops, fan_oneway_out (difference) and fan_total
// (union), on hand-checked cases for the oracle, the generated CPU code and the CUDA kernels, and
// two relations on fuzz graphs that tie them to fan_out and fan_in. The fuzz-graph parity is in
// test_cpu_backend.cpp and test_cuda_backend.cpp (every registered block).
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

const std::vector<std::uint32_t> kBins = {2, 3, 4, 5, 6};

struct Case {
  const char* what;
  Block block;
  std::int32_t window;
  Visibility vis;
  std::vector<std::array<int, 3>> edges;  // src, dst, ts in stream order
  std::uint32_t trigger;
  std::uint64_t size;  // the trigger's instance size (below 2: no instance)
};

// fan_oneway_out = |N_out(u) \ N_in(u) \ {u}|, fan_total = |N_in(v) ∪ N_out(v) \ {v}|, over the
// window (ts > t - window) and the visible edges.
const std::vector<Case> kCases = {
    {"difference: nothing came back", Block::fan_oneway_out, 10, Visibility::stream,
     {{0, 1, 1}, {0, 2, 2}, {0, 3, 3}}, 2, 3},
    {"difference: a counterparty that paid back is removed", Block::fan_oneway_out, 10, Visibility::stream,
     {{0, 1, 1}, {0, 2, 2}, {2, 0, 2}, {0, 3, 3}}, 3, 2},
    {"difference: v paid u before the trigger, so v is removed", Block::fan_oneway_out, 10, Visibility::stream,
     {{1, 0, 1}, {0, 2, 2}, {0, 3, 2}, {0, 1, 3}}, 3, 2},
    {"difference: a return edge at the open lower bound is outside the window", Block::fan_oneway_out, 5,
     Visibility::stream, {{2, 0, 5}, {0, 1, 6}, {0, 2, 7}, {0, 3, 10}}, 3, 3},
    {"difference: a return edge just inside the window", Block::fan_oneway_out, 5, Visibility::stream,
     {{2, 0, 6}, {0, 1, 6}, {0, 2, 7}, {0, 3, 10}}, 3, 2},
    {"difference: an out edge at the open lower bound does not count", Block::fan_oneway_out, 5, Visibility::stream,
     {{0, 1, 5}, {0, 2, 7}, {0, 3, 10}}, 2, 2},
    {"difference: a later return edge of the trigger's timestamp, stream visibility", Block::fan_oneway_out, 10,
     Visibility::stream, {{0, 1, 1}, {0, 2, 2}, {0, 3, 3}, {2, 0, 3}}, 2, 3},
    {"difference: the same, timestamp visibility", Block::fan_oneway_out, 10, Visibility::timestamp,
     {{0, 1, 1}, {0, 2, 2}, {0, 3, 3}, {2, 0, 3}}, 2, 2},
    {"difference: self-loops and parallel edges", Block::fan_oneway_out, 10, Visibility::stream,
     {{0, 0, 1}, {0, 1, 1}, {0, 1, 2}, {0, 2, 2}, {0, 2, 3}}, 4, 2},
    {"difference: a self-loop trigger emits nothing", Block::fan_oneway_out, 10, Visibility::stream,
     {{0, 1, 1}, {0, 2, 1}, {0, 0, 2}}, 2, 0},
    {"union: v's in- and out-neighbours", Block::fan_total, 10, Visibility::stream,
     {{2, 1, 1}, {1, 3, 2}, {0, 1, 3}}, 2, 3},
    {"union: a counterparty on both sides counts once", Block::fan_total, 10, Visibility::stream,
     {{2, 1, 1}, {1, 2, 2}, {0, 1, 3}}, 2, 2},
    {"union: u on both sides alone is size 1", Block::fan_total, 10, Visibility::stream,
     {{1, 0, 2}, {0, 1, 3}}, 1, 1},
    {"union: u on both sides and one more payer", Block::fan_total, 10, Visibility::stream,
     {{1, 0, 2}, {2, 1, 2}, {0, 1, 3}}, 2, 2},
    {"union: self-loops at v and parallel edges", Block::fan_total, 10, Visibility::stream,
     {{1, 1, 1}, {2, 1, 1}, {2, 1, 2}, {1, 3, 2}, {1, 3, 2}, {0, 1, 3}}, 5, 3},
    {"union: the open lower bound on both sides", Block::fan_total, 5, Visibility::stream,
     {{2, 1, 5}, {1, 3, 5}, {1, 4, 6}, {0, 1, 10}}, 3, 2},
    {"union: a later out edge of the trigger's timestamp, stream visibility", Block::fan_total, 10,
     Visibility::stream, {{2, 1, 1}, {0, 1, 3}, {1, 4, 3}}, 1, 2},
    {"union: the same, timestamp visibility", Block::fan_total, 10, Visibility::timestamp,
     {{2, 1, 1}, {0, 1, 3}, {1, 4, 3}}, 1, 3},
    {"union: a self-loop trigger emits nothing", Block::fan_total, 10, Visibility::stream,
     {{2, 1, 1}, {1, 3, 1}, {1, 1, 2}}, 2, 0},
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

// The size an emitted row stands for under unit-width bins (the last bin open), or 0 for none.
std::uint64_t size_of(const std::uint32_t* row, std::span<const std::uint32_t> bins) {
  for (std::size_t i = 0; i < bins.size(); ++i)
    if (row[i]) return bins[i];
  return 0;
}

}  // namespace

TEST_CASE("set ops: fan_oneway_out and fan_total on hand-checked cases, every backend") {
  for (const Case& c : kCases) {
    std::vector<std::uint32_t> want(kBins.size(), 0);
    if (const int i = bin_index(kBins, c.size); i >= 0) want[std::size_t(i)] = 1;
    const auto rows = all_rows(c);
    for (const auto& [backend, r] : rows) {
      INFO(c.what << " (" << block_name(c.block) << ", " << backend << ")");
      const std::vector<std::uint32_t> got(r.begin() + std::ptrdiff_t(c.trigger * kBins.size()),
                                           r.begin() + std::ptrdiff_t((c.trigger + 1) * kBins.size()));
      CHECK(got == want);
      CHECK(r == rows.front().second);  // every trigger's row, not only the case's
    }
  }
}

// N_out(u) \ N_in(u) ⊆ N_out(u) and N_in(v) ⊆ N_in(v) ∪ N_out(v): on every trigger the difference
// is at most fan_out and the union at least fan_in, both of the v1 oracle. The strict cases show
// that the fuzz graphs exercise both ops (the AC1 parity is not vacuous).
TEST_CASE("set ops: fan_oneway_out <= fan_out and fan_total >= fan_in on fuzz graphs, often strictly") {
  std::vector<std::uint32_t> bins;
  for (std::uint32_t b = 2; b <= 60; ++b) bins.push_back(b);  // fuzz graphs have n <= 40: sizes are exact
  std::uint64_t rows = 0, removed = 0, added = 0;
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
    const auto oneway = mine(Block::fan_oneway_out), fan_out = mine(Block::fan_out);
    const auto total = mine(Block::fan_total), fan_in = mine(Block::fan_in);
    for (std::uint64_t e = 0; e < m; ++e) {
      const std::size_t at = e * bins.size();
      const std::uint64_t d = size_of(&oneway[at], bins), fo = size_of(&fan_out[at], bins);
      const std::uint64_t u = size_of(&total[at], bins), fi = size_of(&fan_in[at], bins);
      INFO("seed " << seed << " edge " << e);
      CHECK(d <= fo);
      CHECK(u >= fi);
      ++rows;
      removed += d < fo;
      added += u > fi;
    }
  }
  INFO(rows << " rows, " << removed << " with the difference below fan_out, " << added << " with the union above fan_in");
  CHECK(removed * 10 > rows);
  CHECK(added * 10 > rows);
}

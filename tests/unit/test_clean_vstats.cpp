// R8's vertex statistics on the runtime (SPEC v3 §3.2, F7): CleanVstats on the CPU and, in a CUDA
// build, on the GPU, against the clean oracle bit for bit, on hand-checked sides (n = 0, n = 1,
// equal values, a computed m2 of 0, an underflowing denominator, a self-loop trigger), a hub side
// of thousands of edges, trigger samples and chunk boundaries, and fuzz graphs.
#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <vector>

#include <omp.h>

#include "graph/fuzz_gen.hpp"
#include "graph/index.hpp"
#include "oracle/clean_oracle.hpp"
#include "runtime/clean_vstats.hpp"
#include "runtime/cuda_backend.hpp"

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
  std::uint32_t n = 0;
  Graph() = default;
  Graph(std::initializer_list<Edge> edges) {
    for (const Edge& e : edges) add(e);
  }
  void add(const Edge& e) {
    src.push_back(e.u);
    dst.push_back(e.v);
    ts.push_back(e.t);
    amount.push_back(e.amount);
    n = std::max(n, std::max(e.u, e.v) + 1);
  }
  std::uint64_t m() const { return src.size(); }
};

using Rows = std::vector<double>;

TriggerSel all_rows(std::uint64_t m) {
  TriggerSel s;
  s.last = m;
  return s;
}

Rows oracle_rows(const Graph& g, std::int32_t window, const TriggerSel& sel) {
  Rows out(sel.rows() * kVstatsCols);
  const CleanOracle o({g.src, g.dst, g.ts, g.n}, g.amount);
  if (sel.sampled) o.vstats(window, sel.eids, out.data());
  else o.vstats(window, sel.first, sel.last, out.data());
  return out;
}

// The rows of `sel` on the CPU, or on the GPU with `gpu`.
Rows runtime_rows(const Graph& g, std::int32_t window, const TriggerSel& sel, bool gpu, std::uint64_t* sides = nullptr) {
  const GraphIndex idx(g.src, g.dst, g.ts, g.n, SelfLoops::drop);
  std::optional<DeviceGraph> dg;
  if (gpu) dg.emplace(idx.view());
  CleanVstats vs(idx.view(), g.amount, window, gpu ? &*dg : nullptr);
  Rows out(sel.rows() * kVstatsCols, std::numeric_limits<double>::quiet_NaN());
  vs.rows(sel, out.data());
  if (sides) *sides = vs.sides();
  return out;
}

bool same_bits(const Rows& a, const Rows& b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (std::memcmp(&a[i], &b[i], sizeof(double)) != 0) {
      UNSCOPED_INFO("first difference: row " << i / kVstatsCols << " column " << i % kVstatsCols << ": " << a[i]
                                             << " vs " << b[i]);
      return false;
    }
  return true;
}

// Side k (0 u_out, 1 u_in, 2 v_out, 3 v_in) of row r.
Rows side(const Rows& rows, std::uint64_t r, int k) {
  const auto at = rows.begin() + std::ptrdiff_t(r * kVstatsCols + std::uint64_t(k) * kVstatsPerSide);
  return {at, at + kVstatsPerSide};
}

// Every row on the CPU, and on the GPU in a CUDA build, each equal to the clean oracle's bit for bit.
// Returns the CPU's rows.
Rows checked_rows(const Graph& g, std::int32_t window) {
  const TriggerSel sel = all_rows(g.m());
  const Rows want = oracle_rows(g, window, sel), cpu = runtime_rows(g, window, sel, false);
  CHECK(same_bits(cpu, want));
  if (cuda_built()) CHECK(same_bits(runtime_rows(g, window, sel, true), want));
  return cpu;
}

const Rows kNone(kVstatsPerSide, 0.0);

}  // namespace

TEST_CASE("clean vstats: column names") {
  const auto names = clean_vstats_column_names();
  REQUIRE(names.size() == std::size_t(kVstatsCols));
  CHECK(names[0] == "vs_u_out_fan");
  CHECK(names[3] == "vs_u_out_age_mean");
  CHECK(names[8] == "vs_u_out_amount_mean");
  CHECK(names[kVstatsPerSide] == "vs_u_in_fan");
  CHECK(names[51] == "vs_v_in_amount_kurt");
}

TEST_CASE("clean vstats: small sides, CPU and CUDA = the clean oracle (R8)") {
  // n = 1 (u_out, v_in: the trigger alone, age 0, amount 7) and n = 0 (u_in, v_out).
  {
    const Rows r = checked_rows(Graph{{0, 1, 10, 7}}, 100), one = {1, 1, 1, 0, 0, 0, 0, 0, 7, 7, 0, 0, 0};
    CHECK(same_bits(side(r, 0, 0), one));
    CHECK(same_bits(side(r, 0, 1), kNone));
    CHECK(same_bits(side(r, 0, 2), kNone));
    CHECK(same_bits(side(r, 0, 3), one));
  }
  // Equal values: three amounts 0.1 into 9 at one time. The computed m2 would be 1.9e-34; min = max
  // makes var, skew and kurt 0.
  {
    const double sum = 0.1 + 0.1 + 0.1;
    const Rows r = checked_rows(Graph{{1, 9, 5, 0.1}, {2, 9, 5, 0.1}, {3, 9, 5, 0.1}}, 100);
    for (std::uint64_t e = 0; e < 3; ++e) CHECK(same_bits(side(r, e, 3), {3, 3, 1, 0, 0, 0, 0, 0, sum / 3, sum, 0, 0, 0}));
  }
  // Distinct values whose computed m2 is 0 (the squared deviations underflow).
  {
    const double sum = 1e-200 + 2e-200;
    const Rows r = checked_rows(Graph{{1, 9, 5, 1e-200}, {2, 9, 5, 2e-200}}, 100);
    CHECK(same_bits(side(r, 1, 3), {2, 2, 1, 0, 0, 0, 0, 0, sum / 2, sum, 0, 0, 0}));
  }
  // D-58: a positive m2 whose m2^1.5 and m2^2 underflow: skew and kurt 0, not 0/0.
  {
    const double sum = 1e-160 + 3e-160, mean = sum / 2, d1 = 1e-160 - mean, d2 = 3e-160 - mean;
    const double m2 = (0.0 + d1 * d1 + d2 * d2) / 2;
    REQUIRE(m2 > 0);
    REQUIRE(m2 * std::sqrt(m2) == 0);
    const Rows r = checked_rows(Graph{{1, 9, 5, 1e-160}, {2, 9, 5, 3e-160}}, 100);
    CHECK(same_bits(side(r, 1, 3), {2, 2, 1, 0, 0, 0, 0, 0, mean, sum, m2, 0, 0}));
  }
  // The moments: into 9 at 5, 9, 10, the trigger at t = 10. Ages {0, 1, 5}: mean 2, sum 6,
  // m2 = 14/3, m3 = 6, m4 = 98/3. Amounts {1, 2, 3}: mean 2, sum 6, m2 = 2/3, m3 = 0, m4 = 2/3.
  {
    const double a2 = 14.0 / 3, a4 = 98.0 / 3, b2 = 2.0 / 3;
    const Rows r = checked_rows(Graph{{1, 9, 5, 3}, {2, 9, 9, 1}, {3, 9, 10, 2}}, 100);
    CHECK(same_bits(side(r, 2, 3),
                    {3, 3, 1, 2, 6, a2, 6 / (a2 * std::sqrt(a2)), a4 / (a2 * a2), 2, 6, b2, 0, b2 / (b2 * b2)}));
  }
  // The window (t - W, t]: at W = 4 the edge at 5 is out of the trigger at 9's window (5 = 9 - 4)
  // and the one at 6 in; a later edge at the trigger's time is in (R3).
  {
    const Rows r = checked_rows(Graph{{1, 9, 5, 1}, {2, 9, 6, 2}, {3, 9, 9, 4}, {4, 9, 9, 8}}, 4);
    CHECK(side(r, 2, 3)[1] == 3);
    CHECK(side(r, 2, 3)[9] == 2 + 4 + 8);
    CHECK(side(r, 1, 3)[1] == 2);  // at t = 6: 5 and 6
  }
  // The four sides in order, each with its fan, degree and ratio.
  {
    const Rows r = checked_rows(Graph{{3, 0, 1}, {3, 0, 1}, {4, 0, 1},                         // u_in
                                      {1, 5, 1}, {1, 5, 1}, {1, 5, 1}, {1, 5, 1},             // v_out
                                      {6, 1, 1}, {7, 1, 1}, {8, 1, 1}, {9, 1, 1}, {0, 2, 1},  // v_in, u_out
                                      {0, 1, 2}},
                                100);
    const double want[4][3] = {{2, 2, 1}, {2, 3, 1.5}, {1, 4, 4}, {5, 5, 1}};
    for (int k = 0; k < 4; ++k) {
      INFO("side " << k);
      const Rows s = side(r, 12, k);
      CHECK(same_bits({s[0], s[1], s[2]}, {want[k][0], want[k][1], want[k][2]}));
    }
  }
}

TEST_CASE("clean vstats: a self-loop trigger has its node's loop-free sides (R1, R8)") {
  // e2 is 0 -> 0 at 3 with amount 100, which no window holds.
  const Graph g{{0, 1, 1, 5}, {2, 0, 2, 3}, {0, 0, 3, 100}, {0, 2, 3, 4}};
  const Rows r = checked_rows(g, 100);
  // 0's out-edges at t = 3: e0 (age 2, 5) and e3 (age 0, 4); ages {0, 2}: mean 1, m2 1, m3 0, m4 1;
  // amounts {4, 5}: mean 4.5, m2 0.25, m3 0, m4 0.0625. 0's in-edge: e1 (age 1, 3).
  const Rows out = {2, 2, 1, 1, 2, 1, 0, 1, 4.5, 9, 0.25, 0, 1}, in = {1, 1, 1, 1, 1, 0, 0, 0, 3, 3, 0, 0, 0};
  for (int k : {0, 2}) CHECK(same_bits(side(r, 2, k), out));
  for (int k : {1, 3}) CHECK(same_bits(side(r, 2, k), in));
  CHECK(same_bits(side(r, 3, 0), out));  // e3's u_out: the same window, the loop left out
}

TEST_CASE("clean vstats: hub sides of thousands of edges, negative times, any thread count") {
  // Node 0 sends 6,000 edges to 300 neighbours over 12 windows of W = 1,000 that start below 0,
  // with tied times and amounts; 2,000 edges come back. Windows hold about 500 edges of the hub, and
  // straddle the multiples of W that cut its lists into runs.
  Graph g;
  std::uint64_t s = 42;
  const auto next = [&] {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return s >> 33;
  };
  for (std::int32_t i = 0; i < 8000; ++i) {
    const std::int32_t t = -3000 + i * 12000 / 8000 - std::int32_t(next() % 2);
    const std::uint32_t y = 1 + std::uint32_t(next() % 300);
    const std::uint64_t c = next() % 10;
    const double a = c < 3 ? double(next() % 1000) / 100 : c < 6 ? double(1 + next() % 4) : double(next() % 100000);
    if (i % 4 == 3) g.add({y, 0, t, a});
    else g.add({0, y, t, a});
  }
  std::sort(g.ts.begin(), g.ts.end());  // the times only, so the stream is in ts order
  const Rows r = checked_rows(g, 1000);
  double most = 0;
  for (std::uint64_t e = 0; e < g.m(); ++e) most = std::max(most, side(r, e, g.src[e] == 0 ? 0 : 2)[1]);
  CHECK(most >= 400);
  const int threads = omp_get_max_threads();
  omp_set_num_threads(1);
  const Rows one = runtime_rows(g, 1000, all_rows(g.m()), false);
  omp_set_num_threads(threads);
  CHECK(same_bits(one, r));
}

TEST_CASE("clean vstats: trigger samples and chunk boundaries") {
  // 300,000 edges in runs of 7 equal times, so a run straddles the first chunk's end (2^18 rows).
  Graph g;
  for (std::uint32_t i = 0; i < 300000; ++i)
    g.add({i % 97, (i * 31 + 5) % 89 + 100, std::int32_t(i / 7), double((i * 2654435761u) % 1000) / 8});
  REQUIRE(g.ts[CleanVstats::kChunkRows - 1] == g.ts[CleanVstats::kChunkRows]);
  const TriggerSel all = all_rows(g.m());
  std::uint64_t sides = 0;
  const Rows cpu = runtime_rows(g, 50, all, false, &sides);
  CHECK(sides > 0);
  if (cuda_built()) CHECK(same_bits(runtime_rows(g, 50, all, true), cpu));
  // The rows around the chunk boundary and a sample, against the oracle and the full run.
  TriggerSel sample;
  sample.sampled = true;
  for (std::uint32_t e = 0; e < g.m(); e += 997) sample.eids.push_back(e);
  for (std::uint32_t e = CleanVstats::kChunkRows - 20; e < CleanVstats::kChunkRows + 20; ++e) sample.eids.push_back(e);
  std::sort(sample.eids.begin(), sample.eids.end());
  sample.eids.erase(std::unique(sample.eids.begin(), sample.eids.end()), sample.eids.end());
  sample.first = sample.eids.front();
  sample.last = sample.eids.back() + 1;
  const Rows got = runtime_rows(g, 50, sample, false);
  CHECK(same_bits(got, oracle_rows(g, 50, sample)));
  Rows from_all;
  for (std::uint32_t e : sample.eids)
    from_all.insert(from_all.end(), cpu.begin() + std::ptrdiff_t(e * kVstatsCols),
                    cpu.begin() + std::ptrdiff_t((e + 1) * kVstatsCols));
  CHECK(same_bits(got, from_all));
}

TEST_CASE("clean vstats: fuzz graphs, CPU and CUDA = the clean oracle, dense and sparse") {
  for (const FuzzSetting setting : {FuzzSetting::dense, FuzzSetting::sparse})
    for (std::uint64_t seed = 0; seed < 150; ++seed) {
      INFO(fuzz_setting_name(setting) << " seed " << seed);
      const FuzzGraph fg = fuzz_graph(seed, setting);
      Graph g;
      const std::vector<double> amount = fuzz_amounts(seed, fg.src.size());
      for (std::size_t e = 0; e < fg.src.size(); ++e) g.add({fg.src[e], fg.dst[e], fg.ts[e], amount[e]});
      g.n = fg.n;
      checked_rows(g, fg.window);
    }
}

TEST_CASE("clean vstats: inputs") {
  const Graph g{{0, 1, 1, 2}, {1, 0, 2, 3}};
  const GraphIndex idx(g.src, g.dst, g.ts, g.n, SelfLoops::drop);
  CHECK_THROWS_AS(CleanVstats(idx.view(), std::vector<double>{1}, 10), std::invalid_argument);
  CHECK_THROWS_AS(CleanVstats(idx.view(), g.amount, 0), std::invalid_argument);
  CHECK_THROWS_AS(CleanVstats(idx.view(), std::vector<double>{1, std::numeric_limits<double>::infinity()}, 10),
                  std::invalid_argument);
  CleanVstats vs(idx.view(), g.amount, 10);
  TriggerSel bad;
  bad.last = 3;
  std::vector<double> out(3 * kVstatsCols);
  CHECK_THROWS_AS(vs.rows(bad, out.data()), std::invalid_argument);
}

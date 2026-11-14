// Vertex statistics (SPEC v2 §8 S2): the CPU module against GFP's probes (A43-A45), against a
// naive reference over the edge stream, bit for bit; thread-count independence; CPU = CUDA bit
// for bit in a CUDA build.
#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <omp.h>

#include "graph/fuzz_gen.hpp"
#include "graph/index.hpp"
#include "runtime/cuda_backend.hpp"
#include "runtime/vstats.hpp"

using namespace blz;

namespace {

// Row e's statistics from the definition: per side, the edges f <= e of the node in that
// direction with ts[f] >= ts[e - 1] - W, in eid order; the same arithmetic as vstats_side.
std::vector<double> reference(const FuzzGraph& g, const std::vector<double>& amount, const VstatsParams& p) {
  const std::uint64_t m = g.src.size();
  std::vector<double> out(m * kVstatsCols);
  for (std::uint64_t e = 0; e < m; ++e) {
    const std::int64_t lo = e == 0 ? INT64_MIN : std::int64_t(g.ts[e - 1]) - p.window;
    int side = 0;
    for (std::uint32_t node : {g.src[e], g.dst[e]})
      for (bool out_dir : {true, false}) {
        std::vector<std::uint64_t> w;
        std::set<std::uint32_t> nbrs;
        for (std::uint64_t f = 0; f <= e; ++f)
          if ((out_dir ? g.src[f] : g.dst[f]) == node && g.ts[f] >= lo) {
            w.push_back(f);
            nbrs.insert(out_dir ? g.dst[f] : g.src[f]);
          }
        double* o = out.data() + e * kVstatsCols + side++ * kVstatsPerSide;
        const double n = double(w.size());
        o[0] = double(nbrs.size());
        o[1] = n;
        o[2] = nbrs.empty() ? 0.0 : n / double(nbrs.size());
        for (int col = 0; col < 2; ++col) {
          auto x = [&](std::uint64_t f) { return col == 0 ? double(g.ts[f]) + p.ts_offset : amount[f]; };
          VsMoments mo;
          for (std::uint64_t f : w) mo.sum += x(f);
          if (!w.empty()) {
            const double mu = mo.sum / n;
            for (std::uint64_t f : w) {
              const double d = x(f) - mu, d2 = d * d;
              mo.m2 += d2;
              mo.m3 += d2 * d;
              mo.m4 += d2 * d2;
            }
          }
          vstats_finish(mo, n, o + 3 + 5 * col);
        }
      }
  }
  return out;
}

// The first cell whose bits differ (row * kVstatsCols + column), or -1.
std::int64_t first_diff(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) return 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (std::memcmp(&a[i], &b[i], sizeof(double)) != 0) return std::int64_t(i);
  return -1;
}
bool same_bits(const std::vector<double>& a, const std::vector<double>& b) {
  const std::int64_t i = first_diff(a, b);
  if (i >= 0 && std::size_t(i) < a.size() && a.size() == b.size())
    UNSCOPED_INFO("first difference: row " << i / kVstatsCols << " column " << i % kVstatsCols << ": " << a[std::size_t(i)]
                                            << " vs " << b[std::size_t(i)]);
  return i < 0;
}

std::vector<double> cpu_rows(const GraphIndex& idx, const std::vector<double>& amount, const VstatsParams& p,
                             const TriggerSel& sel) {
  std::vector<double> out(sel.rows() * kVstatsCols);
  vstats_cpu(idx.view(), amount, p, sel, out.data());
  return out;
}

}  // namespace

TEST_CASE("vstats: GFP's column names and order") {
  const auto names = vstats_column_names();
  REQUIRE(names.size() == std::size_t(kVstatsCols));
  CHECK(names[0] == "vs_src_out_fan");
  CHECK(names[2] == "vs_src_out_ratio");
  CHECK(names[3] == "vs_src_out_ts_avg");
  CHECK(names[8] == "vs_src_out_amount_received_avg");
  CHECK(names[12] == "vs_src_out_amount_received_kurtosis");
  CHECK(names[13] == "vs_src_in_fan");
  CHECK(names[51] == "vs_dst_in_amount_received_kurtosis");
}

TEST_CASE("vstats: the guard, the n = 1 case and the moments") {
  double o[5];
  vstats_finish({}, 0, o);
  CHECK((o[0] == 0 && o[1] == 0 && o[4] == 0));
  // {1, 5}: mean 3, population variance 4, kurtosis (not excess) 1.
  vstats_finish({6, 8, 0, 32}, 2, o);
  CHECK((o[0] == 3 && o[1] == 6 && o[2] == 4 && o[3] == 0 && o[4] == 1));
  // {1000, 1006}: variance 9 < 1e-5 * 1003^2, so GFP reports 0 (A44).
  vstats_finish({2006, 18, 0, 162}, 2, o);
  CHECK((o[0] == 1003 && o[1] == 2006 && o[2] == 0 && o[3] == 0 && o[4] == 0));
}

// GFP's rows for the S2 probes. "rule" cases are within S2's tolerance in every column; in "split"
// (vertex_stats_tw < W, A43) and "drift" (GFP's accumulators after eviction, A45) cases only fan
// and degree follow the rule.
TEST_CASE("vstats: the CPU module matches the GFP vertex-statistics probes") {
  std::ifstream in(std::string(BLZ_SOURCE_DIR) + "/tests/golden/gfp_vstats_probes.json");
  REQUIRE(in);
  const auto doc = nlohmann::json::parse(in);
  REQUIRE(doc["columns"].get<std::vector<std::string>>() == vstats_column_names());
  int rule = 0;
  for (const auto& c : doc["cases"]) {
    std::vector<std::uint32_t> src, dst;
    std::vector<std::int32_t> ts;
    std::vector<double> amount;
    std::uint32_t n = 0;
    for (const auto& e : c["edges"]) {
      src.push_back(e[0].get<std::uint32_t>());
      dst.push_back(e[1].get<std::uint32_t>());
      ts.push_back(e[2].get<std::int32_t>());
      amount.push_back(e[3].get<double>());
      n = std::max({n, src.back() + 1, dst.back() + 1});
    }
    const GraphIndex idx(src, dst, ts, n);
    TriggerSel all;
    all.last = src.size();
    const auto got = cpu_rows(idx, amount, {c["window"].get<std::int32_t>(), 0.0}, all);
    const bool full = c["kind"] == "rule";
    rule += full;
    for (std::size_t e = 0; e < src.size(); ++e)
      for (int j = 0; j < kVstatsCols; ++j) {
        const double want = c["output"][e][std::size_t(j)].get<double>(), have = got[e * kVstatsCols + j];
        const bool integer = j % kVstatsPerSide < 2;
        if (!integer && !full) continue;
        INFO(c["id"].get<std::string>() << " row " << e << " " << vstats_column_names()[std::size_t(j)] << ": GFP "
                                        << want << ", blz " << have);
        CHECK((integer ? have == want : std::abs(have - want) <= std::max(1e-9, 1e-6 * std::abs(want))));
      }
  }
  CHECK(rule >= 8);
}

TEST_CASE("vstats: the CPU module equals the definition, bit for bit, at any thread count") {
  const int threads = omp_get_max_threads();
  for (std::uint64_t seed = 0; seed < 300; ++seed) {
    const FuzzGraph g = fuzz_graph(seed);
    const std::uint64_t m = g.src.size();
    const std::vector<double> amount = fuzz_amounts(seed, m);
    const GraphIndex idx(g.src, g.dst, g.ts, g.n);
    const VstatsParams p{g.window, 1.0};
    const auto want = reference(g, amount, p);
    TriggerSel all;
    all.last = m;
    INFO("seed " << seed);
    omp_set_num_threads(1);
    CHECK(same_bits(cpu_rows(idx, amount, p, all), want));
    omp_set_num_threads(threads);
    CHECK(same_bits(cpu_rows(idx, amount, p, all), want));
    TriggerSel range;
    range.first = m / 3;
    range.last = m - m / 4;
    CHECK(same_bits(cpu_rows(idx, amount, p, range),
                    std::vector<double>(want.begin() + range.first * kVstatsCols, want.begin() + range.last * kVstatsCols)));
  }
  omp_set_num_threads(threads);
}

TEST_CASE("vstats: CPU = CUDA, bit for bit") {
  if (!cuda_built()) SKIP("no CUDA backend in this build");
  for (std::uint64_t seed = 0; seed < 300; ++seed) {
    const FuzzGraph g = fuzz_graph(seed);
    const std::uint64_t m = g.src.size();
    const std::vector<double> amount = fuzz_amounts(seed, m);
    const GraphIndex idx(g.src, g.dst, g.ts, g.n);
    const VstatsParams p{g.window, 1.0};
    TriggerSel all;
    all.last = m;
    const auto want = cpu_rows(idx, amount, p, all);
    const DeviceGraph dg(idx.view());
    std::vector<double> got(m * kVstatsCols);
    vstats_cuda(dg, amount, p, 0, m, got.data());
    INFO("seed " << seed);
    CHECK(same_bits(got, want));
  }
}

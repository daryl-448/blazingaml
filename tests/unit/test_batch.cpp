// GFP batch mode (SPEC v2 §8 S1): the batch oracle and the CPU module against the committed GFP
// batch probes, against each other, and B = 1 against v1's stream mode with trigger credit.
#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <omp.h>

#include "blz/edges.hpp"
#include "graph/fuzz_gen.hpp"
#include "graph/index.hpp"
#include "oracle/batch_oracle.hpp"
#include "oracle/oracle.hpp"
#include "runtime/batch.hpp"
#include "runtime/triggers.hpp"

using namespace blz;

namespace {

struct Small {
  std::vector<std::uint32_t> src, dst;
  std::vector<std::int32_t> ts;
  std::uint64_t n = 0;
  EdgeStream view() const { return {src, dst, ts, n}; }
};

Small from_triples(const nlohmann::json& edges) {
  Small g;
  for (const auto& e : edges) {
    g.src.push_back(e[0]);
    g.dst.push_back(e[1]);
    g.ts.push_back(e[2]);
    g.n = std::max<std::uint64_t>(g.n, std::max(g.src.back(), g.dst.back()) + 1);
  }
  return g;
}

std::vector<std::uint32_t> unit_bins() {
  std::vector<std::uint32_t> b;
  for (std::uint32_t i = 2; i <= 33; ++i) b.push_back(i);
  return b;
}

std::vector<std::uint32_t> batch_oracle_rows(const BatchOracle& o, Block b, const BlockParams& p, int k_max,
                                             const TriggerSel& sel) {
  std::vector<std::uint32_t> out(sel.rows() * p.bins.size(), 0);
  o.mine(b, p, k_max, sel, out.data());
  return out;
}

std::vector<std::uint32_t> batch_cpu_rows(const GraphIndex& idx, Block b, const BlockParams& p, int k_max,
                                          std::int32_t tw, const BatchLayout& l, const TriggerSel& sel) {
  std::vector<std::uint32_t> out(sel.rows() * p.bins.size(), 0);
  mine_batch_cpu(idx.view(), b, p, k_max, tw, l, sel, out.data());
  return out;
}

TriggerSel all_of(std::uint64_t m) {
  TriggerSel s;
  s.last = m;
  return s;
}

}  // namespace

TEST_CASE("batch: layouts") {
  CHECK(batch_layout(10, 4, {}).starts == std::vector<std::uint64_t>{0, 4, 8, 10});
  // A cut starts a batch, and the batches after it count from the cut; cuts at 0, at m or past it
  // and repeated cuts change nothing.
  const std::vector<std::uint64_t> cuts = {5, 0, 10, 12, 5};
  CHECK(batch_layout(10, 4, cuts).starts == std::vector<std::uint64_t>{0, 4, 5, 9, 10});
  CHECK(batch_layout(3, 128, {}).starts == std::vector<std::uint64_t>{0, 3});
  CHECK(batch_layout(0, 4, {}).batches() == 0);
  const BatchLayout l = batch_layout(10, 4, cuts);
  CHECK(l.batch_of(0) == 0);
  CHECK(l.batch_of(4) == 1);
  CHECK(l.batch_of(5) == 2);
  CHECK(l.batch_of(9) == 3);
  // cuts: splits cuts at b_fv and at the train/test boundary of the whole stream, also when only
  // a prefix is mined (blz mine --edges).
  const FuzzGraph fg = fuzz_graph(7);
  const std::uint64_t m = fg.ts.size();
  const std::vector<std::uint64_t> splits = {fit_val_boundary(fg.ts), train_size(m)};
  CHECK(batch_layout(BatchSpec{3, true, ""}, m, fg.ts).starts == batch_layout(m, 3, splits).starts);
  CHECK(batch_layout(BatchSpec{3, false, ""}, m, fg.ts).starts == batch_layout(m, 3, {}).starts);
  CHECK(batch_layout(BatchSpec{3, true, ""}, m / 2, fg.ts).starts == batch_layout(m / 2, 3, splits).starts);
  CHECK_THROWS(batch_layout(10, 0, {}));
}

// S1's oracle and CPU module reproduce every committed GFP batch probe that is a plain
// transform() sequence (A34, A35 and A39-A42; A36 re-sends edges and A37 overrides GFP
// parameters, neither of which batch mode models).
TEST_CASE("batch: oracle and cpu match the GFP batch probes") {
  std::ifstream in(std::string(BLZ_SOURCE_DIR) + "/tests/golden/gfp_batch_probes.json");
  REQUIRE(in);
  const auto doc = nlohmann::json::parse(in);
  const auto bins = doc["params"]["bins"].get<std::vector<std::uint32_t>>();
  int ran = 0;
  for (const auto& c : doc["cases"]) {
    const auto& recipe = c["recipe"];
    if (recipe.size() != 1 || !recipe.contains("cuts")) continue;
    ++ran;
    const Small g = from_triples(c["edges"]);
    const std::uint64_t m = g.src.size();
    const auto cuts = recipe["cuts"].get<std::vector<std::uint64_t>>();
    const BatchLayout layout = batch_layout(m, m, cuts);
    const auto& tw = c["tw"];
    const std::int32_t sg_w = tw.is_object() ? tw["sg"].get<std::int32_t>() : tw.get<std::int32_t>();
    const std::int32_t other_w = tw.is_object() ? tw["other"].get<std::int32_t>() : sg_w;
    const std::int32_t time_window = std::max(sg_w, other_w);  // GFP's time_window = -1
    const BatchOracle o(g.view(), layout, time_window);
    const GraphIndex idx(g.src, g.dst, g.ts, g.n);
    for (Block b : kV1Blocks) {
      const BlockParams p{b == Block::sg ? sg_w : other_w, bins};
      const auto want_o = batch_oracle_rows(o, b, p, 10, all_of(m));
      const auto want_c = batch_cpu_rows(idx, b, p, 10, time_window, layout, all_of(m));
      for (std::uint64_t e = 0; e < m; ++e) {
        const auto want = c["output"][e][std::string(block_name(b))].get<std::vector<std::uint32_t>>();
        const auto at = [&](const std::vector<std::uint32_t>& v) {
          return std::vector<std::uint32_t>(v.begin() + e * bins.size(), v.begin() + (e + 1) * bins.size());
        };
        INFO(c["id"].get<std::string>() << " edge " << e << " " << block_name(b));
        CHECK(at(want_o) == want);
        CHECK(at(want_c) == want);
      }
    }
  }
  CHECK(ran >= 35);
}

// One edge per batch: a row's batch is its own edge, so only the trigger is credited and no
// batch edge is later than it. That is v1's stream mode with trigger credit (§4 of v1).
TEST_CASE("batch: B = 1 is stream visibility with trigger credit") {
  const auto bins = unit_bins();
  for (std::uint64_t seed = 0; seed < 300; ++seed) {
    const FuzzGraph fg = fuzz_graph(seed);
    const std::uint64_t m = fg.src.size();
    const EdgeStream s{fg.src, fg.dst, fg.ts, fg.n};
    const Oracle v1(s, Visibility::stream);
    const BatchOracle o(s, batch_layout(m, 1, {}), fg.window);
    for (Block b : kV1Blocks) {
      const BlockParams p{fg.window, bins};
      std::vector<std::uint32_t> want(m * bins.size(), 0);
      v1.mine(b, p, 10, 0, m, want.data());
      INFO("seed " << seed << " " << block_name(b));
      CHECK(batch_oracle_rows(o, b, p, 10, all_of(m)) == want);
    }
  }
}

// The CPU module equals the oracle for any layout, trigger selection and thread count.
TEST_CASE("batch: cpu equals the oracle across layouts, selections and threads") {
  const auto bins = unit_bins();
  const int threads = omp_get_max_threads();
  for (std::uint64_t seed = 0; seed < 40; ++seed) {  // 10,000: the slow fuzz_diff_<batch profile>_cpu_10k
    const FuzzGraph fg = fuzz_graph(seed);
    const std::uint64_t m = fg.src.size();
    if (m < 4) continue;
    const EdgeStream s{fg.src, fg.dst, fg.ts, fg.n};
    const GraphIndex idx(fg.src, fg.dst, fg.ts, fg.n);
    const std::vector<std::uint64_t> cuts = {m / 3, m / 2};
    for (const BatchLayout& l : {batch_layout(m, 1, {}), batch_layout(m, 3, cuts), batch_layout(m, 32, {})}) {
      const BatchOracle o(s, l, fg.window);
      TriggerSel range;
      range.first = m / 4;
      range.last = m - m / 5;
      const TriggerSel sample = select_triggers("sample:40:" + std::to_string(seed), fg.ts);
      for (Block b : kV1Blocks) {
        const BlockParams p{fg.window, bins};
        const auto full = batch_oracle_rows(o, b, p, 10, all_of(m));
        INFO("seed " << seed << " " << block_name(b) << " batches " << l.batches());
        omp_set_num_threads(1);
        CHECK(batch_cpu_rows(idx, b, p, 10, fg.window, l, all_of(m)) == full);
        omp_set_num_threads(threads);
        CHECK(batch_cpu_rows(idx, b, p, 10, fg.window, l, all_of(m)) == full);
        for (const TriggerSel& sel : {range, sample}) {
          std::vector<std::uint32_t> want;
          for (std::uint64_t r = 0; r < sel.rows(); ++r)
            want.insert(want.end(), full.begin() + sel.eid(r) * bins.size(), full.begin() + (sel.eid(r) + 1) * bins.size());
          CHECK(batch_cpu_rows(idx, b, p, 10, fg.window, l, sel) == want);
          CHECK(batch_oracle_rows(o, b, p, 10, sel) == want);
        }
      }
    }
  }
  omp_set_num_threads(threads);
}

// k_max caps tcyc lengths at run time in both implementations.
TEST_CASE("batch: tcyc respects k_max") {
  const Small g = from_triples(nlohmann::json::parse(
      "[[30,31,1],[31,32,2],[32,33,3],[33,34,4],[34,35,5],[35,30,6]]"));  // one 6-cycle (A26)
  const BatchLayout l = batch_layout(6, 6, {});
  const BatchOracle o(g.view(), l, 100);
  const GraphIndex idx(g.src, g.dst, g.ts, g.n);
  const BlockParams p{100, {2, 3, 4, 5, 6, 7}};
  const std::vector<std::uint32_t> none(36, 0);
  CHECK(batch_oracle_rows(o, Block::tcyc, p, 5, all_of(6)) == none);
  CHECK(batch_cpu_rows(idx, Block::tcyc, p, 5, 100, l, all_of(6)) == none);
  std::vector<std::uint32_t> every(36, 0);
  for (int e = 0; e < 6; ++e) every[std::size_t(e) * 6 + 4] = 1;  // each edge of the cycle, size 6
  CHECK(batch_oracle_rows(o, Block::tcyc, p, 6, all_of(6)) == every);
  CHECK(batch_cpu_rows(idx, Block::tcyc, p, 6, 100, l, all_of(6)) == every);
}

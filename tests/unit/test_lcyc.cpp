// lcyc (SPEC v2 §8 S3): the oracle, the generated CPU code and the CUDA kernels against the GFP
// lc-cycle probes A46-A49 (tests/golden/gfp_lcyc_probes.json), and the work limit's error. The
// fuzz-graph parity, with the work limit, is in test_cpu_backend.cpp and test_cuda_backend.cpp.
#include <catch2/catch_test_macros.hpp>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "graph/index.hpp"
#include "oracle/oracle.hpp"
#include "runtime/cpu_backend.hpp"
#include "runtime/cuda_backend.hpp"

using namespace blz;

namespace {

struct Small {
  std::vector<std::uint32_t> src, dst;
  std::vector<std::int32_t> ts;
  std::uint32_t n = 0;
};

Small from_json(const nlohmann::json& edges) {
  Small g;
  for (const auto& e : edges) {
    g.src.push_back(e[0]);
    g.dst.push_back(e[1]);
    g.ts.push_back(e[2]);
    g.n = std::max(g.n, std::max(g.src.back(), g.dst.back()) + 1);
  }
  return g;
}

// Rows of every backend on g: the oracle, the CPU (both schedules), CUDA (every threshold, if built).
std::vector<std::pair<std::string, std::vector<std::uint32_t>>> all_rows(const Small& g, const BlockParams& p) {
  const std::uint64_t m = g.src.size();
  std::vector<std::pair<std::string, std::vector<std::uint32_t>>> out;
  std::vector<std::uint32_t> rows(m * p.bins.size(), 0);
  Oracle({g.src, g.dst, g.ts, g.n}).mine(Block::lcyc, p, compiled_k_max(), 0, m, rows.data());
  out.emplace_back("oracle", rows);
  const GraphIndex idx(g.src, g.dst, g.ts, g.n);
  for (Schedule s : {Schedule::dynamic, Schedule::sorted}) {
    std::fill(rows.begin(), rows.end(), 0);
    mine_cpu(idx.view(), Block::lcyc, p, 0, m, s, rows.data());
    out.emplace_back(std::string("cpu ") + schedule_name(s), rows);
  }
  if (cuda_built()) {
    CudaMiner miner;
    const DeviceGraph dg(idx.view());
    for (std::uint32_t h : {0u, 2u, kAllLight}) {
      miner.mine(dg, Block::lcyc, p, 0, m, h, rows.data());
      out.emplace_back("cuda H=" + std::to_string(h), rows);
    }
  }
  return out;
}

}  // namespace

// A46-A49: blz's lcyc gives the S3 rule's rows ("blz"), which are GFP's ("output") on every rule
// case; the tie cases are where GFP's rank rule uses an edge tied with the trigger (D-38).
TEST_CASE("lcyc matches the GFP lc-cycle probes on every backend (ties as D-38)") {
  std::ifstream in(std::string(BLZ_SOURCE_DIR) + "/tests/golden/gfp_lcyc_probes.json");
  REQUIRE(in);
  const auto doc = nlohmann::json::parse(in);
  REQUIRE(doc["lc_len"].get<int>() == compiled_k_max());
  const auto bins = doc["bins"].get<std::vector<std::uint32_t>>();
  int ties = 0;
  REQUIRE(doc["cases"].size() >= 16);
  for (const auto& c : doc["cases"]) {
    const Small g = from_json(c["edges"]);
    const auto want = c["blz"].get<std::vector<std::vector<std::uint32_t>>>();
    if (c["kind"] == "rule") CHECK(c["output"] == c["blz"]);
    ties += c["kind"] == "tie";
    for (const auto& [backend, rows] : all_rows(g, {c["window"].get<std::int32_t>(), bins})) {
      for (std::size_t e = 0; e < want.size(); ++e) {
        INFO(c["id"].get<std::string>() << " " << backend << " edge " << e);
        CHECK(std::vector<std::uint32_t>(rows.begin() + std::ptrdiff_t(e * bins.size()),
                                         rows.begin() + std::ptrdiff_t((e + 1) * bins.size())) == want[e]);
      }
    }
  }
  CHECK(ties == 3);
}

// The work limit: 10 -> 11 @1, 11 -> 12 @2, 12 -> 10 @3. Only trigger 2 binds anything (C1 = 11,
// one binding), so a limit of 0 stops every backend at eid 2, and a limit of 1 gives its 3-cycle.
TEST_CASE("lcyc: a trigger over the work limit stops the run with its eid, on every backend") {
  const Small g = from_json(nlohmann::json::parse("[[10,11,1],[11,12,2],[12,10,3]]"));
  const std::vector<std::uint32_t> bins = {2, 3, 4};
  BlockParams p{100, bins};
  p.work_limit = 1;
  for (const auto& [backend, rows] : all_rows(g, p)) {
    INFO(backend);
    CHECK(rows == std::vector<std::uint32_t>{0, 0, 0, 0, 0, 0, 0, 1, 0});
  }
  p.work_limit = 0;
  const std::uint64_t m = g.src.size();
  std::vector<std::uint32_t> rows(m * bins.size(), 0);
  const auto stops_at_2 = [&](auto&& run, const std::string& backend) {
    INFO(backend);
    try {
      run();
      FAIL("no WorkLimitExceeded");
    } catch (const WorkLimitExceeded& e) {
      CHECK(e.eid == 2);
      CHECK(e.limit == 0);
      CHECK(std::string(e.what()).find("trigger eid 2 exceeded the work limit of 0 bindings") != std::string::npos);
    }
  };
  stops_at_2([&] { Oracle({g.src, g.dst, g.ts, g.n}).mine(Block::lcyc, p, compiled_k_max(), 0, m, rows.data()); },
             "oracle");
  const GraphIndex idx(g.src, g.dst, g.ts, g.n);
  for (Schedule s : {Schedule::dynamic, Schedule::sorted})
    stops_at_2([&] { mine_cpu(idx.view(), Block::lcyc, p, 0, m, s, rows.data()); }, schedule_name(s));
  if (cuda_built()) {
    CudaMiner miner;
    const DeviceGraph dg(idx.view());
    for (std::uint32_t h : {0u, 2u, kAllLight})
      stops_at_2([&] { miner.mine(dg, Block::lcyc, p, 0, m, h, rows.data()); }, "cuda H=" + std::to_string(h));
  }
}

// Ground-truth coverage (SPEC v2 §5, E9): isolated attempt graphs, eligibility, activation,
// detection and the aggregates, on a hand-checked stream.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <string>
#include <vector>

#include "oracle/oracle.hpp"
#include "runtime/coverage.hpp"
#include "runtime/profile.hpp"

using namespace blz;
using Catch::Matchers::WithinAbs;

namespace {

// eid: src -> dst @ ts, label. Attempt 0 (FAN-OUT): hub 0 pays 1, 2, 3 (eids 1, 3, 5). Attempt 1
// (CYCLE): 4 -> 5 -> 6 -> 4 (eids 2, 4, 6). eid 0 sends to 1 before the hub does, so fan_in fires
// on eid 1 in the full stream but not in isolation. eid 7 is an unlisted positive.
const std::vector<std::uint32_t> kSrc = {7, 0, 4, 0, 5, 0, 6, 8, 9, 1};
const std::vector<std::uint32_t> kDst = {1, 1, 5, 2, 6, 3, 4, 9, 8, 0};
const std::vector<std::int32_t> kTs = {50, 100, 150, 200, 250, 300, 350, 400, 450, 500};
const std::vector<std::uint8_t> kLabel = {0, 1, 1, 1, 1, 1, 1, 1, 0, 0};
constexpr std::uint64_t kBfv = 3, kBtest = 5;  // fit [0, 3), val [3, 5), test [5, 10)

nlohmann::json gt() {
  return nlohmann::json::parse(R"({"attempts": [
    {"id": 0, "typology": "FAN-OUT", "header": "Max 3-degree Fan-Out", "status": "complete", "split": "straddles",
     "span": 200, "rows": [{"status": "unique", "eid": 5}, {"status": "unique", "eid": 1}, {"status": "unique", "eid": 3}]},
    {"id": 1, "typology": "CYCLE", "header": "Max 3 hops", "status": "complete", "split": "straddles", "span": 200,
     "rows": [{"status": "unique", "eid": 2}, {"status": "resolved-duplicate", "eid": 4}, {"status": "unique", "eid": 6}]},
    {"id": 2, "typology": "CYCLE", "header": "Max 2 hops", "status": "unmatched", "split": null, "span": null,
     "rows": [{"status": "unmatched"}]}]})");
}

const Profile& v1_default() {
  static const Profile p = load_profile(std::string(BLZ_SOURCE_DIR) + "/configs/profiles/v1_default.yaml");
  return p;
}

// The full mining under v1_default, by the oracle.
std::vector<std::vector<std::uint32_t>> full_mining(const Profile& prof) {
  const Oracle o(EdgeStream{kSrc, kDst, kTs, 10});
  std::vector<std::vector<std::uint32_t>> out;
  for (const ProfileBlock& pb : prof.blocks) {
    out.emplace_back(kSrc.size() * pb.params.bins.size(), 0);
    o.mine(pb.block, pb.params, prof.k_max, 0, kSrc.size(), out.back().data());
  }
  return out;
}

CoverageInputs inputs(const std::vector<std::vector<std::uint32_t>>& blocks) {
  CoverageInputs in;
  in.graph = {kSrc, kDst, kTs, kLabel};
  in.b_fv = kBfv;
  in.b_test = kBtest;
  in.profile = &v1_default();
  in.attempts = coverage_attempts(gt(), kSrc.size());
  for (const auto& b : blocks) in.blocks.push_back(b);
  return in;
}

std::vector<std::uint64_t> u64s(const nlohmann::ordered_json& j) { return j.get<std::vector<std::uint64_t>>(); }

}  // namespace

TEST_CASE("coverage: percentile_linear is numpy's linear interpolation") {
  CHECK_THAT(percentile_linear({4, 1, 3, 2}, 50), WithinAbs(2.5, 1e-12));
  CHECK_THAT(percentile_linear({4, 1, 3, 2}, 90), WithinAbs(3.7, 1e-12));
  CHECK_THAT(percentile_linear({4, 1, 3, 2}, 10), WithinAbs(1.3, 1e-12));
  CHECK_THAT(percentile_linear({7}, 90), WithinAbs(7.0, 1e-12));
  CHECK(std::isnan(percentile_linear({}, 50)));
}

TEST_CASE("coverage: attempts and isolated graphs") {
  const auto att = coverage_attempts(gt(), kSrc.size());
  REQUIRE(att.size() == 3);
  CHECK(att[0].eids == std::vector<std::uint32_t>{1, 3, 5});  // sorted: stream order
  CHECK(att[1].eids == std::vector<std::uint32_t>{2, 4, 6});  // resolved-duplicate rows count
  CHECK(att[2].eids.empty());
  CHECK(att[2].split == "none");
  CHECK_THROWS(coverage_attempts(gt(), 5));  // eid 5 >= m

  const CoverageGraph g{kSrc, kDst, kTs, kLabel};
  const IsolatedGraph ig = isolated_graph(g, att[1].eids);
  CHECK(ig.n == 3);  // 4, 5, 6 renumbered by first appearance
  CHECK(ig.src == std::vector<std::uint32_t>{0, 1, 2});
  CHECK(ig.dst == std::vector<std::uint32_t>{1, 2, 0});
  CHECK(ig.ts == std::vector<std::int32_t>{150, 250, 350});
  CHECK_THROWS(isolated_graph(g, std::vector<std::uint32_t>{3, 1}));

  // v1_default's blocks: fan_in fan_out deg_in deg_out sg tcyc.
  CHECK(oracle_eligible_rows(isolated_graph(g, att[0].eids), v1_default()) ==
        std::vector<std::uint64_t>{0, 2, 0, 2, 0, 0});
  CHECK(oracle_eligible_rows(ig, v1_default()) == std::vector<std::uint64_t>{0, 0, 0, 0, 0, 1});
}

TEST_CASE("coverage: eligibility, activation, detection and positives") {
  const auto blocks = full_mining(v1_default());
  CoverageInputs in = inputs(blocks);
  //                                    fit             val     test
  const std::vector<std::uint8_t> pred = {255, 255, 255, 1, 0, 1, 1, 0, 1, 0};
  in.pred = pred;
  const auto j = coverage(in);

  CHECK(j["blocks"] == nlohmann::ordered_json({"fan_in", "fan_out", "deg_in", "deg_out", "sg", "tcyc"}));
  CHECK(j["eligibility"] == "oracle");
  const auto& a0 = j["attempts"][0];
  CHECK(u64s(a0["eligible"]) == std::vector<std::uint64_t>{0, 2, 0, 2, 0, 0});
  CHECK(u64s(a0["activated"]) == std::vector<std::uint64_t>{1, 2, 1, 2, 0, 0});  // eid 0's edge into 1
  CHECK(a0["edges"] == nlohmann::ordered_json({{"fit", 1}, {"val", 1}, {"test", 1}}));
  CHECK(a0["predicted"] == nlohmann::ordered_json({{"val", 1}, {"test", 1}}));
  CHECK(u64s(j["attempts"][1]["activated"]) == std::vector<std::uint64_t>{0, 0, 0, 0, 0, 1});
  CHECK(!j["attempts"][2].contains("eligible"));  // not complete

  const auto& all = j["by_typology"]["ALL"];
  CHECK(all["attempts"] == 3);
  CHECK(all["complete"] == 2);
  CHECK(all["status"]["unmatched"] == 1);
  CHECK(all["split"]["straddles"] == 2);
  CHECK(all["split"]["none"] == 1);
  CHECK(all["eligible"]["fan_in"] == 0);
  CHECK(all["activated"]["fan_in"] == 1);
  CHECK(all["eligible"]["any"] == 2);
  CHECK(all["eligible_not_activated"]["any"] == 0);
  CHECK_THAT(all["span_hours"]["p50"].get<double>(), WithinAbs(200.0 / 3600, 1e-12));
  CHECK(all["det_val"]["attempts"] == 2);
  CHECK(all["det_val"]["predicted"] == 1);
  CHECK_THAT(all["det_val"]["mean"].get<double>(), WithinAbs(0.5, 1e-12));
  CHECK_THAT(all["det_test"]["pooled"].get<double>(), WithinAbs(1.0, 1e-12));
  CHECK_THAT(j["by_typology"]["FAN-OUT"]["det_val"]["mean"].get<double>(), WithinAbs(1.0, 1e-12));
  CHECK_THAT(j["by_typology"]["CYCLE"]["det_val"]["mean"].get<double>(), WithinAbs(0.0, 1e-12));
  CHECK(j["by_typology"].back() == all);  // "ALL" comes last

  const auto& p = j["positives"];
  CHECK(p["listed"] == nlohmann::ordered_json({{"fit", 2}, {"val", 2}, {"test", 2}, {"all", 6}}));
  CHECK(p["unlisted"] == nlohmann::ordered_json({{"fit", 0}, {"val", 0}, {"test", 1}, {"all", 1}}));
  // eids 2 (fit) and 4 (val) are cycle edges no block fires on until the cycle closes; eid 7 too.
  CHECK(p["no_block_activates"]["listed"] == nlohmann::ordered_json({{"fit", 1}, {"val", 1}, {"test", 0}, {"all", 2}}));
  CHECK(p["no_block_activates"]["unlisted"]["test"] == 1);
  CHECK(p["predicted"]["val"]["listed"] == nlohmann::ordered_json({{"positives", 2}, {"predicted", 1}}));
  CHECK(p["predicted"]["test"]["unlisted"] == nlohmann::ordered_json({{"positives", 1}, {"predicted", 0}}));
  CHECK(j["base_rates"]["nonzero"]["fan_in"]["edges"]["all"] == 1);
  CHECK(j["base_rates"]["nonzero"]["tcyc"]["positives"]["test"] == 1);
  CHECK(j["base_rates"]["edges"]["all"] == 10);

  const std::string text = coverage_dump(j);
  CHECK(nlohmann::ordered_json::parse(text) == j);
}

TEST_CASE("coverage: input checks") {
  const auto blocks = full_mining(v1_default());
  CoverageInputs in = inputs(blocks);
  const auto j = coverage(in);
  CHECK(j["by_typology"]["ALL"]["det_val"]["mean"].is_null());  // no --pred

  std::vector<std::uint8_t> pred(10, 0);  // scores fit rows
  in.pred = pred;
  CHECK_THROWS(coverage(in));
  const std::vector<std::uint8_t> part = {255, 255, 255, 1, 255, 0, 0, 0, 0, 0};  // scores part of val
  in.pred = part;
  CHECK_THROWS(coverage(in));
  const std::vector<std::uint8_t> test_only = {255, 255, 255, 255, 255, 0, 0, 0, 0, 1};  // det_val is null
  in.pred = test_only;
  const auto k = coverage(in);
  CHECK(k["scored"] == nlohmann::ordered_json({{"val", false}, {"test", true}}));
  CHECK(k["by_typology"]["ALL"]["det_val"]["mean"].is_null());
  CHECK(k["by_typology"]["ALL"]["det_val"]["attempts"] == 2);
  CHECK(k["attempts"][0]["predicted"]["val"].is_null());
}

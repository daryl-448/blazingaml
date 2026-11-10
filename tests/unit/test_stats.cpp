#include <catch2/catch_test_macros.hpp>
#include <vector>

#include "graph/stats.hpp"

using namespace blz;

TEST_CASE("summarize") {
  std::vector<std::uint64_t> v = {0, 1, 2, 3, 4, 8};
  auto j = summarize(v);
  CHECK(j["count"] == 6);
  CHECK(j["min"] == 0);
  CHECK(j["max"] == 8);
  CHECK(j["p50"] == 3);
  CHECK(j["log2_hist"]["0"] == 1);
  CHECK(j["log2_hist"]["1"] == 1);
  CHECK(j["log2_hist"]["2-3"] == 2);
  CHECK(j["log2_hist"]["4-7"] == 1);
  CHECK(j["log2_hist"]["8-15"] == 1);
  CHECK(j["log2_hist"].size() == 5);
  CHECK(summarize(std::vector<std::uint64_t>{})["count"] == 0);
}

TEST_CASE("profile_stream on a small stream") {
  // eid: 0:0->1@0  1:0->1@10  2:0->2@10  3:2->2@20  4:0->1@100  (n = 3)
  std::vector<std::uint32_t> src = {0, 0, 0, 2, 0}, dst = {1, 1, 2, 2, 1};
  std::vector<std::int32_t> ts = {0, 10, 10, 20, 100};
  std::vector<std::uint8_t> label = {0, 1, 0, 0, 1};
  auto p = profile_stream({src, dst, ts, label, 3});
  CHECK(p["graph"]["edges"] == 5);
  CHECK(p["graph"]["self_loops"] == 1);
  CHECK(p["graph"]["train_edges"] == 4);
  CHECK(p["graph"]["test_positives"] == 1);
  CHECK(p["degree"]["out_edges"]["max"] == 4);
  CHECK(p["degree"]["out_distinct_neighbours"]["max"] == 2);
  CHECK(p["pairs"]["distinct_pairs"] == 3);
  CHECK(p["pairs"]["self_loop_pairs"] == 1);
  CHECK(p["pairs"]["multiplicity"]["max"] == 3);
  CHECK(p["ts_ties"]["distinct_timestamps"] == 4);
  CHECK(p["ts_ties"]["edges_tied_with_previous"] == 1);
  CHECK(p["self_loops"]["nodes_with_self_loop"] == 1);
  CHECK(p["self_loops"]["non_loop_edges_dst_has_self_loop"] == 1);
  CHECK(p["self_loops"]["non_loop_edges_src_has_self_loop"] == 0);
  // delta = 21600 covers the whole stream: node 0's out-list occupancies are 1, 2, 3, 4.
  CHECK(p["window_occupancy"]["21600"]["trigger_src_out_edges"]["max"] == 4);
}

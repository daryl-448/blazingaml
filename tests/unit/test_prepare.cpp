#include <catch2/catch_test_macros.hpp>

#include "fixtures.hpp"
#include "graph/prepare.hpp"

using namespace blz;

TEST_CASE("build_edge_table: stable sort, eids, node ids, dictionaries") {
  RawTable raw = parse_csv(kFixtureCsv);
  EdgeTable t = build_edge_table(raw);
  REQUIRE(t.m() == 4);
  // Stream order: ts-sorted, ties keep file order -> rows r1, r3, r0, r2.
  CHECK(t.ts == std::vector<std::int32_t>{0, 0, 1200, 1200});
  CHECK(t.label == std::vector<std::uint8_t>{1, 0, 0, 0});
  CHECK(t.amount_paid == std::vector<double>{1.25, 2000, 10.5, 3});
  // Node ids by first appearance in stream order, source before destination:
  // e0 (r1) 10/A -> 010/A gives 0, 1; e1 (r3) 001/C -> 010/A gives 2; e2 (r0) 010/A -> 001/B
  // gives 3. "10" and "010" are different banks.
  CHECK(t.src == std::vector<std::uint32_t>{0, 2, 1, 3});
  CHECK(t.dst == std::vector<std::uint32_t>{1, 1, 3, 3});
  REQUIRE(t.n() == 4);
  CHECK(t.node_bank == std::vector<std::string>{"10", "010", "001", "001"});
  CHECK(t.node_account == std::vector<std::string>{"A", "A", "C", "B"});
  // One sorted currency dictionary shared by pay and recv; a sorted format dictionary.
  CHECK(t.ccy_dict == std::vector<std::string>{"Bitcoin", "Euro", "US Dollar"});
  CHECK(t.recv_ccy == std::vector<std::uint8_t>{1, 1, 2, 0});
  CHECK(t.pay_ccy == std::vector<std::uint8_t>{2, 1, 2, 0});
  CHECK(t.format_dict == std::vector<std::string>{"ACH", "Cheque", "Reinvestment", "Wire"});
  CHECK(t.format == std::vector<std::uint8_t>{0, 3, 1, 2});
}

TEST_CASE("count_stream and the HI-Small invariants") {
  EdgeTable t = build_edge_table(parse_csv(kFixtureCsv));
  StreamCounts c = count_stream(t);
  CHECK(c.edges == 4);
  CHECK(c.nodes == 4);
  CHECK(c.self_loops == 1);
  CHECK(c.positives == 1);
  CHECK(c.train_positives == 1);  // train = eid < floor(0.8 * 4) = 3
  CHECK(c.test_positives == 0);
  CHECK(train_size(5'078'345) == 4'062'676);
  // The fixture is not HI-Small: every count check fails, and the sort check passes.
  CHECK(check_hi_small_invariants(t).size() == 7);
  t.ts = {5, 0, 1200, 1200};
  CHECK(check_hi_small_invariants(t).size() == 8);
}

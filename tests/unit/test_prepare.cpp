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
  // SPEC v2 §3.5: each edge's 0-based data row in the CSV.
  CHECK(t.file_row == std::vector<std::uint32_t>{1, 3, 0, 2});
  CHECK(check_file_row(t.file_row, t.ts).empty());
}

TEST_CASE("check_file_row: a permutation that reproduces the stable sort (SPEC v2 §3.5)") {
  const std::vector<std::int32_t> ts = {0, 0, 5, 5, 9};
  CHECK(check_file_row(std::vector<std::uint32_t>{1, 3, 0, 4, 2}, ts).empty());
  CHECK_FALSE(check_file_row(std::vector<std::uint32_t>{3, 1, 0, 4, 2}, ts).empty());  // a tie out of file order
  CHECK_FALSE(check_file_row(std::vector<std::uint32_t>{1, 1, 0, 4, 2}, ts).empty());  // not a permutation
  CHECK_FALSE(check_file_row(std::vector<std::uint32_t>{1, 3, 0, 4, 5}, ts).empty());  // out of range
  CHECK_FALSE(check_file_row(std::vector<std::uint32_t>{1, 3, 0, 4}, ts).empty());     // wrong length
  const std::vector<std::int32_t> unsorted = {0, 5, 0};
  CHECK_FALSE(check_file_row(std::vector<std::uint32_t>{0, 1, 2}, unsorted).empty());
}

TEST_CASE("fit_val_boundary: the first eid >= floor(0.6 m) that starts a timestamp (SPEC v2 §3.3)") {
  // m = 10, floor(0.6 m) = 6.
  CHECK(fit_val_boundary(std::vector<std::int32_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) == 6);
  CHECK(fit_val_boundary(std::vector<std::int32_t>{0, 1, 2, 3, 4, 5, 5, 5, 8, 9}) == 8);  // 6, 7 continue ts 5
  CHECK(fit_val_boundary(std::vector<std::int32_t>{0, 1, 2, 3, 4, 5, 5, 5, 5, 5}) == 10);  // none: m
  CHECK(fit_val_boundary(std::vector<std::int32_t>{3}) == 0);                             // eid 0 starts one
  CHECK(fit_val_boundary(std::vector<std::int32_t>{}) == 0);
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
  // SPEC v2 §3.3: floor(0.6 * 4) = 2 starts ts 1200, so fit = {0, 1}, validation = {2}.
  CHECK(c.fit_val_boundary == 2);
  CHECK(c.fit_positives == 1);
  CHECK(c.val_positives == 0);
  CHECK(c.concatenated_keys == 4);  // "10A", "010A", "001C", "001B"
  // The fixture is not HI-Small: every count check fails (7 of v1, 5 of v2), and the sort and
  // file_row checks pass.
  CHECK(check_hi_small_invariants(t).size() == 12);
  t.ts = {5, 0, 1200, 1200};
  CHECK(check_hi_small_invariants(t).size() == 14);  // + the sort check, + file_row's
}

TEST_CASE("concatenated node keys count the (bank, account) pairs that collide (SPEC v2 §4.4 R1)") {
  const std::string csv = std::string(kCsvHeader) + "\n"
      "2022/09/01 00:00,01,2A,0,12A,1,Euro,1,Euro,ACH,0\n"   // "01" + "2A" == "0" + "12A"
      "2022/09/01 00:00,01,2A,02,X,1,Euro,1,Euro,ACH,0\n";
  const EdgeTable t = build_edge_table(parse_csv(csv));
  const StreamCounts c = count_stream(t);
  CHECK(c.nodes == 3);
  CHECK(c.concatenated_keys == 2);
}

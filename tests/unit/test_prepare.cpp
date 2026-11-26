#include <catch2/catch_test_macros.hpp>

#include "fixtures.hpp"
#include <algorithm>

#include "graph/prepare.hpp"

using namespace blz;

TEST_CASE("build_edge_table: stable sort, eids, node ids, dictionaries") {
  EdgeTable t = prepare_text(kFixtureCsv);
  REQUIRE(t.m() == 4);
  // SPEC v3 §6.1: the epoch is midnight of the earliest row's day, 2022-09-01 (D-1's).
  CHECK(t.epoch == 1'661'990'400);
  CHECK(t.csv_sha256 == sha256(kFixtureCsv.data(), kFixtureCsv.size()));
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

TEST_CASE("compute_splits: legacy_hi_small and temporal_60_20_20 (SPEC v3 §6.3)") {
  using R = SplitRule;
  // m = 10, floor(0.6 m) = 6, floor(0.8 m) = 8.
  const std::vector<std::int32_t> distinct = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  CHECK(compute_splits(R::legacy_hi_small, distinct) == Splits{6, 8});
  CHECK(compute_splits(R::temporal_60_20_20, distinct) == Splits{6, 8});
  // A tie at each cut: eids 5, 6 share ts 5 and eids 7, 8 share ts 7. Both rules move b_fv to
  // eid 7; only the temporal rule moves b_test, to eid 9.
  const std::vector<std::int32_t> ties = {0, 1, 2, 3, 4, 5, 5, 7, 7, 9};
  CHECK(compute_splits(R::legacy_hi_small, ties) == Splits{7, 8});
  CHECK(compute_splits(R::temporal_60_20_20, ties) == Splits{7, 9});
  // No timestamp starts at or after floor(0.6 m): the temporal splits are empty past fit, and the
  // legacy b_fv stops at its b_test.
  const std::vector<std::int32_t> tail = {0, 1, 2, 3, 4, 5, 5, 5, 5, 5};
  CHECK(compute_splits(R::temporal_60_20_20, tail) == Splits{10, 10});
  CHECK(compute_splits(R::legacy_hi_small, tail) == Splits{8, 8});
  // Eid 0 starts a timestamp.
  CHECK(compute_splits(R::temporal_60_20_20, std::vector<std::int32_t>{3}) == Splits{0, 0});
  CHECK(compute_splits(R::legacy_hi_small, std::vector<std::int32_t>{}) == Splits{0, 0});
  CHECK(split_rule_name(R::temporal_60_20_20) == "temporal_60_20_20");
  CHECK(parse_split_rule("legacy_hi_small") == R::legacy_hi_small);
  CHECK_FALSE(parse_split_rule("temporal").has_value());
}

TEST_CASE("build_edge_table: the same stream through every read buffer size") {
  const EdgeTable want = prepare_text(kFixtureCsv);
  for (std::size_t buffer = 1; buffer <= 40; ++buffer) {
    INFO("buffer " << buffer);
    const EdgeTable t = prepare_text(kFixtureCsv, kIbmSchema, buffer);
    CHECK(t.src == want.src);
    CHECK(t.dst == want.dst);
    CHECK(t.ts == want.ts);
    CHECK(t.file_row == want.file_row);
    CHECK(t.node_bank == want.node_bank);
    CHECK(t.node_account == want.node_account);
    CHECK(t.ccy_dict == want.ccy_dict);
    CHECK(t.recv_ccy == want.recv_ccy);
    CHECK(t.csv_sha256 == want.csv_sha256);
  }
}

TEST_CASE("build_edge_table: the epoch and the int32 range (SPEC v3 §6.1)") {
  const std::string ibm(kIbmSchema.header), tx(kTransxionSchema.header);
  const std::string row = ",1,A,2,B,1,Euro,1,Euro,ACH,0\n";
  // The earliest row's day at midnight, wherever that row is in the file.
  EdgeTable t = prepare_text(ibm + "\n2022/09/04 01:00" + row + "2022/09/03 05:00" + row);
  CHECK(t.epoch == 1'661'990'400 + 2 * 86400);
  CHECK(t.ts == std::vector<std::int32_t>{5 * 3600, 86400 + 3600});
  CHECK(t.file_row == std::vector<std::uint32_t>{1, 0});
  t = prepare_text(tx + "\n2025-01-01 00:00:29" + row, kTransxionSchema);
  CHECK(t.epoch == 1'735'689'600);
  CHECK(t.ts == std::vector<std::int32_t>{29});
  // 2^31 - 1 seconds after the epoch fits; one more does not.
  t = prepare_text(tx + "\n1970-01-01 00:00:00" + row + "2038-01-19 03:14:07" + row, kTransxionSchema);
  CHECK(t.ts.back() == INT32_MAX);
  CHECK_THROWS_AS(prepare_text(tx + "\n1970-01-01 00:00:00" + row + "2038-01-19 03:14:08" + row, kTransxionSchema),
                  std::runtime_error);
  CHECK_THROWS_AS(prepare_text(ibm + "\n2022/09/01 00:00" + row + "2100/01/01 00:00" + row), std::runtime_error);
  // Before 1970: the epoch is still the earliest day's midnight.
  t = prepare_text(ibm + "\n1969/12/31 23:59" + row);
  CHECK(t.epoch == -86400);
  CHECK(t.ts == std::vector<std::int32_t>{86340});
  CHECK(prepare_text(ibm + "\n").m() == 0);
}

TEST_CASE("count_invariants and check_invariants (SPEC v3 §6.1)") {
  const EdgeTable t = prepare_text(kFixtureCsv);
  // legacy: floor(0.6 * 4) = 2 starts ts 1200, so fit = {0, 1}, validation = {2}, test = {3}.
  const Splits s = compute_splits(SplitRule::legacy_hi_small, t.ts);
  REQUIRE(s == Splits{2, 3});
  const Invariants got = count_invariants(t, s, true);
  const Invariants want = {{"edges", 4},         {"nodes", 4},         {"self_loops", 1},     {"distinct_timestamps", 2},
                           {"positives", 1},     {"b_fv", 2},          {"b_test", 3},         {"fit_positives", 1},
                           {"val_positives", 0}, {"test_positives", 0}, {"concatenated_keys", 4}};  // "10A", "010A", ...
  CHECK(got == want);
  CHECK(count_invariants(t, s, false).count("concatenated_keys") == 0);
  for (const auto& [k, v] : got)
    CHECK(std::find(kInvariantNames.begin(), kInvariantNames.end(), k) != kInvariantNames.end());

  CHECK(check_invariants(got, want).empty());
  Invariants wrong = want;
  wrong["self_loops"] = 2;
  CHECK(check_invariants(got, wrong) == std::vector<std::string>{"self_loops: counted 1, pinned 2"});
  Invariants missing = want;
  missing.erase("nodes");
  CHECK(check_invariants(got, missing) == std::vector<std::string>{"nodes: counted 4, not pinned"});
  Invariants extra = want;
  extra["unresolved_accounts"] = 0;
  CHECK(check_invariants(got, extra) ==
        std::vector<std::string>{"unresolved_accounts: pinned 0, not counted for this dataset"});
}

TEST_CASE("concatenated node keys count the (bank, account) pairs that collide (SPEC v2 §4.4 R1)") {
  const std::string csv = std::string(kIbmSchema.header) + "\n"
      "2022/09/01 00:00,01,2A,0,12A,1,Euro,1,Euro,ACH,0\n"   // "01" + "2A" == "0" + "12A"
      "2022/09/01 00:00,01,2A,02,X,1,Euro,1,Euro,ACH,0\n";
  const EdgeTable t = prepare_text(csv);
  const Invariants c = count_invariants(t, compute_splits(SplitRule::legacy_hi_small, t.ts), true);
  CHECK(c.at("nodes") == 3);
  CHECK(c.at("concatenated_keys") == 2);
}

TEST_CASE("resolve_node_tables: accounts no row names, rows that name no account (SPEC v3 §6.1)") {
  const std::string csv = std::string(kTransxionSchema.header) + "\n"
      "2025-01-01 00:00:00,1,A,2,B,1,CNY,1,CNY,Card,0\n"
      "2025-01-01 00:00:01,2,B,3,C,1,CNY,1,CNY,Card,0\n";
  const EdgeTable t = prepare_text(csv, kTransxionSchema);
  const std::vector<std::pair<std::string, std::string>> rows = {{"1", "A"}, {"3", "C"}, {"9", "Z"}, {"2", "A"}};
  const NodeTableCounts c = resolve_node_tables(t, rows);
  CHECK(c.unresolved_accounts == 1);
  CHECK(c.unresolved_examples == std::vector<std::string>{"2,B"});
  CHECK(c.unmatched_node_rows == 2);
  CHECK(c.unmatched_examples == std::vector<std::string>{"9,Z", "2,A"});
  CHECK(resolve_node_tables(t, rows, 1).unmatched_examples.size() == 1);
}

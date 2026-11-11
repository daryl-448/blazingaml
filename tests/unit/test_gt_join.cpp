// The ground-truth join (SPEC v2 §3.5) on a fixture with every row status and attempt status.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <string>

#include "graph/gt_join.hpp"
#include "graph/prepare.hpp"

using namespace blz;

namespace {

// Data rows in file order; the stream order is r1 r3 r4 r5 r0 r2 r6 r7 (eids 0..7).
const std::string kRows[] = {
    "2022/09/01 00:10,001,A,002,B,100.00,US Dollar,100.00,US Dollar,ACH,1",  // r0 ts 600, eid 4
    "2022/09/01 00:00,003,C,004,D,5.00,Euro,5.00,Euro,Wire,1",               // r1 ts 0, eid 0
    "2022/09/01 00:20,005,E,006,F,7.00,Euro,7.00,Euro,Cash,0",               // r2 ts 1200, eid 5
    "2022/09/01 00:00,003,C,004,D,5.00,Euro,5.00,Euro,Wire,1",               // r3 = r1, eid 1
    "2022/09/01 00:05,007,G,008,H,9.00,Euro,9.00,Euro,ACH,0",                // r4 ts 300, eid 2
    "2022/09/01 00:05,007,G,008,H,9.00,Euro,9.00,Euro,ACH,0",                // r5 = r4, eid 3
    "2022/09/01 00:30,009,I,010,J,12.5,Euro,12.5,Euro,ACH,1",                // r6 ts 1800, eid 6
    "2022/09/01 00:40,011,K,012,L,1.00,Euro,1.00,Euro,ACH,1",                // r7 ts 2400, eid 7
};

std::string csv(std::size_t nrows = 8) {
  std::string s = "Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,"
                  "Amount Paid,Payment Currency,Payment Format,Is Laundering\n";
  for (std::size_t i = 0; i < nrows; ++i) s += kRows[i] + "\n";
  return s;
}

const std::string kPatterns =
    "BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 2-degree Fan-Out\n"   // line 1
    + kRows[0] + "\n"                                                 // unique, eid 4 (val)
    + kRows[1] + "\r\n"                                               // resolved, first of r1/r3: eid 0 (fit)
    "END LAUNDERING ATTEMPT - FAN-OUT\n"
    "\n"
    "BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 3 hops\n"                 // line 6
    + kRows[1] + "\n"                                                 // resolved: eid 1 (fit)
    "2022/09/01 00:30,009,I,010,J,12.50,Euro,12.50,Euro,ACH,1\n"     // rule fields: eid 6 (test)
    "END LAUNDERING ATTEMPT - CYCLE\n"
    "\n"
    "BEGIN LAUNDERING ATTEMPT - STACK\n"                              // line 11
    + kRows[4] + "\n"                                                 // ambiguous: candidates 2, 3
    + kRows[2] + "\n"                                                 // unique: eid 5 (val)
    "END LAUNDERING ATTEMPT - STACK\n"
    "\n"
    "BEGIN LAUNDERING ATTEMPT - RANDOM:  Random\n"                    // line 16
    "2022/09/01 00:50,099,Z,098,Y,1.00,Euro,1.00,Euro,ACH,1\n"       // unmatched
    "END LAUNDERING ATTEMPT - RANDOM\n";

}  // namespace

TEST_CASE("gt-join: row statuses, rules, attempts, splits and unlisted positives") {
  const std::string text = csv();
  const EdgeTable t = build_edge_table(parse_csv(text));
  REQUIRE(t.file_row == std::vector<std::uint32_t>{1, 3, 4, 5, 0, 2, 6, 7});
  const auto j = gt_join(text, kPatterns, {t.file_row, t.ts, t.label});

  // m = 8: train = [0, 6), floor(0.6 m) = 4 starts ts 600, so fit = [0, 4), val = [4, 6).
  CHECK(j["splits"].dump() == R"({"fit":[0,4],"val":[4,6],"test":[6,8]})");
  const auto& s = j["summary"];
  CHECK(s["attempts"] == 4);
  CHECK(s["listed_rows"] == 7);
  CHECK(s["rows_by_status"].dump() == R"({"ambiguous":1,"resolved-duplicate":2,"unique":3,"unmatched":1})");
  CHECK(s["rows_by_rule"].dump() == R"({"fields":1,"none":1,"text":5})");
  CHECK(s["attempts_by_status"].dump() == R"({"complete":2,"partial":1,"unmatched":1})");
  CHECK(s["positives"] == 5);
  CHECK(s["mapped_eids"] == 5);
  CHECK(s["mapped_eids_with_label_0"] == 1);  // r2
  CHECK(s["unlisted_positives"] == 1);
  CHECK(s["unlisted_positives_that_are_ambiguous_candidates"] == 0);
  CHECK(j["unlisted_positive_eids"].dump() == "[7]");
  CHECK(s["attempts_by_typology"]["CYCLE"].dump() ==
        R"({"attempts":1,"complete":1,"partial":0,"unmatched":0,"rows":2,"fit":0,"val":0,"test":0,"straddles":1})");

  const auto& a = j["attempts"];
  REQUIRE(a.size() == 4);
  CHECK(a[0]["typology"] == "FAN-OUT");
  CHECK(a[0]["header"] == "Max 2-degree Fan-Out");
  CHECK(a[0]["line"] == 1);
  CHECK(a[0]["status"] == "complete");
  CHECK(a[0]["split"] == "straddles");
  CHECK(a[0]["accounts"] == 4);
  CHECK(a[0]["ts_min"] == 0);
  CHECK(a[0]["span"] == 600);
  CHECK(a[0]["rows"].dump() ==
        R"([{"line":2,"status":"unique","rule":"text","eid":4},)"
        R"({"line":3,"status":"resolved-duplicate","rule":"text","eid":0}])");
  CHECK(a[1]["rows"].dump() ==
        R"([{"line":7,"status":"resolved-duplicate","rule":"text","eid":1},)"
        R"({"line":8,"status":"unique","rule":"fields","eid":6}])");
  CHECK(a[2]["header"] == "");
  CHECK(a[2]["status"] == "partial");
  CHECK(a[2]["split"] == "val");
  CHECK(a[2]["rows"][0].dump() == R"({"line":12,"status":"ambiguous","rule":"text","eid":null,"candidates":[2,3]})");
  CHECK(a[3]["status"] == "unmatched");
  CHECK(a[3]["split"].is_null());
  CHECK(a[3]["rows"][0].dump() == R"({"line":17,"status":"unmatched","rule":"none","eid":null})");

  // The committed form is valid JSON with the same content.
  CHECK(nlohmann::ordered_json::parse(gt_join_dump(j)) == j);
}

TEST_CASE("gt-join: a CSV or Patterns.txt that does not fit is an error") {
  const std::string text = csv();
  const EdgeTable t = build_edge_table(parse_csv(text));
  const GtGraph g{t.file_row, t.ts, t.label};
  CHECK_THROWS_WITH(gt_join(csv(7), kPatterns, g),
                    "gt-join: the CSV has 7 data rows, the graph m = 8 (not the CSV the graph was prepared from?)");
  CHECK_THROWS_WITH(gt_join(text + kRows[0] + "\n", kPatterns, g),
                    "gt-join: the CSV has more data rows than the graph's m = 8");
  CHECK_THROWS_WITH(gt_join("x\n" + text, kPatterns, g), "csv line 1: unexpected header");
  CHECK_THROWS_WITH(gt_join(text, "BEGIN LAUNDERING ATTEMPT - CYCLE\nBEGIN LAUNDERING ATTEMPT - CYCLE\n", g),
                    "patterns line 2: BEGIN inside an attempt");
  CHECK_THROWS_WITH(gt_join(text, "BEGIN LAUNDERING ATTEMPT - CYCLE\nEND LAUNDERING ATTEMPT - STACK\n", g),
                    "patterns line 2: END does not match its BEGIN");
  CHECK_THROWS_WITH(gt_join(text, kRows[0] + "\n", g), "patterns line 1: a transaction outside an attempt");
  CHECK_THROWS_WITH(gt_join(text, "BEGIN LAUNDERING ATTEMPT - CYCLE\n1,2,3\n", g),
                    "patterns line 2: expected 11 comma-separated fields");
  CHECK_THROWS_WITH(gt_join(text, "BEGIN LAUNDERING ATTEMPT - CYCLE\n", g), "patterns: the last attempt has no END");
}

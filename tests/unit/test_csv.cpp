#include <catch2/catch_test_macros.hpp>

#include "fixtures.hpp"
#include "io/csv.hpp"

using namespace blz;

TEST_CASE("parse_timestamp") {
  CHECK(parse_timestamp("2022/09/01 00:00") == 0);
  CHECK(parse_timestamp("2022/09/01 00:20") == 1200);
  CHECK(parse_timestamp("2022/09/02 01:01") == 86400 + 3660);
  CHECK(parse_timestamp("2022/09/18 16:18") == kMaxTimestamp);
  CHECK_THROWS_AS(parse_timestamp("2022/09/18 16:19"), std::invalid_argument);  // past range
  CHECK_THROWS_AS(parse_timestamp("2022/08/31 23:59"), std::invalid_argument);  // before epoch
  CHECK_THROWS_AS(parse_timestamp("2022/09/31 00:00"), std::invalid_argument);
  CHECK_THROWS_AS(parse_timestamp("2022/09/01 24:00"), std::invalid_argument);
  CHECK_THROWS_AS(parse_timestamp("2022/09/01 0:00"), std::invalid_argument);
  CHECK_THROWS_AS(parse_timestamp("2022-09-01 00:00"), std::invalid_argument);
  CHECK_THROWS_AS(parse_timestamp("2022/09/0a 00:00"), std::invalid_argument);
}

TEST_CASE("parse_csv fixture") {
  RawTable t = parse_csv(kFixtureCsv);
  REQUIRE(t.size() == 4);
  CHECK(t.ts == std::vector<std::int32_t>{1200, 0, 1200, 0});
  CHECK(t.from_bank[0] == "010");  // leading zeros kept
  CHECK(t.from_bank[1] == "10");
  CHECK(t.to_account[1] == "A");
  CHECK(t.format[1] == "ACH");
  CHECK(t.label == std::vector<std::uint8_t>{0, 1, 0, 0});
  CHECK(t.amount_received[0] == 10.5);
  CHECK(t.amount_paid[1] == 1.25);
  CHECK(t.amount_received[3] == 2000.0);
}

TEST_CASE("parse_csv without a trailing newline") {
  std::string s = kFixtureCsv;
  s.pop_back();
  CHECK(parse_csv(s).size() == 4);
}

TEST_CASE("parse_csv errors carry the line number") {
  const std::string hdr(kCsvHeader);
  auto line_of = [](const std::string& text) {
    try {
      parse_csv(text);
    } catch (const CsvError& e) {
      return e.line;
    }
    return std::size_t(0);
  };
  const std::string good = "2022/09/01 00:00,1,A,2,B,1,Euro,1,Euro,ACH,0\n";
  CHECK(line_of("Timestamp,Bank\n" + good) == 1);
  CHECK(line_of(hdr + "\n" + good + "2022/09/01 00:00,1,A,2,B,1,Euro,1,Euro,ACH\n") == 3);
  CHECK(line_of(hdr + "\n" + good + good + "2022/09/01 00:00,1,A,2,B,1,Euro,1,Euro,ACH,0,9\n") == 4);
  CHECK(line_of(hdr + "\n2022/09/01 00:00,1,A,2,B,x,Euro,1,Euro,ACH,0\n") == 2);
  CHECK(line_of(hdr + "\n2022/09/01 00:00,1,A,2,B,1,Euro,1,Euro,ACH,2\n") == 2);
  CHECK(line_of(hdr + "\n2022/09/01 00:00,,A,2,B,1,Euro,1,Euro,ACH,0\n") == 2);
  CHECK(line_of(hdr + "\n2022/09/01 00:00,\"1\",A,2,B,1,Euro,1,Euro,ACH,0\n") == 2);
  CHECK(line_of(hdr + "\n2022/13/01 00:00,1,A,2,B,1,Euro,1,Euro,ACH,0\n") == 2);
  CHECK(line_of(hdr + "\n" + good + "\n" + good) == 3);  // blank line inside the file
}

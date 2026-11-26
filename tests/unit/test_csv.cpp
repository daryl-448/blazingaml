#include <catch2/catch_test_macros.hpp>
#include <sstream>

#include "fixtures.hpp"
#include "io/csv.hpp"

using namespace blz;

namespace {

constexpr std::int64_t k20220901 = 1'661'990'400;  // 2022-09-01 00:00 in unix seconds
constexpr std::int64_t k20250101 = 1'735'689'600;

const std::string kIbmHeader(kIbmSchema.header);
const std::string kTxHeader(kTransxionSchema.header);

std::vector<std::string> lines(const std::string& text, std::size_t buffer) {
  std::istringstream in(text);
  LineReader r(in, buffer);
  std::vector<std::string> out;
  std::string_view l;
  while (r.next(l)) {
    out.emplace_back(l);
    CHECK(r.line_number() == out.size());
  }
  return out;
}

}  // namespace

TEST_CASE("parse_timestamp: both schemas, unix seconds with no time zone (SPEC v3 §6.1)") {
  CHECK(parse_timestamp(kIbmSchema, "2022/09/01 00:00") == k20220901);
  CHECK(parse_timestamp(kIbmSchema, "2022/09/01 00:20") == k20220901 + 1200);
  CHECK(parse_timestamp(kIbmSchema, "2022/09/02 01:01") == k20220901 + 86400 + 3660);
  CHECK(parse_timestamp(kIbmSchema, "2024/02/29 00:00") == 1'709'164'800);
  CHECK(parse_timestamp(kIbmSchema, "1969/12/31 23:59") == -60);
  for (const char* bad : {"2022/09/31 00:00", "2023/02/29 00:00", "2022/09/01 24:00", "2022/09/01 0:00",
                          "2022-09-01 00:00", "2022/09/0a 00:00", "2022/09/01 00:00:00", "2022/13/01 00:00"})
    CHECK_THROWS_AS(parse_timestamp(kIbmSchema, bad), std::invalid_argument);

  CHECK(parse_timestamp(kTransxionSchema, "2025-01-01 00:00:29") == k20250101 + 29);
  CHECK(parse_timestamp(kTransxionSchema, "2025-12-31 22:59:18") == k20250101 + 364 * 86400 + 22 * 3600 + 59 * 60 + 18);
  for (const char* bad : {"2025/01/01 00:00:29", "2025-01-01 00:00", "2025-01-01 00:00:60", "2025-01-01T00:00:00",
                          "2025-01-01 00:00:2x"})
    CHECK_THROWS_AS(parse_timestamp(kTransxionSchema, bad), std::invalid_argument);

  CHECK(find_schema("ibm") == &kIbmSchema);
  CHECK(find_schema("transxion") == &kTransxionSchema);
  CHECK(find_schema("IBM") == nullptr);
  CHECK(civil_date(k20220901 / 86400) == "2022-09-01");
  CHECK(civil_date(0) == "1970-01-01");
  CHECK(civil_date(-1) == "1969-12-31");
  CHECK(civil_date(1'709'164'800 / 86400) == "2024-02-29");
}

TEST_CASE("LineReader: lines across buffer boundaries, CRLF, a final line with no newline") {
  const std::string text = "a,b\r\n\nccccccccccccc\nd\r\nlast";
  const std::vector<std::string> want = {"a,b", "", "ccccccccccccc", "d", "last"};
  for (std::size_t buffer : {1, 2, 3, 4, 5, 7, 13, 14, 64})
    CHECK(lines(text, buffer) == want);
  CHECK(lines("x\n", 1) == std::vector<std::string>{"x"});  // nothing after the final '\n'
  CHECK(lines("", 4).empty());
  // A '\r' and its '\n' in different reads.
  CHECK(lines("ab\r\ncd\r\n", 3) == std::vector<std::string>{"ab", "cd"});
}

TEST_CASE("read_csv: the fixture in file order, and the bytes' SHA-256") {
  RawTable t = read_text(kFixtureCsv);
  REQUIRE(t.size() == 4);
  CHECK(t.ts == std::vector<std::int64_t>{k20220901 + 1200, k20220901, k20220901 + 1200, k20220901});
  CHECK(t.nodes[t.src[0]] == "010,A");  // leading zeros kept
  CHECK(t.nodes[t.src[1]] == "10,A");
  CHECK(t.src[1] != t.dst[1]);          // "10" and "010" are different banks
  CHECK(t.src[2] == t.dst[2]);          // r2 is a self-loop
  CHECK(t.nodes.size() == 4);
  CHECK(t.formats[t.format[1]] == "ACH");
  CHECK(t.currencies[t.recv_ccy[2]] == "Bitcoin");
  CHECK(t.label == std::vector<std::uint8_t>{0, 1, 0, 0});
  CHECK(t.amount_received[0] == 10.5);
  CHECK(t.amount_paid[1] == 1.25);
  CHECK(t.amount_received[3] == 2000.0);
  CHECK(t.sha256 == sha256(kFixtureCsv.data(), kFixtureCsv.size()));

  std::string s = kFixtureCsv;
  s.pop_back();  // no trailing newline
  CHECK(read_text(s).size() == 4);
  CHECK(read_text(kFixtureCsv + "\n").size() == 4);  // one empty last line
}

TEST_CASE("read_csv: every buffer size gives the same table") {
  const RawTable want = read_text(kFixtureCsv);
  for (std::size_t buffer = 1; buffer <= kFixtureCsv.size() + 1; ++buffer) {
    INFO("buffer " << buffer);
    const RawTable t = read_text(kFixtureCsv, kIbmSchema, buffer);
    CHECK(t.ts == want.ts);
    CHECK(t.src == want.src);
    CHECK(t.dst == want.dst);
    CHECK(t.amount_paid == want.amount_paid);
    CHECK(t.format == want.format);
    CHECK(t.sha256 == want.sha256);
  }
}

TEST_CASE("read_csv: the TransXion schema") {
  const std::string csv = kTxHeader + "\n"
      "2025-01-01 00:00:40,3186407,A007397,3295718,A010077,58.45,CNY,58.51,CNY,Cash,0\n"
      "2025-01-01 00:00:29,3295718,A009852,2657408,A010012,4.4,CHF,8.06,AUD,Mobile,1\n";
  const RawTable t = read_text(csv, kTransxionSchema);
  REQUIRE(t.size() == 2);
  CHECK(t.ts == std::vector<std::int64_t>{k20250101 + 40, k20250101 + 29});
  CHECK(t.nodes[t.dst[0]] == "3295718,A010077");
  CHECK(t.label == std::vector<std::uint8_t>{0, 1});
  // Each schema's header is its own.
  CHECK_THROWS_AS(read_text(csv, kIbmSchema), CsvError);
  CHECK_THROWS_AS(read_text(kFixtureCsv, kTransxionSchema), CsvError);
}

TEST_CASE("read_csv errors carry the line number") {
  auto line_of = [](const std::string& text) {
    try {
      read_text(text);
    } catch (const CsvError& e) {
      return e.line;
    }
    return std::size_t(0);
  };
  const std::string hdr = kIbmHeader;
  const std::string good = "2022/09/01 00:00,1,A,2,B,1,Euro,1,Euro,ACH,0\n";
  CHECK(line_of("Timestamp,Bank\n" + good) == 1);
  CHECK(line_of(kTxHeader + "\n" + good) == 1);
  CHECK(line_of(hdr + "\n" + good + "2022/09/01 00:00,1,A,2,B,1,Euro,1,Euro,ACH\n") == 3);
  CHECK(line_of(hdr + "\n" + good + good + "2022/09/01 00:00,1,A,2,B,1,Euro,1,Euro,ACH,0,9\n") == 4);
  CHECK(line_of(hdr + "\n2022/09/01 00:00,1,A,2,B,x,Euro,1,Euro,ACH,0\n") == 2);
  CHECK(line_of(hdr + "\n2022/09/01 00:00,1,A,2,B,1,Euro,1,Euro,ACH,2\n") == 2);
  CHECK(line_of(hdr + "\n2022/09/01 00:00,,A,2,B,1,Euro,1,Euro,ACH,0\n") == 2);
  CHECK(line_of(hdr + "\n2022/09/01 00:00,\"1\",A,2,B,1,Euro,1,Euro,ACH,0\n") == 2);
  CHECK(line_of(hdr + "\n2022/13/01 00:00,1,A,2,B,1,Euro,1,Euro,ACH,0\n") == 2);
  CHECK(line_of(hdr + "\n" + good + "\n" + good) == 3);  // an empty line inside the file
  CHECK(line_of(hdr + "\n" + good + "\n\n") == 3);       // two at the end
  std::string many = hdr + "\n";
  for (int i = 0; i < 257; ++i) many += "2022/09/01 00:00,1,A,2,B,1,C" + std::to_string(i) + ",1,Euro,ACH,0\n";
  CHECK(line_of(many) == 257);  // a 257th currency
}

TEST_CASE("for_each_node_row: named columns, quoted fields, CRLF") {
  const std::string table =
      "id,bank_account_number,bank,description\r\n"
      "c1,A01,3468205,\"Small store, snacks\"\r\n"
      "c2,A02,1047286,\"a \"\"quoted\"\" word\"\r\n"
      "c3,A03,1047286,plain";
  std::vector<std::pair<std::string, std::string>> rows;
  for (std::size_t buffer : {std::size_t(3), kCsvBuffer}) {
    rows.clear();
    std::istringstream in(table);
    for_each_node_row(in, "bank", "bank_account_number",
                      [&](std::string_view b, std::string_view a) { rows.emplace_back(b, a); }, buffer);
    CHECK(rows == std::vector<std::pair<std::string, std::string>>{{"3468205", "A01"}, {"1047286", "A02"},
                                                                     {"1047286", "A03"}});
  }
  auto fails = [](const std::string& text, std::string_view bank = "bank") {
    std::istringstream in(text);
    try {
      for_each_node_row(in, bank, "acct", [](std::string_view, std::string_view) {});
    } catch (const CsvError& e) {
      return e.line;
    }
    return std::size_t(0);
  };
  CHECK(fails("bank,acct\n1,A\n", "branch") == 1);   // no such column
  CHECK(fails("bank,acct\n1,A,x\n") == 2);           // a field too many
  CHECK(fails("bank,acct\n1,\"A\n") == 2);           // unterminated quote
  CHECK(fails("bank,acct\n1,\"A\"x\n") == 2);        // text after a closing quote
  CHECK(fails("bank,acct\n1,A\n") == 0);
}

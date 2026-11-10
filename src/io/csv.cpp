#include "io/csv.hpp"

#include <array>
#include <charconv>

namespace blz {
namespace {

// Days from 1970-01-01 to y-m-d in the proleptic Gregorian calendar (H. Hinnant's algorithm).
constexpr std::int64_t days_from_civil(std::int64_t y, unsigned m, unsigned d) {
  y -= m <= 2;
  const std::int64_t era = (y >= 0 ? y : y - 399) / 400;
  const unsigned yoe = unsigned(y - era * 400);
  const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
  const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
  return era * 146097 + std::int64_t(doe) - 719468;
}
constexpr std::int64_t kEpochDays = days_from_civil(2022, 9, 1);

bool leap(int y) { return (y % 4 == 0 && y % 100 != 0) || y % 400 == 0; }

int digits(std::string_view s, std::size_t pos, std::size_t n) {
  int v = 0;
  for (std::size_t i = pos; i < pos + n; ++i) {
    if (s[i] < '0' || s[i] > '9') throw std::invalid_argument("bad timestamp: " + std::string(s));
    v = v * 10 + (s[i] - '0');
  }
  return v;
}

constexpr std::size_t kFields = 11;

}  // namespace

std::int32_t parse_timestamp(std::string_view s) {
  // YYYY/MM/DD HH:MM
  if (s.size() != 16 || s[4] != '/' || s[7] != '/' || s[10] != ' ' || s[13] != ':')
    throw std::invalid_argument("bad timestamp: " + std::string(s));
  int y = digits(s, 0, 4), mo = digits(s, 5, 2), d = digits(s, 8, 2);
  int h = digits(s, 11, 2), mi = digits(s, 14, 2);
  static constexpr std::array<int, 12> mdays = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  if (mo < 1 || mo > 12 || d < 1 || d > mdays[mo - 1] + (mo == 2 && leap(y)) || h > 23 || mi > 59)
    throw std::invalid_argument("bad timestamp: " + std::string(s));
  std::int64_t secs = (days_from_civil(y, unsigned(mo), unsigned(d)) - kEpochDays) * 86400 +
                      std::int64_t(h) * 3600 + std::int64_t(mi) * 60;
  if (secs < 0 || secs > kMaxTimestamp)
    throw std::invalid_argument("timestamp out of range: " + std::string(s));
  return std::int32_t(secs);
}

RawTable parse_csv(std::string_view text) {
  RawTable t;
  std::size_t pos = 0, line = 0;
  auto next_line = [&](std::string_view& out) {
    if (pos >= text.size()) return false;
    std::size_t end = text.find('\n', pos);
    if (end == std::string_view::npos) end = text.size();
    out = text.substr(pos, end - pos);
    if (!out.empty() && out.back() == '\r') out.remove_suffix(1);
    pos = end + 1;
    ++line;
    return true;
  };

  std::string_view row;
  if (!next_line(row) || row != kCsvHeader) throw CsvError(1, "unexpected header");

  const std::size_t approx = text.size() / 90 + 16;
  for (auto* v : {&t.from_bank, &t.from_account, &t.to_bank, &t.to_account, &t.recv_ccy,
                  &t.pay_ccy, &t.format})
    v->reserve(approx);
  t.ts.reserve(approx);
  t.amount_received.reserve(approx);
  t.amount_paid.reserve(approx);
  t.label.reserve(approx);

  std::array<std::string_view, kFields> f;
  while (next_line(row)) {
    if (row.empty() && pos >= text.size()) break;  // trailing newline at EOF
    std::size_t nf = 0, start = 0;
    for (std::size_t i = 0; i <= row.size(); ++i) {
      if (i == row.size() || row[i] == ',') {
        if (nf == kFields) throw CsvError(line, "too many fields");
        f[nf++] = row.substr(start, i - start);
        start = i + 1;
      } else if (row[i] == '"') {
        throw CsvError(line, "quoted fields are not supported");
      }
    }
    if (nf != kFields) throw CsvError(line, "expected 11 fields, got " + std::to_string(nf));
    for (std::size_t i = 0; i < kFields; ++i)
      if (f[i].empty()) throw CsvError(line, "empty field " + std::to_string(i + 1));

    try {
      t.ts.push_back(parse_timestamp(f[0]));
    } catch (const std::invalid_argument& e) {
      throw CsvError(line, e.what());
    }
    auto num = [&](std::string_view s) {
      double v = 0;
      auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
      if (ec != std::errc() || p != s.data() + s.size())
        throw CsvError(line, "bad amount: " + std::string(s));
      return v;
    };
    t.from_bank.push_back(f[1]);
    t.from_account.push_back(f[2]);
    t.to_bank.push_back(f[3]);
    t.to_account.push_back(f[4]);
    t.amount_received.push_back(num(f[5]));
    t.recv_ccy.push_back(f[6]);
    t.amount_paid.push_back(num(f[7]));
    t.pay_ccy.push_back(f[8]);
    t.format.push_back(f[9]);
    if (f[10] != "0" && f[10] != "1") throw CsvError(line, "bad label: " + std::string(f[10]));
    t.label.push_back(std::uint8_t(f[10][0] - '0'));
  }
  return t;
}

}  // namespace blz

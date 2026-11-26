#include "io/csv.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdio>
#include <cstring>

namespace blz {
namespace {

// Days from 1970-01-01 to y-m-d in the proleptic Gregorian calendar, and back (H. Hinnant's
// algorithms).
constexpr std::int64_t days_from_civil(std::int64_t y, unsigned m, unsigned d) {
  y -= m <= 2;
  const std::int64_t era = (y >= 0 ? y : y - 399) / 400;
  const unsigned yoe = unsigned(y - era * 400);
  const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
  const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
  return era * 146097 + std::int64_t(doe) - 719468;
}

bool leap(int y) { return (y % 4 == 0 && y % 100 != 0) || y % 400 == 0; }

constexpr std::size_t kFields = 11;

}  // namespace

const CsvSchema* find_schema(std::string_view name) {
  for (const CsvSchema& s : kSchemas)
    if (s.name == name) return &s;
  return nullptr;
}

std::int64_t parse_timestamp(const CsvSchema& s, std::string_view t) {
  // YYYY-MM-DD HH:MM[:SS], '-' being the schema's date separator
  const auto bad = [&] { return std::invalid_argument("bad timestamp: " + std::string(t)); };
  const std::size_t len = s.seconds ? 19 : 16;
  if (t.size() != len || t[4] != s.date_sep || t[7] != s.date_sep || t[10] != ' ' || t[13] != ':' ||
      (s.seconds && t[16] != ':'))
    throw bad();
  const auto num = [&](std::size_t pos, std::size_t n) {
    int v = 0;
    for (std::size_t i = pos; i < pos + n; ++i) {
      if (t[i] < '0' || t[i] > '9') throw bad();
      v = v * 10 + (t[i] - '0');
    }
    return v;
  };
  const int y = num(0, 4), mo = num(5, 2), d = num(8, 2), h = num(11, 2), mi = num(14, 2);
  const int sec = s.seconds ? num(17, 2) : 0;
  static constexpr std::array<int, 12> mdays = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  if (mo < 1 || mo > 12 || d < 1 || d > mdays[mo - 1] + (mo == 2 && leap(y)) || h > 23 || mi > 59 || sec > 59)
    throw bad();
  return days_from_civil(y, unsigned(mo), unsigned(d)) * 86400 + std::int64_t(h) * 3600 + std::int64_t(mi) * 60 + sec;
}

std::string civil_date(std::int64_t days) {
  days += 719468;
  const std::int64_t era = (days >= 0 ? days : days - 146096) / 146097;
  const unsigned doe = unsigned(days - era * 146097);
  const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
  const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  const unsigned mp = (5 * doy + 2) / 153;
  const unsigned d = doy - (153 * mp + 2) / 5 + 1, m = mp < 10 ? mp + 3 : mp - 9;
  const std::int64_t y = std::int64_t(yoe) + era * 400 + (m <= 2);
  char buf[32];
  std::snprintf(buf, sizeof buf, "%04lld-%02u-%02u", (long long)y, m, d);
  return buf;
}

LineReader::LineReader(std::istream& in, std::size_t buffer_bytes, Sha256* sha)
    : in_(in), buf_(std::max<std::size_t>(buffer_bytes, 1)), sha_(sha) {}

bool LineReader::fill() {
  in_.read(buf_.data(), std::streamsize(buf_.size()));
  if (in_.bad()) throw std::runtime_error("read failed");
  pos_ = 0;
  end_ = std::size_t(in_.gcount());
  if (sha_) sha_->update(buf_.data(), end_);
  return end_ > 0;
}

bool LineReader::next(std::string_view& line) {
  if (carry_out_) {
    carry_.clear();
    carry_out_ = false;
  }
  for (;;) {
    if (pos_ < end_) {
      const char* p = buf_.data() + pos_;
      if (const auto* nl = static_cast<const char*>(std::memchr(p, '\n', end_ - pos_))) {
        pos_ += std::size_t(nl - p) + 1;
        if (carry_.empty()) {
          line = std::string_view(p, std::size_t(nl - p));
        } else {
          carry_.append(p, nl);
          line = carry_;
          carry_out_ = true;
        }
        break;
      }
      carry_.append(p, end_ - pos_);
      pos_ = end_;
    }
    if (!fill()) {
      if (carry_.empty()) return false;
      line = carry_;
      carry_out_ = true;
      break;
    }
  }
  if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
  ++line_;
  return true;
}

std::uint32_t Interner::id(std::string_view s) {
  if (const auto it = ids_.find(s); it != ids_.end()) return it->second;
  const auto id = std::uint32_t(keys_.size());
  keys_.push_back(&ids_.emplace(std::string(s), id).first->first);
  return id;
}

RawTable read_csv(std::istream& in, const CsvSchema& schema, std::size_t buffer_bytes) {
  std::uint64_t newlines = 0;
  {
    std::vector<char> buf(std::max<std::size_t>(buffer_bytes, 1));
    while (in.read(buf.data(), std::streamsize(buf.size())) || in.gcount() > 0)
      newlines += std::uint64_t(std::count(buf.data(), buf.data() + in.gcount(), '\n'));
    if (in.bad()) throw std::runtime_error("read failed");
    in.clear();
    if (!in.seekg(0)) throw std::runtime_error("read_csv: the stream is not seekable");
  }

  RawTable t;
  // At most one data row per '\n': the header holds one unless it is the only line.
  for (auto* v : {&t.src, &t.dst}) v->reserve(newlines);
  for (auto* v : {&t.label, &t.recv_ccy, &t.pay_ccy, &t.format}) v->reserve(newlines);
  t.ts.reserve(newlines);
  t.amount_received.reserve(newlines);
  t.amount_paid.reserve(newlines);

  Sha256 sha;
  LineReader r(in, buffer_bytes, &sha);
  std::string_view row;
  if (!r.next(row) || row != schema.header) throw CsvError(1, "unexpected header");

  std::array<std::string_view, kFields> f;
  std::string key;
  std::size_t blank = 0;  // the last empty line, an error unless no line follows
  auto code = [&](Interner& dict, std::string_view v, std::size_t line) {
    const std::uint32_t c = dict.id(v);
    if (c > 255) throw CsvError(line, "a dictionary has more than 256 entries");
    return std::uint8_t(c);
  };
  while (r.next(row)) {
    const std::size_t line = r.line_number();
    if (blank) throw CsvError(blank, "empty line");
    if (row.empty()) {
      blank = line;
      continue;
    }
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
    if (t.size() == newlines) throw std::runtime_error("read_csv: the stream changed between its two reads");

    try {
      t.ts.push_back(parse_timestamp(schema, f[0]));
    } catch (const std::invalid_argument& e) {
      throw CsvError(line, e.what());
    }
    auto num = [&](std::string_view s) {
      double v = 0;
      auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
      if (ec != std::errc() || p != s.data() + s.size()) throw CsvError(line, "bad amount: " + std::string(s));
      return v;
    };
    key.assign(f[1]).append(1, ',').append(f[2]);
    t.src.push_back(t.nodes.id(key));
    key.assign(f[3]).append(1, ',').append(f[4]);
    t.dst.push_back(t.nodes.id(key));
    t.amount_received.push_back(num(f[5]));
    t.recv_ccy.push_back(code(t.currencies, f[6], line));
    t.amount_paid.push_back(num(f[7]));
    t.pay_ccy.push_back(code(t.currencies, f[8], line));
    t.format.push_back(code(t.formats, f[9], line));
    if (f[10] != "0" && f[10] != "1") throw CsvError(line, "bad label: " + std::string(f[10]));
    t.label.push_back(std::uint8_t(f[10][0] - '0'));
  }
  t.sha256 = sha.finish();
  return t;
}

void for_each_node_row(std::istream& in, std::string_view bank_column, std::string_view account_column,
                       const std::function<void(std::string_view, std::string_view)>& f,
                       std::size_t buffer_bytes) {
  LineReader r(in, buffer_bytes);
  std::vector<std::string> fields;
  auto split = [&](std::string_view l) {
    fields.assign(1, "");
    for (std::size_t i = 0; i < l.size(); ++i) {
      if (l[i] == ',') {
        fields.emplace_back();
      } else if (l[i] == '"' && fields.back().empty()) {
        for (++i;; ++i) {
          if (i == l.size()) throw CsvError(r.line_number(), "unterminated quote");
          if (l[i] == '"') {
            if (i + 1 < l.size() && l[i + 1] == '"') ++i;
            else break;
          }
          fields.back() += l[i];
        }
        if (i + 1 < l.size() && l[i + 1] != ',') throw CsvError(r.line_number(), "text after a closing quote");
      } else {
        fields.back() += l[i];
      }
    }
  };
  std::string_view l;
  if (!r.next(l)) throw CsvError(1, "no header");
  split(l);
  const auto col = [&](std::string_view name) {
    const auto it = std::find(fields.begin(), fields.end(), name);
    if (it == fields.end()) throw CsvError(1, "no column '" + std::string(name) + "'");
    return std::size_t(it - fields.begin());
  };
  const std::size_t ncols = fields.size(), bi = col(bank_column), ai = col(account_column);
  while (r.next(l)) {
    split(l);
    if (fields.size() != ncols)
      throw CsvError(r.line_number(), "expected " + std::to_string(ncols) + " fields, got " + std::to_string(fields.size()));
    f(fields[bi], fields[ai]);
  }
}

}  // namespace blz

#pragma once
#include <cstdint>
#include <functional>
#include <istream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "common/sha256.hpp"

namespace blz {

struct CsvError : std::runtime_error {
  CsvError(std::size_t line, const std::string& what)
      : std::runtime_error("line " + std::to_string(line) + ": " + what), line(line) {}
  std::size_t line;
};

// A transactions file's layout (SPEC v3 §6.1). Every schema has the same 11 columns in the same
// order: timestamp, from bank, from account, to bank, to account, amount received, receiving
// currency, amount paid, payment currency, payment format, is laundering. Schemas differ in the
// header's account names and the timestamp format, YYYY<date_sep>MM<date_sep>DD HH:MM[:SS].
struct CsvSchema {
  std::string_view name, header;
  char date_sep;
  bool seconds;
  bool concatenated_keys;  // pin IBM Multi-GNN's bank + account key count (SPEC v2 §4.4 R1)
};
inline constexpr CsvSchema kSchemas[] = {
    {"ibm",
     "Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,"
     "Amount Paid,Payment Currency,Payment Format,Is Laundering",
     '/', false, true},
    {"transxion",
     "Timestamp,From Bank,From Account,To Bank,To Account,Amount Received,Receiving Currency,"
     "Amount Paid,Payment Currency,Payment Format,Is Laundering",
     '-', true, false},
};
inline constexpr const CsvSchema& kIbmSchema = kSchemas[0];
inline constexpr const CsvSchema& kTransxionSchema = kSchemas[1];

// The schema called `name`, or nullptr.
const CsvSchema* find_schema(std::string_view name);

// A timestamp in the schema's format -> seconds since 1970-01-01 00:00, no time zone. Throws
// std::invalid_argument if it is malformed or not a calendar date and time.
std::int64_t parse_timestamp(const CsvSchema& s, std::string_view text);

// Days since 1970-01-01 -> "YYYY-MM-DD".
std::string civil_date(std::int64_t days);

// The read buffer of `blz prepare` (SPEC v3 §6.2).
inline constexpr std::size_t kCsvBuffer = std::size_t(1) << 22;

// The lines of a stream read through a buffer of a fixed size, each without its '\n' and a
// trailing '\r'. A line that spans reads is copied into a carry string, so memory is the buffer
// plus the longest such line. Every byte read is fed to `sha` when it is given.
class LineReader {
 public:
  LineReader(std::istream& in, std::size_t buffer_bytes, Sha256* sha = nullptr);
  // The next line, valid until the next call; false at the end of the stream. A final line with
  // no '\n' is a line; nothing after a final '\n' is.
  bool next(std::string_view& line);
  std::size_t line_number() const { return line_; }  // 1-based number of the last line returned

 private:
  bool fill();
  std::istream& in_;
  std::vector<char> buf_;
  std::size_t pos_ = 0, end_ = 0, line_ = 0;
  std::string carry_;
  bool carry_out_ = false;
  Sha256* sha_;
};

// Distinct strings in the order they first appear: id -> string and string -> id.
class Interner {
 public:
  Interner() = default;
  Interner(Interner&&) = default;
  Interner& operator=(Interner&&) = default;
  Interner(const Interner&) = delete;
  Interner& operator=(const Interner&) = delete;

  std::uint32_t id(std::string_view s);  // s's id; a new one, size(), when s is new
  std::size_t size() const { return keys_.size(); }
  std::string_view operator[](std::uint32_t id) const { return *keys_[id]; }

 private:
  struct Hash {
    using is_transparent = void;
    std::size_t operator()(std::string_view s) const noexcept { return std::hash<std::string_view>{}(s); }
  };
  std::unordered_map<std::string, std::uint32_t, Hash, std::equal_to<>> ids_;
  std::vector<const std::string*> keys_;  // into ids_'s nodes, which never move
};

// A transactions CSV in file order (SPEC v3 §6.2): node ids and dictionary codes are provisional,
// by first appearance in file order, and timestamps are unix seconds.
struct RawTable {
  std::vector<std::int64_t> ts;
  std::vector<std::uint32_t> src, dst;
  std::vector<std::uint8_t> label;
  std::vector<double> amount_received, amount_paid;
  std::vector<std::uint8_t> recv_ccy, pay_ccy, format;
  Interner nodes;  // "bank,account": a field never holds ','
  Interner currencies, formats;
  Sha256Digest sha256{};  // of every byte of the stream
  std::size_t size() const { return ts.size(); }
};

// Reads the stream twice through a buffer of `buffer_bytes`: once to count its lines (so every
// column is allocated once, at its final size), then to parse it. The stream must be seekable.
// An empty line is an error unless it is the last line. Throws CsvError (with the 1-based line
// number) on any malformed row or header.
RawTable read_csv(std::istream& in, const CsvSchema& schema, std::size_t buffer_bytes = kCsvBuffer);

// Calls f(bank, account) for every data row of a node table (SPEC v3 §6.1: TransXion's
// person.csv and merchant.csv), by the header's column names. A field may be quoted ("a, b",
// with "" for a quote). Throws CsvError on a missing column or a malformed row.
void for_each_node_row(std::istream& in, std::string_view bank_column, std::string_view account_column,
                       const std::function<void(std::string_view, std::string_view)>& f,
                       std::size_t buffer_bytes = kCsvBuffer);

}  // namespace blz

#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace blz {

struct CsvError : std::runtime_error {
  CsvError(std::size_t line, const std::string& what)
      : std::runtime_error("line " + std::to_string(line) + ": " + what), line(line) {}
  std::size_t line;
};

// HI-Small_Trans.csv rows in file order (SPEC §3.1). Views point into the parsed text.
struct RawTable {
  std::vector<std::int32_t> ts;
  std::vector<std::string_view> from_bank, from_account, to_bank, to_account;
  std::vector<double> amount_received, amount_paid;
  std::vector<std::string_view> recv_ccy, pay_ccy, format;
  std::vector<std::uint8_t> label;
  std::size_t size() const { return ts.size(); }
};

inline constexpr std::string_view kCsvHeader =
    "Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,"
    "Amount Paid,Payment Currency,Payment Format,Is Laundering";
inline constexpr std::int32_t kMaxTimestamp = 1'527'480;  // 2022/09/18 16:18

// "YYYY/MM/DD HH:MM" -> seconds since 2022/09/01 00:00 (D-1). Throws std::invalid_argument if
// malformed or outside [0, kMaxTimestamp].
std::int32_t parse_timestamp(std::string_view s);

// Parses the whole file text. Throws CsvError (with the 1-based line number) on any malformed
// row or header.
RawTable parse_csv(std::string_view text);

}  // namespace blz

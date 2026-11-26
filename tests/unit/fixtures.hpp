#pragma once
#include <filesystem>
#include <sstream>
#include <string>

#include "graph/prepare.hpp"

// Rows deliberately out of timestamp order, with ties, a self-loop, leading-zero banks and
// CRLF line endings on one row.
inline const std::string kFixtureCsv =
    "Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,"
    "Amount Paid,Payment Currency,Payment Format,Is Laundering\n"
    "2022/09/01 00:20,010,A,001,B,10.5,US Dollar,10.5,US Dollar,Cheque,0\n"         // r0 ts 1200
    "2022/09/01 00:00,10,A,010,A,1,Euro,1.25,US Dollar,ACH,1\r\n"                   // r1 ts 0
    "2022/09/01 00:20,001,B,001,B,3,Bitcoin,3,Bitcoin,Reinvestment,0\n"             // r2 ts 1200 (self-loop)
    "2022/09/01 00:00,001,C,010,A,2e3,Euro,2000,Euro,Wire,0\n";                     // r3 ts 0

inline std::string test_tmp(const std::string& name) {
  std::filesystem::create_directories(BLZ_TEST_TMPDIR);
  return std::string(BLZ_TEST_TMPDIR) + "/" + name;
}

// CSV text through read_csv.
inline blz::RawTable read_text(const std::string& csv, const blz::CsvSchema& schema = blz::kIbmSchema,
                               std::size_t buffer_bytes = blz::kCsvBuffer) {
  std::istringstream in(csv);
  return blz::read_csv(in, schema, buffer_bytes);
}

// CSV text through read_csv and build_edge_table, as `blz prepare` reads a file.
inline blz::EdgeTable prepare_text(const std::string& csv, const blz::CsvSchema& schema = blz::kIbmSchema,
                                   std::size_t buffer_bytes = blz::kCsvBuffer) {
  return blz::build_edge_table(read_text(csv, schema, buffer_bytes));
}

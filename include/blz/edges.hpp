#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace blz {

// The prepared transaction stream (SPEC §3): edges in stream order, so eid = index.
struct EdgeTable {
  std::vector<std::uint32_t> src, dst;
  std::vector<std::int32_t> ts;  // seconds since 2022-09-01 00:00 (D-1)
  std::vector<std::uint8_t> label;
  std::vector<double> amount_paid, amount_received;
  std::vector<std::uint8_t> pay_ccy, recv_ccy, format;  // codes into the dictionaries
  std::vector<std::string> ccy_dict, format_dict;        // sorted by string
  std::vector<std::string> node_bank, node_account;      // node id -> raw key (D-2, D-3)
  std::array<std::uint8_t, 32> csv_sha256{};

  std::size_t m() const { return src.size(); }
  std::size_t n() const { return node_bank.size(); }
};

// Train split: eid < floor(0.8 m) (SPEC §3.4).
inline std::uint64_t train_size(std::uint64_t m) { return m * 4 / 5; }

}  // namespace blz

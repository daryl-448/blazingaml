#pragma once
#include <array>
#include <cstdint>
#include <span>
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
  std::vector<std::uint32_t> file_row;  // eid -> 0-based data row of the CSV (SPEC v2 §3.5, D-31)
  std::array<std::uint8_t, 32> csv_sha256{};

  std::size_t m() const { return src.size(); }
  std::size_t n() const { return node_bank.size(); }
};

// Train split: eid < floor(0.8 m) (SPEC §3.4).
inline std::uint64_t train_size(std::uint64_t m) { return m * 4 / 5; }

// The fit/validation boundary b_fv (SPEC v2 §3.3, D-17): the first eid >= floor(0.6 m) whose
// timestamp differs from its predecessor's (eid 0 has none and counts as a boundary), or m if
// there is none. fit = [0, b_fv), validation = [b_fv, train_size(m)), test = [train_size(m), m).
inline std::uint64_t fit_val_boundary(std::span<const std::int32_t> ts) {
  const std::uint64_t m = ts.size();
  for (std::uint64_t e = m * 3 / 5; e < m; ++e)
    if (e == 0 || ts[e] != ts[e - 1]) return e;
  return m;
}

}  // namespace blz

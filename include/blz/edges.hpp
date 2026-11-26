#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
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

// The named splits (SPEC v3 §6.3): fit = [0, b_fv), validation = [b_fv, b_test), test =
// [b_test, m), train = fit + validation; b_fv <= b_test <= m.
struct Splits {
  std::uint64_t b_fv = 0, b_test = 0;
  bool operator==(const Splits&) const = default;
};

// How a dataset's splits are cut (SPEC v3 §6.3).
//   legacy_hi_small (SPEC §3.4, V2 §3.3, D-17): b_test = floor(0.8 m) and
//     b_fv = min(timestamp_start(floor(0.6 m)), b_test). HI-Small keeps it.
//   temporal_60_20_20: b_fv = timestamp_start(floor(0.6 m)), b_test = timestamp_start(floor(0.8 m)),
//     so no timestamp straddles either cut.
enum class SplitRule : std::uint8_t { legacy_hi_small, temporal_60_20_20 };

inline constexpr std::array<std::string_view, 2> kSplitRuleNames = {"legacy_hi_small", "temporal_60_20_20"};
inline std::string_view split_rule_name(SplitRule r) { return kSplitRuleNames[std::size_t(r)]; }
inline std::optional<SplitRule> parse_split_rule(std::string_view s) {
  for (std::size_t i = 0; i < kSplitRuleNames.size(); ++i)
    if (s == kSplitRuleNames[i]) return SplitRule(i);
  return std::nullopt;
}

// The first eid >= e that starts a timestamp (eid 0 does, and so does an eid whose timestamp
// differs from its predecessor's), or m if there is none.
inline std::uint64_t timestamp_start(std::span<const std::int32_t> ts, std::uint64_t e) {
  const std::uint64_t m = ts.size();
  for (; e < m; ++e)
    if (e == 0 || ts[e] != ts[e - 1]) return e;
  return m;
}

inline Splits compute_splits(SplitRule rule, std::span<const std::int32_t> ts) {
  const std::uint64_t m = ts.size(), b_fv = timestamp_start(ts, m * 3 / 5);
  if (rule == SplitRule::legacy_hi_small) return {std::min(b_fv, m * 4 / 5), m * 4 / 5};
  return {b_fv, timestamp_start(ts, m * 4 / 5)};
}

}  // namespace blz

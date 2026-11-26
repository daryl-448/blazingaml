#include "graph/prepare.hpp"

#include <algorithm>
#include <array>
#include <climits>
#include <set>
#include <stdexcept>
#include <string_view>
#include <unordered_set>

namespace blz {
namespace {

// col in stream order, out[e] = col[file row of e]; col is freed.
template <class T>
std::vector<T> permuted(std::vector<T>& col, const std::vector<std::int64_t>& keys) {
  std::vector<T> out(keys.size());
  for (std::size_t e = 0; e < keys.size(); ++e) out[e] = col[std::uint32_t(keys[e])];
  std::vector<T>().swap(col);
  return out;
}

// The dictionary sorted by string, and code[c] = provisional code c's place in it.
std::vector<std::string> sorted_dict(const Interner& d, std::array<std::uint8_t, 256>& code) {
  std::vector<std::string> sorted;
  for (std::uint32_t c = 0; c < d.size(); ++c) sorted.emplace_back(d[c]);
  std::sort(sorted.begin(), sorted.end());
  for (std::uint32_t c = 0; c < d.size(); ++c)
    code[c] = std::uint8_t(std::lower_bound(sorted.begin(), sorted.end(), d[c]) - sorted.begin());
  return sorted;
}

}  // namespace

EdgeTable build_edge_table(RawTable&& raw) {
  const std::size_t m = raw.size();
  if (m >= (std::size_t(1) << 32)) throw std::runtime_error("too many edges for uint32 ids");
  EdgeTable t;
  t.csv_sha256 = raw.sha256;
  if (m) {
    const auto [lo, hi] = std::minmax_element(raw.ts.begin(), raw.ts.end());
    const std::int64_t day = *lo / 86400 - (*lo % 86400 < 0);
    t.epoch = day * 86400;
    if (*hi - t.epoch > INT32_MAX)
      throw std::runtime_error("the timestamps span " + std::to_string(*hi - t.epoch) + " s from the epoch " +
                               civil_date(day) + ", more than int32 seconds (SPEC v3 §6.1)");
  }

  // The sort keys (ts - epoch) << 32 | file row, in ts's place. They are distinct, so sorting
  // them gives the stable sort by timestamp.
  std::vector<std::int64_t> keys = std::move(raw.ts);
  for (std::size_t r = 0; r < m; ++r) keys[r] = ((keys[r] - t.epoch) << 32) | std::int64_t(r);
  std::sort(keys.begin(), keys.end());

  // Node ids by first appearance in stream order, source before destination (D-3).
  const std::uint32_t none = UINT32_MAX;
  std::vector<std::uint32_t> id(raw.nodes.size(), none);
  std::uint32_t next = 0;
  for (const std::int64_t k : keys) {
    const std::uint32_t r(k);
    if (id[raw.src[r]] == none) id[raw.src[r]] = next++;
    if (id[raw.dst[r]] == none) id[raw.dst[r]] = next++;
  }
  t.src = permuted(raw.src, keys);
  for (std::uint32_t& x : t.src) x = id[x];
  t.dst = permuted(raw.dst, keys);
  for (std::uint32_t& x : t.dst) x = id[x];
  t.node_bank.resize(next);
  t.node_account.resize(next);
  for (std::uint32_t p = 0; p < raw.nodes.size(); ++p) {
    const std::string_view key = raw.nodes[p];
    const std::size_t comma = key.find(',');
    t.node_bank[id[p]] = key.substr(0, comma);
    t.node_account[id[p]] = key.substr(comma + 1);
  }
  raw.nodes = Interner();

  std::array<std::uint8_t, 256> ccy{}, fmt{};
  t.ccy_dict = sorted_dict(raw.currencies, ccy);
  t.format_dict = sorted_dict(raw.formats, fmt);
  t.label = permuted(raw.label, keys);
  t.amount_paid = permuted(raw.amount_paid, keys);
  t.amount_received = permuted(raw.amount_received, keys);
  t.pay_ccy = permuted(raw.pay_ccy, keys);
  for (std::uint8_t& c : t.pay_ccy) c = ccy[c];
  t.recv_ccy = permuted(raw.recv_ccy, keys);
  for (std::uint8_t& c : t.recv_ccy) c = ccy[c];
  t.format = permuted(raw.format, keys);
  for (std::uint8_t& c : t.format) c = fmt[c];

  t.ts.resize(m);
  t.file_row.resize(m);
  for (std::size_t e = 0; e < m; ++e) {
    t.ts[e] = std::int32_t(keys[e] >> 32);
    t.file_row[e] = std::uint32_t(keys[e]);
  }
  return t;
}

Invariants count_invariants(const EdgeTable& t, const Splits& s, bool concatenated_keys) {
  std::uint64_t self_loops = 0, distinct = 0, fit = 0, val = 0, test = 0;
  for (std::size_t e = 0; e < t.m(); ++e) {
    self_loops += t.src[e] == t.dst[e];
    distinct += e == 0 || t.ts[e] != t.ts[e - 1];
    if (t.label[e]) ++(e < s.b_fv ? fit : e < s.b_test ? val : test);
  }
  Invariants c = {{"edges", t.m()},         {"nodes", t.n()},        {"self_loops", self_loops},
                  {"distinct_timestamps", distinct},                {"positives", fit + val + test},
                  {"b_fv", s.b_fv},         {"b_test", s.b_test},    {"fit_positives", fit},
                  {"val_positives", val},   {"test_positives", test}};
  if (concatenated_keys) {
    std::unordered_set<std::string> keys;
    keys.reserve(t.n());
    for (std::size_t x = 0; x < t.n(); ++x) keys.insert(t.node_bank[x] + t.node_account[x]);
    c["concatenated_keys"] = keys.size();
  }
  return c;
}

NodeTableCounts resolve_node_tables(const EdgeTable& t, std::span<const std::pair<std::string, std::string>> rows,
                                    std::size_t examples) {
  using Key = std::pair<std::string_view, std::string_view>;
  std::set<Key> accounts, listed;
  for (std::size_t x = 0; x < t.n(); ++x) accounts.insert({t.node_bank[x], t.node_account[x]});
  NodeTableCounts c;
  for (const auto& [bank, account] : rows) {
    listed.insert({bank, account});
    if (accounts.count({bank, account})) continue;
    ++c.unmatched_node_rows;
    if (c.unmatched_examples.size() < examples) c.unmatched_examples.push_back(bank + "," + account);
  }
  for (std::size_t x = 0; x < t.n(); ++x) {
    if (listed.count({t.node_bank[x], t.node_account[x]})) continue;
    ++c.unresolved_accounts;
    if (c.unresolved_examples.size() < examples) c.unresolved_examples.push_back(t.node_bank[x] + "," + t.node_account[x]);
  }
  return c;
}

std::vector<std::string> check_invariants(const Invariants& got, const Invariants& pinned) {
  std::vector<std::string> bad;
  for (const auto& [k, v] : got) {
    const auto it = pinned.find(k);
    if (it == pinned.end()) bad.push_back(k + ": counted " + std::to_string(v) + ", not pinned");
    else if (it->second != v)
      bad.push_back(k + ": counted " + std::to_string(v) + ", pinned " + std::to_string(it->second));
  }
  for (const auto& [k, v] : pinned)
    if (!got.count(k)) bad.push_back(k + ": pinned " + std::to_string(v) + ", not counted for this dataset");
  return bad;
}

std::string check_file_row(std::span<const std::uint32_t> file_row, std::span<const std::int32_t> ts) {
  const std::size_t m = ts.size();
  if (file_row.size() != m) return "file_row has " + std::to_string(file_row.size()) + " entries, m = " + std::to_string(m);
  std::vector<bool> seen(m, false);
  for (std::size_t e = 0; e < m; ++e) {
    if (file_row[e] >= m || seen[file_row[e]])
      return "file_row is not a permutation of 0..m-1 (eid " + std::to_string(e) + ")";
    seen[file_row[e]] = true;
    if (e > 0 && !(ts[e - 1] < ts[e] || (ts[e - 1] == ts[e] && file_row[e - 1] < file_row[e])))
      return "file_row does not reproduce the stable sort at eid " + std::to_string(e);
  }
  return "";
}

}  // namespace blz

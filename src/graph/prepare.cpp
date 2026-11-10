#include "graph/prepare.hpp"

#include <algorithm>
#include <map>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace blz {
namespace {

// The distinct values of the columns, sorted by string.
std::vector<std::string> make_dict(std::initializer_list<const std::vector<std::string_view>*> cols) {
  std::map<std::string_view, int> seen;
  for (auto* c : cols)
    for (auto v : *c) seen.emplace(v, 0);
  if (seen.size() > 256) throw std::runtime_error("dictionary has more than 256 entries");
  std::vector<std::string> dict;
  for (auto& [k, _] : seen) dict.emplace_back(k);
  return dict;
}

std::uint8_t code(const std::vector<std::string>& dict, std::string_view v) {
  auto it = std::lower_bound(dict.begin(), dict.end(), v);
  return std::uint8_t(it - dict.begin());
}

}  // namespace

EdgeTable build_edge_table(const RawTable& raw) {
  const std::size_t m = raw.size();
  if (m >= (std::size_t(1) << 32)) throw std::runtime_error("too many edges for uint32 ids");
  std::vector<std::uint32_t> order(m);
  std::iota(order.begin(), order.end(), 0u);
  std::stable_sort(order.begin(), order.end(),
                   [&](std::uint32_t a, std::uint32_t b) { return raw.ts[a] < raw.ts[b]; });

  EdgeTable t;
  t.ccy_dict = make_dict({&raw.recv_ccy, &raw.pay_ccy});
  t.format_dict = make_dict({&raw.format});
  for (auto* v : {&t.src, &t.dst}) v->resize(m);
  t.ts.resize(m);
  t.label.resize(m);
  t.amount_paid.resize(m);
  t.amount_received.resize(m);
  t.pay_ccy.resize(m);
  t.recv_ccy.resize(m);
  t.format.resize(m);

  std::unordered_map<std::string, std::uint32_t> ids;
  ids.reserve(m / 4);
  std::string key;
  auto node = [&](std::string_view bank, std::string_view account) {
    key.assign(bank);
    key += ',';  // fields never contain ',', so the key is unambiguous
    key += account;
    auto [it, inserted] = ids.try_emplace(key, std::uint32_t(t.node_bank.size()));
    if (inserted) {
      t.node_bank.emplace_back(bank);
      t.node_account.emplace_back(account);
    }
    return it->second;
  };

  for (std::size_t e = 0; e < m; ++e) {
    const std::uint32_t r = order[e];
    t.src[e] = node(raw.from_bank[r], raw.from_account[r]);
    t.dst[e] = node(raw.to_bank[r], raw.to_account[r]);
    t.ts[e] = raw.ts[r];
    t.label[e] = raw.label[r];
    t.amount_paid[e] = raw.amount_paid[r];
    t.amount_received[e] = raw.amount_received[r];
    t.pay_ccy[e] = code(t.ccy_dict, raw.pay_ccy[r]);
    t.recv_ccy[e] = code(t.ccy_dict, raw.recv_ccy[r]);
    t.format[e] = code(t.format_dict, raw.format[r]);
  }
  return t;
}

StreamCounts count_stream(const EdgeTable& t) {
  StreamCounts c;
  c.edges = t.m();
  c.nodes = t.n();
  const std::uint64_t train = train_size(t.m());
  for (std::size_t e = 0; e < t.m(); ++e) {
    c.self_loops += t.src[e] == t.dst[e];
    if (t.label[e]) (e < train ? c.train_positives : c.test_positives) += 1;
  }
  c.positives = c.train_positives + c.test_positives;
  return c;
}

std::vector<std::string> check_hi_small_invariants(const EdgeTable& t) {
  std::vector<std::string> bad;
  for (std::size_t e = 1; e < t.m(); ++e)
    if (t.ts[e - 1] > t.ts[e]) {
      bad.push_back("stream not sorted by ts at eid " + std::to_string(e));
      break;
    }
  const StreamCounts c = count_stream(t);
  auto expect = [&](const char* what, std::uint64_t got, std::uint64_t want) {
    if (got != want)
      bad.push_back(std::string(what) + ": got " + std::to_string(got) + ", expected " +
                    std::to_string(want));
  };
  expect("edges", c.edges, 5'078'345);
  expect("nodes", c.nodes, 515'088);
  expect("positives", c.positives, 5'177);
  expect("self-loops", c.self_loops, 591'212);
  expect("test edges", c.edges - train_size(c.edges), 1'015'669);
  expect("test positives", c.test_positives, 1'797);
  expect("train positives", c.train_positives, 3'380);
  return bad;
}

}  // namespace blz

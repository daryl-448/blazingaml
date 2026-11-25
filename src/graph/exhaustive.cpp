#include "graph/exhaustive.hpp"

#include <algorithm>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace blz {

ExhaustiveBound parse_exhaustive_bound(const std::string& s) {
  ExhaustiveBound b;
  bool m = false, n = false, t = false;
  std::stringstream ss(s);
  std::string item;
  const auto bad = [&](const std::string& why) {
    return std::invalid_argument("bound '" + s + "': " + why + " (expected m=<edges>,n=<nodes>,t=<timestamps>)");
  };
  while (std::getline(ss, item, ',')) {
    const auto eq = item.find('=');
    if (eq == std::string::npos) throw bad("'" + item + "' is not key=value");
    const std::string k = item.substr(0, eq), v = item.substr(eq + 1);
    long long x = 0;
    try {
      std::size_t used = 0;
      x = std::stoll(v, &used);
      if (used != v.size()) throw std::invalid_argument(v);
    } catch (const std::exception&) {
      throw bad("'" + v + "' is not an integer");
    }
    if (x < 1) throw bad(k + " must be >= 1");
    if (k == "m") b.m_max = std::uint32_t(x), m = true;
    else if (k == "n") b.n_max = std::uint32_t(x), n = true;
    else if (k == "t") b.t = std::int32_t(x), t = true;
    else throw bad("unknown key '" + k + "'");
  }
  if (!m || !n || !t) throw bad("m, n and t are all needed");
  return b;
}

std::string bound_name(const ExhaustiveBound& b) {
  return "m=" + std::to_string(b.m_max) + ",n=" + std::to_string(b.n_max) + ",t=" + std::to_string(b.t);
}

ExhaustiveGraphs::ExhaustiveGraphs(ExhaustiveBound b) : b_(b) {
  // Enumeration, not a sampler: a bound past these is years of checking.
  if (b.m_max < 1 || b.m_max > 16 || b.n_max < 1 || b.n_max > 32 || b.t < 1 || b.t > 64)
    throw std::invalid_argument("exhaustive bound " + bound_name(b) + ": need 1 <= m <= 16, 1 <= n <= 32, 1 <= t <= 64");
  constexpr std::uint64_t kMax = std::uint64_t(1) << 62;
  const auto checked_mul = [&](std::uint64_t x, std::uint64_t y) {
    if (y != 0 && x > kMax / y) throw std::invalid_argument("exhaustive bound " + bound_name(b) + ": too many graphs");
    return x * y;
  };
  // completions_[r][u]: the ways to write r more node ids when u ids are in use: each one an old id
  // (u choices) or the next new one (if u < n_max).
  const std::uint32_t L = 2 * b.m_max;
  completions_.assign(L, std::vector<std::uint64_t>(b.n_max + 2, 0));
  for (std::uint32_t u = 0; u <= b.n_max; ++u) completions_[0][u] = 1;
  for (std::uint32_t r = 1; r < L; ++r)
    for (std::uint32_t u = 1; u <= b.n_max; ++u)
      completions_[r][u] = checked_mul(u, completions_[r - 1][u]) + (u < b.n_max ? completions_[r - 1][u + 1] : 0);
  // The ts sequences of length m: ts[0] = 0, non-decreasing, below t, in lexicographic order.
  ts_.resize(b.m_max);
  for (std::uint32_t m = 1; m <= b.m_max; ++m) {
    std::vector<std::int32_t> s(m, 0);
    for (;;) {
      ts_[m - 1].insert(ts_[m - 1].end(), s.begin(), s.end());
      std::uint32_t p = m;  // the last position that can grow; the rest restart at its value
      while (p > 1 && s[p - 1] == b.t - 1) --p;
      if (p == 1) break;
      ++s[p - 1];
      for (std::uint32_t q = p; q < m; ++q) s[q] = s[p - 1];
    }
  }
  first_.assign(b.m_max + 1, 0);
  for (std::uint32_t m = 1; m <= b.m_max; ++m) {
    const std::uint64_t strings = completions_[2 * m - 1][1];  // node 0 first, then 2m - 1 ids
    first_[m] = first_[m - 1] + checked_mul(strings, ts_[m - 1].size() / m);
    if (first_[m] > kMax) throw std::invalid_argument("exhaustive bound " + bound_name(b) + ": too many graphs");
  }
}

std::uint32_t ExhaustiveGraphs::edges(std::uint64_t i) const {
  if (i >= size()) throw std::out_of_range("exhaustive graph index past the bound");
  return std::uint32_t(std::upper_bound(first_.begin(), first_.end(), i) - first_.begin());
}

EdgeList ExhaustiveGraphs::graph(std::uint64_t i) const {
  const std::uint32_t m = edges(i);
  const std::uint64_t seqs = ts_[m - 1].size() / m, j = i - first_[m - 1];
  std::uint64_t rank = j / seqs;
  // Unrank the node string in lexicographic order: at each position the old ids 0..used-1, each
  // followed by its completions, then the new id `used`.
  const std::uint32_t L = 2 * m;
  std::vector<std::uint32_t> id(L, 0);
  std::uint32_t used = 1;
  for (std::uint32_t p = 1; p < L; ++p) {
    const std::uint64_t block = completions_[L - 1 - p][used];
    if (rank < used * block) {
      id[p] = std::uint32_t(rank / block);
      rank %= block;
    } else {
      rank -= used * block;
      id[p] = used++;
    }
  }
  EdgeList g;
  g.n = used;
  g.src.resize(m);
  g.dst.resize(m);
  for (std::uint32_t k = 0; k < m; ++k) g.src[k] = id[2 * k], g.dst[k] = id[2 * k + 1];
  const auto ts = ts_[m - 1].begin() + std::ptrdiff_t((j % seqs) * m);
  g.ts.assign(ts, ts + m);
  return g;
}

GraphUnion disjoint_union(const ExhaustiveGraphs& gs, std::uint64_t first, std::uint64_t last) {
  if (first > last || last > gs.size()) throw std::invalid_argument("disjoint_union: bad graph range");
  struct Edge {
    std::uint32_t src, dst;
    std::int32_t ts;
    std::uint64_t graph;
    std::uint32_t local;
  };
  std::vector<Edge> edges;
  std::uint32_t n = 0;
  for (std::uint64_t i = first; i < last; ++i) {
    const EdgeList g = gs.graph(i);
    for (std::uint32_t k = 0; k < g.src.size(); ++k) edges.push_back({n + g.src[k], n + g.dst[k], g.ts[k], i, k});
    n += g.n;
  }
  // Concatenated in (member, local eid) order, so a stable sort by ts merges by (ts, member, local eid).
  std::stable_sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) { return a.ts < b.ts; });
  GraphUnion u;
  u.g.n = n;
  std::vector<std::int64_t> last_local(last - first, -1);
  for (const Edge& e : edges) {
    // The locality argument's stream-visibility step: each member keeps its own eid order.
    std::int64_t& prev = last_local[e.graph - first];
    if (std::int64_t(e.local) != prev + 1) throw std::logic_error("disjoint_union: a member's eid order changed");
    prev = e.local;
    u.g.src.push_back(e.src);
    u.g.dst.push_back(e.dst);
    u.g.ts.push_back(e.ts);
    u.graph.push_back(e.graph);
    u.local.push_back(e.local);
  }
  return u;
}

}  // namespace blz

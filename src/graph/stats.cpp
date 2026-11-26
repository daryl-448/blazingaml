#include "graph/stats.hpp"

#include <algorithm>
#include <bit>
#include <map>
#include <vector>

#include "blz/edges.hpp"

namespace blz {
namespace {

// Per-node incident edge ids in eid order (counting sort), for one direction.
struct Adjacency {
  std::vector<std::uint64_t> off;
  std::vector<std::uint32_t> eid;
};

Adjacency build(std::span<const std::uint32_t> key, std::uint64_t n) {
  Adjacency a;
  a.off.assign(n + 1, 0);
  for (auto k : key) ++a.off[k + 1];
  for (std::uint64_t i = 0; i < n; ++i) a.off[i + 1] += a.off[i];
  a.eid.resize(key.size());
  std::vector<std::uint64_t> fill(a.off.begin(), a.off.end() - 1);
  for (std::size_t e = 0; e < key.size(); ++e) a.eid[fill[key[e]]++] = std::uint32_t(e);
  return a;
}

// For every edge, the number of edges in its node's list with ts in [t - delta, t] and eid <= it.
std::vector<std::uint64_t> occupancy(const Adjacency& a, std::span<const std::int32_t> ts,
                                     std::int64_t delta) {
  std::vector<std::uint64_t> occ(ts.size());
  const std::uint64_t n = a.off.size() - 1;
#pragma omp parallel for schedule(dynamic, 1024)
  for (std::uint64_t x = 0; x < n; ++x) {
    std::uint64_t lo = a.off[x];
    for (std::uint64_t j = a.off[x]; j < a.off[x + 1]; ++j) {
      const std::int64_t t = ts[a.eid[j]];
      while (ts[a.eid[lo]] < t - delta) ++lo;
      occ[a.eid[j]] = j - lo + 1;
    }
  }
  return occ;
}

// Edges and distinct neighbours per node, for one direction.
void degrees(const Adjacency& a, std::span<const std::uint32_t> other,
             std::vector<std::uint64_t>& deg, std::vector<std::uint64_t>& distinct) {
  const std::uint64_t n = a.off.size() - 1;
  deg.resize(n);
  distinct.resize(n);
#pragma omp parallel
  {
    std::vector<std::uint32_t> nb;
#pragma omp for schedule(dynamic, 1024)
    for (std::uint64_t x = 0; x < n; ++x) {
      nb.clear();
      for (std::uint64_t j = a.off[x]; j < a.off[x + 1]; ++j) nb.push_back(other[a.eid[j]]);
      std::sort(nb.begin(), nb.end());
      deg[x] = nb.size();
      distinct[x] = std::uint64_t(std::unique(nb.begin(), nb.end()) - nb.begin());
    }
  }
}

}  // namespace

nlohmann::json summarize(std::span<const std::uint64_t> values) {
  nlohmann::json j;
  j["count"] = values.size();
  if (values.empty()) return j;
  std::vector<std::uint64_t> v(values.begin(), values.end());
  std::sort(v.begin(), v.end());
  long double sum = 0;
  std::map<int, std::uint64_t> log2;
  for (auto x : v) {
    sum += x;
    ++log2[x == 0 ? -1 : int(std::bit_width(x)) - 1];
  }
  auto q = [&](double p) { return v[std::min(v.size() - 1, std::size_t(p * double(v.size())))]; };
  j["min"] = v.front();
  j["max"] = v.back();
  j["mean"] = double(sum / v.size());
  j["p50"] = q(0.5);
  j["p90"] = q(0.9);
  j["p99"] = q(0.99);
  j["p99_9"] = q(0.999);
  nlohmann::json h = nlohmann::json::object();
  for (auto [k, c] : log2) {
    const std::uint64_t lo = k < 0 ? 0 : std::uint64_t(1) << k, hi = k < 0 ? 0 : 2 * lo - 1;
    h[lo == hi ? std::to_string(lo) : std::to_string(lo) + "-" + std::to_string(hi)] = c;
  }
  j["log2_hist"] = h;
  return j;
}

nlohmann::json profile_stream(const StreamView& g) {
  const std::size_t m = g.src.size();
  nlohmann::json out;
  std::uint64_t positives = 0, self_loops = 0, test_pos = 0;
  const std::uint64_t train = g.splits.b_test;
  for (std::size_t e = 0; e < m; ++e) {
    positives += g.label[e];
    self_loops += g.src[e] == g.dst[e];
    if (e >= train) test_pos += g.label[e];
  }
  out["graph"] = {{"edges", m},           {"nodes", g.n},
                  {"positives", positives}, {"self_loops", self_loops},
                  {"train_edges", train},   {"test_edges", m - train},
                  {"train_positives", positives - test_pos}, {"test_positives", test_pos}};

  const Adjacency outs = build(g.src, g.n), ins = build(g.dst, g.n);
  {
    std::vector<std::uint64_t> deg, distinct;
    degrees(outs, g.dst, deg, distinct);
    out["degree"]["out_edges"] = summarize(deg);
    out["degree"]["out_distinct_neighbours"] = summarize(distinct);
    degrees(ins, g.src, deg, distinct);
    out["degree"]["in_edges"] = summarize(deg);
    out["degree"]["in_distinct_neighbours"] = summarize(distinct);
  }

  for (std::int64_t delta : {21'600, 86'400}) {
    auto& w = out["window_occupancy"][std::to_string(delta)];
    w["trigger_src_out_edges"] = summarize(occupancy(outs, g.ts, delta));
    w["trigger_dst_in_edges"] = summarize(occupancy(ins, g.ts, delta));
  }
  out["window_occupancy"]["note"] =
      "per trigger e=(u->v,t): u's out-edges (v's in-edges) with ts in [t - delta, t] and "
      "eid <= e, including e and self-loops";

  {
    std::vector<std::uint64_t> pairs(m);
    for (std::size_t e = 0; e < m; ++e) pairs[e] = (std::uint64_t(g.src[e]) << 32) | g.dst[e];
    std::sort(pairs.begin(), pairs.end());
    std::vector<std::uint64_t> mult;
    std::uint64_t self_pairs = 0;
    for (std::size_t i = 0; i < m;) {
      std::size_t j = i;
      while (j < m && pairs[j] == pairs[i]) ++j;
      mult.push_back(j - i);
      self_pairs += (pairs[i] >> 32) == (pairs[i] & 0xffffffffu);
      i = j;
    }
    out["pairs"] = {{"distinct_pairs", mult.size()},
                    {"self_loop_pairs", self_pairs},
                    {"multiplicity", summarize(mult)}};
  }

  {
    std::vector<std::uint64_t> group;
    std::uint64_t tied_with_prev = 0;
    for (std::size_t i = 0; i < m;) {
      std::size_t j = i;
      while (j < m && g.ts[j] == g.ts[i]) ++j;
      group.push_back(j - i);
      tied_with_prev += j - i - 1;
      i = j;
    }
    out["ts_ties"] = {{"distinct_timestamps", group.size()},
                      {"edges_tied_with_previous", tied_with_prev},
                      {"edges_per_timestamp", summarize(group)}};
  }

  {
    std::vector<std::uint8_t> has_loop(g.n, 0);
    for (std::size_t e = 0; e < m; ++e)
      if (g.src[e] == g.dst[e]) has_loop[g.src[e]] = 1;
    std::uint64_t nodes = 0, src_end = 0, dst_end = 0, either = 0;
    for (auto h : has_loop) nodes += h;
    for (std::size_t e = 0; e < m; ++e) {
      if (g.src[e] == g.dst[e]) continue;
      const bool a = has_loop[g.src[e]], b = has_loop[g.dst[e]];
      src_end += a;
      dst_end += b;
      either += a || b;
    }
    out["self_loops"] = {
        {"edges", self_loops},
        {"nodes_with_self_loop", nodes},
        {"non_loop_edges_src_has_self_loop", src_end},
        {"non_loop_edges_dst_has_self_loop", dst_end},
        {"non_loop_edges_either_endpoint_has_self_loop", either},
        {"note", "a node 'has a self-loop' if it has one anywhere in the stream"}};
  }
  return out;
}

}  // namespace blz

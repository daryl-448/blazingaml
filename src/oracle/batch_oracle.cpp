#include "oracle/batch_oracle.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <stdexcept>
#include <utility>

namespace blz {

// One batch [r.lo, r.hi) and one block: the rows being filled, and the credit rule.
struct BatchOracle::Ctx {
  Block b;
  const BlockParams& p;
  int k_max;
  BatchRange r;
  std::uint32_t* rows;  // (r.hi - r.lo) * nb
  std::size_t nb;
  std::uint64_t saturated = 0;

  // Instances credit each of their edges that is in the batch; earlier rows were already
  // returned by earlier transform() calls (A35).
  void credit(std::uint64_t f, int bin) {
    if (f >= r.lo && f < r.hi && add_saturating(rows[(f - r.lo) * nb + std::size_t(bin)])) ++saturated;
  }
};

BatchOracle::BatchOracle(const EdgeStream& s, BatchLayout layout, std::int32_t time_window)
    : s_(s), layout_(std::move(layout)), time_window_(time_window), out_(s.n), in_(s.n) {
  if (s.dst.size() != s.src.size() || s.ts.size() != s.src.size())
    throw std::invalid_argument("batch oracle: src/dst/ts sizes differ");
  if (s.src.size() > std::numeric_limits<std::uint32_t>::max())
    throw std::invalid_argument("batch oracle: too many edges for u32 edge ids");
  if (layout_.edges() != s.src.size()) throw std::invalid_argument("batch oracle: the layout does not cover the stream");
  if (time_window_ <= 0) throw std::invalid_argument("batch oracle: time_window must be > 0");
  for (std::uint32_t e = 0; e < s.src.size(); ++e) {
    if (s.src[e] >= s.n || s.dst[e] >= s.n) throw std::invalid_argument("batch oracle: node id >= n");
    if (e > 0 && s.ts[e] < s.ts[e - 1]) throw std::invalid_argument("batch oracle: ts not sorted");
    out_[s.src[e]].push_back(e);
    in_[s.dst[e]].push_back(e);
  }
}

std::uint64_t BatchOracle::mine(Block b, const BlockParams& p, int k_max, const TriggerSel& sel,
                                std::uint32_t* out) const {
  if (k_max < 2) throw std::invalid_argument("batch oracle: k_max < 2");
  if (!is_v1_block(b)) throw std::invalid_argument("batch oracle: block " + std::string(block_name(b)) + " has no batch semantics");
  return run_batches(layout_, s_.ts, time_window_, sel, p.bins.size(), out,
                     [&](const BatchRange& r, std::uint32_t* rows) { return mine_batch(b, p, k_max, r, rows); });
}

std::uint64_t BatchOracle::mine_batch(Block b, const BlockParams& p, int k_max, const BatchRange& r,
                                      std::uint32_t* rows) const {
  Ctx c{b, p, k_max, r, rows, p.bins.size()};
  // Every edge of the batch searches, self-loops included (S1 rules 1 and 3).
  for (std::uint64_t e = r.lo; e < r.hi; ++e) {
    if (b == Block::sg) sg(c, std::uint32_t(e));
    else if (b == Block::tcyc) tcyc(c, std::uint32_t(e));
    else fan_or_degree(c, std::uint32_t(e));
  }
  return c.saturated;
}

// S1 rule 1 (fan, degree): the instance of trigger e = (u -> v, t) is the set W of non-loop graph
// edges into v (out of u) with t - delta < ts <= t; edges of the batch with ts = t count whatever
// their eid. Fan's size is W's distinct neighbours, degree's |W|. Each edge of W in the batch is
// credited. A self-loop trigger computes its instance too; it is never in W itself.
void BatchOracle::fan_or_degree(Ctx& c, std::uint32_t e) const {
  const bool in = c.b == Block::fan_in || c.b == Block::deg_in;
  const bool fan = c.b == Block::fan_in || c.b == Block::fan_out;
  const std::int64_t t = s_.ts[e], lower = t - c.p.window;  // open: ts > lower
  const std::uint32_t self = in ? s_.dst[e] : s_.src[e];
  std::vector<std::uint32_t> w;
  std::set<std::uint32_t> nbrs;
  for (std::uint32_t f : in ? in_[self] : out_[self]) {
    if (f >= c.r.hi) break;  // the graph ends with the batch; the lists are in eid order
    if (s_.ts[f] <= lower || s_.ts[f] > t || s_.src[f] == s_.dst[f]) continue;
    w.push_back(f);
    nbrs.insert(in ? s_.src[f] : s_.dst[f]);
  }
  const int bin = bin_index(c.p.bins, fan ? nbrs.size() : w.size());
  if (bin < 0) return;
  for (std::uint32_t f : w) c.credit(f, bin);
}

// S1 rule 2 (tcyc): trigger e's instances are its stream-mode ones (§4.6 of v1: every other
// cycle edge has ts < t, so the batch's later edges never take part). Each instance (a node
// sequence v = c_0, ..., c_{k-1} = u) credits e and every window edge c_{i-1} -> c_i that some
// valid timestamp assignment uses. Written from the definition: the DFS keeps every feasible
// arrival time (as Oracle::tcyc does), and the participants are found from the full sets of
// feasible arrival and departure times per hop.
void BatchOracle::tcyc(Ctx& c, std::uint32_t e) const {
  using Ts = std::int64_t;
  const std::uint32_t u = s_.src[e], v = s_.dst[e];
  if (u == v) return;  // self-loops are not cycles
  const Ts t = s_.ts[e], lower = t - c.p.window;  // closed: ts >= lower
  // A list's window edges: lower <= ts < t. They all precede the trigger, so they are in the graph.
  auto window = [&](const std::vector<std::uint32_t>& list) {
    std::vector<std::uint32_t> w;
    for (std::uint32_t f : list) {
      if (s_.ts[f] >= t) break;
      if (s_.ts[f] >= lower) w.push_back(f);
    }
    return w;
  };

  // ld[y]: the latest time a temporal walk inside the window can leave y and still reach u before
  // t (Oracle::tcyc's pruning; it does not change the result).
  std::map<std::uint32_t, Ts> ld;
  {
    std::priority_queue<std::pair<Ts, std::uint32_t>> pq;
    ld[u] = t;
    pq.push({t, u});
    while (!pq.empty()) {
      auto [l, y] = pq.top();
      pq.pop();
      if (ld[y] != l) continue;
      for (std::uint32_t f : window(in_[y])) {
        if (s_.ts[f] >= l) break;
        const std::uint32_t x = s_.src[f];
        if (x == y) continue;
        auto it = ld.find(x);
        if (it == ld.end() || it->second < s_.ts[f]) {
          ld[x] = s_.ts[f];
          pq.push({s_.ts[f], x});
        }
      }
    }
  }
  if (!ld.contains(v)) return;

  auto credit_cycle = [&](const std::vector<std::uint32_t>& seq) {  // seq = c_0 = v, ..., c_{k-1} = u
    const int bin = bin_index(c.p.bins, seq.size());
    if (bin < 0) return;
    const std::size_t hops = seq.size() - 1;
    std::vector<std::vector<std::uint32_t>> h(hops);
    for (std::size_t i = 0; i < hops; ++i)
      for (std::uint32_t f : window(out_[seq[i]]))
        if (s_.dst[f] == seq[i + 1]) h[i].push_back(f);
    // arr[i]: the arrival times at c_{i+1} of valid prefixes ending with hop i; dep[i]: the times
    // of hop i from which a valid suffix reaches u.
    std::vector<std::set<Ts>> arr(hops), dep(hops);
    auto after_some = [](const std::set<Ts>& xs, Ts x) { return !xs.empty() && *xs.begin() < x; };
    auto before_some = [](const std::set<Ts>& xs, Ts x) { return !xs.empty() && *xs.rbegin() > x; };
    for (std::size_t i = 0; i < hops; ++i)
      for (std::uint32_t f : h[i])
        if (i == 0 || after_some(arr[i - 1], s_.ts[f])) arr[i].insert(s_.ts[f]);
    for (std::size_t i = hops; i-- > 0;)
      for (std::uint32_t f : h[i])
        if (i == hops - 1 || before_some(dep[i + 1], s_.ts[f])) dep[i].insert(s_.ts[f]);
    c.credit(e, bin);
    for (std::size_t i = 0; i < hops; ++i)
      for (std::uint32_t f : h[i])
        if ((i == 0 || after_some(arr[i - 1], s_.ts[f])) && (i == hops - 1 || before_some(dep[i + 1], s_.ts[f])))
          c.credit(f, bin);
  };

  std::vector<std::uint32_t> path = {v};
  auto dfs = [&](auto& self, const std::vector<Ts>& arrivals) -> void {
    std::map<std::uint32_t, std::vector<Ts>> next;
    for (std::uint32_t f : window(out_[path.back()])) {
      const Ts t1 = s_.ts[f];
      if (std::any_of(arrivals.begin(), arrivals.end(), [&](Ts a) { return a < t1; })) next[s_.dst[f]].push_back(t1);
    }
    for (const auto& [y, ys] : next) {
      if (y == u) {
        if (int(path.size()) + 1 <= c.k_max) {
          std::vector<std::uint32_t> seq = path;
          seq.push_back(u);
          credit_cycle(seq);
        }
        continue;
      }
      if (int(path.size()) + 1 >= c.k_max) continue;
      if (std::find(path.begin(), path.end(), y) != path.end()) continue;  // simple cycles only
      auto it = ld.find(y);
      if (it == ld.end() || !std::any_of(ys.begin(), ys.end(), [&](Ts a) { return a < it->second; })) continue;
      path.push_back(y);
      self(self, ys);
      path.pop_back();
    }
  };
  dfs(dfs, {std::numeric_limits<Ts>::min()});
}

// S1 rule 3 (SG). Trigger e = (a -> b, t), self-loops included, delta = the SG window. Views of
// the graph G (the edges of earlier batches with ts >= kept_from, plus the batch's own):
//   up  = G's edges with ts >= t - delta (no upper bound: the batch's later edges are in it);
//   win = up's edges with ts <= t.
// Pairs: gather (s, w = b, excluded x = w) for each s != a, b with s -> a in up; scatter
// (s = a, w, x = s) for each w != a, b with b -> w in up. I = {y != x : s -> y and y -> w in win}.
// When |I| >= 2 the pair is reported, unless a batch edge f with ts > t suppresses it:
//   gather:  f = y -> w (y != x) with s -> y in G,   or f = s -> y (y != x) with y -> w in up;
//   scatter: f = s -> y (y != x) with y -> w in G,   or f = y -> w (y != x) with s -> y in up.
// A report credits each batch edge of win that is s -> y or y -> w with y in I.
void BatchOracle::sg(Ctx& c, std::uint32_t e) const {
  using Ts = std::int64_t;
  constexpr Ts kNone = std::numeric_limits<Ts>::max();
  const std::uint32_t a = s_.src[e], b = s_.dst[e];
  const Ts t = s_.ts[e], lower = t - c.p.window;  // closed
  const std::uint64_t hi = c.r.hi;
  auto in_range = [&](std::uint32_t f, Ts from, Ts to) { return f < hi && s_.ts[f] >= from && s_.ts[f] <= to; };
  // Whether the graph has an edge x -> y with from <= ts <= to; scans the shorter list.
  auto exists = [&](std::uint32_t x, std::uint32_t y, Ts from, Ts to) {
    const auto& lx = out_[x];
    const auto& ly = in_[y];
    if (lx.size() <= ly.size())
      return std::any_of(lx.begin(), lx.end(), [&](std::uint32_t f) { return s_.dst[f] == y && in_range(f, from, to); });
    return std::any_of(ly.begin(), ly.end(), [&](std::uint32_t f) { return s_.src[f] == x && in_range(f, from, to); });
  };
  auto in_nbrs = [&](std::uint32_t x, Ts from, Ts to) {
    std::set<std::uint32_t> r;
    for (std::uint32_t f : in_[x])
      if (in_range(f, from, to)) r.insert(s_.src[f]);
    return r;
  };
  auto out_nbrs = [&](std::uint32_t x, Ts from, Ts to) {
    std::set<std::uint32_t> r;
    for (std::uint32_t f : out_[x])
      if (in_range(f, from, to)) r.insert(s_.dst[f]);
    return r;
  };
  struct Pair {
    std::uint32_t s, w, x;
    bool gather;
  };
  std::vector<Pair> pairs;
  for (std::uint32_t s : in_nbrs(a, lower, kNone))
    if (s != a && s != b) pairs.push_back({s, b, b, true});
  for (std::uint32_t w : out_nbrs(b, lower, kNone))
    if (w != a && w != b) pairs.push_back({a, w, a, false});

  for (const Pair& pr : pairs) {
    std::set<std::uint32_t> inter;
    const std::set<std::uint32_t> so = out_nbrs(pr.s, lower, t), wi = in_nbrs(pr.w, lower, t);
    for (std::uint32_t y : so)
      if (y != pr.x && wi.contains(y)) inter.insert(y);
    const int bin = bin_index(c.p.bins, inter.size());
    if (bin < 0) continue;
    bool suppressed = false;
    for (std::uint64_t f = c.r.lo; f < hi && !suppressed; ++f) {
      if (s_.ts[f] <= t) continue;
      const std::uint32_t fs = s_.src[f], fd = s_.dst[f];
      if (fd == pr.w && fs != pr.x)  // y -> w, y = fs
        suppressed = exists(pr.s, fs, pr.gather ? c.r.kept_from : lower, kNone);
      if (!suppressed && fs == pr.s && fd != pr.x)  // s -> y, y = fd
        suppressed = exists(fd, pr.w, pr.gather ? lower : c.r.kept_from, kNone);
    }
    if (suppressed) continue;
    for (std::uint64_t f = c.r.lo; f < hi; ++f)
      if (in_range(std::uint32_t(f), lower, t) &&
          ((s_.src[f] == pr.s && inter.contains(s_.dst[f])) || (s_.dst[f] == pr.w && inter.contains(s_.src[f]))))
        c.credit(f, bin);
  }
}

}  // namespace blz

#include "oracle/oracle.hpp"

#include <algorithm>
#include <atomic>
#include <iterator>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <stdexcept>
#include <utility>

namespace blz {

Oracle::Oracle(const EdgeStream& s, Visibility vis) : s_(s), vis_(vis), out_(s.n), in_(s.n) {
  if (s.dst.size() != s.src.size() || s.ts.size() != s.src.size())
    throw std::invalid_argument("oracle: src/dst/ts sizes differ");
  if (s.src.size() > std::numeric_limits<std::uint32_t>::max())
    throw std::invalid_argument("oracle: too many edges for u32 edge ids");
  for (std::uint32_t e = 0; e < s.src.size(); ++e) {
    if (s.src[e] >= s.n || s.dst[e] >= s.n) throw std::invalid_argument("oracle: node id >= n");
    if (e > 0 && s.ts[e] < s.ts[e - 1]) throw std::invalid_argument("oracle: ts not sorted");
    out_[s.src[e]].push_back(e);
    in_[s.dst[e]].push_back(e);
  }
}

// The part of a node's list inside W_P(e) (SPEC §4.1): visible (eid <= e under stream
// visibility, ts <= t under timestamp visibility, SPEC v2 §6.1), and ts > t - delta (open) or
// ts >= t - delta (closed). Lists are in eid order, hence also in ts order (§3.3).
Oracle::Window Oracle::window(const std::vector<std::uint32_t>& list, const Trigger& tr) const {
  const std::uint32_t* b = list.data();
  const std::uint32_t* e = b + list.size();
  const std::uint32_t* hi = vis_ == Visibility::stream
                                ? std::upper_bound(b, e, tr.e)
                                : std::partition_point(b, e, [&](std::uint32_t x) { return s_.ts[x] <= tr.t; });
  const std::uint32_t* lo = std::partition_point(b, hi, [&](std::uint32_t x) {
    return tr.closed ? s_.ts[x] < tr.lo_ts : s_.ts[x] <= tr.lo_ts;
  });
  return {lo, hi};
}

// Whether some window edge x -> y exists. Scans the shorter of x's out-window and y's in-window.
bool Oracle::has_edge(std::uint32_t x, std::uint32_t y, const Trigger& tr) const {
  const Window wo = window(out_[x], tr), wi = window(in_[y], tr);
  if (wo.size() <= wi.size()) return std::any_of(wo.begin, wo.end, [&](auto f) { return s_.dst[f] == y; });
  return std::any_of(wi.begin, wi.end, [&](auto f) { return s_.src[f] == x; });
}

// SPEC §4.3, §4.4.
void Oracle::fan_or_degree(Block b, const Trigger& tr, std::vector<std::uint64_t>& sizes) const {
  const bool in = b == Block::fan_in || b == Block::deg_in;
  const std::uint32_t self = in ? tr.v : tr.u;
  const Window w = window(in ? in_[self] : out_[self], tr);
  std::set<std::uint32_t> nbrs;
  std::uint64_t edges = 0;
  for (auto f = w.begin; f != w.end; ++f) {
    const std::uint32_t x = in ? s_.src[*f] : s_.dst[*f];
    if (x == self) continue;
    nbrs.insert(x);
    ++edges;
  }
  sizes.push_back(b == Block::fan_in || b == Block::fan_out ? nbrs.size() : edges);
}

// SPEC §4.5: gather role (s in N_in(u) \ {u, v}, w = v, I excludes w) plus, with `scatter`,
// the scatter role (s = u, w in N_out(v) \ {u, v}, I excludes s).
void Oracle::sg(const Trigger& tr, bool scatter, std::vector<std::uint64_t>& sizes) const {
  auto in_nbrs = [&](std::uint32_t x) {
    std::set<std::uint32_t> r;
    const Window w = window(in_[x], tr);
    for (auto f = w.begin; f != w.end; ++f) r.insert(s_.src[*f]);
    return r;
  };
  auto out_nbrs = [&](std::uint32_t x) {
    std::set<std::uint32_t> r;
    const Window w = window(out_[x], tr);
    for (auto f = w.begin; f != w.end; ++f) r.insert(s_.dst[*f]);
    return r;
  };
  auto emit = [&](std::uint32_t s, std::uint32_t w, std::uint32_t excluded) {
    std::uint64_t n = 0;
    for (std::uint32_t x : in_nbrs(w))  // I = N_out(s) ∩ N_in(w), minus `excluded`
      if (x != excluded && has_edge(s, x, tr)) ++n;
    sizes.push_back(n);
  };
  for (std::uint32_t s : in_nbrs(tr.u))
    if (s != tr.u && s != tr.v) emit(s, tr.v, tr.v);
  if (!scatter) return;
  for (std::uint32_t w : out_nbrs(tr.v))
    if (w != tr.u && w != tr.v) emit(tr.u, w, tr.u);
}

// SPEC v2 §6.3 (sg_ordered): for each source s in N_in(u) \ {u, v} with a window edge s -> u at
// ts < t, the instance (s, w = v) with
//   I_ord = { y != w : some window edges (s -> y)@a and (y -> w)@b have a < b }.
// Written from the definition: every witness pair is enumerated, so the earliest/latest-witness
// lowering of order constraints (§6.2) is tested, not assumed.
void Oracle::sg_ordered(const Trigger& tr, std::vector<std::uint64_t>& sizes) const {
  const std::uint32_t w = tr.v;
  std::map<std::uint32_t, std::vector<std::int64_t>> into_w;  // y -> the times of its edges y -> w
  const Window ww = window(in_[w], tr);
  for (auto f = ww.begin; f != ww.end; ++f) into_w[s_.src[*f]].push_back(s_.ts[*f]);
  std::set<std::uint32_t> sources;
  const Window wu = window(in_[tr.u], tr);
  for (auto f = wu.begin; f != wu.end; ++f)
    if (s_.ts[*f] < tr.t) sources.insert(s_.src[*f]);
  for (std::uint32_t s : sources) {
    if (s == tr.u || s == w) continue;
    std::map<std::uint32_t, std::vector<std::int64_t>> from_s;  // y -> the times of its edges s -> y
    const Window ws = window(out_[s], tr);
    for (auto f = ws.begin; f != ws.end; ++f) from_s[s_.dst[*f]].push_back(s_.ts[*f]);
    std::uint64_t n = 0;
    for (const auto& [y, as] : from_s) {
      if (y == w) continue;
      auto it = into_w.find(y);
      if (it == into_w.end()) continue;
      bool ordered = false;
      for (std::int64_t a : as)
        for (std::int64_t b : it->second) ordered = ordered || a < b;
      n += ordered;
    }
    sizes.push_back(n);
  }
}

// SPEC §4.6: distinct simple node sequences v = c_0, ..., c_{k-1} = u (2 <= k <= k_max) with
// window edges c_{i-1} -> c_i at t - delta <= t_1 < ... < t_{k-1} < t.
//
// The DFS runs over node sequences, so each sequence is reached at most once. For the current
// prefix it carries the full set F of arrival times at its last node over every valid timestamp
// assignment of the prefix, considering all parallel edges. It deliberately does not reduce F to
// its minimum, so the earliest-witness rule (§4.6) is not assumed here; the generated backends
// rely on it and are tested against this. A sequence counts when its final hop into u has a
// feasible timestamp.
//
// The search is pruned, without changing the result, by ld[y]: the latest time a temporal walk
// inside the window can leave y and still reach u before t. A prefix ending at y with arrival
// times F can only be extended to u if some f in F has f < ld[y].
//
// Every hop other than the trigger needs ts < t, so timestamp visibility, which only adds edges
// at ts = t, cannot change the result (SPEC v2 §6.1); the metamorphic test checks it.
void Oracle::tcyc(const Trigger& tr, int k_max, std::vector<std::uint64_t>& sizes) const {
  using Ts = std::int64_t;
  std::map<std::uint32_t, Ts> ld;
  {
    std::priority_queue<std::pair<Ts, std::uint32_t>> pq;  // latest first
    ld[tr.u] = tr.t;
    pq.push({tr.t, tr.u});
    while (!pq.empty()) {
      auto [l, y] = pq.top();
      pq.pop();
      if (ld[y] != l) continue;  // stale
      const Window w = window(in_[y], tr);
      for (auto f = w.begin; f != w.end && s_.ts[*f] < l; ++f) {
        const std::uint32_t x = s_.src[*f];
        if (x == y) continue;
        auto it = ld.find(x);
        if (it == ld.end() || it->second < s_.ts[*f]) {
          ld[x] = s_.ts[*f];
          pq.push({s_.ts[*f], x});
        }
      }
    }
  }
  if (!ld.contains(tr.v)) return;

  std::vector<std::uint32_t> path = {tr.v};
  auto dfs = [&](auto& self, const std::vector<Ts>& arrivals) -> void {
    // Feasible arrival times at each distinct out-neighbour y of the prefix's last node.
    std::map<std::uint32_t, std::vector<Ts>> next;
    const Window w = window(out_[path.back()], tr);
    for (auto f = w.begin; f != w.end && s_.ts[*f] < tr.t; ++f) {
      const Ts t1 = s_.ts[*f];
      if (std::any_of(arrivals.begin(), arrivals.end(), [&](Ts a) { return a < t1; }))
        next[s_.dst[*f]].push_back(t1);
    }
    for (const auto& [y, ys] : next) {
      if (y == tr.u) {
        if (int(path.size()) + 1 <= k_max) sizes.push_back(path.size() + 1);
        continue;
      }
      if (int(path.size()) + 1 >= k_max) continue;
      if (std::find(path.begin(), path.end(), y) != path.end()) continue;  // simple cycles only
      auto it = ld.find(y);
      if (it == ld.end() ||
          !std::any_of(ys.begin(), ys.end(), [&](Ts a) { return a < it->second; }))
        continue;
      path.push_back(y);
      self(self, ys);
      path.pop_back();
    }
  };
  dfs(dfs, {std::numeric_limits<Ts>::min()});
}

// SPEC v2 §8 S3 (lcyc): the simple cycles through the trigger u -> v of length 2..k_max in any
// time order. A cycle is a node sequence v = c_0, c_1, ..., c_h, u of distinct nodes, h + 2 <=
// k_max, with an edge c_i -> c_{i+1} for every hop and c_h -> u, each in the window with ts < t
// (an edge tied with the trigger never takes part, A49). Parallel edges collapse: one cycle per
// node sequence. Each extension of the path to a new node is one binding; more than `limit` of
// them abort the trigger (the same count as the generated code's).
bool Oracle::lcyc(const Trigger& tr, int k_max, std::uint64_t limit, std::vector<std::uint64_t>& sizes) const {
  std::vector<std::uint32_t> path = {tr.u, tr.v};  // u closes the cycle and is never a hop
  std::uint64_t work = 0;
  bool aborted = false;
  const auto successors = [&](std::uint32_t x) {
    std::set<std::uint32_t> ys;
    const Window w = window(out_[x], tr);
    for (auto f = w.begin; f != w.end && s_.ts[*f] < tr.t; ++f) ys.insert(s_.dst[*f]);
    return ys;
  };
  const auto dfs = [&](const auto& self, std::uint32_t x, int hops) -> void {  // hops: the nodes c_1.. so far
    for (std::uint32_t y : successors(x)) {
      if (aborted) return;
      if (y == tr.u) {
        sizes.push_back(std::uint64_t(hops) + 2);
        continue;
      }
      if (std::find(path.begin(), path.end(), y) != path.end() || hops + 1 > k_max - 2) continue;
      if (++work > limit) {
        aborted = true;
        return;
      }
      path.push_back(y);
      self(self, y, hops + 1);
      path.pop_back();
    }
  };
  dfs(dfs, tr.v, 0);
  return !aborted;
}

// SPEC v2 §8 S4, from the definitions, over the window's distinct neighbours (an open lower bound,
// as fan): fan_oneway_out = |N_out(u) \ N_in(u) \ {u}|, the counterparties u paid that paid
// nothing back to u; fan_total = |N_in(v) ∪ N_out(v) \ {v}|, every counterparty of v.
void Oracle::fan_sets(Block b, const Trigger& tr, std::vector<std::uint64_t>& sizes) const {
  const std::uint32_t x = b == Block::fan_total ? tr.v : tr.u;
  std::set<std::uint32_t> in, out;
  const Window wi = window(in_[x], tr), wo = window(out_[x], tr);
  for (auto f = wi.begin; f != wi.end; ++f) in.insert(s_.src[*f]);
  for (auto f = wo.begin; f != wo.end; ++f) out.insert(s_.dst[*f]);
  in.erase(x);
  out.erase(x);
  std::vector<std::uint32_t> r;
  if (b == Block::fan_oneway_out) std::set_difference(out.begin(), out.end(), in.begin(), in.end(), std::back_inserter(r));
  else std::set_union(in.begin(), in.end(), out.begin(), out.end(), std::back_inserter(r));
  sizes.push_back(r.size());
}

// SPEC v2 §8 S5 (tchain): the distinct simple node sequences c_{k-1}, ..., c_1, c_0 = u ending in
// the trigger u -> v (2 <= k <= k_max edges, v included, all nodes distinct) with window edges
// c_i -> c_{i-1} at t - delta <= t_{k-1} < ... < t_1 < t. counts[k] gets the number of sequences
// with k edges.
//
// The mirror of tcyc's search: the DFS runs backward over node sequences, so each is reached once.
// For the current prefix it carries the full set D of the times its head's chain edge can take
// over every valid timestamp assignment of the suffix, considering all parallel edges, and does
// not reduce D to its maximum, so the latest-witness rule that the generated code relies on is
// tested, not assumed. An edge x -> head extends the prefix when its time is below some d in D.
// Every hop needs ts < t, so timestamp visibility cannot change the result (as for tcyc).
void Oracle::tchain(const Trigger& tr, int k_max, std::vector<std::uint64_t>& counts) const {
  using Ts = std::int64_t;
  std::vector<std::uint32_t> path = {tr.v, tr.u};
  auto dfs = [&](auto& self, const std::vector<Ts>& departures) -> void {
    // The feasible times of an edge into the head from each distinct in-neighbour x.
    std::map<std::uint32_t, std::vector<Ts>> prev;
    const Window w = window(in_[path.back()], tr);
    for (auto f = w.begin; f != w.end; ++f) {
      const Ts t1 = s_.ts[*f];
      if (std::any_of(departures.begin(), departures.end(), [&](Ts d) { return t1 < d; }))
        prev[s_.src[*f]].push_back(t1);
    }
    for (const auto& [x, xs] : prev) {
      if (std::find(path.begin(), path.end(), x) != path.end()) continue;  // simple chains only
      ++counts[path.size()];  // path.size() nodes before x: path.size() edges with x
      if (int(path.size()) + 1 > k_max) continue;
      path.push_back(x);
      self(self, xs);
      path.pop_back();
    }
  };
  if (k_max >= 2) dfs(dfs, {tr.t});
}

// SPEC v2 §8 S5 (gs), from the definition: the trigger x -> w out of hub x completes one instance
// of size min(|A|, |B|), A = the distinct sources s that sent to x before t (a window edge
// s -> x with ts < t), s not in {x, w}; B = the distinct sinks d that x paid in the window, d != x
// (w included). An open lower bound, as fan.
void Oracle::gs(const Trigger& tr, std::vector<std::uint64_t>& sizes) const {
  const std::uint32_t x = tr.u, w = tr.v;
  std::set<std::uint32_t> a, b;
  const Window wi = window(in_[x], tr), wo = window(out_[x], tr);
  for (auto f = wi.begin; f != wi.end; ++f)
    if (s_.ts[*f] < tr.t && s_.src[*f] != x && s_.src[*f] != w) a.insert(s_.src[*f]);
  for (auto f = wo.begin; f != wo.end; ++f)
    if (s_.dst[*f] != x) b.insert(s_.dst[*f]);
  sizes.push_back(std::min(a.size(), b.size()));
}

std::uint64_t Oracle::mine_one(Block b, const BlockParams& p, int k_max, std::uint64_t e, std::uint32_t* row,
                               bool gather_only, bool* aborted) const {
  const Trigger tr{std::uint32_t(e), s_.src[e], s_.dst[e], s_.ts[e], std::int64_t(s_.ts[e]) - p.window,
                   closed_lower_bound(b)};
  if (tr.u == tr.v) return 0;  // self-loop triggers emit nothing (§4.1)
  std::vector<std::uint64_t> sizes;
  switch (b) {
    case Block::sg: sg(tr, !gather_only, sizes); break;
    case Block::tcyc: tcyc(tr, k_max, sizes); break;
    case Block::sg_ord: sg_ordered(tr, sizes); break;
    case Block::fan_oneway_out:
    case Block::fan_total: fan_sets(b, tr, sizes); break;
    case Block::gs: gs(tr, sizes); break;
    case Block::tchain: {  // counted per length: a hub can end millions of chains
      std::vector<std::uint64_t> counts(std::size_t(std::max(k_max, 1)) + 1, 0);
      tchain(tr, k_max, counts);
      std::uint64_t saturated = 0;
      for (std::size_t k = 0; k < counts.size(); ++k) {
        const int i = bin_index(p.bins, k);
        if (i < 0 || counts[k] == 0) continue;
        const std::uint64_t room = UINT32_MAX - row[i], add = std::min<std::uint64_t>(room, counts[k]);
        row[i] += std::uint32_t(add);
        saturated += counts[k] - add;
      }
      return saturated;
    }
    case Block::lcyc:
      if (!lcyc(tr, k_max, p.work_limit, sizes)) {
        if (aborted) *aborted = true;
        return 0;
      }
      break;
    default: fan_or_degree(b, tr, sizes);
  }
  std::uint64_t saturated = 0;
  for (std::uint64_t sz : sizes) {
    const int i = bin_index(p.bins, sz);
    if (i >= 0 && add_saturating(row[i])) ++saturated;
  }
  return saturated;
}

namespace {

// The work limit's report (SPEC v2 §8 S3) names the smallest aborted eid, whatever the thread
// count: a trigger is skipped only if its eid is larger than an abort already seen.
void lower_to(std::atomic<std::uint64_t>& a, std::uint64_t e) {
  for (std::uint64_t v = a.load(); e < v && !a.compare_exchange_weak(v, e);) {}
}

}  // namespace

std::uint64_t Oracle::mine(Block b, const BlockParams& p, int k_max, std::uint64_t first,
                           std::uint64_t last, std::uint32_t* out) const {
  if (first > last || last > s_.src.size()) throw std::invalid_argument("oracle: bad trigger range");
  if (k_max < 2) throw std::invalid_argument("oracle: k_max < 2");
  const std::size_t nb = p.bins.size();
  std::uint64_t saturated = 0;
  std::atomic<std::uint64_t> aborted{UINT64_MAX};
#pragma omp parallel for schedule(dynamic, 256) reduction(+ : saturated)
  for (std::uint64_t e = first; e < last; ++e) {
    if (e > aborted.load(std::memory_order_relaxed)) continue;
    bool ab = false;
    saturated += mine_one(b, p, k_max, e, out + (e - first) * nb, false, &ab);
    if (ab) lower_to(aborted, e);
  }
  if (aborted != UINT64_MAX) throw WorkLimitExceeded("oracle, block " + std::string(block_name(b)), aborted, p.work_limit);
  return saturated;
}

std::uint64_t Oracle::mine(Block b, const BlockParams& p, int k_max, std::span<const std::uint32_t> eids,
                           std::uint32_t* out) const {
  if (k_max < 2) throw std::invalid_argument("oracle: k_max < 2");
  for (std::uint32_t e : eids)
    if (e >= s_.src.size()) throw std::invalid_argument("oracle: trigger eid >= m");
  const std::size_t nb = p.bins.size();
  std::uint64_t saturated = 0;
  std::atomic<std::uint64_t> aborted{UINT64_MAX};
#pragma omp parallel for schedule(dynamic, 256) reduction(+ : saturated)
  for (std::size_t r = 0; r < eids.size(); ++r) {
    if (eids[r] > aborted.load(std::memory_order_relaxed)) continue;
    bool ab = false;
    saturated += mine_one(b, p, k_max, eids[r], out + r * nb, false, &ab);
    if (ab) lower_to(aborted, eids[r]);
  }
  if (aborted != UINT64_MAX) throw WorkLimitExceeded("oracle, block " + std::string(block_name(b)), aborted, p.work_limit);
  return saturated;
}

std::uint64_t Oracle::mine_sg_gather(const BlockParams& p, std::uint64_t first, std::uint64_t last,
                                     std::uint32_t* out) const {
  if (first > last || last > s_.src.size()) throw std::invalid_argument("oracle: bad trigger range");
  const std::size_t nb = p.bins.size();
  std::uint64_t saturated = 0;
#pragma omp parallel for schedule(dynamic, 256) reduction(+ : saturated)
  for (std::uint64_t e = first; e < last; ++e)
    saturated += mine_one(Block::sg, p, 2, e, out + (e - first) * nb, true);
  return saturated;
}

}  // namespace blz

#include "oracle/oracle.hpp"

#include <algorithm>
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

std::uint64_t Oracle::mine_one(Block b, const BlockParams& p, int k_max, std::uint64_t e, std::uint32_t* row,
                               bool gather_only) const {
  const Trigger tr{std::uint32_t(e), s_.src[e], s_.dst[e], s_.ts[e], std::int64_t(s_.ts[e]) - p.window,
                   closed_lower_bound(b)};
  if (tr.u == tr.v) return 0;  // self-loop triggers emit nothing (§4.1)
  std::vector<std::uint64_t> sizes;
  switch (b) {
    case Block::sg: sg(tr, !gather_only, sizes); break;
    case Block::tcyc: tcyc(tr, k_max, sizes); break;
    case Block::sg_ord: sg_ordered(tr, sizes); break;
    default: fan_or_degree(b, tr, sizes);
  }
  std::uint64_t saturated = 0;
  for (std::uint64_t sz : sizes) {
    const int i = bin_index(p.bins, sz);
    if (i >= 0 && add_saturating(row[i])) ++saturated;
  }
  return saturated;
}

std::uint64_t Oracle::mine(Block b, const BlockParams& p, int k_max, std::uint64_t first,
                           std::uint64_t last, std::uint32_t* out) const {
  if (first > last || last > s_.src.size()) throw std::invalid_argument("oracle: bad trigger range");
  if (k_max < 2) throw std::invalid_argument("oracle: k_max < 2");
  const std::size_t nb = p.bins.size();
  std::uint64_t saturated = 0;
#pragma omp parallel for schedule(dynamic, 256) reduction(+ : saturated)
  for (std::uint64_t e = first; e < last; ++e) saturated += mine_one(b, p, k_max, e, out + (e - first) * nb);
  return saturated;
}

std::uint64_t Oracle::mine(Block b, const BlockParams& p, int k_max, std::span<const std::uint32_t> eids,
                           std::uint32_t* out) const {
  if (k_max < 2) throw std::invalid_argument("oracle: k_max < 2");
  for (std::uint32_t e : eids)
    if (e >= s_.src.size()) throw std::invalid_argument("oracle: trigger eid >= m");
  const std::size_t nb = p.bins.size();
  std::uint64_t saturated = 0;
#pragma omp parallel for schedule(dynamic, 256) reduction(+ : saturated)
  for (std::size_t r = 0; r < eids.size(); ++r) saturated += mine_one(b, p, k_max, eids[r], out + r * nb);
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

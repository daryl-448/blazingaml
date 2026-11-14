#include "runtime/batch.hpp"

#include <limits>
#include <queue>
#include <utility>

#include "blz/edges.hpp"

namespace blz {

BatchLayout batch_layout(std::uint64_t m, std::uint64_t size, std::span<const std::uint64_t> cuts) {
  if (size < 1) throw std::invalid_argument("batch layout: size must be >= 1");
  std::vector<std::uint64_t> stops;
  for (std::uint64_t c : cuts)
    if (c > 0 && c < m) stops.push_back(c);
  std::sort(stops.begin(), stops.end());
  stops.erase(std::unique(stops.begin(), stops.end()), stops.end());
  stops.push_back(m);
  BatchLayout l;
  std::uint64_t a = 0;
  for (std::uint64_t stop : stops) {
    for (; a < stop; a = std::min(a + size, stop)) l.starts.push_back(a);
  }
  l.starts.push_back(m);
  return l;
}

BatchLayout batch_layout(const BatchSpec& spec, std::uint64_t m, std::span<const std::int32_t> full_ts) {
  if (m > full_ts.size()) throw std::invalid_argument("batch layout: m > the stream's length");
  std::vector<std::uint64_t> cuts;
  if (spec.cut_at_splits) cuts = {fit_val_boundary(full_ts), train_size(full_ts.size())};
  return batch_layout(m, spec.size, cuts);
}

namespace {

using Ts = std::int64_t;

// One batch and one block on the indexed graph.
struct Ctx {
  const GraphView& g;
  const BlockParams& p;
  int k_max;
  BatchRange r;
  std::uint32_t* rows;
  std::size_t nb;
  std::uint64_t saturated = 0;

  std::uint32_t max_eid() const { return std::uint32_t(r.hi - 1); }  // the graph ends with the batch
  void credit(std::uint32_t f, int bin) {
    if (f >= r.lo && add_saturating(rows[(f - r.lo) * nb + std::size_t(bin)])) ++saturated;
  }
  // Positions of node x's list entries whose eid is in the batch.
  Range batch_entries(const TCsrView& l, std::uint32_t x) const {
    const std::uint32_t a = l.off[x], b = l.off[x + 1];
    return {r.lo == 0 ? a : l.first_eid_after(a, b, std::uint32_t(r.lo - 1)), l.first_eid_after(a, b, max_eid())};
  }
};

// S1 rule 1: the window range of v.in (u.out) with t - delta < ts <= t and eid < hi; the batch's
// entries are the range's tail, because a list is in eid order.
void fan_or_degree(Ctx& c, Block b, std::uint32_t e) {
  const bool in = b == Block::fan_in || b == Block::deg_in;
  const bool fan = b == Block::fan_in || b == Block::fan_out;
  const TCsrView& l = in ? c.g.in : c.g.out;
  const std::uint32_t self = in ? c.g.dst[e] : c.g.src[e];
  const Ts t = c.g.ts[e];
  const Range w = l.range(self, t - c.p.window + 1, t, c.max_eid());
  const auto keep = [&](std::uint32_t x) { return x != self; };
  const std::uint32_t size = fan ? count_distinct(l, w, keep) : count_edges(l, w, keep);
  const int bin = bin_index(c.p.bins, size);
  if (bin < 0) return;
  for (std::uint32_t j = w.hi; j > w.lo && l.eid[j - 1] >= c.r.lo; --j)
    if (l.nbr[j - 1] != self) c.credit(l.eid[j - 1], bin);
}

// Per-thread scratch for tcyc: the latest departures (ld) of the touched nodes.
struct CycScratch {
  std::vector<Ts> ld;
  std::vector<std::uint32_t> touched;
  std::vector<std::uint32_t> path;  // c_0 = v, ..., the current prefix
  std::vector<Ts> arrival;          // arrival[i]: the earliest arrival at path[i] (path[0]: none)
};

// S1 rule 2, on the index. The DFS walks node sequences with earliest witnesses (§4.6 of v1),
// pruned by ld. For a found sequence, hop i's earliest arrival a_i comes from the DFS and its
// latest departure l_i from a backward pass of pair_latest; an edge of hop i takes part in some
// valid assignment exactly when a_{i-1} < ts < l_{i+1}.
void tcyc(Ctx& c, CycScratch& cs, std::uint32_t e) {
  const GraphView& g = c.g;
  const std::uint32_t u = g.src[e], v = g.dst[e];
  if (u == v) return;
  const Ts t = g.ts[e], lower = t - c.p.window;
  if (cs.ld.size() < g.n) cs.ld.assign(g.n, kNoTs);
  for (std::uint32_t x : cs.touched) cs.ld[x] = kNoTs;
  cs.touched.clear();
  {
    std::priority_queue<std::pair<Ts, std::uint32_t>> pq;
    cs.ld[u] = t;
    cs.touched.push_back(u);
    pq.push({t, u});
    while (!pq.empty()) {
      auto [l, y] = pq.top();
      pq.pop();
      if (cs.ld[y] != l) continue;
      const Range w = g.in.range(y, lower, l - 1, kAnyEid);  // lower <= ts < l (< t): in the graph
      for (std::uint32_t j = w.lo; j < w.hi; ++j) {
        const std::uint32_t x = g.in.nbr[j];
        if (x == y) continue;
        if (cs.ld[x] == kNoTs) cs.touched.push_back(x);
        if (cs.ld[x] < g.in.ts[j]) {
          cs.ld[x] = g.in.ts[j];
          pq.push({g.in.ts[j], x});
        }
      }
    }
  }
  if (cs.ld[v] == kNoTs) return;

  std::vector<Ts> late;  // per found cycle: l_i
  auto credit_cycle = [&]() {
    // The sequence is cs.path (c_0 .. c_{k-2}) then u; hop i (0-based) is path[i] -> path[i+1] (or u).
    const std::size_t hops = cs.path.size();
    const int bin = bin_index(c.p.bins, hops + 1);
    if (bin < 0) return;
    auto node = [&](std::size_t i) { return i < cs.path.size() ? cs.path[i] : u; };
    late.assign(hops, kNoTs);
    Ts next = t;
    for (std::size_t i = hops; i-- > 0;) {
      late[i] = g.pout.pair_latest(node(i), node(i + 1), lower, next - 1, kAnyEid);
      next = late[i];
    }
    c.credit(e, bin);
    for (std::size_t i = 0; i < hops; ++i) {
      const Ts from = i == 0 ? lower : cs.arrival[i] + 1;  // ts > a_{i-1}: the arrival at node(i)
      const Ts to = i + 1 < hops ? late[i + 1] - 1 : t - 1;
      const Range w = g.out.range(node(i), from, to, kAnyEid);
      for (std::uint32_t j = w.hi; j > w.lo && g.out.eid[j - 1] >= c.r.lo; --j)
        if (g.out.nbr[j - 1] == node(i + 1)) c.credit(g.out.eid[j - 1], bin);
    }
  };

  cs.path.assign(1, v);
  cs.arrival.assign(1, std::numeric_limits<Ts>::min());
  auto dfs = [&](auto& self) -> void {
    const std::uint32_t x = cs.path.back();
    const Ts from = cs.path.size() == 1 ? lower : cs.arrival.back() + 1;
    const Range w = g.out.range(x, from, t - 1, kAnyEid);
    for (std::uint32_t j = w.lo; j < w.hi; ++j) {
      if (!g.out.distinct_first(j, w.lo)) continue;  // each neighbour at its earliest witness
      const std::uint32_t y = g.out.nbr[j];
      const Ts ty = g.out.ts[j];
      if (y == u) {
        if (int(cs.path.size()) + 1 <= c.k_max) credit_cycle();
        continue;
      }
      if (int(cs.path.size()) + 1 >= c.k_max) continue;
      if (std::find(cs.path.begin(), cs.path.end(), y) != cs.path.end()) continue;
      if (cs.ld[y] == kNoTs || !(ty < cs.ld[y])) continue;
      cs.path.push_back(y);
      cs.arrival.push_back(ty);
      self(self);
      cs.path.pop_back();
      cs.arrival.pop_back();
    }
  };
  dfs(dfs);
}

// S1 rule 3, on the index (see BatchOracle::sg for the rule). The batch's later edges of a node
// are a suffix of its batch entries, because the batch is in ts order.
void sg(Ctx& c, std::uint32_t e) {
  const GraphView& g = c.g;
  const std::uint32_t a = g.src[e], b = g.dst[e], me = c.max_eid();
  const Ts t = g.ts[e], lower = t - c.p.window;
  auto has = [&](std::uint32_t x, std::uint32_t y, Ts from, Ts to) { return g.pout.pair_exists(x, y, from, to, me) != kNoTs; };
  auto later = [&](const TCsrView& l, std::uint32_t x) {
    const Range r = c.batch_entries(l, x);
    return Range{l.first_ts_at_least(r.lo, r.hi, t + 1), r.hi};
  };
  auto pair = [&](std::uint32_t s, std::uint32_t w, std::uint32_t x, bool gather) {
    // |I|: the distinct window neighbours of the shorter side whose other edge exists.
    const Range so = g.out.range(s, lower, t, me), wi = g.in.range(w, lower, t, me);
    std::uint32_t size = 0;
    if (so.size() <= wi.size()) {
      for (std::uint32_t j = so.lo; j < so.hi; ++j)
        if (g.out.distinct_first(j, so.lo) && g.out.nbr[j] != x && has(g.out.nbr[j], w, lower, t)) ++size;
    } else {
      for (std::uint32_t j = wi.lo; j < wi.hi; ++j)
        if (g.in.distinct_first(j, wi.lo) && g.in.nbr[j] != x && has(s, g.in.nbr[j], lower, t)) ++size;
    }
    const int bin = bin_index(c.p.bins, size);
    if (bin < 0) return;
    const Range lw = later(g.in, w), ls = later(g.out, s);
    for (std::uint32_t j = lw.lo; j < lw.hi; ++j)  // f = y -> w
      if (const std::uint32_t y = g.in.nbr[j]; y != x && has(s, y, gather ? c.r.kept_from : lower, kNoUpper)) return;
    for (std::uint32_t j = ls.lo; j < ls.hi; ++j)  // f = s -> y
      if (const std::uint32_t y = g.out.nbr[j]; y != x && has(y, w, gather ? lower : c.r.kept_from, kNoUpper)) return;
    // Credits: the batch's window edges s -> y and y -> w with y in I. An edge s -> w is in both
    // lists, but only one side can hold it: y = w needs w != x, y = s needs s != x.
    const Range bs = c.batch_entries(g.out, s), bw = c.batch_entries(g.in, w);
    for (std::uint32_t j = bs.lo; j < bs.hi && g.out.ts[j] <= t; ++j)
      if (const std::uint32_t y = g.out.nbr[j]; g.out.ts[j] >= lower && y != x && has(y, w, lower, t))
        c.credit(g.out.eid[j], bin);
    for (std::uint32_t j = bw.lo; j < bw.hi && g.in.ts[j] <= t; ++j)
      if (const std::uint32_t y = g.in.nbr[j]; g.in.ts[j] >= lower && y != x && has(s, y, lower, t))
        c.credit(g.in.eid[j], bin);
  };
  const Range sa = g.in.range(a, lower, kNoUpper, me);  // gather: s -> a in up
  for (std::uint32_t j = sa.lo; j < sa.hi; ++j)
    if (const std::uint32_t s = g.in.nbr[j]; g.in.distinct_first(j, sa.lo) && s != a && s != b) pair(s, b, b, true);
  const Range wb = g.out.range(b, lower, kNoUpper, me);  // scatter: b -> w in up
  for (std::uint32_t j = wb.lo; j < wb.hi; ++j)
    if (const std::uint32_t w = g.out.nbr[j]; g.out.distinct_first(j, wb.lo) && w != a && w != b) pair(a, w, a, false);
}

}  // namespace

std::uint64_t mine_batch_cpu(const GraphView& g, Block b, const BlockParams& p, int k_max,
                             std::int32_t time_window, const BatchLayout& layout, const TriggerSel& sel,
                             std::uint32_t* out) {
  if (k_max < 2) throw std::invalid_argument("batch mode: k_max < 2");
  if (!is_v1_block(b)) throw std::invalid_argument("batch mode: block " + std::string(block_name(b)) + " has no batch semantics");
  if (layout.edges() != g.m) throw std::invalid_argument("batch mode: the layout does not cover the graph");
  const std::span<const std::int32_t> ts(g.ts, g.m);
  return run_batches(layout, ts, time_window, sel, p.bins.size(), out, [&](const BatchRange& r, std::uint32_t* rows) {
    thread_local CycScratch cs;
    Ctx c{g, p, k_max, r, rows, p.bins.size()};
    for (std::uint64_t e = r.lo; e < r.hi; ++e) {
      if (b == Block::sg) sg(c, std::uint32_t(e));
      else if (b == Block::tcyc) tcyc(c, cs, std::uint32_t(e));
      else fan_or_degree(c, b, std::uint32_t(e));
    }
    return c.saturated;
  });
}

}  // namespace blz

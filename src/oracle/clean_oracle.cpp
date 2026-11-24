// Built with -ffp-contract=off (CMakeLists.txt): the vertex statistics run R8's float64 operations
// with no fused multiply-add.
#include "oracle/clean_oracle.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <iterator>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>

namespace blz {

Quirks parse_quirks(std::string_view list) {
  Quirks q;
  if (list.empty()) return q;
  for (std::size_t at = 0;;) {
    const std::size_t comma = list.find(',', at);
    const std::string_view id = list.substr(at, comma == std::string_view::npos ? std::string_view::npos : comma - at);
    const bool all = id == "all";
    bool known = all;
    const auto turn_on = [&](std::string_view name, bool& flag) {
      if (all || id == name) flag = known = true;
    };
    turn_on("Q1", q.q1_sg_self_loops);
    turn_on("Q2", q.q2_vstats_self_loops);
    turn_on("Q3", q.q3_closed_windows);
    turn_on("Q4", q.q4_stream_visibility);
    turn_on("Q5", q.q5_strict_order);
    turn_on("Q6", q.q6_lcyc_drops_ties);
    turn_on("Q9", q.q9_vstats_gfp_window);
    turn_on("Q10", q.q10_vstats_gfp_arith);
    if (!known)
      throw std::invalid_argument("unknown quirk '" + std::string(id) +
                                  "' (known: Q1, Q2, Q3, Q4, Q5, Q6, Q9, Q10, all; SPEC v3 §3.1)");
    if (comma == std::string_view::npos) return q;
    at = comma + 1;
  }
}

CleanOracle::CleanOracle(const EdgeStream& s, std::span<const double> amount, Quirks q)
    : s_(s), amount_(amount), q_(q), out_(s.n), in_(s.n) {
  if (s.dst.size() != s.src.size() || s.ts.size() != s.src.size())
    throw std::invalid_argument("clean oracle: src/dst/ts sizes differ");
  if (s.src.size() > std::numeric_limits<std::uint32_t>::max())
    throw std::invalid_argument("clean oracle: too many edges for u32 edge ids");
  if (!amount.empty() && amount.size() != s.src.size())
    throw std::invalid_argument("clean oracle: one amount per edge is needed");
  for (std::uint32_t e = 0; e < s.src.size(); ++e) {
    if (s.src[e] >= s.n || s.dst[e] >= s.n) throw std::invalid_argument("clean oracle: node id >= n");
    if (e > 0 && s.ts[e] < s.ts[e - 1]) throw std::invalid_argument("clean oracle: ts not sorted");
    out_[s.src[e]].push_back(e);
    in_[s.dst[e]].push_back(e);
  }
}

// R2: W_P(e) = (t - delta_P, t]. Q3: the legacy closed bound [t - delta, t] for its closed blocks.
// Q1: sg and sg_ord keep self-loops.
CleanOracle::Trigger CleanOracle::pattern_trigger(Block b, const BlockParams& p, std::uint32_t e) const {
  return {e,
          s_.src[e],
          s_.dst[e],
          s_.ts[e],
          std::int64_t(s_.ts[e]) - p.window,
          q_.q3_closed_windows && closed_lower_bound(b),
          q_.q1_sg_self_loops && (b == Block::sg || b == Block::sg_ord)};
}

// R2 for vertex statistics: (t - W, t]. Q9: GFP's [ts[e - 1] - W, t], unbounded below for eid 0.
// Q2: self-loops count.
CleanOracle::Trigger CleanOracle::vstats_trigger(std::int32_t window, std::uint32_t e) const {
  Trigger tr{e, s_.src[e], s_.dst[e], s_.ts[e], std::int64_t(s_.ts[e]) - window, false, q_.q2_vstats_self_loops};
  if (q_.q9_vstats_gfp_window) {
    tr.lo = e == 0 ? std::numeric_limits<std::int64_t>::min() : std::int64_t(s_.ts[e - 1]) - window;
    tr.closed = true;
  }
  return tr;
}

// The entries of one node's list in the trigger's window. Lists are in eid order, hence in ts
// order (SPEC §3.3), so the visible part is a prefix and the window a contiguous run of it.
std::vector<std::uint32_t> CleanOracle::window(const std::vector<std::uint32_t>& list, const Trigger& tr) const {
  const auto visible = [&](std::uint32_t f) {
    return q_.q4_stream_visibility ? f <= tr.e : s_.ts[f] <= tr.t;  // R3: t' <= t. Q4: eid' <= e
  };
  const auto below = [&](std::uint32_t f) { return tr.closed ? s_.ts[f] < tr.lo : s_.ts[f] <= tr.lo; };  // R2
  const auto hi = std::partition_point(list.begin(), list.end(), visible);
  const auto lo = std::partition_point(list.begin(), hi, below);
  std::vector<std::uint32_t> w;
  for (auto f = lo; f != hi; ++f)
    if (tr.loops || s_.src[*f] != s_.dst[*f]) w.push_back(*f);  // R1: no self-loop in any window
  return w;
}

// R4: hops in non-decreasing time order, a <= b. Q5: a < b.
bool CleanOracle::in_order(std::int64_t a, std::int64_t b) const { return q_.q5_strict_order ? a < b : a <= b; }

// R7: fan_in = |N_in(v)|, fan_out = |N_out(u)| (distinct neighbours, the trigger's included);
// deg_in, deg_out = the window edges into v, out of u (parallel edges and the trigger included).
void CleanOracle::fan_or_degree(Block b, const Trigger& tr, Sizes& sizes) const {
  const bool in = b == Block::fan_in || b == Block::deg_in;
  const std::vector<std::uint32_t> w = window(in ? in_[tr.v] : out_[tr.u], tr);
  std::set<std::uint32_t> nbrs;
  for (std::uint32_t f : w) nbrs.insert(in ? s_.src[f] : s_.dst[f]);
  ++sizes[b == Block::fan_in || b == Block::fan_out ? nbrs.size() : w.size()];
}

// R7: fan_total = |N_in(v) ∪ N_out(v)|, fan_oneway_out = |N_out(u) \ N_in(u)|.
void CleanOracle::fan_sets(Block b, const Trigger& tr, Sizes& sizes) const {
  const std::uint32_t x = b == Block::fan_total ? tr.v : tr.u;
  std::set<std::uint32_t> in, out;
  for (std::uint32_t f : window(in_[x], tr)) in.insert(s_.src[f]);
  for (std::uint32_t f : window(out_[x], tr)) out.insert(s_.dst[f]);
  std::vector<std::uint32_t> r;
  if (b == Block::fan_total) std::set_union(in.begin(), in.end(), out.begin(), out.end(), std::back_inserter(r));
  else std::set_difference(out.begin(), out.end(), in.begin(), in.end(), std::back_inserter(r));
  ++sizes[r.size()];
}

// R7 (gs): for the trigger x -> w, A = the distinct sources s not in {x, w} with s -> x in the
// window and ts <= t (Q5: ts < t); B = fan_out's set; gs = min(|A|, |B|).
void CleanOracle::gs(const Trigger& tr, Sizes& sizes) const {
  const std::uint32_t x = tr.u, w = tr.v;
  std::set<std::uint32_t> a, b;
  for (std::uint32_t f : window(in_[x], tr))
    if (in_order(s_.ts[f], tr.t) && s_.src[f] != x && s_.src[f] != w) a.insert(s_.src[f]);
  for (std::uint32_t f : window(out_[x], tr)) b.insert(s_.dst[f]);
  ++sizes[std::min(a.size(), b.size())];
}

// R6a: an instance (s, w), s != w, of size |I(s, w)|, I(s, w) = (N_out(s) ∩ N_in(w)) \ {s, w} over
// the window. The gather role of a trigger x -> w reports (s, w) for each source s in
// N_in(x) \ {x, w}; with `scatter`, the scatter role of a trigger s -> x reports (s, w) for each
// sink w in N_out(x) \ {s, x}.
void CleanOracle::sg(const Trigger& tr, bool scatter, Sizes& sizes) const {
  const auto in_nbrs = [&](std::uint32_t x) {
    std::set<std::uint32_t> r;
    for (std::uint32_t f : window(in_[x], tr)) r.insert(s_.src[f]);
    return r;
  };
  const auto out_nbrs = [&](std::uint32_t x) {
    std::set<std::uint32_t> r;
    for (std::uint32_t f : window(out_[x], tr)) r.insert(s_.dst[f]);
    return r;
  };
  const auto size = [&](std::uint32_t s, std::uint32_t w, bool gather) {
    const std::set<std::uint32_t> from_s = out_nbrs(s), into_w = in_nbrs(w);
    std::set<std::uint32_t> i;
    std::set_intersection(from_s.begin(), from_s.end(), into_w.begin(), into_w.end(), std::inserter(i, i.end()));
    if (q_.q1_sg_self_loops) {
      i.erase(gather ? w : s);  // Q1: v1 excludes only the role's far endpoint (A10, A30)
    } else {
      i.erase(s);
      i.erase(w);
    }
    return i.size();
  };
  {
    const std::uint32_t x = tr.u, w = tr.v;  // gather: the trigger is x -> w
    for (std::uint32_t s : in_nbrs(x))
      if (s != x && s != w) ++sizes[size(s, w, true)];
  }
  if (!scatter) return;
  const std::uint32_t s = tr.u, x = tr.v;  // scatter: the trigger is s -> x
  for (std::uint32_t w : out_nbrs(x))
    if (w != s && w != x) ++sizes[size(s, w, false)];
}

// R6b: a trigger x -> w reports, for each source s not in {x, w} with an edge s -> x in the window
// (Q5: before t), the instance (s, w) of size |I(s, w)|,
//   I(s, w) = { y not in {s, w} : window edges s -> y at t_1 and y -> w at t_2, t_1 <= t_2 }.
// Every witness pair is enumerated.
void CleanOracle::sg_ord(const Trigger& tr, Sizes& sizes) const {
  const std::uint32_t x = tr.u, w = tr.v;
  std::map<std::uint32_t, std::vector<std::int64_t>> into_w;  // y -> the times of its edges y -> w
  for (std::uint32_t f : window(in_[w], tr)) into_w[s_.src[f]].push_back(s_.ts[f]);
  std::set<std::uint32_t> sources;
  for (std::uint32_t f : window(in_[x], tr))
    if (in_order(s_.ts[f], tr.t)) sources.insert(s_.src[f]);
  for (std::uint32_t s : sources) {
    if (s == x || s == w) continue;
    std::map<std::uint32_t, std::vector<std::int64_t>> from_s;  // y -> the times of its edges s -> y
    for (std::uint32_t f : window(out_[s], tr)) from_s[s_.dst[f]].push_back(s_.ts[f]);
    std::uint64_t n = 0;
    for (const auto& [y, t1s] : from_s) {
      if (y == w) continue;
      if (y == s && !q_.q1_sg_self_loops) continue;  // Q1: v2 lets s join I through its self-loop
      const auto it = into_w.find(y);
      if (it == into_w.end()) continue;
      bool ordered = false;
      for (std::int64_t t1 : t1s)
        for (std::int64_t t2 : it->second) ordered = ordered || in_order(t1, t2);
      n += ordered;
    }
    ++sizes[n];
  }
}

// R4: the distinct node sequences v = c_0, ..., c_{k-1} = u of distinct nodes (2 <= k <= k_max)
// with window edges c_{i-1} -> c_i at t_1 <= t_2 <= ... <= t_{k-1} <= t (Q5: strict), each of
// size k (R10).
//
// The search runs over node sequences, so each is reached once. For the current prefix it carries
// every time at which some valid timestamp assignment of the prefix arrives at its last node, over
// all parallel edges; a hop needs some arrival no later than it. It never reduces the arrivals to
// their earliest, so the earliest-witness rule is tested, not assumed.
void CleanOracle::tcyc(const Trigger& tr, int k_max, Sizes& sizes) const {
  using Ts = std::int64_t;
  std::vector<std::uint32_t> path = {tr.v};
  const auto dfs = [&](const auto& self, const std::vector<Ts>& arrivals) -> void {
    std::map<std::uint32_t, std::vector<Ts>> next;  // out-neighbour y -> the feasible times of a hop to y
    for (std::uint32_t f : window(out_[path.back()], tr)) {
      const Ts t1 = s_.ts[f];
      if (!in_order(t1, tr.t)) continue;
      if (std::any_of(arrivals.begin(), arrivals.end(), [&](Ts a) { return in_order(a, t1); }))
        next[s_.dst[f]].push_back(t1);
    }
    for (const auto& [y, times] : next) {
      if (y == tr.u) {
        if (int(path.size()) + 1 <= k_max) ++sizes[path.size() + 1];
        continue;
      }
      if (int(path.size()) + 1 >= k_max) continue;
      if (std::find(path.begin(), path.end(), y) != path.end()) continue;
      path.push_back(y);
      self(self, times);
      path.pop_back();
    }
  };
  dfs(dfs, {std::numeric_limits<Ts>::min()});
}

// R10, R4: the distinct simple node paths c_{k-1} -> ... -> c_1 -> c_0 = u -> v ending in the
// trigger (2 <= k <= k_max edges, all nodes distinct, v included), with window edges c_i -> c_{i-1}
// at t_{k-1} <= ... <= t_1 <= t (Q5: strict), each of size k.
//
// The mirror of tcyc's search, backward from u: for the current prefix it carries every time its
// head's chain edge can take over the valid timestamp assignments of the suffix, and an edge
// x -> head extends the prefix when it is no later than one of them. It never reduces them to their
// latest, so the latest-witness rule is tested, not assumed.
void CleanOracle::tchain(const Trigger& tr, int k_max, Sizes& sizes) const {
  using Ts = std::int64_t;
  std::vector<std::uint32_t> path = {tr.v, tr.u};
  const auto dfs = [&](const auto& self, const std::vector<Ts>& departures) -> void {
    std::map<std::uint32_t, std::vector<Ts>> prev;  // in-neighbour x -> the feasible times of its edge x -> head
    for (std::uint32_t f : window(in_[path.back()], tr)) {
      const Ts t1 = s_.ts[f];
      if (std::any_of(departures.begin(), departures.end(), [&](Ts d) { return in_order(t1, d); }))
        prev[s_.src[f]].push_back(t1);
    }
    for (const auto& [x, times] : prev) {
      if (std::find(path.begin(), path.end(), x) != path.end()) continue;
      ++sizes[path.size()];  // path.size() nodes before x: a chain of path.size() edges
      if (int(path.size()) + 1 > k_max) continue;
      path.push_back(x);
      self(self, times);
      path.pop_back();
    }
  };
  if (k_max >= 2) dfs(dfs, {tr.t});
}

// R5: the simple node cycles u -> v = c_0 -> c_1 -> ... -> c_h -> u through the trigger, of length
// h + 2 <= k_max, whose other edges are all in the window, in any time order, ties included (Q6:
// none at ts = t). Parallel edges collapse: one cycle per node sequence.
// R11: each extension of the path to a new node is one binding; more than `limit` abort.
bool CleanOracle::lcyc(const Trigger& tr, int k_max, std::uint64_t limit, Sizes& sizes) const {
  std::vector<std::uint32_t> path = {tr.u, tr.v};
  std::uint64_t work = 0;
  bool aborted = false;
  const auto successors = [&](std::uint32_t x) {
    std::set<std::uint32_t> ys;
    for (std::uint32_t f : window(out_[x], tr)) {
      if (q_.q6_lcyc_drops_ties && s_.ts[f] == tr.t) continue;
      ys.insert(s_.dst[f]);
    }
    return ys;
  };
  const auto dfs = [&](const auto& self, std::uint32_t x, int hops) -> void {  // hops: c_1 .. c_h so far
    for (std::uint32_t y : successors(x)) {
      if (aborted) return;
      if (y == tr.u) {
        ++sizes[std::uint64_t(hops) + 2];
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

namespace {

// The blocks whose pattern emits one instance per trigger: only they can be count blocks (R7).
bool one_instance(Block b) {
  switch (b) {
    case Block::fan_in:
    case Block::fan_out:
    case Block::deg_in:
    case Block::deg_out:
    case Block::fan_oneway_out:
    case Block::fan_total:
    case Block::gs: return true;
    default: return false;
  }
}

// R7, R10: a count block's one column gets the size; a histogram block's bin gets one per instance
// (min size 2, Q16). Counts saturate at UINT32_MAX: a histogram counts each lost increment, a count
// block each clamped row.
std::uint64_t emit(const BlockParams& p, const std::map<std::uint64_t, std::uint64_t>& sizes, std::uint32_t* row) {
  std::uint64_t saturated = 0;
  for (const auto& [size, k] : sizes) {
    if (p.bins.empty()) {
      if (size > UINT32_MAX - row[0]) {
        row[0] = UINT32_MAX;
        ++saturated;
      } else {
        row[0] += std::uint32_t(size);
      }
      continue;
    }
    const int i = bin_index(p.bins, size);
    if (i < 0) continue;
    const std::uint64_t add = std::min<std::uint64_t>(UINT32_MAX - row[i], k);
    row[i] += std::uint32_t(add);
    saturated += k - add;
  }
  return saturated;
}

// R11's report names the smallest aborted eid, whatever the thread count: a trigger is skipped only
// if its eid is larger than an abort already seen.
void lower_to(std::atomic<std::uint64_t>& a, std::uint64_t e) {
  for (std::uint64_t v = a.load(); e < v && !a.compare_exchange_weak(v, e);) {}
}

void check_params(Block b, const BlockParams& p, int k_max) {
  if (k_max < 2) throw std::invalid_argument("clean oracle: k_max < 2");
  if (p.bins.empty() && !one_instance(b))
    throw std::invalid_argument("clean oracle: block " + std::string(block_name(b)) +
                                " has several instances per trigger, so it needs bins (R7)");
}

}  // namespace

std::uint64_t CleanOracle::mine_one(Block b, const BlockParams& p, int k_max, std::uint32_t e, std::uint32_t* row,
                                    bool gather_only, bool& aborted) const {
  const Trigger tr = pattern_trigger(b, p, e);
  if (tr.u == tr.v) return 0;  // R1: a self-loop trigger's pattern rows are all zero
  Sizes sizes;
  switch (b) {
    case Block::fan_in:
    case Block::fan_out:
    case Block::deg_in:
    case Block::deg_out: fan_or_degree(b, tr, sizes); break;
    case Block::fan_oneway_out:
    case Block::fan_total: fan_sets(b, tr, sizes); break;
    case Block::gs: gs(tr, sizes); break;
    case Block::sg: sg(tr, !gather_only, sizes); break;
    case Block::sg_ord: sg_ord(tr, sizes); break;
    case Block::tcyc: tcyc(tr, k_max, sizes); break;
    case Block::tchain: tchain(tr, k_max, sizes); break;
    case Block::lcyc:
      if (!lcyc(tr, k_max, p.work_limit, sizes)) {
        aborted = true;  // R11: no rows
        return 0;
      }
      break;
  }
  return emit(p, sizes, row);
}

std::uint64_t CleanOracle::mine(Block b, const BlockParams& p, int k_max, std::uint64_t first, std::uint64_t last,
                                std::uint32_t* out) const {
  if (first > last || last > s_.src.size()) throw std::invalid_argument("clean oracle: bad trigger range");
  std::vector<std::uint32_t> eids(last - first);
  for (std::uint64_t e = first; e < last; ++e) eids[e - first] = std::uint32_t(e);
  return mine(b, p, k_max, eids, out);
}

std::uint64_t CleanOracle::mine(Block b, const BlockParams& p, int k_max, std::span<const std::uint32_t> eids,
                                std::uint32_t* out) const {
  check_params(b, p, k_max);
  for (std::uint32_t e : eids)
    if (e >= s_.src.size()) throw std::invalid_argument("clean oracle: trigger eid >= m");
  const std::size_t cols = block_cols(p);
  std::uint64_t saturated = 0;
  std::atomic<std::uint64_t> aborted{UINT64_MAX};
#pragma omp parallel for schedule(dynamic, 256) reduction(+ : saturated)
  for (std::size_t r = 0; r < eids.size(); ++r) {
    if (eids[r] > aborted.load(std::memory_order_relaxed)) continue;
    bool ab = false;
    saturated += mine_one(b, p, k_max, eids[r], out + r * cols, false, ab);
    if (ab) lower_to(aborted, eids[r]);
  }
  if (aborted != UINT64_MAX)
    throw WorkLimitExceeded("clean oracle, block " + std::string(block_name(b)), aborted, p.work_limit);
  return saturated;
}

std::uint64_t CleanOracle::mine_sg_gather(const BlockParams& p, std::uint64_t first, std::uint64_t last,
                                          std::uint32_t* out) const {
  if (first > last || last > s_.src.size()) throw std::invalid_argument("clean oracle: bad trigger range");
  check_params(Block::sg, p, 2);
  const std::size_t cols = block_cols(p);
  std::uint64_t saturated = 0;
#pragma omp parallel for schedule(dynamic, 256) reduction(+ : saturated)
  for (std::uint64_t e = first; e < last; ++e) {
    bool ab = false;
    saturated += mine_one(Block::sg, p, 2, std::uint32_t(e), out + (e - first) * cols, true, ab);
  }
  return saturated;
}

// R8 for one value column: mean, sum, var, skew, kurt into out[0..4]. The values are accumulated in
// ascending order (Q10: eid order), the sum first and then the central moments about the mean.
void CleanOracle::moments(std::vector<double> x, double* out) const {
  if (!q_.q10_vstats_gfp_arith) std::sort(x.begin(), x.end());
  const double n = double(x.size());
  double sum = 0.0;
  for (double xi : x) sum = sum + xi;
  const double mean = sum / n;
  out[0] = mean;
  out[1] = sum;
  out[2] = out[3] = out[4] = 0.0;
  if (!q_.q10_vstats_gfp_arith && x.front() == x.back()) return;  // every value equal (n = 1 included)
  double m2s = 0.0, m3s = 0.0, m4s = 0.0;
  for (double xi : x) {
    const double d = xi - mean;
    m2s = m2s + d * d;
    m3s = m3s + (d * d) * d;
    m4s = m4s + (d * d) * (d * d);
  }
  const double m2 = m2s / n, m3 = m3s / n, m4 = m4s / n;
  // Q10: GFP's guard reports a variance below 1e-5 * mean^2 as 0, with its skew and kurtosis (A44).
  if (q_.q10_vstats_gfp_arith && (!(m2 > 0) || m2 < kVstatsVarGuard * (mean * mean))) return;
  out[2] = m2;
  // R8, D-58: skew and kurtosis are 0 when their denominator is 0 (the computed m2 is 0, or m2^1.5 or
  // m2^2 underflows): never 0/0. Q10 keeps GFP's division, which its guard protects.
  const double d3 = m2 * std::sqrt(m2), d4 = m2 * m2;
  const bool zero = !q_.q10_vstats_gfp_arith;
  out[3] = zero && d3 == 0 ? 0.0 : m3 / d3;
  out[4] = zero && d4 == 0 ? 0.0 : m4 / d4;
}

// R8 for one side, node x's out- or in-edges in the window: fan (distinct neighbours), deg, ratio,
// then the age t - t' (Q10: the absolute t') and the amount.
void CleanOracle::vstats_side(const Trigger& tr, std::uint32_t x, bool out_edges, double* out) const {
  std::fill(out, out + kVstatsPerSide, 0.0);
  const std::vector<std::uint32_t> w = window(out_edges ? out_[x] : in_[x], tr);
  if (w.empty()) return;  // n = 0: every column 0
  std::set<std::uint32_t> nbrs;
  std::vector<double> age, amount;
  for (std::uint32_t f : w) {
    nbrs.insert(out_edges ? s_.dst[f] : s_.src[f]);
    age.push_back(q_.q10_vstats_gfp_arith ? double(s_.ts[f]) : double(tr.t - s_.ts[f]));
    amount.push_back(amount_[f]);
  }
  const double n = double(w.size());
  out[0] = double(nbrs.size());
  out[1] = n;
  out[2] = n / double(nbrs.size());
  moments(age, out + 3);
  moments(amount, out + 8);
}

void CleanOracle::vstats_row(std::int32_t window, std::uint32_t e, double* out) const {
  const Trigger tr = vstats_trigger(window, e);
  vstats_side(tr, tr.u, true, out);
  vstats_side(tr, tr.u, false, out + kVstatsPerSide);
  vstats_side(tr, tr.v, true, out + 2 * kVstatsPerSide);
  vstats_side(tr, tr.v, false, out + 3 * kVstatsPerSide);
}

void CleanOracle::vstats(std::int32_t window, std::uint64_t first, std::uint64_t last, double* out) const {
  if (first > last || last > s_.src.size()) throw std::invalid_argument("clean oracle: bad trigger range");
  std::vector<std::uint32_t> eids(last - first);
  for (std::uint64_t e = first; e < last; ++e) eids[e - first] = std::uint32_t(e);
  vstats(window, eids, out);
}

void CleanOracle::vstats(std::int32_t window, std::span<const std::uint32_t> eids, double* out) const {
  if (amount_.size() != s_.src.size()) throw std::invalid_argument("clean oracle: vstats needs one amount per edge");
  if (window <= 0) throw std::invalid_argument("clean oracle: vstats window must be > 0");
  for (std::uint32_t e : eids)
    if (e >= s_.src.size()) throw std::invalid_argument("clean oracle: trigger eid >= m");
#pragma omp parallel for schedule(dynamic, 1024)
  for (std::size_t r = 0; r < eids.size(); ++r) vstats_row(window, eids[r], out + r * kVstatsCols);
}

}  // namespace blz

#include "graph/index_contracts.hpp"

#include <algorithm>
#include <exception>
#include <span>
#include <vector>

#include "blz/reach.hpp"

namespace blz {

const char* contract_name(Contract c) {
  switch (c) {
    case Contract::list: return "list";
    case Contract::prev_same: return "prev_same";
    case Contract::range: return "range";
    case Contract::head_range: return "head_range";
    case Contract::distinct_first: return "distinct_first";
    case Contract::distinct_first_ts: return "distinct_first_ts";
    case Contract::pair_exists: return "pair_exists";
    case Contract::pair_latest: return "pair_latest";
    case Contract::search: return "search";
  }
  return "?";
}

namespace {

std::string bound_text(std::int64_t v) {
  if (v == kNoUpper) return "kNoUpper";
  if (v == INT64_MIN) return "INT64_MIN";
  return std::to_string(v);
}
std::string eid_text(std::uint32_t e) { return e == kAnyEid ? "kAnyEid" : std::to_string(e); }
std::string ts_text(std::int64_t v) { return v == kNoTs ? "kNoTs" : std::to_string(v); }

std::string eids_text(std::span<const std::uint32_t> eids) {
  std::string s = "{";
  for (std::size_t k = 0; k < eids.size(); ++k) s += (k ? "," : "") + std::to_string(eids[k]);
  return s + "}";
}

std::string edges_text(const EdgeList& g) {
  std::string s;
  for (std::size_t e = 0; e < g.src.size(); ++e)
    s += (e ? "  " : "") + std::to_string(e) + ": " + std::to_string(g.src[e]) + "->" + std::to_string(g.dst[e]) + "@" +
         std::to_string(g.ts[e]);
  return s;
}

// One direction of one graph: the naive lists, the index's view of them, and the tally.
struct Direction {
  const EdgeList& g;
  const TCsrView& L;
  const PCsrView& P;
  bool out;
  std::vector<std::vector<std::uint32_t>> list;  // per node: its edges in eid order (the naive list)
  IndexContracts& res;
  std::uint64_t id;
  SelfLoops mode;

  std::string where() const;  // "graph 12 (n = 3; edges ...), self-loops dropped, out-lists"

  std::uint32_t nbr(std::uint32_t e) const { return out ? g.dst[e] : g.src[e]; }

  // `what` builds the violation's text, only for the first one.
  template <class What>
  void count(Contract c, bool ok, What what) {
    ++res.checks[std::size_t(c)];
    if (ok) return;
    ++res.violations[std::size_t(c)];
    if (res.first.empty()) res.first = where() + ": " + contract_name(c) + ": " + what();
  }

  // Whether r is the positions of x's list (verified by `list`) whose entries pass `in`, i.e. k with
  // in(k) are exactly a + k in [r.lo, r.hi), and r lies inside the list.
  template <class In>
  static bool same_range(Range r, std::uint32_t a, std::uint32_t len, In in) {
    if (r.lo > r.hi || r.lo < a || r.hi > a + len) return false;
    for (std::uint32_t k = 0; k < len; ++k)
      if (in(k) != (a + k >= r.lo && a + k < r.hi)) return false;
    return true;
  }

  void check() {
    const std::uint32_t n = g.n;
    const std::uint64_t m = g.src.size();
    // The query values: every timestamp and its neighbours, every eid 0..m, and the sentinels.
    std::vector<std::int64_t> ts_q;
    for (std::int32_t t : g.ts)
      for (std::int64_t d = -1; d <= 1; ++d) ts_q.push_back(std::int64_t(t) + d);
    std::sort(ts_q.begin(), ts_q.end());
    ts_q.erase(std::unique(ts_q.begin(), ts_q.end()), ts_q.end());
    std::vector<std::int64_t> lo_q = ts_q, hi_q = ts_q;
    lo_q.insert(lo_q.begin(), INT64_MIN);
    hi_q.push_back(kNoUpper);
    std::vector<std::uint32_t> eid_q;
    for (std::uint32_t e = 0; e <= m; ++e) eid_q.push_back(e);
    eid_q.push_back(kAnyEid);

    // list: the builder's arrays hold each node's list. The other contracts read positions, so a
    // node whose list is wrong is not queried further.
    std::uint64_t total = 0;
    for (const auto& l : list) total += l.size();
    bool offsets = L.off[0] == 0 && L.off[n] == total;
    for (std::uint32_t x = 0; x < n && offsets; ++x) offsets = L.off[x + 1] >= L.off[x];
    std::vector<bool> listed(n, false);
    for (std::uint32_t x = 0; x < n; ++x) {
      bool ok = offsets && L.off[x + 1] - L.off[x] == list[x].size();
      for (std::size_t k = 0; ok && k < list[x].size(); ++k) {
        const std::uint32_t j = L.off[x] + std::uint32_t(k), e = list[x][k];
        ok = L.eid[j] == e && L.nbr[j] == nbr(e) && L.ts[j] == g.ts[e];
      }
      listed[x] = ok;
      count(Contract::list, ok, [&] {
        return "node " + std::to_string(x) + "'s entries [" + std::to_string(L.off[x]) + ", " +
               std::to_string(L.off[x + 1]) + ") are not its edges " + eids_text(list[x]);
      });
    }
    for (std::uint32_t x = 0; x < n; ++x) {
      if (!listed[x]) continue;
      const std::vector<std::uint32_t>& lx = list[x];
      const std::uint32_t a = L.off[x], len = std::uint32_t(lx.size());
      const auto ts = [&](std::uint32_t k) { return std::int64_t(g.ts[lx[k]]); };
      const auto same = [&](std::uint32_t k, std::uint32_t q) { return nbr(lx[k]) == nbr(lx[q]); };
      for (std::uint32_t k = 0; k < len; ++k) {
        std::int32_t want = -1;
        for (std::uint32_t q = 0; q < k; ++q)
          if (same(q, k)) want = std::int32_t(a + q);
        count(Contract::prev_same, L.prev_same[a + k] == want, [&] {
          return "prev_same[" + std::to_string(a + k) + "] (eid " + std::to_string(lx[k]) + ") = " +
                 std::to_string(L.prev_same[a + k]) + ", want " + std::to_string(want);
        });
      }
      // range and head_range: the entries of x's list inside the bounds.
      const auto selected = [&](std::int64_t lo, std::int64_t hi, std::uint32_t me) {
        return [=, &lx, &ts](std::uint32_t k) { return ts(k) >= lo && ts(k) <= hi && lx[k] <= me; };
      };
      const auto want_text = [&](std::int64_t lo, std::int64_t hi, std::uint32_t me) {
        std::vector<std::uint32_t> w;
        for (std::uint32_t k = 0; k < len; ++k)
          if (selected(lo, hi, me)(k)) w.push_back(lx[k]);
        return "; x's list starts at " + std::to_string(a) + " and is " + eids_text(lx) + ", so want the eids " +
               eids_text(w);
      };
      for (std::int64_t hi : hi_q)
        for (std::uint32_t me : eid_q) {
          for (std::int64_t lo : lo_q) {
            const Range r = L.range(x, lo, hi, me);
            count(Contract::range, same_range(r, a, len, selected(lo, hi, me)), [&] {
              return "range(" + std::to_string(x) + ", " + bound_text(lo) + ", " + bound_text(hi) + ", " + eid_text(me) +
                     ") = [" + std::to_string(r.lo) + ", " + std::to_string(r.hi) + ")" + want_text(lo, hi, me);
            });
          }
          const Range r = L.head_range(x, hi, me);
          count(Contract::head_range, r.lo == a && same_range(r, a, len, selected(INT64_MIN, hi, me)), [&] {
            return "head_range(" + std::to_string(x) + ", " + bound_text(hi) + ", " + eid_text(me) + ") = [" +
                   std::to_string(r.lo) + ", " + std::to_string(r.hi) + ")" + want_text(INT64_MIN, hi, me);
          });
        }
      const auto bool_text = [](bool b) { return std::string(b ? "true" : "false"); };
      // distinct_first(j, lo) for every scan start lo of x's list, distinct_first_ts(j, lo_ts).
      for (std::uint32_t lo = 0; lo <= len; ++lo)
        for (std::uint32_t k = lo; k < len; ++k) {
          bool want = true;
          for (std::uint32_t q = lo; q < k; ++q) want = want && !same(q, k);
          count(Contract::distinct_first, L.distinct_first(a + k, a + lo) == want, [&] {
            return "distinct_first(" + std::to_string(a + k) + ", " + std::to_string(a + lo) + ") = " + bool_text(!want) +
                   "; x's list starts at " + std::to_string(a) + " and is " + eids_text(lx);
          });
        }
      for (std::uint32_t k = 0; k < len; ++k)
        for (std::int64_t lo : lo_q) {
          bool want = true;
          for (std::uint32_t q = 0; q < k; ++q) want = want && !(same(q, k) && ts(q) >= lo);
          count(Contract::distinct_first_ts, L.distinct_first_ts(a + k, lo) == want, [&] {
            return "distinct_first_ts(" + std::to_string(a + k) + ", " + bound_text(lo) + ") = " + bool_text(!want) +
                   "; x's list starts at " + std::to_string(a) + " and is " + eids_text(lx);
          });
        }
      // pair_exists and pair_latest for every y, those x has no edge with included.
      for (std::uint32_t y = 0; y < n; ++y)
        for (std::int64_t lo : lo_q)
          for (std::int64_t hi : hi_q)
            for (std::uint32_t me : eid_q) {
              std::int64_t earliest = kNoTs, latest = kNoTs;
              for (std::uint32_t k = 0; k < len; ++k) {
                if (nbr(lx[k]) != y || !selected(lo, hi, me)(k)) continue;
                if (earliest == kNoTs) earliest = ts(k);
                latest = ts(k);
              }
              const auto args = [&] {
                return "(" + std::to_string(x) + ", " + std::to_string(y) + ", " + bound_text(lo) + ", " + bound_text(hi) +
                       ", " + eid_text(me) + ") = ";
              };
              const std::int64_t pe = P.pair_exists(x, y, lo, hi, me), pl = P.pair_latest(x, y, lo, hi, me);
              count(Contract::pair_exists, pe == earliest,
                    [&] { return "pair_exists" + args() + ts_text(pe) + ", want " + ts_text(earliest); });
              count(Contract::pair_latest, pl == latest,
                    [&] { return "pair_latest" + args() + ts_text(pl) + ", want " + ts_text(latest); });
            }
    }
  }

  // search, on the in-lists: for trigger e = U -> V at t and window delta, ReachScratch::search with
  // tcyc_trigger_reach's arguments against ldJ (verify/lean/Blz/Prune.lean) over Prune.lean's search
  // set E, the visible edges with t - delta <= ts <= t - 1, after the same number of rounds.
  void check_search(std::int32_t windows) {
    constexpr int kHops = 9;                  // tcyc_trigger_reach's rounds
    constexpr std::int64_t kBot = INT64_MIN;  // ldJ's bot: not reached
    ReachScratch reach(g.n);
    std::vector<std::int64_t> ld(g.n), next(g.n);
    std::vector<std::uint32_t> E;
    for (std::uint32_t e = 0; e < g.src.size(); ++e) {
      const std::uint32_t U = g.src[e];
      if (U == g.dst[e]) continue;  // tcyc_trigger_reach returns before its search
      const std::int64_t t = g.ts[e];
      for (const bool ts_vis : {false, true})
        for (std::int32_t delta = 1; delta <= windows; ++delta) {
          const std::uint32_t vis_e = ts_vis ? kAnyEid : e;
          const std::int64_t vis_t = ts_vis ? t : kNoUpper;
          const std::int64_t lo = t - delta, hi = min_ts(t - 1, vis_t);
          reach.search(L, U, t, lo, hi, vis_e, kHops);
          E.clear();
          for (std::uint32_t f = 0; f < g.src.size(); ++f)
            if ((ts_vis ? g.ts[f] <= t : f <= e) && g.ts[f] >= lo && g.ts[f] <= t - 1) E.push_back(f);
          // ldJ 0 = t at U, bot elsewhere; ldJ (r + 1) = relaxRound E U (ldJ r).
          std::fill(ld.begin(), ld.end(), kBot);
          ld[U] = t;
          for (int r = 0; r < kHops; ++r) {
            next = ld;
            for (const std::uint32_t f : E) {
              const std::uint32_t x = g.src[f], y = g.dst[f];
              if (x != U && x != y && g.ts[f] < ld[y]) next[x] = std::max<std::int64_t>(next[x], g.ts[f]);
            }
            if (next == ld) break;  // a fixed point: the later rounds change nothing
            ld.swap(next);
          }
          // keep(x, h) must hold at every threshold below ldJ that tcyc_trigger_reach tests: t - delta - 1
          // at the root, then hop times in [lo, t - 1].
          std::uint32_t bad = g.n;
          std::int64_t bad_h = 0;
          for (std::uint32_t x = 0; x < g.n; ++x) {
            if (ld[x] == kBot) continue;
            res.search_nodes += x != U;
            for (std::int64_t h = lo - 1; h < ld[x] && bad == g.n; ++h)
              if (!reach.keep(x, h)) bad = x, bad_h = h;
          }
          count(Contract::search, bad == g.n, [&] {
            std::string want;
            for (std::uint32_t x = 0; x < g.n; ++x)
              if (ld[x] != kBot) want += (want.empty() ? "" : ", ") + std::to_string(x) + " " + std::to_string(ld[x]);
            return "trigger " + std::to_string(e) + " (" + std::to_string(U) + "->" + std::to_string(g.dst[e]) + "@" +
                   std::to_string(t) + "), " + (ts_vis ? "timestamp" : "stream") + " visibility, window " +
                   std::to_string(delta) + ": after search(in, " + std::to_string(U) + ", " + std::to_string(t) + ", " +
                   std::to_string(lo) + ", " + bound_text(hi) + ", " + eid_text(vis_e) + ", " + std::to_string(kHops) +
                   "), keep(" + std::to_string(bad) + ", " + std::to_string(bad_h) + ") is false; want ld >= " +
                   std::to_string(ld[bad]) + ", as ldJ after " + std::to_string(kHops) + " rounds gives (node ld: " +
                   want + ")";
          });
        }
    }
  }
};

std::string Direction::where() const {
  return "graph " + std::to_string(id) + " (n = " + std::to_string(g.n) + "; edges, eid: u->v@t: " + edges_text(g) +
         "), self-loops " + (mode == SelfLoops::drop ? "dropped" : "kept") + ", " + (out ? "out" : "in") + "-lists";
}

}  // namespace

void check_index_contracts(const EdgeList& g, std::uint64_t id, SelfLoops mode, std::int32_t windows,
                           IndexContracts& res) {
  const GraphIndex index(g.src, g.dst, g.ts, g.n, mode);
  const GraphView& v = index.view();
  const bool had = !res.first.empty();
  for (const bool out : {true, false}) {
    Direction d{g, out ? v.out : v.in, out ? v.pout : v.pin, out, std::vector<std::vector<std::uint32_t>>(g.n), res,
                id, mode};
    for (std::uint32_t e = 0; e < g.src.size(); ++e)
      if (mode == SelfLoops::keep || g.src[e] != g.dst[e]) d.list[out ? g.src[e] : g.dst[e]].push_back(e);
    d.check();
    if (!out) d.check_search(windows);
  }
  if (!had && !res.first.empty()) res.first_graph = id;
}

IndexContracts check_index_contracts(const ExhaustiveGraphs& gs) {
  IndexContracts res;
  std::exception_ptr error;
#pragma omp parallel
  {
    IndexContracts mine;
#pragma omp for schedule(dynamic, 256)
    for (std::int64_t i = 0; i < std::int64_t(gs.size()); ++i) try {
        const EdgeList g = gs.graph(std::uint64_t(i));
        // The first violation of this thread's smallest graph: a later graph's never replaces it.
        IndexContracts one;
        for (const SelfLoops mode : {SelfLoops::keep, SelfLoops::drop})
          check_index_contracts(g, std::uint64_t(i), mode, gs.bound().t, one);
        ++mine.graphs;
        mine.search_nodes += one.search_nodes;
        for (std::size_t c = 0; c < kContracts; ++c) {
          mine.checks[c] += one.checks[c];
          mine.violations[c] += one.violations[c];
        }
        if (!one.first.empty() && std::uint64_t(i) < mine.first_graph) {
          mine.first_graph = std::uint64_t(i);
          mine.first = one.first;
        }
      } catch (...) {
#pragma omp critical
        if (!error) error = std::current_exception();
      }
#pragma omp critical
    {
      res.graphs += mine.graphs;
      res.search_nodes += mine.search_nodes;
      for (std::size_t c = 0; c < kContracts; ++c) {
        res.checks[c] += mine.checks[c];
        res.violations[c] += mine.violations[c];
      }
      if (mine.first_graph < res.first_graph) {
        res.first_graph = mine.first_graph;
        res.first = mine.first;
      }
    }
  }
  if (error) std::rethrow_exception(error);
  return res;
}

}  // namespace blz

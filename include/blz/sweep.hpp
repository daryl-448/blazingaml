#pragma once
// The CPU sweep driver (SPEC v2 §7.1, D-29), host only. The generated OpenMP driver of a pattern
// planned as a sweep (a fan or degree count over the trigger's own list) calls sweep_count.
//
// For node x, the entries of x's list in the count's direction are exactly the triggers whose
// src (out-lists) or dst (in-lists) is x, and their windows are positions [lo_j, hi_j) of that
// same list, with both ends non-decreasing in j:
//   - hi_j = j + 1 under stream visibility (eid <= e: the list is in eid order), or the first
//     position with ts > t_j under timestamp visibility;
//   - lo_j = the first position with ts >= lower(t_j), as range() finds it, capped at hi_j.
// So one forward pass keeps the window's count as entries enter and leave: an edge count, or for
// a distinct count, per-neighbour counts in a per-thread dense array plus a distinct counter. An
// entry enters the state only if it passes the element filters. Trigger j emits once, if `out`
// selects it. The dense array is cleared by walking the last window, so a list costs O(length).
#include <algorithm>
#include <cstdint>
#include <vector>

#include "blz/primitives.hpp"
#include "blz/registry.hpp"

namespace blz {

// Nodes per dynamic-schedule chunk. The nodes come longest list first.
inline constexpr int kSweepChunk = 16;

// The nodes with a non-empty list in l, longest first, ties in node order (a stable counting
// sort on the length).
inline std::vector<std::uint32_t> sweep_order(const TCsrView& l, std::uint32_t n) {
  std::uint32_t longest = 0;
  for (std::uint32_t x = 0; x < n; ++x) longest = std::max(longest, l.off[x + 1] - l.off[x]);
  std::vector<std::uint32_t> start(std::size_t(longest) + 1, 0);  // bucket longest - len
  for (std::uint32_t x = 0; x < n; ++x)
    if (const std::uint32_t len = l.off[x + 1] - l.off[x]) ++start[longest - len + 1];
  for (std::size_t i = 1; i < start.size(); ++i) start[i] += start[i - 1];
  std::vector<std::uint32_t> order(start.back());
  for (std::uint32_t x = 0; x < n; ++x)
    if (const std::uint32_t len = l.off[x + 1] - l.off[x]) order[start[longest - len]++] = x;
  return order;
}

// The sweep over list direction l of g, under stream or timestamp visibility. lower(t) is the
// window's inclusive lower bound for a trigger at time t, keep(x, y) whether entry y of x's list
// enters the count (the element filters), and emit(e, count, sink) runs trigger e's filters and
// emit. Distinct: count distinct neighbours, else entries. Uses the current OpenMP thread count;
// returns the emits lost to saturation.
template <bool Distinct, class Lower, class Keep, class Emit>
std::uint64_t sweep_count(const GraphView& g, const TCsrView& l, bool ts_visibility, const BlockOut& out,
                          Lower lower, Keep keep, Emit emit) {
  const std::vector<std::uint32_t> order = sweep_order(l, g.n);
  const std::int64_t nodes = std::int64_t(order.size());
  std::uint64_t saturated = 0;
#pragma omp parallel reduction(+ : saturated)
  {
    std::vector<std::uint32_t> seen(Distinct ? g.n : 0);  // per-neighbour counts in the window
#pragma omp for schedule(dynamic, kSweepChunk)
    for (std::int64_t i = 0; i < nodes; ++i) {
      const std::uint32_t x = order[std::size_t(i)];
      const std::uint32_t a = l.off[x], b = l.off[x + 1];
      std::uint32_t lo = a, hi = a;  // the window [lo, hi); its entries are in the state
      std::int64_t count = 0;
      for (std::uint32_t j = a; j < b; ++j) {
        const std::int64_t t = l.ts[j];
        for (; hi < b && (ts_visibility ? std::int64_t(l.ts[hi]) <= t : hi <= j); ++hi)
          if (keep(x, l.nbr[hi])) {
            if constexpr (Distinct) count += seen[l.nbr[hi]]++ == 0;
            else ++count;
          }
        const std::int64_t lo_ts = lower(t);
        for (; lo < hi && std::int64_t(l.ts[lo]) < lo_ts; ++lo)
          if (keep(x, l.nbr[lo])) {
            if constexpr (Distinct) count -= --seen[l.nbr[lo]] == 0;
            else --count;
          }
        const std::uint32_t e = l.eid[j];
        if (!out.selects(e)) continue;
        RowSink sink = out.sink(e);
        emit(e, count, sink);
        saturated += sink.saturated;
      }
      if constexpr (Distinct)
        for (std::uint32_t k = lo; k < hi; ++k) seen[l.nbr[k]] = 0;
    }
  }
  return saturated;
}

}  // namespace blz

#pragma once
// Reachability pruning for chain patterns (SPEC v2 §8 S6, D-43): host-only scratch that the
// generated <pattern>_trigger_reach (in <stem>_cpu.cpp) uses, one per OpenMP thread.
//
// search() runs, per trigger, a backward search from the target U over the in-lists and gives
// each node x its latest departure time ld[x]: the largest time of an edge x -> y that starts a
// temporal walk x -> y -> ... -> U whose times strictly increase, lie in [lo, hi] and are visible
// (eid <= max_eid). ld[U] = t, so an edge into U needs ts < t (hi <= t - 1 in any case). The walk
// may repeat nodes; `hops` rounds find every walk of up to `hops` edges (and some longer ones, see
// below), so ld over-approximates every simple path the
// pattern can close: a binding of node x by a hop at time h can only lead to an emit if
// ld[x] > h, and keep(x, h) is exactly that test.
//
// The search is a label-correcting relaxation in rounds: round r scans the in-lists of the nodes
// whose ld rose in round r - 1 (U for r = 1), each from the window's lower end to ld[y] - 1, and
// raises ld[x] for every edge x -> y it finds. A value raised in the same round may be read later
// in that round; that only raises ld further (a longer walk), which keeps more and stays exact.
// So `hops` bounds the rounds (the cost), not the walks' length.
#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "blz/primitives.hpp"

namespace blz {

class ReachScratch {
 public:
  explicit ReachScratch(std::uint32_t n) : mark_(n, 0), queued_(n, 0), ld_(n, 0) {}

  void search(const TCsrView& in, std::uint32_t target, std::int64_t t, std::int64_t lo, std::int64_t hi,
              std::uint32_t max_eid, int hops) {
    epoch_ = bump(epoch_, mark_);
    mark_[target] = epoch_;
    ld_[target] = std::int32_t(t);
    frontier_.assign(1, target);
    for (int r = 0; r < hops && !frontier_.empty(); ++r) {
      round_ = bump(round_, queued_);
      next_.clear();
      for (const std::uint32_t y : frontier_) {
        const std::int64_t lim = min_ts(hi, std::int64_t(ld_[y]) - 1);
        if (lim < lo) continue;
        const Range rg = in.range(y, lo, lim, max_eid);
        for (std::uint32_t j = rg.hi; j-- > rg.lo;) {  // latest first: one raise per source
          const std::uint32_t x = in.nbr[j];
          if (x == y || x == target) continue;
          const std::int32_t ts = in.ts[j];
          if (mark_[x] == epoch_ && ld_[x] >= ts) continue;
          mark_[x] = epoch_;
          ld_[x] = ts;
          if (queued_[x] != round_) {
            queued_[x] = round_;
            next_.push_back(x);
          }
        }
      }
      std::swap(frontier_, next_);
    }
  }

  // Whether x can depart towards the target strictly after time h (ld[x] > h).
  bool keep(std::uint32_t x, std::int64_t h) const { return mark_[x] == epoch_ && std::int64_t(ld_[x]) > h; }

 private:
  // The next stamp; on wrap-around every stamp in `marks` is cleared, so no old stamp matches.
  static std::uint32_t bump(std::uint32_t s, std::vector<std::uint32_t>& marks) {
    if (++s == 0) {
      std::fill(marks.begin(), marks.end(), 0);
      s = 1;
    }
    return s;
  }

  std::uint32_t epoch_ = 0, round_ = 0;
  std::vector<std::uint32_t> mark_, queued_;
  std::vector<std::int32_t> ld_;
  std::vector<std::uint32_t> frontier_, next_;
};

}  // namespace blz

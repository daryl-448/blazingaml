#pragma once
// GFP batch mode (SPEC v2 §8 S1): the batch layout, the driver both implementations share, and
// the CPU module. The oracle's literal version is oracle/batch_oracle.hpp.
//
// A batch is one GFP transform() call. Its rows are returned at the end of the call, so a row
// collects every instance found while its batch is processed that has the row's edge as a
// participant. Rows of different batches never interact, so the batches are independent: each
// is computed whole, into a buffer of its own, and the rows a trigger selection asks for are
// copied out. That is the "per-thread buffers merged in eid order" of §8 S1: the buffers are
// disjoint, so the merge is a copy and the counts are deterministic.
#include <algorithm>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "blz/primitives.hpp"
#include "runtime/features.hpp"
#include "runtime/profile.hpp"
#include "runtime/triggers.hpp"

namespace blz {

// Batch k is eids [starts[k], starts[k + 1]); starts[0] = 0, starts.back() = m.
struct BatchLayout {
  std::vector<std::uint64_t> starts;

  std::uint64_t batches() const { return starts.empty() ? 0 : starts.size() - 1; }
  std::uint64_t edges() const { return starts.empty() ? 0 : starts.back(); }
  // The batch that holds eid e (< edges()).
  std::uint64_t batch_of(std::uint64_t e) const {
    return std::uint64_t(std::upper_bound(starts.begin(), starts.end(), e) - starts.begin()) - 1;
  }
};

// Consecutive batches of `size` edges over [0, m), each cut short at the next cut, so a new batch
// starts at every cut in (0, m) (the R6 layout of tools/gfp/dg2_features.py).
BatchLayout batch_layout(std::uint64_t m, std::uint64_t size, std::span<const std::uint64_t> cuts);

// The layout of a batch profile on the prefix [0, m) of a stream whose timestamps are `full_ts`
// (m <= full_ts.size(); `blz mine --edges`). `cuts: splits` cuts at b_fv and at the train/test
// boundary of the whole stream (SPEC v2 §3.3), as R6 does; cuts at or past m are dropped.
BatchLayout batch_layout(const BatchSpec& spec, std::uint64_t m, std::span<const std::int32_t> full_ts);

// One batch as both implementations see it: eids [lo, hi), and GFP's graph during the call.
// Eviction: at each transform() call GFP drops the edges with ts < (the largest ts before the
// call) - time_window; the first call drops nothing, and the batch's own edges are never dropped
// (A37). Only SG's suppression rule can see the difference (SPEC v2 §8 S1).
struct BatchRange {
  std::uint64_t lo = 0, hi = 0;
  std::int64_t kept_from = 0;  // graph edges of earlier batches have ts >= kept_from
};

inline BatchRange batch_range(const BatchLayout& l, std::uint64_t k, std::span<const std::int32_t> ts,
                              std::int32_t time_window) {
  BatchRange r{l.starts[k], l.starts[k + 1], INT64_MIN};
  if (r.lo > 0) r.kept_from = std::int64_t(ts[r.lo - 1]) - time_window;
  return r;
}

// Runs fn(range, rows) for every batch that holds a row of `sel`, in parallel over batches (each
// call on one thread, with `rows` = (hi - lo) * nb zeroed cells for eids lo..hi-1), and copies the
// selected rows to `out` (sel.rows() * nb cells, row r = sel.eid(r); the caller zeroes it).
// Returns the sum of fn's results (saturated increments).
template <class Fn>
std::uint64_t run_batches(const BatchLayout& l, std::span<const std::int32_t> ts, std::int32_t time_window,
                          const TriggerSel& sel, std::size_t nb, std::uint32_t* out, Fn fn) {
  if (sel.last > l.edges() || sel.first > sel.last) throw std::invalid_argument("batch mode: bad trigger range");
  std::vector<std::uint64_t> need;  // batch ids, ascending
  if (sel.sampled) {
    for (std::uint32_t e : sel.eids)
      if (const std::uint64_t k = l.batch_of(e); need.empty() || need.back() != k) need.push_back(k);
  } else if (sel.first < sel.last) {
    for (std::uint64_t k = l.batch_of(sel.first); k <= l.batch_of(sel.last - 1); ++k) need.push_back(k);
  }
  std::uint64_t saturated = 0;
#pragma omp parallel reduction(+ : saturated)
  {
    std::vector<std::uint32_t> rows;
#pragma omp for schedule(dynamic, 1)
    for (std::int64_t i = 0; i < std::int64_t(need.size()); ++i) {
      const BatchRange r = batch_range(l, need[std::size_t(i)], ts, time_window);
      rows.assign((r.hi - r.lo) * nb, 0);
      saturated += fn(r, rows.data());
      if (!sel.sampled) {
        for (std::uint64_t e = std::max(r.lo, sel.first); e < std::min(r.hi, sel.last); ++e)
          std::copy_n(rows.data() + (e - r.lo) * nb, nb, out + (e - sel.first) * nb);
      } else {
        auto it = std::lower_bound(sel.eids.begin(), sel.eids.end(), std::uint32_t(r.lo));
        for (; it != sel.eids.end() && *it < r.hi; ++it)
          std::copy_n(rows.data() + (*it - r.lo) * nb, nb, out + std::size_t(it - sel.eids.begin()) * nb);
      }
    }
  }
  return saturated;
}

// The CPU module of batch mode: block b's rows under GFP batch mode, for the triggers `sel`,
// adding into `out` (sel.rows() * p.bins.size() cells, zeroed by the caller). `k_max` caps tcyc
// lengths at run time (GFP has no cap). Uses the current OpenMP thread count. Returns the number
// of saturated increments.
std::uint64_t mine_batch_cpu(const GraphView& g, Block b, const BlockParams& p, int k_max,
                             std::int32_t time_window, const BatchLayout& layout, const TriggerSel& sel,
                             std::uint32_t* out);

}  // namespace blz

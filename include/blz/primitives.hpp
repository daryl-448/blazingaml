#pragma once
// The graph view and the primitives that generated pattern code is built from (SPEC §5.3).
// Everything here is BLZ_HD (host and device) and header-only, so the same loop nest compiles
// for the OpenMP driver and for CUDA (P5). No standard algorithms: they are not device code.
#include <cstdint>

#if defined(__CUDACC__)
#define BLZ_HD __host__ __device__
#else
#define BLZ_HD
#endif

namespace blz {

// pair_exists: no witness. A real timestamp is an int32, so it is never INT64_MIN.
inline constexpr std::int64_t kNoTs = INT64_MIN;
// range / pair_exists: no upper time bound.
inline constexpr std::int64_t kNoUpper = INT64_MAX;
// range / pair_exists: no eid bound (timestamp visibility, SPEC v2 §6.1). Edge ids are < m <= 2^32 - 1.
inline constexpr std::uint32_t kAnyEid = UINT32_MAX;

BLZ_HD inline std::int64_t max_ts(std::int64_t a, std::int64_t b) { return a < b ? b : a; }
BLZ_HD inline std::int64_t min_ts(std::int64_t a, std::int64_t b) { return b < a ? b : a; }
// min() of the DSL's int expressions (SPEC v2 §8 S5).
BLZ_HD inline std::int64_t min_int(std::int64_t a, std::int64_t b) { return b < a ? b : a; }

// Positions [lo, hi) of one node's T-CSR list.
struct Range {
  std::uint32_t lo = 0, hi = 0;
  BLZ_HD std::uint32_t size() const { return hi - lo; }
};

// T-CSR, one direction (SPEC §5.1). Node x's entries are off[x]..off[x+1], sorted by eid and
// therefore by ts. Out-lists hold dst as nbr, in-lists src. prev_same[j] is the absolute position
// of the previous entry of the same list with the same nbr, or -1. Self-loops are kept.
struct TCsrView {
  const std::uint32_t* off = nullptr;  // n + 1
  const std::uint32_t* nbr = nullptr;  // m
  const std::int32_t* ts = nullptr;    // m
  const std::uint32_t* eid = nullptr;  // m
  const std::int32_t* prev_same = nullptr;  // m

  // x's entries with lo_ts <= ts <= hi_ts and eid <= max_eid (stream visibility). The bounds are
  // inclusive; strict bounds are normalized by the planner (SPEC §7.2). The lower end is the
  // paper's find_start; hi_ts == kNoUpper skips the time search on the upper end.
  BLZ_HD Range range(std::uint32_t x, std::int64_t lo_ts, std::int64_t hi_ts, std::uint32_t max_eid) const {
    const std::uint32_t a = off[x];
    std::uint32_t b = first_eid_after(a, off[x + 1], max_eid);
    if (hi_ts != kNoUpper) b = first_ts_at_least(a, b, hi_ts + 1);  // first ts > hi_ts
    return {first_ts_at_least(a, b, lo_ts), b};
  }

  // range() without the lower bound: x's entries from the list head (SPEC v2 §7.2, find_start
  // disabled; the window's lower bound is then a per-entry skip).
  BLZ_HD Range head_range(std::uint32_t x, std::int64_t hi_ts, std::uint32_t max_eid) const {
    const std::uint32_t a = off[x];
    std::uint32_t b = first_eid_after(a, off[x + 1], max_eid);
    if (hi_ts != kNoUpper) b = first_ts_at_least(a, b, hi_ts + 1);
    return {a, b};
  }

  // The first position in [a, b) with ts >= v, or b.
  BLZ_HD std::uint32_t first_ts_at_least(std::uint32_t a, std::uint32_t b, std::int64_t v) const {
    while (a < b) {
      const std::uint32_t mid = a + (b - a) / 2;
      if (std::int64_t(ts[mid]) < v) a = mid + 1;
      else b = mid;
    }
    return a;
  }

  // The first position in [a, b) with eid > e, or b.
  BLZ_HD std::uint32_t first_eid_after(std::uint32_t a, std::uint32_t b, std::uint32_t e) const {
    while (a < b) {
      const std::uint32_t mid = a + (b - a) / 2;
      if (eid[mid] <= e) a = mid + 1;
      else b = mid;
    }
    return a;
  }

  // Inside a scanned range starting at lo: whether entry j is the first (earliest) occurrence of
  // its neighbour, so each neighbour is visited once, at its earliest witness (SPEC §5.1).
  BLZ_HD bool distinct_first(std::uint32_t j, std::uint32_t lo) const {
    return prev_same[j] < std::int32_t(lo);
  }

  // The same test for a scan that starts at the list head (find_start disabled): the window
  // starts at the first entry with ts >= lo_ts, so the previous entry of the same neighbour lies
  // before it exactly when it is absent or has ts < lo_ts.
  BLZ_HD bool distinct_first_ts(std::uint32_t j, std::int64_t lo_ts) const {
    const std::int32_t k = prev_same[j];
    return k < 0 || std::int64_t(ts[k]) < lo_ts;
  }
};

// P-CSR, one direction (SPEC §5.2). Node x's distinct neighbours are nbr[off[x]..off[x+1]),
// sorted by node id; pair i's edges are run_ts / run_eid [run[i], run[i+1]), sorted by eid and
// therefore by ts.
struct PCsrView {
  const std::uint32_t* off = nullptr;   // n + 1
  const std::uint32_t* nbr = nullptr;   // pairs
  const std::uint32_t* run = nullptr;   // pairs + 1
  const std::int32_t* run_ts = nullptr;    // m
  const std::uint32_t* run_eid = nullptr;  // m

  // The earliest ts of an edge between x and y in this direction (x -> y for out, y -> x for in)
  // with lo_ts <= ts <= hi_ts and eid <= max_eid, or kNoTs. Two binary searches: the first run
  // entry with ts >= lo_ts has the smallest eid of all that do, so it alone decides visibility.
  BLZ_HD std::int64_t pair_exists(std::uint32_t x, std::uint32_t y, std::int64_t lo_ts, std::int64_t hi_ts,
                                  std::uint32_t max_eid) const {
    const std::uint32_t l = pair_index(x, y);
    if (l == kNoPair) return kNoTs;
    std::uint32_t a = run[l], b = run[l + 1];
    while (a < b) {
      const std::uint32_t mid = a + (b - a) / 2;
      if (std::int64_t(run_ts[mid]) < lo_ts) a = mid + 1;
      else b = mid;
    }
    if (a == run[l + 1] || std::int64_t(run_ts[a]) > hi_ts || run_eid[a] > max_eid) return kNoTs;
    return run_ts[a];
  }

  // The latest ts of an edge between x and y in this direction with the same bounds, or kNoTs
  // (SPEC v2 §6.2: the latest witness of an order constraint). The run is sorted by eid and so by
  // ts, so the entries with eid <= max_eid and ts <= hi_ts are a prefix of it, and the last one
  // decides. Two binary searches after the pair's, one per upper bound that is set.
  BLZ_HD std::int64_t pair_latest(std::uint32_t x, std::uint32_t y, std::int64_t lo_ts, std::int64_t hi_ts,
                                  std::uint32_t max_eid) const {
    const std::uint32_t l = pair_index(x, y);
    if (l == kNoPair) return kNoTs;
    std::uint32_t a = run[l], b = run[l + 1];
    if (max_eid != kAnyEid) {  // the first entry with eid > max_eid
      while (a < b) {
        const std::uint32_t mid = a + (b - a) / 2;
        if (run_eid[mid] <= max_eid) a = mid + 1;
        else b = mid;
      }
      b = a;
      a = run[l];
    }
    if (hi_ts != kNoUpper) {  // the first entry with ts > hi_ts
      while (a < b) {
        const std::uint32_t mid = a + (b - a) / 2;
        if (std::int64_t(run_ts[mid]) <= hi_ts) a = mid + 1;
        else b = mid;
      }
      b = a;
      a = run[l];
    }
    if (b == a || std::int64_t(run_ts[b - 1]) < lo_ts) return kNoTs;
    return run_ts[b - 1];
  }

 private:
  static constexpr std::uint32_t kNoPair = UINT32_MAX;
  // The pair index of (x, y) (a position in nbr), or kNoPair: a binary search over x's neighbours.
  BLZ_HD std::uint32_t pair_index(std::uint32_t x, std::uint32_t y) const {
    std::uint32_t l = off[x], h = off[x + 1];
    while (l < h) {
      const std::uint32_t mid = l + (h - l) / 2;
      if (nbr[mid] < y) l = mid + 1;
      else h = mid;
    }
    return l == off[x + 1] || nbr[l] != y ? kNoPair : l;
  }
};

// The indexed graph (SPEC §5): the edge arrays in stream order plus both directions of T-CSR
// and P-CSR. Plain pointers, so the view can be copied to the device as is.
struct GraphView {
  std::uint32_t m = 0, n = 0;
  const std::uint32_t* src = nullptr;
  const std::uint32_t* dst = nullptr;
  const std::int32_t* ts = nullptr;
  TCsrView out, in;
  PCsrView pout, pin;
};

// A pattern's runtime parameters: its window span (seconds), taken from its block's config, and
// the profile's visibility (SPEC v2 §6.1, D-22). Generated code turns the visibility into the
// upper end of every range and probe once per trigger: eid <= e (stream) or ts <= t (timestamp).
struct Params {
  std::int64_t span = 0;
  bool ts_visibility = false;  // false: stream visibility (v1); true: timestamp visibility
  // Bindings one trigger may make in a pattern compiled with `work_limit: true` (SPEC v2 §8 S3).
  std::uint64_t work_limit = UINT64_MAX;
};

// count_distinct / count_edges (SPEC §5.3): the number of distinct neighbours, or of edges, in
// range r whose neighbour passes keep(nbr). Fan and degree counts in O(window). With `stride` > 1
// only positions r.lo + first, r.lo + first + stride, ... are counted: the CUDA heavy kernel's
// lanes split one scan this way and sum their partial counts (SPEC §7.5). distinct_first still
// tests against r.lo, so the partial counts add up to the full count.
template <class Keep>
BLZ_HD inline std::uint32_t count_distinct(const TCsrView& l, Range r, Keep keep, std::uint32_t first = 0,
                                           std::uint32_t stride = 1) {
  std::uint32_t n = 0;
  for (std::uint32_t j = r.lo + first; j < r.hi; j += stride)
    if (l.distinct_first(j, r.lo) && keep(l.nbr[j])) ++n;
  return n;
}

template <class Keep>
BLZ_HD inline std::uint32_t count_edges(const TCsrView& l, Range r, Keep keep, std::uint32_t first = 0,
                                        std::uint32_t stride = 1) {
  std::uint32_t n = 0;
  for (std::uint32_t j = r.lo + first; j < r.hi; j += stride)
    if (keep(l.nbr[j])) ++n;
  return n;
}

}  // namespace blz

#pragma once
#include <cstdint>
#include <span>
#include <vector>

#include "runtime/features.hpp"

namespace blz {

// An edge stream in stream order (eid = index, ts non-decreasing; SPEC §3.3).
struct EdgeStream {
  std::span<const std::uint32_t> src, dst;
  std::span<const std::int32_t> ts;
  std::uint64_t n = 0;  // node ids are < n
};

// The reference oracle (SPEC §9): each §4 definition computed literally, per trigger, from
// per-node edge-id lists and std::set/std::map. It deliberately shares no index structure with
// the generated backends (no T-CSR, P-CSR or prev_same), and its tcyc search is exhaustive over
// parallel edges instead of relying on the earliest-witness rule.
//
// `vis` selects the visible edges (SPEC v2 §6.1): stream (eid' <= e, v1) or timestamp (t' <= t).
// Only the window's upper end changes; everything else in §4 of v1 is the same.
class Oracle {
 public:
  explicit Oracle(const EdgeStream& s, Visibility vis = Visibility::stream);

  // Mines block `b` for triggers [first, last): row (e - first) of `out` (bins.size() columns,
  // zeroed by the caller) gets trigger e's histogram. `k_max` caps tcyc lengths. Parallel over
  // triggers with OpenMP. Returns the number of saturated increments (SPEC §4.2).
  std::uint64_t mine(Block b, const BlockParams& p, int k_max, std::uint64_t first,
                     std::uint64_t last, std::uint32_t* out) const;
  // The same for the triggers `eids` (a sample, SPEC v2 §11): row r of `out` gets eids[r].
  std::uint64_t mine(Block b, const BlockParams& p, int k_max, std::span<const std::uint32_t> eids,
                     std::uint32_t* out) const;
  // The pattern sg_gather alone (the gather role of §4.5) for triggers [first, last), as `mine`
  // does it for block sg. The order-relaxation test (SPEC v2 §3.4) bounds sg_ord with it.
  std::uint64_t mine_sg_gather(const BlockParams& p, std::uint64_t first, std::uint64_t last,
                               std::uint32_t* out) const;

 private:
  struct Window {
    const std::uint32_t* begin;  // edge ids of one node's list, in eid order
    const std::uint32_t* end;
    std::size_t size() const { return std::size_t(end - begin); }
  };
  struct Trigger {
    std::uint32_t e, u, v;
    std::int64_t t, lo_ts;  // lo_ts = t - delta
    bool closed;
  };

  // Trigger e's histogram into `row`; returns the number of saturated increments. `gather_only`
  // (block sg): the gather role alone.
  std::uint64_t mine_one(Block b, const BlockParams& p, int k_max, std::uint64_t e, std::uint32_t* row,
                         bool gather_only = false) const;
  Window window(const std::vector<std::uint32_t>& list, const Trigger& tr) const;
  bool has_edge(std::uint32_t x, std::uint32_t y, const Trigger& tr) const;

  void fan_or_degree(Block b, const Trigger& tr, std::vector<std::uint64_t>& sizes) const;
  void sg(const Trigger& tr, bool scatter, std::vector<std::uint64_t>& sizes) const;
  void sg_ordered(const Trigger& tr, std::vector<std::uint64_t>& sizes) const;
  void tcyc(const Trigger& tr, int k_max, std::vector<std::uint64_t>& sizes) const;

  EdgeStream s_;
  Visibility vis_;
  std::vector<std::vector<std::uint32_t>> out_, in_;  // per node: incident edge ids, eid order
};

}  // namespace blz

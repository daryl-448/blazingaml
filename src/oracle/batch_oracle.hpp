#pragma once
#include <cstdint>
#include <vector>

#include "oracle/oracle.hpp"
#include "runtime/batch.hpp"

namespace blz {

// The reference oracle of GFP batch mode (SPEC v2 §8 S1), written literally from the addendum's
// rules with per-node edge-id lists and std::set, like Oracle (§9 of v1). It shares no index
// structure with the CPU module (runtime/batch.cpp), and its tcyc keeps every feasible arrival
// and departure time instead of the earliest and latest ones.
class BatchOracle {
 public:
  // `time_window` is GFP's (Profile::batch_time_window()); `layout` covers the whole stream.
  BatchOracle(const EdgeStream& s, BatchLayout layout, std::int32_t time_window);

  // Block b's rows for the triggers `sel` (whole batches are computed, the selected rows copied),
  // adding into `out` (sel.rows() * p.bins.size() cells, zeroed by the caller). `k_max` caps
  // tcyc lengths. Parallel over batches with OpenMP. Returns the saturated increments.
  std::uint64_t mine(Block b, const BlockParams& p, int k_max, const TriggerSel& sel, std::uint32_t* out) const;

 private:
  struct Ctx;  // one batch and one block
  std::uint64_t mine_batch(Block b, const BlockParams& p, int k_max, const BatchRange& r, std::uint32_t* rows) const;
  void fan_or_degree(Ctx& c, std::uint32_t e) const;
  void tcyc(Ctx& c, std::uint32_t e) const;
  void sg(Ctx& c, std::uint32_t e) const;

  EdgeStream s_;
  BatchLayout layout_;
  std::int32_t time_window_;
  std::vector<std::vector<std::uint32_t>> out_, in_;  // per node: incident edge ids, eid order
};

}  // namespace blz

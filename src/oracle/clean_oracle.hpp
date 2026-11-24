#pragma once
#include <cstdint>
#include <map>
#include <span>
#include <string_view>
#include <vector>

#include "oracle/oracle.hpp"
#include "runtime/features.hpp"
#include "runtime/vstats.hpp"

namespace blz {

// The GFP / v2 behaviours of SPEC v3 §3.1 that the clean rules replace, each one switchable for
// E16 (§3.6). Clean = all false. Q7 (the sg_ord block) and Q8 (count vs histogram) are profile
// choices, not quirks of the oracle.
struct Quirks {
  bool q1_sg_self_loops = false;      // sg and sg_ord see self-loops, with v1's role-dependent exclusions
  bool q2_vstats_self_loops = false;  // vertex statistics count self-loops
  bool q3_closed_windows = false;     // sg, sg_ord, tcyc, lcyc, tchain use t' >= t - delta
  bool q4_stream_visibility = false;  // eid' <= e instead of t' <= t
  bool q5_strict_order = false;       // tcyc, tchain, sg_ord (sources ts < t, a < b) and gs's A (ts < t): strict
  bool q6_lcyc_drops_ties = false;    // lcyc drops every edge with ts == t
  bool q9_vstats_gfp_window = false;  // vstats window anchored at ts[e - 1] - W (eid 0: unbounded), closed
  bool q10_vstats_gfp_arith = false;  // absolute ts, eid-order accumulation, the 1e-5 guard (A44)
};

// "" (clean), a comma-separated list of ledger ids ("Q1,Q3") or "all". Throws std::invalid_argument
// on any other name.
Quirks parse_quirks(std::string_view list);

// The clean reference (SPEC v3 §3.2): each rule R1-R11 computed literally, per trigger, from its own
// per-node edge-id lists and std::set/std::map. It shares no index structure with the generated
// backends, and its ordered searches (tcyc, tchain, sg_ord) enumerate every parallel edge instead of
// relying on the earliest- or latest-witness rule. With every quirk on it is the legacy Oracle at
// stream visibility, and its vertex statistics are vstats_cpu's.
class CleanOracle {
 public:
  // `amount` (Amount Received by eid) is read by the vertex statistics only.
  CleanOracle(const EdgeStream& s, std::span<const double> amount = {}, Quirks q = {});

  // Mines block `b` for triggers [first, last): row (e - first) of `out` (block_cols(p) columns)
  // gets trigger e's row added, so the caller zeroes `out`. A block without bins is a count block
  // (R7), which only the blocks with one instance per trigger have. `k_max` caps tcyc, lcyc and
  // tchain. Parallel over triggers with OpenMP. Returns the number of saturated increments (R10).
  // lcyc throws WorkLimitExceeded, with the smallest such eid, when a trigger makes more than
  // p.work_limit bindings (R11).
  std::uint64_t mine(Block b, const BlockParams& p, int k_max, std::uint64_t first, std::uint64_t last,
                     std::uint32_t* out) const;
  // The same for the triggers `eids`: row r of `out` gets eids[r].
  std::uint64_t mine(Block b, const BlockParams& p, int k_max, std::span<const std::uint32_t> eids,
                     std::uint32_t* out) const;
  // Block sg's gather role alone (R6a), which bounds sg_ord (R6b).
  std::uint64_t mine_sg_gather(const BlockParams& p, std::uint64_t first, std::uint64_t last,
                               std::uint32_t* out) const;

  // R8: kVstatsCols doubles per row. Sides u_out, u_in, v_out, v_in; per side fan, deg, ratio, then
  // the age t - t' and the amount, each mean, sum, var, skew, kurt.
  void vstats(std::int32_t window, std::uint64_t first, std::uint64_t last, double* out) const;
  void vstats(std::int32_t window, std::span<const std::uint32_t> eids, double* out) const;

 private:
  struct Trigger {
    std::uint32_t e, u, v;
    std::int64_t t, lo;  // the window's lower bound: lo < t' (closed: lo <= t')
    bool closed;         // Q3, Q9
    bool loops;          // self-loops are window edges (Q1, Q2)
  };
  using Sizes = std::map<std::uint64_t, std::uint64_t>;  // instance size -> instances

  Trigger pattern_trigger(Block b, const BlockParams& p, std::uint32_t e) const;
  Trigger vstats_trigger(std::int32_t window, std::uint32_t e) const;
  std::vector<std::uint32_t> window(const std::vector<std::uint32_t>& list, const Trigger& tr) const;
  bool in_order(std::int64_t a, std::int64_t b) const;

  // Trigger e's row into `row`; returns the number of saturated increments. `aborted` (lcyc): set
  // when the trigger made more than p.work_limit bindings.
  std::uint64_t mine_one(Block b, const BlockParams& p, int k_max, std::uint32_t e, std::uint32_t* row,
                         bool gather_only, bool& aborted) const;
  void fan_or_degree(Block b, const Trigger& tr, Sizes& sizes) const;
  void fan_sets(Block b, const Trigger& tr, Sizes& sizes) const;
  void gs(const Trigger& tr, Sizes& sizes) const;
  void sg(const Trigger& tr, bool scatter, Sizes& sizes) const;
  void sg_ord(const Trigger& tr, Sizes& sizes) const;
  void tcyc(const Trigger& tr, int k_max, Sizes& sizes) const;
  void tchain(const Trigger& tr, int k_max, Sizes& sizes) const;
  bool lcyc(const Trigger& tr, int k_max, std::uint64_t limit, Sizes& sizes) const;  // false: over the limit

  void vstats_row(std::int32_t window, std::uint32_t e, double* out) const;
  void vstats_side(const Trigger& tr, std::uint32_t x, bool out_edges, double* out) const;
  void moments(std::vector<double> x, double* out) const;

  EdgeStream s_;
  std::span<const double> amount_;
  Quirks q_;
  std::vector<std::vector<std::uint32_t>> out_, in_;  // per node: incident edge ids in eid order, self-loops too
};

}  // namespace blz

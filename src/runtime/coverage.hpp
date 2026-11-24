#pragma once
// Ground-truth coverage (SPEC v2 §5, E9): for each listed laundering attempt (results/gt_join.json,
// §3.5) and each block of a profile, whether the attempt's own edges can make the block fire
// (structural eligibility), whether the block fires on them in the full mining (activation), and
// which of its validation and test edges a classifier flags (detection).
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "oracle/oracle.hpp"
#include "runtime/profile.hpp"

namespace blz {

// What coverage needs from the prepared stream (eid = index).
struct CoverageGraph {
  std::span<const std::uint32_t> src, dst;
  std::span<const std::int32_t> ts;
  std::span<const std::uint8_t> label;
};

// One attempt of a gt-join output.
struct CoverageAttempt {
  std::int64_t id = 0;
  std::string typology, header;
  std::string status;  // complete | partial | unmatched (§3.5)
  std::string split;   // fit | val | test | straddles, or "none" (no mapped eid)
  std::int64_t span = 0;
  std::vector<std::uint32_t> eids;  // the mapped eids (unique and resolved rows), ascending
};

// The attempts of results/gt_join.json, in file order. Throws std::runtime_error on a malformed
// file or a mapped eid >= m.
std::vector<CoverageAttempt> coverage_attempts(const nlohmann::json& gt, std::uint64_t m);

// An attempt's edges alone: eids ascending, so stream order (and ts order) is kept; nodes
// renumbered 0..n-1 by first appearance, the source before the destination. Every SPEC §4 rule is
// invariant under the renumbering (§3.4 relabeling), so row i is what eid eids[i] gets when the
// stream holds only these edges.
struct IsolatedGraph {
  std::vector<std::uint32_t> eids, src, dst;
  std::vector<std::int32_t> ts;
  std::uint64_t n = 0;
  EdgeStream stream() const { return {src, dst, ts, n}; }
};
IsolatedGraph isolated_graph(const CoverageGraph& g, std::span<const std::uint32_t> eids);

// Structural eligibility (§5): for each block of `prof` (in its order), the number of the isolated
// graph's edges whose row is non-zero when the profile's oracle (ProfileOracle) mines that graph
// alone with the profile's windows, bins, k_max and visibility.
std::vector<std::uint64_t> oracle_eligible_rows(const IsolatedGraph& ig, const Profile& prof);

// Prediction codes of `blz coverage --pred` (u8[m]).
inline constexpr std::uint8_t kPredNegative = 0, kPredPositive = 1, kPredNotScored = 255;

struct CoverageInputs {
  CoverageGraph graph;
  std::uint64_t b_fv = 0, b_test = 0;  // fit = [0, b_fv), val = [b_fv, b_test), test = [b_test, m)
  const Profile* profile = nullptr;
  std::vector<CoverageAttempt> attempts;
  // The full mining under the profile: one block per profile block, in its order, m x bins
  // (row = eid).
  std::vector<std::span<const std::uint32_t>> blocks;
  // u8[m] prediction codes, or empty (no detection). Fit rows must be kNotScored: they are
  // training data. A split is either scored (every row 0 or 1) or not (every row 255).
  std::span<const std::uint8_t> pred;
};

// The coverage of one profile:
// - `blocks`: the profile's block names, which index every per-block array;
// - `attempts`: per attempt, its edges per split and, for a complete attempt, `eligible` and
//   `activated` (non-zero rows per block) and `predicted` (predicted-positive edges per scored
//   split);
// - `by_typology` (alphabetical, then "ALL"): attempt counts by status and split, span
//   percentiles in hours (numpy's linear interpolation, complete attempts), per block (and "any")
//   the complete attempts eligible, activated, and eligible but not activated, and `det_val`,
//   `det_test`: over the complete attempts with edges in the split, their count, edges, predicted
//   positives, the mean of the per-attempt shares and the pooled share (null when the split is
//   not scored or no attempt has an edge in it);
// - `positives`: listed (mapped by any attempt) and unlisted positives per split, those that no
//   block activates, and predicted positives per scored split;
// - `base_rates`: per block (and "any"), the non-zero rows among all edges and among positives,
//   per split.
// Throws std::runtime_error on inconsistent inputs.
nlohmann::ordered_json coverage(const CoverageInputs& in);

// The coverage as text: one line per attempt, so the file diffs well.
std::string coverage_dump(const nlohmann::ordered_json& j);

// numpy.percentile(x, q) with linear interpolation; x need not be sorted. NaN for an empty x.
double percentile_linear(std::vector<double> x, double q);

}  // namespace blz

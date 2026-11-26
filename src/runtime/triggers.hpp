#pragma once
// Trigger selection (SPEC §8 `--triggers`, extended by SPEC v2 §11): which rows a mining run
// computes. Every trigger always sees the whole graph; the selection only chooses rows.
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "blz/edges.hpp"

namespace blz {

struct TriggerSel {
  std::string spec;                // as given: all | train | test | fit | val | range:a:b | sample:<pct>:<seed>
  std::uint64_t first = 0, last = 0;  // the eid range; for a sample, the range its eids lie in
  bool sampled = false;
  double percent = 0;              // sample only
  std::uint64_t seed = 0;          // sample only
  std::vector<std::uint32_t> eids; // sample only: the selected eids, ascending

  // Output rows: last - first, or the sample's size. Row r belongs to eid first + r, or eids[r].
  std::uint64_t rows() const { return sampled ? eids.size() : last - first; }
  std::uint32_t eid(std::uint64_t r) const { return sampled ? eids[r] : std::uint32_t(first + r); }
  nlohmann::json json() const;  // for the manifest; a sample's eids are not included
};

// The selection `spec` names on a graph of m edges whose named splits are `splits` (SPEC v3
// §6.3; a graph cut short by --edges keeps its whole stream's splits, V2 §4.4 R5):
//   all = [0, m); train = [0, b_test); test = [b_test, m); fit = [0, b_fv); val = [b_fv, b_test);
//   range:a:b = [a, b);
//   sample:<pct>:<seed> = each eid independently with probability pct/100 (0 < pct <= 100),
//   decided by a hash of (seed, eid): reproducible, and independent of m and of other eids.
// Throws std::runtime_error on a malformed spec, or a range or named split past m.
TriggerSel select_triggers(const std::string& spec, std::uint64_t m, const Splits& splits);

// Whether eid `e` is in sample (percent, seed).
bool sample_selected(double percent, std::uint64_t seed, std::uint64_t e);

}  // namespace blz

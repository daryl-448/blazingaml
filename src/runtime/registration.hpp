#pragma once
#include <span>
#include <string>

#include "runtime/features.hpp"

namespace blz {

// What the generated registry (SPEC §7.6, P4) records about one compiled pattern.
struct PatternRef {
  std::string name;
  std::string block;
  std::string block_loc;  // "patterns/<p>.yaml:line:col" of its `block:` value
};

// The start-up check of every registered pattern against the feature config (SPEC §6.7):
// E402 when a pattern's block is not a known block or is missing from the config. Throws
// std::runtime_error "<block_loc>: error E402: ..." on the first violation.
void check_registration(std::span<const PatternRef> patterns, const FeatureConfig& cfg,
                        const std::string& config_name);

}  // namespace blz

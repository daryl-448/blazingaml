#pragma once
#include <cstdint>
#include <span>

#include <nlohmann/json.hpp>

#include "blz/edges.hpp"

namespace blz {

struct StreamView {
  std::span<const std::uint32_t> src, dst;
  std::span<const std::int32_t> ts;
  std::span<const std::uint8_t> label;
  std::uint64_t n = 0;
  Splits splits;  // train = [0, b_test), test = [b_test, m)
};

// Summary of a non-negative integer distribution: count, min, max, mean, p50/p90/p99/p99.9 and a
// log2 histogram keyed by value range: "0", "1", "2-3", "4-7", ...
nlohmann::json summarize(std::span<const std::uint64_t> values);

// The E0 dataset profile without the GFP part (SPEC §12.2). Window occupancy uses the closed
// lower bound ts >= t - delta and stream visibility (eid <= trigger).
nlohmann::json profile_stream(const StreamView& g);

}  // namespace blz

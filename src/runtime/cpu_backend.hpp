#pragma once
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "blz/registry.hpp"
#include "runtime/features.hpp"

namespace blz {

// Trigger order for the OpenMP drivers (SPEC §7.4).
enum class Schedule {
  dynamic,  // eid order, schedule(dynamic, kDynamicChunk)
  sorted,   // descending work estimate (ties by eid), schedule(dynamic, kSortedChunk)
};
inline constexpr int kDynamicChunk = 256;
inline constexpr int kSortedChunk = 16;
const char* schedule_name(Schedule s);
Schedule parse_schedule(const std::string& s);

// The registered patterns that emit into block b, in registry order.
std::vector<const PatternInfo*> patterns_for(Block b);

// The start-up checks of the generated backends (SPEC §7.6): E402 for every registered pattern
// (check_registration), and the config's k_max must equal the K_MAX const of every pattern that
// declares one, since K_MAX is compiled in. Throws std::runtime_error.
void check_backend_config(const FeatureConfig& cfg, const std::string& config_name);

// The K_MAX the library was compiled with (the const of the patterns that declare it).
int compiled_k_max();

struct CpuBlockTiming {
  double schedule_seconds = 0;  // trigger list, plus the work estimates and sort for `sorted`
  double loop_seconds = 0;      // the drivers' trigger loops
  std::uint64_t saturated = 0;
};

// Mines block b for the triggers [first, last) with every pattern of the block, adding into
// `out` (last - first rows of p.bins.size() columns, zeroed by the caller). Uses the current
// OpenMP thread count. Throws if no pattern emits into b.
CpuBlockTiming mine_cpu(const GraphView& g, Block b, const BlockParams& p, std::uint64_t first,
                        std::uint64_t last, Schedule schedule, std::uint32_t* out);

}  // namespace blz

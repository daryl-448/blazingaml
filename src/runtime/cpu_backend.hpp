#pragma once
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "blz/registry.hpp"
#include "runtime/features.hpp"
#include "runtime/triggers.hpp"

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

// The start-up checks of the generated backends (SPEC §7.6): E402 for every registered pattern of
// a v1 block (check_registration), and the config's k_max must equal the K_MAX const of every
// pattern that declares one, since K_MAX is compiled in. Throws std::runtime_error.
void check_backend_config(const FeatureConfig& cfg, const std::string& config_name);

// The K_MAX the library was compiled with (the const of the patterns that declare it).
int compiled_k_max();

struct CpuBlockTiming {
  double schedule_seconds = 0;  // trigger list, plus the work estimates and sort for `sorted`
  double loop_seconds = 0;      // the drivers' trigger loops
  double sweep_seconds = 0;     // the part of loop_seconds spent in sweep drivers (SPEC v2 §7.1)
  std::uint64_t saturated = 0;
};

// Mines one block with the patterns `pats` (a profile's list for the block, SPEC v2 §3.1; their
// emits are summed) under visibility `vis` (the profile's, SPEC v2 §6.1) for the triggers `sel`,
// adding into `out` (sel.rows() rows of block_cols(p) columns, zeroed by the caller; row r
// belongs to sel.eid(r)); p without bins is a count block (SPEC v3 R7). Uses the current OpenMP
// thread count. Throws if `pats` is empty. A pattern planned as a sweep (SPEC v2 §7.1) ignores
// `schedule`: it walks every list entry, so its cost does not shrink with a trigger sample.
CpuBlockTiming mine_cpu(const GraphView& g, std::span<const PatternInfo* const> pats, const BlockParams& p,
                        Visibility vis, const TriggerSel& sel, Schedule schedule, std::uint32_t* out);

// The v1 form: block b with every registered pattern of the block, triggers [first, last),
// stream visibility.
CpuBlockTiming mine_cpu(const GraphView& g, Block b, const BlockParams& p, std::uint64_t first,
                        std::uint64_t last, Schedule schedule, std::uint32_t* out);

}  // namespace blz

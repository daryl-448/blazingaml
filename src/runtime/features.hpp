#pragma once
#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace blz {

// Output blocks: the six v1 blocks in GFP column order (SPEC §4.2), then sg_ord (SPEC v2 §6.3),
// which GFP does not have, lcyc (GFP's lc-cycle, SPEC v2 §8 S3), fan_oneway_out and fan_total,
// the features of the set ops difference and union (SPEC v2 §8 S4), and tchain and gs, the
// typologies of SPEC v2 §8 S5 (temporal chains, gather-scatter).
enum class Block { fan_in, fan_out, deg_in, deg_out, sg, tcyc, sg_ord, lcyc, fan_oneway_out, fan_total, tchain, gs };
inline constexpr std::array<Block, 12> kAllBlocks = {
    Block::fan_in, Block::fan_out, Block::deg_in, Block::deg_out,         Block::sg,        Block::tcyc,
    Block::sg_ord, Block::lcyc,    Block::fan_oneway_out, Block::fan_total, Block::tchain, Block::gs};
// The v1 blocks: what a v1 feature config covers (E402) and what GFP computes.
inline constexpr std::array<Block, 6> kV1Blocks = {Block::fan_in,  Block::fan_out, Block::deg_in,
                                                   Block::deg_out, Block::sg,      Block::tcyc};
inline bool is_v1_block(Block b) { return int(b) <= int(Block::tcyc); }

std::string_view block_name(Block b);
std::optional<Block> parse_block(std::string_view name);

// Window lower bound (SPEC §4.1): fan and degree (and S4's fan_oneway_out and fan_total, and S5's
// gs) are open (t' > t - delta), SG, tcyc, sg_ord, lcyc and S5's tchain closed (t' >= t - delta).
// Fixed by the semantics, not by the config.
inline bool closed_lower_bound(Block b) {
  return b == Block::sg || b == Block::tcyc || b == Block::sg_ord || b == Block::lcyc || b == Block::tchain;
}

// Which edges a trigger e = (u -> v, t) may see (SPEC §4.1 of v1, §6.1 of v2): stream = eid' <=
// eid (the default, GFP stream mode); timestamp = t' <= t, so later edges of the same timestamp
// too. The window lower bound, the self-loop rule and min size 2 do not depend on it.
enum class Visibility { stream, timestamp };
const char* visibility_name(Visibility v);  // defined in runtime/profile.cpp

// The semantics family of a pattern or profile (SPEC v3 §3.2): legacy (v1/v2, with their quirks)
// or clean (R1–R11). A profile never mixes them (E403).
enum class Family { legacy, clean };
const char* family_name(Family f);  // defined in runtime/profile.cpp

// The per-trigger work limit of lcyc (SPEC v2 §8 S3): bindings one trigger may make before the
// run stops with that trigger's eid. Every backend counts the same bindings, so they all stop on
// the same triggers. No other block counts work.
inline constexpr std::uint64_t kDefaultWorkLimit = 100'000'000;

struct BlockParams {
  std::int32_t window = 0;           // delta, seconds (> 0)
  std::vector<std::uint32_t> bins;   // strictly increasing, last bin open-ended
  std::uint64_t work_limit = kDefaultWorkLimit;
  bool reach_prune = false;  // the CPU's reachability pruning (SPEC v2 §8 S6): a speed option, same rows
};

// A block's columns: a count block (SPEC v3 R7, no bins) has one, the count; a histogram block one
// per bin.
inline std::uint32_t block_cols(std::span<const std::uint32_t> bins) {
  return bins.empty() ? 1u : std::uint32_t(bins.size());
}
inline std::uint32_t block_cols(const BlockParams& p) { return block_cols(p.bins); }

// Thrown when a trigger exceeds the work limit (SPEC v2 §8 S3). `eid` is the smallest such
// trigger among those that ran (a driver skips the triggers it has not started yet).
struct WorkLimitExceeded : std::runtime_error {
  WorkLimitExceeded(const std::string& where, std::uint64_t eid, std::uint64_t limit);
  std::uint64_t eid, limit;
};

// A feature config (configs/features_*.yaml, SPEC §4.8).
struct FeatureConfig {
  int k_max = 10;  // tcyc length cap (D-7)
  std::array<std::optional<BlockParams>, kAllBlocks.size()> blocks;

  const std::optional<BlockParams>& operator[](Block b) const { return blocks[int(b)]; }
  std::optional<BlockParams>& operator[](Block b) { return blocks[int(b)]; }
};

// Throws std::runtime_error with the file name on any schema error.
FeatureConfig load_feature_config(const std::string& path);
FeatureConfig parse_feature_config(std::string_view yaml_text, const std::string& name);

// Minimum instance size of every pattern (SPEC §4.2).
inline constexpr std::uint64_t kMinInstanceSize = 2;

// The bin an instance of `size` falls in, or -1 if it is dropped (size < 2 or size < b_0).
inline int bin_index(std::span<const std::uint32_t> bins, std::uint64_t size) {
  if (size < kMinInstanceSize || bins.empty() || size < bins[0]) return -1;
  int i = int(bins.size()) - 1;
  while (size < bins[i]) --i;
  return i;
}

// Saturating increment (SPEC §4.2). Returns true if the counter was already saturated.
inline bool add_saturating(std::uint32_t& c) {
  if (c == UINT32_MAX) return true;
  ++c;
  return false;
}

// "<block>[b_i,b_{i+1})" ... "<block>[b_last,inf)" (SPEC §4.2); "<block>" for a count block (no bins).
std::vector<std::string> column_names(Block b, std::span<const std::uint32_t> bins);

}  // namespace blz

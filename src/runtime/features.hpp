#pragma once
#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace blz {

// Output blocks in GFP column order (SPEC §4.2).
enum class Block { fan_in, fan_out, deg_in, deg_out, sg, tcyc };
inline constexpr std::array<Block, 6> kAllBlocks = {Block::fan_in, Block::fan_out, Block::deg_in,
                                                    Block::deg_out, Block::sg,     Block::tcyc};

std::string_view block_name(Block b);
std::optional<Block> parse_block(std::string_view name);

// Window lower bound (SPEC §4.1): fan and degree are open (t' > t - delta), SG and tcyc closed
// (t' >= t - delta). Fixed by the semantics, not by the config.
inline bool closed_lower_bound(Block b) { return b == Block::sg || b == Block::tcyc; }

struct BlockParams {
  std::int32_t window = 0;           // delta, seconds (> 0)
  std::vector<std::uint32_t> bins;   // strictly increasing, last bin open-ended
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

// "<block>[b_i,b_{i+1})" ... "<block>[b_last,inf)" (SPEC §4.2).
std::vector<std::string> column_names(Block b, std::span<const std::uint32_t> bins);

}  // namespace blz

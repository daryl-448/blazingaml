#include "runtime/features.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

namespace blz {

std::string_view block_name(Block b) {
  static constexpr std::array<std::string_view, kAllBlocks.size()> names = {
      "fan_in", "fan_out", "deg_in", "deg_out", "sg", "tcyc", "sg_ord", "lcyc", "fan_oneway_out", "fan_total"};
  return names[int(b)];
}

WorkLimitExceeded::WorkLimitExceeded(const std::string& where, std::uint64_t eid_, std::uint64_t limit_)
    : std::runtime_error(where + ": trigger eid " + std::to_string(eid_) + " exceeded the work limit of " +
                         std::to_string(limit_) + " bindings (SPEC v2 §8 S3); the run stops, no row is truncated; "
                         "raise it with --work-limit"),
      eid(eid_),
      limit(limit_) {}

std::optional<Block> parse_block(std::string_view name) {
  for (Block b : kAllBlocks)
    if (block_name(b) == name) return b;
  return std::nullopt;
}

std::vector<std::string> column_names(Block b, std::span<const std::uint32_t> bins) {
  std::vector<std::string> out;
  const std::string base(block_name(b));
  for (std::size_t i = 0; i < bins.size(); ++i) {
    const std::string hi = i + 1 < bins.size() ? std::to_string(bins[i + 1]) : "inf";
    out.push_back(base + "[" + std::to_string(bins[i]) + "," + hi + ")");
  }
  return out;
}

FeatureConfig parse_feature_config(std::string_view yaml_text, const std::string& name) {
  auto fail = [&](const std::string& msg) -> void {
    throw std::runtime_error(name + ": " + msg);
  };
  FeatureConfig cfg;
  try {
    YAML::Node root = YAML::Load(std::string(yaml_text));
    if (!root.IsMap()) fail("top level must be a map");
    for (auto kv : root) {
      const auto key = kv.first.as<std::string>();
      if (key != "k_max" && key != "blocks") fail("unknown key '" + key + "'");
    }
    if (root["k_max"]) cfg.k_max = root["k_max"].as<int>();
    if (cfg.k_max < 2) fail("k_max must be >= 2");
    const YAML::Node blocks = root["blocks"];
    if (!blocks || !blocks.IsMap() || blocks.size() == 0) fail("'blocks' must be a non-empty map");
    for (auto kv : blocks) {
      const auto bname = kv.first.as<std::string>();
      const auto b = parse_block(bname);
      if (!b) fail("unknown block '" + bname + "'");
      if (cfg[*b]) fail("duplicate block '" + bname + "'");
      const YAML::Node p = kv.second;
      if (!p.IsMap()) fail(bname + ": must be a map");
      for (auto f : p) {
        const auto k = f.first.as<std::string>();
        if (k != "window" && k != "bins") fail(bname + ": unknown key '" + k + "'");
      }
      BlockParams bp;
      if (!p["window"] || !p["bins"]) fail(bname + ": needs 'window' and 'bins'");
      bp.window = p["window"].as<std::int32_t>();
      if (bp.window <= 0) fail(bname + ": window must be > 0");
      bp.bins = p["bins"].as<std::vector<std::uint32_t>>();
      if (bp.bins.empty()) fail(bname + ": bins must be non-empty");
      for (std::size_t i = 1; i < bp.bins.size(); ++i)
        if (bp.bins[i] <= bp.bins[i - 1]) fail(bname + ": bins must be strictly increasing");
      cfg[*b] = std::move(bp);
    }
  } catch (const YAML::Exception& e) {
    throw std::runtime_error(name + ": " + e.what());
  }
  return cfg;
}

FeatureConfig load_feature_config(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::stringstream ss;
  ss << in.rdbuf();
  return parse_feature_config(ss.str(), path);
}

}  // namespace blz

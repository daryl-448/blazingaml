#pragma once
// Semantic profiles (SPEC v2 §3.1): everything that determines feature values, in one file.
// A profile is `configs/profiles/<name>.yaml` (`version: 2`); a v1 feature config (no `version`)
// loads as the profile it implies. The profile id is content-derived and identifies the
// semantics, not the build.
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "blz/registry.hpp"
#include "graph/index.hpp"
#include "runtime/features.hpp"

namespace blz {

// Visibility (and visibility_name) is in runtime/features.hpp, so the oracle can take it.

// The backends a profile can be run on (E404 names them).
enum class Backend { oracle, cpu, cuda, hybrid };  // hybrid: cpu + cuda (SPEC v2 §8 S6)
const char* backend_name(Backend b);
Backend parse_backend(const std::string& s);

struct ProfileBlock {
  Block block;
  std::vector<const PatternInfo*> patterns;  // in listed order; their emits are summed
  BlockParams params;  // no bins: a count block (SPEC v3 R7), clean profiles only
  std::string loc;  // "<file>:<line>:<col>" of the block's key
};

struct Profile {
  std::string file;  // the file it was loaded from
  std::string name;  // `profile:`; for a v1 feature config, the file's stem
  int format = 2;    // 2: a profile; 1: a v1 feature config (SPEC v2 §3.1 "Compatibility")
  Family family = Family::legacy;  // `family:` (SPEC v3 §3.2); v1 configs are legacy
  Visibility visibility = Visibility::stream;
  int k_max = 10;
  std::vector<ProfileBlock> blocks;  // output order = listed order
  std::string family_loc, visibility_loc, k_max_loc;  // "<file>:<line>:<col>" of the values

  const ProfileBlock* find(Block b) const;
  // The index the profile mines on: a clean profile's has no self-loops (SPEC v3 R1).
  SelfLoops self_loops() const { return family == Family::clean ? SelfLoops::drop : SelfLoops::keep; }
  // Stream visibility: a row depends only on the edges before it (§3.3).
  bool causal() const;
  // Which edges a row may see, for the manifest (§3.3: a non-causal profile MUST say so).
  std::string row_sees() const;
  // The windows, bins and k_max as a v1 FeatureConfig.
  FeatureConfig feature_config() const;
  // The resolved profile: name, semantics, block order, and each block's patterns (name, spec
  // file, spec SHA-256), window and bins.
  nlohmann::json resolved() const;
  // What the id hashes (§3.1): visibility, credit, k_max, every block's window, bins and patterns
  // (each replaced by its spec's SHA-256), plus `block_order`, and `family` for a clean profile
  // only, so no legacy id moves (SPEC v3 AC-V3-0). The name and file are labels and are left out,
  // so a v1 config and the profile it implies have one id.
  nlohmann::json canonical() const;
  // The first 16 hex digits of SHA-256 over canonical().dump() (sorted keys, no whitespace).
  std::string id() const;
};

// Loads a profile or a v1 feature config, resolving pattern names against `registry`. Throws
// std::runtime_error "<file>:<line>:<col>: <message>" on a schema error, and with
// "error E403: " (§3.1) when the profile names an unknown pattern, lists a pattern under a block
// it does not emit into, has a visibility that conflicts with a pattern that pins its own, or
// breaks a family rule (SPEC v3 §3.2): a clean profile has visibility: timestamp and only clean
// patterns, a legacy one only legacy patterns, and a block without bins (a count block, R7) is
// in a clean profile and has only one_instance patterns.
// A v1 config gets, per block, every registered legacy pattern whose block matches, in registry
// order.
Profile load_profile(const std::string& path, std::span<const PatternInfo> registry = registered_patterns());
Profile parse_profile(std::string_view yaml_text, const std::string& file,
                      std::span<const PatternInfo> registry = registered_patterns());

// Every block that a registered legacy pattern emits into, in GFP order, with those patterns, and
// k_max = the compiled K_MAX; lcyc only from a profile that lists it (SPEC v2 §8 S3). Windows and bins are left empty: fuzz graphs bring their own.
// The default of `blz fuzz-diff` and `blz metamorphic` without --profile (the v1 behaviour).
Profile registry_profile(std::span<const PatternInfo> registry = registered_patterns());
// The same for the clean patterns, lcyc included (SPEC v3 §4.4: a clean test profile enables
// every pattern): family clean, visibility timestamp.
Profile clean_registry_profile(std::span<const PatternInfo> registry = registered_patterns());

// The start-up check of a profile against a backend (§3.1, §6.4 of v2). Throws
// "<file>:<line>:<col>: error E404: ..." when the backend cannot run the profile:
//   - oracle: a clean profile (the oracle computes the legacy definitions), or a block whose
//     patterns are not the set the oracle implements for it (§9 of v1 computes each block's
//     definition directly);
//   - cpu, cuda: a k_max other than the compiled K_MAX of a listed pattern that declares it.
// For a v1 config on cpu or cuda it runs the v1 start-up checks instead of the K_MAX one
// (check_backend_config: E402 and the K_MAX message), so v1 configs behave as in v1.
void check_profile_backend(const Profile& p, Backend b);

}  // namespace blz

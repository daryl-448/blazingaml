// Semantic profiles (SPEC v2 §3.1): the loader, the profile id (a v1 config and the profile it
// implies share one), E403 at load and E404 per backend.
#include <catch2/catch_test_macros.hpp>
#include <string>
#include <vector>

#include "blz/registry.hpp"
#include "runtime/profile.hpp"

using namespace blz;

namespace {

const std::string kSrc = BLZ_SOURCE_DIR;

std::string error_of(const std::string& yaml, std::span<const PatternInfo> reg = registered_patterns()) {
  try {
    parse_profile(yaml, "p.yaml", reg);
  } catch (const std::runtime_error& e) {
    return e.what();
  }
  return "ok";
}

std::string backend_error(const Profile& p, Backend b) {
  try {
    check_profile_backend(p, b);
  } catch (const std::runtime_error& e) {
    return e.what();
  }
  return "ok";
}

// A minimal v2 profile with the given body after `blocks:`.
std::string profile(const std::string& blocks, const std::string& head = "visibility: stream\ncredit: trigger\nk_max: 10\n") {
  return "version: 2\nprofile: t\n" + head + "blocks:\n" + blocks;
}

}  // namespace

TEST_CASE("profiles: the frozen profiles load and match the v1 configs' ids (SPEC v2 §3.1)") {
  const Profile d = load_profile(kSrc + "/configs/profiles/v1_default.yaml");
  const Profile p = load_profile(kSrc + "/configs/profiles/v1_parity.yaml");
  const Profile d1 = load_profile(kSrc + "/configs/features_default.yaml");
  const Profile p1 = load_profile(kSrc + "/configs/features_parity.yaml");
  CHECK(d.format == 2);
  CHECK(d1.format == 1);
  CHECK(d.name == "v1_default");
  CHECK(d1.name == "features_default");
  CHECK(d.id().size() == 16);
  CHECK(d.id() == d1.id());
  CHECK(p.id() == p1.id());
  CHECK(d.id() != p.id());
  CHECK(d.canonical() == d1.canonical());
  CHECK(d.causal());

  REQUIRE(d.blocks.size() == 6);
  const std::vector<Block> order = {Block::fan_in, Block::fan_out, Block::deg_in, Block::deg_out, Block::sg, Block::tcyc};
  for (std::size_t i = 0; i < order.size(); ++i) CHECK(d.blocks[i].block == order[i]);
  const ProfileBlock* sg = d.find(Block::sg);
  REQUIRE(sg != nullptr);
  REQUIRE(sg->patterns.size() == 2);
  CHECK(std::string(sg->patterns[0]->name) == "sg_gather");
  CHECK(std::string(sg->patterns[1]->name) == "sg_scatter");
  CHECK(sg->params.window == 21600);
  CHECK(sg->params.bins.size() == 10);
  CHECK(d.find(Block::tcyc)->params.bins.size() == 9);
  CHECK(p.find(Block::fan_in)->params.bins.size() == 32);

  const auto r = d.resolved();
  CHECK(r["profile"] == "v1_default");
  CHECK(r["blocks"]["sg"]["patterns"][0]["spec_sha256"].get<std::string>().size() == 64);
  CHECK(r["block_order"].size() == 6);
  CHECK(r["credit"] == "trigger");
  // The canonical form leaves out the labels.
  CHECK_FALSE(d.canonical().contains("profile"));
  CHECK(d.canonical()["credit"] == "trigger");

  // Every block runs on every backend.
  for (Backend b : {Backend::oracle, Backend::cpu, Backend::cuda}) {
    CHECK(backend_error(d, b) == "ok");
    CHECK(backend_error(p, b) == "ok");
    CHECK(backend_error(d1, b) == "ok");
  }
}

TEST_CASE("profiles: the id follows every semantic field and the block order") {
  const std::string fan = "  fan_in: {patterns: [fan_in], window: 86400, bins: [2, 3]}\n";
  const std::string deg = "  deg_in: {patterns: [deg_in], window: 86400, bins: [2, 3]}\n";
  const std::string base = parse_profile(profile(fan + deg), "a.yaml").id();
  // The name and the file are labels.
  std::string renamed = profile(fan + deg);
  renamed.replace(renamed.find("profile: t"), 10, "profile: other");
  CHECK(parse_profile(renamed, "b.yaml").id() == base);
  CHECK(parse_profile(profile(deg + fan), "a.yaml").id() != base);
  CHECK(parse_profile(profile(fan), "a.yaml").id() != base);
  CHECK(parse_profile(profile("  fan_in: {patterns: [fan_in], window: 86400, bins: [2, 4]}\n" + deg), "a.yaml").id() != base);
  CHECK(parse_profile(profile("  fan_in: {patterns: [fan_in], window: 3600, bins: [2, 3]}\n" + deg), "a.yaml").id() != base);
  CHECK(parse_profile(profile(fan + deg, "visibility: stream\ncredit: trigger\nk_max: 12\n"), "a.yaml").id() != base);
  CHECK(parse_profile(profile(fan + deg, "visibility: timestamp\ncredit: trigger\nk_max: 10\n"), "a.yaml").id() != base);
  // A changed spec changes the id: the patterns enter it by their spec's SHA-256.
  std::vector<PatternInfo> reg(registered_patterns().begin(), registered_patterns().end());
  for (PatternInfo& pi : reg)
    if (std::string(pi.name) == "fan_in") pi.spec_sha256 = "0000000000000000000000000000000000000000000000000000000000000000";
  CHECK(parse_profile(profile(fan + deg), "a.yaml", reg).id() != base);
}

TEST_CASE("profiles: E403 at load, with the file, line and column of the pattern") {
  CHECK(error_of(profile("  fan_in: {patterns: [fan_inn], window: 86400, bins: [2]}\n")) ==
        "p.yaml:7:23: error E403: unknown pattern 'fan_inn': no compiled pattern has this name");
  CHECK(error_of(profile("  fan_in: {patterns: [fan_out], window: 86400, bins: [2]}\n")) ==
        "p.yaml:7:23: error E403: pattern 'fan_out' (patterns/fan_out.yaml) emits into block 'fan_out', not 'fan_in'");
  // A pattern that pins visibility: timestamp (§6.1) in a stream profile, and the other way round.
  std::vector<PatternInfo> reg(registered_patterns().begin(), registered_patterns().end());
  for (PatternInfo& pi : reg)
    if (std::string(pi.name) == "tcyc") {
      pi.visibility = "timestamp";
      pi.visibility_loc = "patterns/tcyc.yaml:3:13";
    }
  CHECK(error_of(profile("  tcyc:\n    patterns: [tcyc]\n    window: 86400\n    bins: [2]\n"), reg) ==
        "p.yaml:8:16: error E403: pattern 'tcyc' pins visibility: timestamp (patterns/tcyc.yaml:3:13), but the "
        "profile has visibility: stream (p.yaml:3:13)");
  CHECK(error_of(profile("  tcyc: {patterns: [tcyc], window: 86400, bins: [2]}\n",
                         "visibility: timestamp\ncredit: trigger\nk_max: 10\n"),
                 reg) == "ok");
  // The library's specs pin nothing.
  for (const PatternInfo& pi : registered_patterns()) CHECK(std::string(pi.visibility).empty());
}

TEST_CASE("profiles: schema errors carry the location") {
  CHECK(error_of("version: 3\n") ==
        "p.yaml:1:10: a profile has version: 2 (a v1 feature config has no version key), found 3");
  CHECK(error_of(profile("  fan_in: {patterns: [fan_in], window: 86400, bins: [2]}\n") + "extra: 1\n") ==
        "p.yaml:8:1: unknown key 'extra' in the profile");
  CHECK(error_of(profile("  fan_in: {patterns: [fan_in], window: 86400, bins: [3, 2]}\n")) ==
        "p.yaml:7:57: fan_in: bins must be strictly increasing");
  CHECK(error_of(profile("  fan_in: {patterns: [fan_in], window: 0, bins: [2]}\n")) ==
        "p.yaml:7:40: fan_in: window must be > 0");
  CHECK(error_of(profile("  fan_inx: {patterns: [fan_in], window: 1, bins: [2]}\n")) ==
        "p.yaml:7:3: unknown block 'fan_inx'");
  CHECK(error_of(profile("  fan_in: {patterns: [fan_in, fan_in], window: 1, bins: [2]}\n")) ==
        "p.yaml:7:31: fan_in: pattern 'fan_in' is listed twice");
  CHECK(error_of(profile("  fan_in: {patterns: [fan_in], window: 1, bins: [2]}\n",
                         "visibility: batch\ncredit: trigger\nk_max: 10\n")) ==
        "p.yaml:3:13: visibility must be stream or timestamp, found 'batch'");
  CHECK(error_of(profile("  fan_in: {patterns: [fan_in], window: 1, bins: [2]}\n",
                         "visibility: stream\ncredit: participants\nk_max: 10\n")) ==
        "p.yaml:4:9: credit must be trigger, found 'participants'");
  CHECK(error_of(profile("  fan_in: {patterns: [fan_in], window: 1, bins: [2]}\n",
                         "visibility: stream\ncredit: trigger\nbatch: {size: 4, cuts: none}\nk_max: 10\n")) ==
        "p.yaml:5:1: unknown key 'batch' in the profile");
  CHECK(error_of(profile("  fan_in: {patterns: [fan_in], window: 1, bins: [2]}\n",
                         "visibility: stream\ncredit: trigger\nk_max: 1\n")) == "p.yaml:5:8: k_max must be >= 2");
  CHECK(error_of("version: 2\nprofile: 1x\n").find("p.yaml:2:10: 'profile' must be an identifier") == 0);
  CHECK(error_of("version: 2\nprofile: t\n").find("needs 'visibility'") != std::string::npos);
}

TEST_CASE("profiles: E404 names the backend and the profile key") {
  const std::string fan = "  fan_in: {patterns: [fan_in], window: 86400, bins: [2]}\n";
  const Profile ts = parse_profile(profile(fan, "visibility: timestamp\ncredit: trigger\nk_max: 10\n"), "p.yaml");
  CHECK_FALSE(ts.causal());
  for (Backend b : {Backend::oracle, Backend::cpu, Backend::cuda})  // the oracle since P8, cpu and cuda since P10
    CHECK(backend_error(ts, b) == "ok");
  // The oracle computes sg as sg_gather + sg_scatter; one of them alone is a different block.
  const Profile half = parse_profile(profile("  sg: {patterns: [sg_gather], window: 21600, bins: [2]}\n"), "p.yaml");
  CHECK(backend_error(half, Backend::oracle) ==
        "p.yaml:7:3: error E404: the oracle computes block sg as the patterns [sg_gather, sg_scatter] together; it "
        "cannot run another pattern list");
  CHECK(backend_error(half, Backend::cpu) == "ok");
  // tcyc was compiled with K_MAX = 10; the patterns that declare no K_MAX run at any k_max.
  const std::string k12 = "visibility: stream\ncredit: trigger\nk_max: 12\n";
  const Profile tc = parse_profile(profile("  tcyc: {patterns: [tcyc], window: 86400, bins: [2]}\n", k12), "p.yaml");
  CHECK(backend_error(tc, Backend::oracle) == "ok");
  CHECK(backend_error(tc, Backend::cpu) ==
        "p.yaml:5:8: error E404: the cpu backend cannot run k_max: 12: pattern 'tcyc' (patterns/tcyc.yaml) was "
        "compiled with K_MAX = 10");
  CHECK(backend_error(parse_profile(profile(fan, k12), "p.yaml"), Backend::cpu) == "ok");
  // A v1 config keeps the v1 start-up checks: E402 for a block it leaves out, then K_MAX.
  const Profile v1 = parse_profile("k_max: 12\nblocks:\n  fan_in: {window: 86400, bins: [2]}\n", "cfg.yaml");
  CHECK(v1.format == 1);
  CHECK(backend_error(v1, Backend::cpu) ==
        "patterns/deg_in.yaml:4:8: error E402: pattern 'deg_in' emits into block 'deg_in', which the feature config "
        "cfg.yaml does not define");
  CHECK(backend_error(v1, Backend::oracle) == "ok");
}

TEST_CASE("profiles: registry_profile is every registered block with the compiled K_MAX") {
  const Profile r = registry_profile();
  CHECK(r.k_max == 10);
  CHECK(r.blocks.size() == 11);  // every block but lcyc
  REQUIRE(r.find(Block::sg_ord) != nullptr);
  CHECK(std::string(r.find(Block::sg_ord)->patterns.at(0)->name) == "sg_ordered");
  CHECK(r.find(Block::sg)->patterns.size() == 2);
  CHECK(r.find(Block::fan_in)->params.bins.empty());
}

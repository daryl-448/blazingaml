#include <catch2/catch_test_macros.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/registration.hpp"

using namespace blz;

namespace {

std::string error_of(const std::vector<PatternRef>& pats, const FeatureConfig& cfg) {
  try {
    check_registration(pats, cfg, "cfg.yaml");
  } catch (const std::runtime_error& e) {
    return e.what();
  }
  return "ok";
}

}  // namespace

TEST_CASE("registration: a block missing from the feature config is E402 (SPEC §6.7, §13.1)") {
  const FeatureConfig cfg = parse_feature_config(
      "blocks:\n  fan_in: {window: 10, bins: [2, 3]}\n  sg: {window: 5, bins: [2]}\n", "cfg.yaml");
  const std::vector<PatternRef> ok = {{"fan_in", "fan_in", "patterns/fan_in.yaml:4:8"},
                                      {"sg_gather", "sg", "patterns/sg_gather.yaml:4:8"},
                                      {"sg_scatter", "sg", "patterns/sg_scatter.yaml:4:8"}};
  CHECK(error_of(ok, cfg) == "ok");

  auto missing = ok;
  missing.push_back({"tcyc", "tcyc", "patterns/tcyc.yaml:4:8"});
  CHECK(error_of(missing, cfg) ==
        "patterns/tcyc.yaml:4:8: error E402: pattern 'tcyc' emits into block 'tcyc', which the "
        "feature config cfg.yaml does not define");

  const std::vector<PatternRef> unknown = {{"p", "fanin", "x.yaml:4:8"}};
  CHECK(error_of(unknown, cfg).rfind("x.yaml:4:8: error E402: ", 0) == 0);
}

// SPEC v2 §8 S7, multi-pattern fusion: blzc's fusion plan (plan_fusions, emit_fused_cuda) and the
// runtime's choice of fused units (cuda_units). The fused kernels' rows are test_pack_fuse.cpp's.
#include <catch2/catch_test_macros.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "codegen/codegen.hpp"
#include "runtime/fusion.hpp"
#include "runtime/profile.hpp"

using namespace blz;

namespace {

std::vector<ir::Pattern> library(const std::vector<std::string>& names) {
  std::vector<ir::Pattern> out;
  for (const std::string& n : names) out.push_back(ir::compile_file(std::string(BLZ_SOURCE_DIR) + "/patterns/" + n + ".yaml"));
  return out;
}

const PatternInfo* pattern(const std::string& name) {
  for (const PatternInfo& p : registered_patterns())
    if (p.name == name) return &p;
  FAIL("no pattern " << name);
  return nullptr;
}

std::vector<std::string> names(const std::vector<CudaUnit>& units) {
  std::vector<std::string> out;
  for (const CudaUnit& u : units) out.push_back(u.name());
  return out;
}

}  // namespace

TEST_CASE("fusion: blzc fuses sweep-shaped members over one list into one scan (D-46)") {
  const auto pats = library({"fan_in", "deg_in", "fan_out", "deg_out"});
  const auto fs = codegen::plan_fusions(pats, {{"fan_in", "deg_in"}, {"deg_out", "fan_out"}});
  REQUIRE(fs.size() == 2);
  CHECK(fs[0].id == "fan_in__deg_in");
  CHECK(fs[0].name == "fan_in+deg_in");
  CHECK(fs[0].members == std::vector<std::size_t>{0, 1});
  CHECK(fs[0].slots == std::vector<std::uint32_t>{0, 1});
  CHECK(fs[0].first_scans == std::vector<std::string>{"dst.in", "dst.in"});
  CHECK(fs[1].name == "deg_out+fan_out");  // --fuse order
  CHECK(fs[1].first_scans == std::vector<std::string>{"src.out", "src.out"});

  const std::vector<std::string> stems = {"fan_in", "deg_in", "fan_out", "deg_out"};
  const std::string scan = codegen::emit_fused_cuda(fs[0], pats).text;
  CHECK(scan.find("fused_fan_in__deg_in_trigger(g, p, e, lane, cuda::kWarpSize, out);") != std::string::npos);
  CHECK(scan.find("g.in.distinct_first(j, m0_lo) && m0_keep(y)) ++m0_i_n;") != std::string::npos);  // fan_in: distinct
  CHECK(scan.find("if (j >= m1_lo && m1_keep(y)) ++m1_i_n;") != std::string::npos);                  // deg_in: edges
  CHECK(scan.find("}(sinks[1]);") != std::string::npos);

  const std::string reg = codegen::emit_registry(pats, stems, fs).text;
  CHECK(reg.find("\"fan_in+deg_in\"") != std::string::npos);
  CHECK(reg.find("std::span<const FusionInfo> registered_fusions() { return kFusions; }") != std::string::npos);
}

TEST_CASE("fusion: blzc rejects bad --fuse groups") {
  const auto pats = library({"fan_in", "deg_in", "fan_out", "sg_gather", "sg_scatter", "lcyc"});
  const auto rejects = [&](const std::vector<std::vector<std::string>>& groups, const std::string& why) {
    try {
      codegen::plan_fusions(pats, groups);
      FAIL("accepted");
    } catch (const std::runtime_error& e) {
      INFO(e.what());
      CHECK(std::string(e.what()).find(why) != std::string::npos);
    }
  };
  rejects({{"fan_in"}}, "a fusion has 2 to 4 patterns");
  rejects({{"fan_in", "nope"}}, "no pattern named 'nope'");
  rejects({{"fan_in", "deg_in"}, {"deg_in", "lcyc"}}, "'deg_in' is already in a fusion");
  rejects({{"fan_in", "lcyc"}}, "work_limit: true");
  // sg_gather and sg_scatter scan X.in and Y.out with binding stages: not one sweep (D-46).
  rejects({{"sg_gather", "sg_scatter"}}, "'sg_gather' is not a count over one trigger list");
  rejects({{"fan_in", "fan_out"}}, "'fan_out' scans src.out, 'fan_in' scans dst.in");
}

TEST_CASE("fusion: cuda_units fuses exactly the blocks whose patterns are a fusion's slots") {
  if (registered_fusions().empty()) SKIP("built without blzc --fuse");
  const Profile prof = load_profile(std::string(BLZ_SOURCE_DIR) + "/configs/profiles/v1_default.yaml");
  std::vector<const ProfileBlock*> all;
  for (const ProfileBlock& b : prof.blocks) all.push_back(&b);
  CHECK(names(cuda_units(all, false)) == std::vector<std::string>{"fan_in", "fan_out", "deg_in", "deg_out", "sg", "tcyc"});
  const auto fused = cuda_units(all, true);
  CHECK(names(fused) == std::vector<std::string>{"fan_in+deg_in", "fan_out+deg_out", "sg", "tcyc"});
  // Each fused unit's blocks are in slot order: slot s writes blocks[s].
  CHECK(fused[0].blocks == std::vector<const ProfileBlock*>{prof.find(Block::fan_in), prof.find(Block::deg_in)});
  CHECK(fused[1].blocks == std::vector<const ProfileBlock*>{prof.find(Block::fan_out), prof.find(Block::deg_out)});

  // A selection without a fusion's partner block stays unfused; the order is the selection's.
  const std::vector<const ProfileBlock*> some = {prof.find(Block::deg_out), prof.find(Block::fan_in), prof.find(Block::fan_out)};
  CHECK(names(cuda_units(some, true)) == std::vector<std::string>{"fan_out+deg_out", "fan_in"});

  // A block with other patterns than the slot's members is not the slot's block.
  ProfileBlock fan_in = *prof.find(Block::fan_in);
  fan_in.patterns = {pattern("fan_in"), pattern("fan_total")};
  CHECK(names(cuda_units(std::vector<const ProfileBlock*>{&fan_in, prof.find(Block::deg_in)}, true)) ==
        std::vector<std::string>{"fan_in", "deg_in"});
}

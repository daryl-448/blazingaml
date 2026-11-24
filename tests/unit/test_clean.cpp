// The clean family's runtime (SPEC v3 §3.2): count blocks (R7) and the loop-free index (R1), on
// hand-checked graphs, and the CPU backend against the clean oracle on fuzz graphs.
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <string>
#include <vector>

#include "blz/registry.hpp"
#include "graph/index.hpp"
#include "runtime/cpu_backend.hpp"
#include "runtime/fuzz_diff.hpp"
#include "runtime/profile.hpp"

using namespace blz;

namespace {

const PatternInfo* pattern(const std::string& name) {
  for (const PatternInfo& p : registered_patterns())
    if (p.name == name) return &p;
  FAIL("no pattern " << name);
  return nullptr;
}

struct Edges {
  std::vector<std::uint32_t> src, dst;
  std::vector<std::int32_t> ts;
  std::uint32_t n;
};

// One block's rows over every trigger, added into `rows` (sized by the caller).
CpuBlockTiming mine(const Edges& g, SelfLoops loops, const std::vector<std::string>& names, const BlockParams& p,
                    Visibility vis, Schedule s, std::vector<std::uint32_t>& rows) {
  const GraphIndex index(g.src, g.dst, g.ts, g.n, loops);
  std::vector<const PatternInfo*> pats;
  for (const std::string& n : names) pats.push_back(pattern(n));
  TriggerSel all;
  all.last = g.src.size();
  return mine_cpu(index.view(), pats, p, vis, all, s, rows.data());
}

std::vector<std::uint32_t> count_rows(const Edges& g, const std::string& name, Schedule s = Schedule::dynamic,
                                      SelfLoops loops = SelfLoops::drop) {
  std::vector<std::uint32_t> rows(g.src.size(), 0);
  mine(g, loops, {name}, {100, {}}, Visibility::timestamp, s, rows);
  return rows;
}

// e3 is a self-loop; e0 and e1 tie at t = 10.
const Edges kG1 = {{1, 2, 2, 0, 3, 0}, {0, 0, 0, 0, 4, 2}, {10, 10, 11, 12, 13, 14}, 5};

}  // namespace

TEST_CASE("count rows: an emit adds its size, min_size aside; a sum past UINT32_MAX clamps and loses one emit") {
  std::uint32_t row = 0;
  RowSink s{&row, nullptr, 1, true};
  s.emit(0, 2);
  CHECK(row == 0);
  s.emit(1, 2);
  CHECK(row == 1);
  s.emit(5, 2);
  CHECK(row == 6);
  CHECK(s.saturated == 0);
  row = UINT32_MAX - 3;
  s.emit(3, 2);
  CHECK(row == UINT32_MAX);
  CHECK(s.saturated == 0);
  s.emit(1, 2);
  CHECK(row == UINT32_MAX);
  CHECK(s.saturated == 1);
  row = UINT32_MAX - 1;
  s.emit(7, 2);
  CHECK(row == UINT32_MAX);
  CHECK(s.saturated == 2);
}

TEST_CASE("histogram rows: an emit below min_size or b_0 is dropped") {
  const std::uint32_t bins[] = {2, 4};
  std::uint32_t row[2] = {};
  RowSink s{row, bins, 2};
  for (std::int64_t size : {0, 1, 2, 3, 4, 9}) s.emit(size, 2);
  CHECK(row[0] == 2);
  CHECK(row[1] == 2);
  s.emit(3, 4);  // a spec's min_size above 2 (SPEC §6.6)
  CHECK(row[0] == 2);
}

TEST_CASE("count blocks on a hand-checked graph: 0, 1 and 2+, ties seen (R3), a self-loop trigger's row is 0 (R1)") {
  for (const Schedule s : {Schedule::dynamic, Schedule::sorted}) {
    INFO("schedule " << schedule_name(s));
    // Sweep plans (SPEC v2 §7.1).
    CHECK(count_rows(kG1, "clean_fan_in", s) == std::vector<std::uint32_t>{2, 2, 2, 0, 1, 1});
    CHECK(count_rows(kG1, "clean_deg_in", s) == std::vector<std::uint32_t>{2, 2, 3, 0, 1, 1});
    CHECK(count_rows(kG1, "clean_fan_out", s) == std::vector<std::uint32_t>{1, 1, 1, 0, 1, 1});
    CHECK(count_rows(kG1, "clean_deg_out", s) == std::vector<std::uint32_t>{1, 1, 2, 0, 1, 1});
    // Per-trigger plans. gs for e5 = 0 -> 2: A = {1} (2 is W; 0 -> 0 is no source, R1), B = {2}.
    CHECK(count_rows(kG1, "clean_gs", s) == std::vector<std::uint32_t>{0, 0, 0, 0, 0, 1});
    CHECK(count_rows(kG1, "clean_fan_total", s) == std::vector<std::uint32_t>{2, 2, 2, 0, 1, 1});
    CHECK(count_rows(kG1, "clean_fan_oneway_out", s) == std::vector<std::uint32_t>{1, 1, 1, 0, 1, 0});
  }
  // On an index that kept 0 -> 0, the same spec would count 0 as a source and a sink of 0.
  CHECK(count_rows(kG1, "clean_gs", Schedule::dynamic, SelfLoops::keep).back() == 2);
  // Every histogram block of a clean pattern leaves the self-loop trigger's row zero too.
  for (const char* name : {"clean_sg_gather", "clean_sg_scatter", "clean_sg_ord", "clean_tcyc", "clean_lcyc",
                           "clean_tchain"}) {
    INFO(name);
    std::vector<std::uint32_t> rows(kG1.src.size() * 2, 0);
    mine(kG1, SelfLoops::drop, {name}, {100, {2, 3}}, Visibility::timestamp, Schedule::dynamic, rows);
    CHECK(rows[3 * 2] == 0);
    CHECK(rows[3 * 2 + 1] == 0);
  }
}

TEST_CASE("count blocks saturate at UINT32_MAX, one lost emit per clamped row") {
  for (const char* name : {"clean_fan_in", "clean_gs"}) {  // a sweep driver, a per-trigger one
    for (const Schedule s : {Schedule::dynamic, Schedule::sorted}) {
      INFO(name << " " << schedule_name(s));
      std::vector<std::uint32_t> rows(kG1.src.size(), UINT32_MAX - 1);  // the drivers add into the rows
      const CpuBlockTiming tm = mine(kG1, SelfLoops::drop, {name}, {100, {}}, Visibility::timestamp, s, rows);
      const std::uint32_t M = UINT32_MAX;
      if (std::string(name) == "clean_fan_in") {
        CHECK(rows == std::vector<std::uint32_t>{M, M, M, M - 1, M, M});  // + 2, 2, 2, 0, 1, 1
        CHECK(tm.saturated == 3);
      } else {
        CHECK(rows == std::vector<std::uint32_t>{M - 1, M - 1, M - 1, M - 1, M - 1, M});  // only e5 emits
        CHECK(tm.saturated == 0);
      }
    }
  }
}

TEST_CASE("the loop-free index: sg no longer counts an endpoint through its self-loop (R1, R6a)") {
  // s = 1 has a self-loop and pays 3, 2 and 4; 4 and the trigger e5 = 2 -> 3 pay w = 3.
  const Edges g = {{1, 1, 1, 1, 4, 2}, {1, 3, 2, 4, 3, 3}, {1, 2, 3, 4, 5, 6}, 5};
  const auto sg_rows = [&](SelfLoops loops, const std::vector<std::string>& pats, Visibility vis) {
    std::vector<std::uint32_t> rows(g.src.size() * 2, 0);
    mine(g, loops, pats, {100, {2, 3}}, vis, Schedule::dynamic, rows);
    return rows;
  };
  // Clean: e5's I(1, 3) = N_out(1) ∩ N_in(3) = {2, 4}, one instance of size 2; e4 = 4 -> 3 has
  // I(1, 3) = {4} only, no instance.
  CHECK(sg_rows(SelfLoops::drop, {"clean_sg_gather", "clean_sg_scatter"}, Visibility::timestamp) ==
        std::vector<std::uint32_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0});
  // The clean spec has no filter for it: on an index with 1 -> 1, s itself joins I, so e4 gets
  // {1, 4} and e5 {1, 2, 4}.
  const std::vector<std::uint32_t> with_s = {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1};
  CHECK(sg_rows(SelfLoops::keep, {"clean_sg_gather", "clean_sg_scatter"}, Visibility::timestamp) == with_s);
  // v1's gather role (Q1) counts s through its self-loop the same way.
  CHECK(sg_rows(SelfLoops::keep, {"sg_gather", "sg_scatter"}, Visibility::stream) == with_s);
}

// AC-V3-1's fast share (the gate is `blz fuzz-diff --profile configs/profiles/clean_all.yaml`): both
// schedules against the clean oracle. lcyc apart, on fewer graphs at a lower work limit, where some
// graphs are limited (R11: inconclusive for lcyc, their stop eids compared) and the others complete.
TEST_CASE("clean_all: the CPU backend equals the clean oracle on fuzz graphs, both schedules (SPEC v3 F6, R11)") {
  const Profile all = load_profile(std::string(BLZ_SOURCE_DIR) + "/configs/profiles/clean_all.yaml");
  Profile rest = all, lcyc = all;
  std::erase_if(rest.blocks, [](const ProfileBlock& pb) { return pb.block == Block::lcyc; });
  std::erase_if(lcyc.blocks, [](const ProfileBlock& pb) { return pb.block != Block::lcyc; });
  const auto check = [](const Profile& prof, const FuzzDiff& d, std::uint64_t graphs) {
    CHECK(d.graphs == graphs);
    for (const ProfileBlock& pb : prof.blocks) {
      INFO(block_name(pb.block));
      CHECK(d.mismatched_rows[int(pb.block)] == 0);
    }
    CHECK(d.bad_seeds.empty());
  };
  const FuzzDiff r = fuzz_diff_cpu(rest, 0, 200, {});
  check(rest, r, 200);
  CHECK(r.limited_seeds.empty());
  const FuzzDiff l = fuzz_diff_cpu(lcyc, 0, 100, {}, 100'000);
  check(lcyc, l, 100);
  CHECK(l.limited_graphs[int(Block::lcyc)] > 0);
  CHECK(l.limited_graphs[int(Block::lcyc)] < 50);
  CHECK(l.limited_graphs[int(Block::lcyc)] == l.limited_seeds.size());
  // The CPU's sorted schedule against its dynamic one (--reference cpu), every block.
  check(all, fuzz_diff_cpu(all, 0, 60, {}, 100'000, Reference::cpu), 60);
  // The sparse-time setting (SPEC v3 §4.4), every block, each with nonzero rows (path hit).
  const FuzzDiff sp = fuzz_diff_cpu(all, 0, 1000, {}, kDefaultWorkLimit, Reference::oracle, FuzzSetting::sparse);
  check(all, sp, 1000);
  CHECK(sp.limited_seeds.empty());
  CHECK(sp.hits.multi_component_graphs == 1000);
  for (const ProfileBlock& pb : all.blocks) {
    INFO(block_name(pb.block));
    CHECK(sp.nonzero_rows[int(pb.block)] > 0);
  }
}

TEST_CASE("clean profiles: clean_default, clean_wc_hi_small and clean_all (SPEC v3 §3.3, F5)") {
  const std::string dir = std::string(BLZ_SOURCE_DIR) + "/configs/profiles/";
  const auto cols = [](const Profile& p) {
    std::uint32_t c = 0;
    for (const ProfileBlock& pb : p.blocks) c += block_cols(pb.params);
    return c;
  };
  const Profile d = load_profile(dir + "clean_default.yaml");
  const Profile w = load_profile(dir + "clean_wc_hi_small.yaml");
  const Profile a = load_profile(dir + "clean_all.yaml");
  for (const Profile* p : {&d, &w, &a}) {
    INFO(p->name);
    CHECK(p->family == Family::clean);
    CHECK(p->visibility == Visibility::timestamp);
    CHECK(p->k_max == 10);
    for (Block b : {Block::fan_in, Block::fan_out, Block::deg_in, Block::deg_out})
      CHECK(p->find(b)->params.bins.empty());
    CHECK(p->find(Block::sg)->patterns.size() == 2);
  }
  CHECK(cols(d) == 33);
  CHECK(cols(w) == 33);
  CHECK(cols(a) == 54);
  CHECK(d.find(Block::sg)->params.window == 21600);
  CHECK(w.find(Block::fan_in)->params.window == 320400);
  CHECK(w.find(Block::sg_ord)->params.window == 342000);
  CHECK(w.find(Block::tcyc)->params.window == 320400);
  REQUIRE(a.blocks.size() == kAllBlocks.size());  // every clean block
  for (Block b : {Block::gs, Block::fan_oneway_out, Block::fan_total}) CHECK(a.find(b)->params.bins.empty());
  CHECK(a.find(Block::lcyc)->params.bins == a.find(Block::tcyc)->params.bins);
  CHECK(a.find(Block::tchain)->params.bins == a.find(Block::tcyc)->params.bins);
  CHECK(d.id() != w.id());
}

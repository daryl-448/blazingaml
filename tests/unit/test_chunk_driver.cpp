// SPEC v3 H7: the chunk driver. On every backend this build has, chunks of 1, 2, 3 and more than the
// selection's rows give the rows of one chunk, for the whole stream, a range and (oracle, cpu) a
// sample, fused CUDA units included; the backends agree; and a work-limit abort (R11) leaves no
// block file.
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "fixtures.hpp"
#include "graph/fuzz_gen.hpp"
#include "runtime/chunk_driver.hpp"
#include "runtime/profile.hpp"

using namespace blz;

namespace {

const std::string kProfiles = std::string(BLZ_SOURCE_DIR) + "/configs/profiles/";

// The profile with every window (vstats' too) at w, as a fuzz graph is mined.
Profile at_window(Profile p, std::int32_t w) {
  for (ProfileBlock& b : p.blocks) b.params.window = w;
  if (p.vstats) p.vstats->window = w;
  return p;
}

std::vector<Backend> backends() {
  if (cuda_built()) return {Backend::oracle, Backend::cpu, Backend::cuda, Backend::hybrid};
  return {Backend::oracle, Backend::cpu};
}

struct Mined {
  std::map<std::string, std::vector<std::uint32_t>> u32;
  std::map<std::string, std::vector<double>> f64;
  std::vector<BlockRun> runs;
};

Mined mine(const FuzzGraph& fg, const std::vector<double>& amount, const Profile& prof, const MineOptions& o,
           const TriggerSel& sel) {
  ChunkDriver d(EdgeStream{fg.src, fg.dst, fg.ts, fg.n}, amount, prof, o);
  std::vector<const ProfileBlock*> blocks;
  for (const ProfileBlock& b : prof.blocks) blocks.push_back(&b);
  MemorySink sink;
  Mined m;
  m.runs = d.run(blocks, prof.vstats.has_value(), sel, sink);
  m.u32 = std::move(sink.u32);
  m.f64 = std::move(sink.f64);
  return m;
}

// Bit for bit: vstats rows compare as bytes.
bool same(const Mined& a, const Mined& b) {
  if (a.u32 != b.u32 || a.f64.size() != b.f64.size()) return false;
  for (const auto& [k, v] : a.f64) {
    const auto it = b.f64.find(k);
    if (it == b.f64.end() || it->second.size() != v.size() ||
        std::memcmp(v.data(), it->second.data(), v.size() * sizeof(double)) != 0)
      return false;
  }
  return true;
}

MineOptions options(Backend be) {
  MineOptions o;
  o.backend = be;
  o.heavy_threshold = 4;  // cuda, hybrid: both kernels on fuzz graphs
  o.cpu_threshold = 4;    // hybrid: both sides
  if (be == Backend::hybrid) o.schedule = Schedule::sorted;
  return o;
}

}  // namespace

TEST_CASE("chunk driver: any chunk size gives one chunk's rows, on every backend and selection") {
  const std::vector<Profile> profiles = {load_profile(kProfiles + "clean_all.yaml"),
                                         load_profile(kProfiles + "v1_parity.yaml")};
  std::uint64_t fused = 0, whole = 0, nonzero = 0;
  for (const Profile& base : profiles)
    for (const FuzzSetting setting : {FuzzSetting::dense, FuzzSetting::sparse})
      for (std::uint64_t seed = 0, used = 0; used < 3; ++seed) {
        const FuzzGraph fg = fuzz_graph(seed, setting);
        // Small graphs: a chunk of one row runs every block's kernels once per row.
        if (fg.src.size() < 20 || fg.src.size() > 80) continue;
        ++used;
        const Profile prof = at_window(base, fg.window);
        const std::vector<double> amount = fuzz_amounts(fg.seed, fg.src.size());
        const std::uint64_t m = fg.src.size();
        const Splits splits = compute_splits(SplitRule::legacy_hi_small, fg.ts);
        std::vector<TriggerSel> sels = {select_triggers("all", m, splits),
                                        select_triggers("range:" + std::to_string(m / 4) + ":" + std::to_string(m - m / 5), m, splits),
                                        select_triggers("sample:40:" + std::to_string(seed), m, splits)};
        std::optional<Mined> oracle_all;
        for (const Backend be : backends()) {
          for (const TriggerSel& sel : sels) {
            if (sel.sampled && (be == Backend::cuda || be == Backend::hybrid)) continue;  // an eid range only
            INFO(base.name << " " << fuzz_setting_name(setting) << " seed " << seed << " " << backend_name(be) << " "
                           << sel.spec);
            MineOptions o = options(be);
            const Mined one = mine(fg, amount, prof, o, sel);  // the default chunk: more than m rows
            for (const BlockRun& r : one.runs) {
              CHECK(r.out.rows == sel.rows());
              CHECK(r.out.chunks == (sel.rows() > 0));
              whole += r.whole;
            }
            if (be == Backend::oracle && sel.spec == "all") oracle_all = one;
            if (sel.spec == "all") CHECK(same(one, *oracle_all));  // the backends agree
            for (const std::uint64_t chunk : {std::uint64_t(1), std::uint64_t(2), std::uint64_t(3), sel.rows() + 5}) {
              INFO("chunk " << chunk);
              o.chunk_rows = chunk;
              const Mined got = mine(fg, amount, prof, o, sel);
              CHECK(same(got, one));
              for (const BlockRun& r : got.runs) {
                CHECK(r.out.chunks == (sel.rows() + chunk - 1) / chunk);
                nonzero += r.out.instances > 0;
              }
            }
            if (be == Backend::cuda) {  // fused (SPEC v2 §8 S7) and packed, chunked too
              o.fuse = o.pack = true;
              for (const std::uint64_t chunk : {std::uint64_t(1), std::uint64_t(3), kDefaultChunkRows}) {
                o.chunk_rows = chunk;
                const Mined got = mine(fg, amount, prof, o, sel);
                CHECK(same(got, one));
                for (const BlockRun& r : got.runs) fused += r.fusion != nullptr;
              }
            }
          }
        }
      }
  CHECK((fused > 0) == cuda_built());
  CHECK(whole > 0);    // the cpu's sweep blocks, mined once and delivered in chunks
  CHECK(nonzero > 0);  // the rows compared are not all zero
}

TEST_CASE("chunk driver: a work-limit abort (R11) leaves no block file, at any chunk size") {
  const Profile all = load_profile(kProfiles + "clean_all.yaml");
  // A graph whose first trigger over the limit has eid >= 3, so chunks of 1 and 2 rows reach the
  // lcyc stream before the abort.
  std::optional<FuzzGraph> g;
  std::uint64_t stop = 0;
  for (std::uint64_t seed = 0; seed < 200 && !g; ++seed) {
    const FuzzGraph fg = fuzz_graph(seed);
    MineOptions o = options(Backend::oracle);
    o.work_limit = 1;
    try {
      mine(fg, fuzz_amounts(fg.seed, fg.src.size()), at_window(all, fg.window), o,
           select_triggers("all", fg.src.size(), compute_splits(SplitRule::legacy_hi_small, fg.ts)));
    } catch (const WorkLimitExceeded& x) {
      if (x.eid >= 3) {
        g = fg;
        stop = x.eid;
      }
    }
  }
  REQUIRE(g);
  const Profile prof = at_window(all, g->window);
  const std::vector<double> amount = fuzz_amounts(g->seed, g->src.size());
  const TriggerSel sel = select_triggers("all", g->src.size(), compute_splits(SplitRule::legacy_hi_small, g->ts));
  std::vector<const ProfileBlock*> blocks;
  for (const ProfileBlock& b : prof.blocks) blocks.push_back(&b);
  const std::string dir = test_tmp("chunk_abort");
  for (const Backend be : backends())
    for (const std::uint64_t chunk : {std::uint64_t(1), std::uint64_t(2), kDefaultChunkRows}) {
      INFO(backend_name(be) << " chunk " << chunk);
      std::filesystem::remove_all(dir);
      std::filesystem::create_directories(dir);
      MineOptions o = options(be);
      o.work_limit = 1;
      o.chunk_rows = chunk;
      ChunkDriver d(EdgeStream{g->src, g->dst, g->ts, g->n}, amount, prof, o);
      NpySink sink(dir);
      std::uint64_t finished = 0;
      try {
        d.run(blocks, true, sel, sink, [&](const BlockRun&) { ++finished; });
        FAIL("no abort");
      } catch (const WorkLimitExceeded& x) {
        CHECK(x.eid == stop);
      }
      CHECK(finished == 0);
      CHECK(std::filesystem::is_empty(dir));
    }
}

#include "runtime/metamorphic.hpp"

#include <algorithm>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string_view>

#include "graph/fuzz_gen.hpp"
#include "graph/index.hpp"
#include "oracle/profile_oracle.hpp"
#include "runtime/cpu_backend.hpp"
#include "runtime/cuda_backend.hpp"
#include "runtime/hybrid.hpp"

namespace blz {

namespace {

struct Rng {  // splitmix64, as in graph/fuzz_gen.cpp
  std::uint64_t s;
  std::uint64_t next() {
    std::uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }
  std::uint64_t below(std::uint64_t k) { return next() % k; }
};

struct Graph {
  std::vector<std::uint32_t> src, dst;
  std::vector<std::int32_t> ts;
  std::uint32_t n = 0;
  std::uint64_t m() const { return src.size(); }
};

// One run of the backend over a graph: the oracle and cpu have one; cuda one per light/heavy
// threshold; the hybrid one per pair of a GPU threshold and a CPU threshold.
struct Run {
  std::uint32_t gpu = 0, cpu = 0;
};
using BlockBins = std::array<std::vector<std::uint32_t>, kAllBlocks.size()>;

// A graph's rows on one backend, rows[r][block] for run r. A block that reached the work limit in
// some run is limited: inconclusive (SPEC v3 R11), its rows are not compared.
struct Mined {
  std::vector<std::array<std::vector<std::uint32_t>, kAllBlocks.size()>> rows;
  std::array<bool, kAllBlocks.size()> limited{};
};

Mined mine_all(const Graph& g, const Profile& prof, Backend be, std::int32_t window, const BlockBins& bins,
               std::uint64_t work_limit, std::span<const Run> runs, CudaMiner* miner) {
  const std::uint64_t m = g.m();
  std::optional<ProfileOracle> oracle;
  std::optional<GraphIndex> index;
  std::optional<DeviceGraph> dg;
  if (be == Backend::oracle) oracle.emplace(EdgeStream{g.src, g.dst, g.ts, g.n}, prof);
  else index.emplace(g.src, g.dst, g.ts, g.n, prof.self_loops());
  if (miner) dg.emplace(index->view());
  TriggerSel all;
  all.last = m;
  Mined out;
  out.rows.resize(runs.size());
  for (std::size_t t = 0; t < runs.size(); ++t)
    for (const ProfileBlock& pb : prof.blocks) {
      BlockParams p{window, bins[int(pb.block)]};
      p.work_limit = work_limit;
      std::vector<std::uint32_t>& r = out.rows[t][int(pb.block)];
      r.assign(m * block_cols(p), 0);
      try {
        switch (be) {
          case Backend::oracle:
            // check_profile_backend holds for the profiles under test, so an sg block of one pattern
            // is the order-relaxation bound, the gather role alone (the oracle's sg is both roles).
            if (pb.block == Block::sg && pb.patterns.size() == 1) oracle->mine_sg_gather(p, 0, m, r.data());
            else oracle->mine(pb.block, p, 0, m, r.data());
            break;
          case Backend::cpu:
            mine_cpu(index->view(), pb.patterns, p, prof.visibility, all, Schedule::dynamic, r.data());
            break;
          case Backend::cuda: miner->mine(*dg, pb.patterns, p, prof.visibility, 0, m, runs[t].gpu, r.data()); break;
          case Backend::hybrid:
            p.reach_prune = true;  // as blz mine --backend hybrid runs it (SPEC v2 §8 S6)
            mine_hybrid(index->view(), *dg, *miner, pb.patterns, p, prof.visibility, 0, m, runs[t].cpu, runs[t].gpu,
                        Schedule::sorted, r.data());
            break;
        }
      } catch (const WorkLimitExceeded&) {
        out.limited[int(pb.block)] = true;
      }
    }
  return out;
}

}  // namespace

const char* meta_test_name(MetaTest t) {
  switch (t) {
    case MetaTest::prefix_closure: return "prefix_closure";
    case MetaTest::relabeling: return "relabeling";
    case MetaTest::time_shift: return "time_shift";
    case MetaTest::visibility_invariance: return "visibility_invariance";
    case MetaTest::order_relaxation: return "order_relaxation";
  }
  return "?";
}

bool meta_test_applies(MetaTest t, const Profile& prof, [[maybe_unused]] Backend be) {
  switch (t) {
    // A clean profile's visibility is timestamp, fixed (SPEC v3 R3), so it has nothing to flip.
    case MetaTest::visibility_invariance:
      return prof.family == Family::legacy && (prof.find(Block::tcyc) || prof.find(Block::tchain));
    case MetaTest::order_relaxation: return prof.find(Block::sg_ord) != nullptr;
    default: return true;
  }
}

MetamorphicResult metamorphic(const Profile& prof, Backend be, std::uint64_t s0, std::uint64_t s1,
                              std::span<const std::uint32_t> bins, std::span<const std::uint32_t> thresholds,
                              std::span<const std::uint32_t> cpu_thresholds, std::uint64_t work_limit) {
  check_profile_backend(prof, be);
  const bool gpu = be == Backend::cuda || be == Backend::hybrid;
  if (gpu && thresholds.empty()) throw std::invalid_argument("metamorphic: cuda and hybrid need thresholds");
  if (be == Backend::hybrid && cpu_thresholds.empty())
    throw std::invalid_argument("metamorphic: the hybrid needs CPU thresholds");
  std::vector<Run> runs;
  if (be == Backend::cuda)
    for (std::uint32_t h : thresholds) runs.push_back({h, 0});
  else if (be == Backend::hybrid)
    for (std::uint32_t h : thresholds)
      for (std::uint32_t c : cpu_thresholds) runs.push_back({h, c});
  else
    runs.push_back({});
  // tcyc and tchain (SPEC v2 §8 S5) alone, under the other visibility: every hop of either is
  // strictly before t. Every backend runs both visibilities (P10), so the check of `prof` covers
  // it; checking the reduced profile itself would fail a v1 config's E402.
  std::optional<Profile> flipped;
  if (meta_test_applies(MetaTest::visibility_invariance, prof, be)) {
    flipped = prof;
    flipped->visibility = prof.visibility == Visibility::stream ? Visibility::timestamp : Visibility::stream;
    flipped->blocks.clear();
    for (Block b : {Block::tcyc, Block::tchain})
      if (const ProfileBlock* pb = prof.find(b)) flipped->blocks.push_back(*pb);
  }
  BlockBins block_bins;
  for (const ProfileBlock& pb : prof.blocks) try {
      block_bins[int(pb.block)] = fuzz_bins(prof, pb, bins);
    } catch (const std::invalid_argument& e) {
      throw std::invalid_argument(std::string("metamorphic: ") + e.what());
    }
  // Order relaxation: sg_gather alone (the profile's family's: clean_sg_gather for a clean
  // profile, SPEC v3 R6), at sg_ord's window and bins, mined by the same backend.
  std::optional<Profile> gather;
  BlockBins gather_bins;
  if (meta_test_applies(MetaTest::order_relaxation, prof, be)) {
    const std::string_view want = prof.family == Family::clean ? "clean_sg_gather" : "sg_gather";
    const PatternInfo* sg_gather = nullptr;
    for (const PatternInfo& pi : registered_patterns())
      if (std::string_view(pi.name) == want) sg_gather = &pi;
    if (!sg_gather) throw std::runtime_error("metamorphic: order relaxation needs the pattern " + std::string(want));
    gather = prof;
    gather->blocks = {ProfileBlock{Block::sg, {sg_gather}, prof.find(Block::sg_ord)->params, prof.file}};
    gather_bins[int(Block::sg)] = block_bins[int(Block::sg_ord)];
  }
  MetamorphicResult res;
  std::optional<CudaMiner> miner;
  if (gpu) miner.emplace(be == Backend::hybrid);  // hybrid: the GPU's host thread sleeps while the CPU works

  auto one_graph = [&](std::uint64_t seed, MetamorphicResult& out) {
    const FuzzGraph fg = fuzz_graph(seed);
    Graph base{fg.src, fg.dst, fg.ts, fg.n};
    const std::uint64_t m = base.m();
    CudaMiner* mp = miner ? &*miner : nullptr;
    const Mined want = mine_all(base, prof, be, fg.window, block_bins, work_limit, runs, mp);
    // R11: a block the base graph limits is inconclusive for this graph, so the transformed graphs
    // leave it out.
    std::array<bool, kAllBlocks.size()> limited = want.limited;
    Profile live = prof;
    std::erase_if(live.blocks, [&](const ProfileBlock& pb) { return limited[int(pb.block)]; });
    const auto mine = [&](const Graph& g, const Profile& p, const BlockBins& bb) {
      return mine_all(g, p, be, fg.window, bb, work_limit, runs, mp);
    };
    Rng rng{seed ^ 0x6d657461ULL};  // "meta"
    bool bad = false;
    // Rows of `got` (a graph whose eid e is the base's eid e, for e < rows) vs the base, for the
    // blocks of `which` (by default, every block the base completed).
    auto compare = [&](MetaTest test, const Mined& got, std::uint64_t rows, const Profile* which = nullptr) {
      for (std::size_t t = 0; t < want.rows.size(); ++t)
        for (const ProfileBlock& pb : (which ? which : &live)->blocks) {
          if (want.limited[int(pb.block)] || got.limited[int(pb.block)]) {
            limited[int(pb.block)] = true;
            continue;
          }
          const auto& w = want.rows[t][int(pb.block)];
          const auto& g = got.rows[t][int(pb.block)];
          const std::size_t nb = block_cols(block_bins[int(pb.block)]);
          for (std::uint64_t e = 0; e < rows; ++e)
            if (!std::equal(g.begin() + std::ptrdiff_t(e * nb), g.begin() + std::ptrdiff_t((e + 1) * nb),
                            w.begin() + std::ptrdiff_t(e * nb))) {
              ++out.mismatched_rows[int(test)][int(pb.block)];
              bad = true;
            }
          out.rows_compared[int(test)] += rows;
        }
    };

    for (int cut = 0; cut < 2; ++cut) {
      std::uint64_t k = 1 + rng.below(m - 1);  // m >= 5
      if (prof.visibility == Visibility::timestamp)
        while (k < m && base.ts[k] == base.ts[k - 1]) ++k;
      Graph pre{{base.src.begin(), base.src.begin() + std::ptrdiff_t(k)},
                {base.dst.begin(), base.dst.begin() + std::ptrdiff_t(k)},
                {base.ts.begin(), base.ts.begin() + std::ptrdiff_t(k)},
                base.n};
      compare(MetaTest::prefix_closure, mine(pre, live, block_bins), k);
    }

    std::vector<std::uint32_t> perm(base.n);
    std::iota(perm.begin(), perm.end(), 0u);
    for (std::uint32_t i = base.n - 1; i > 0; --i) std::swap(perm[i], perm[rng.below(i + 1)]);
    Graph rel = base;
    for (std::uint64_t e = 0; e < m; ++e) {
      rel.src[e] = perm[base.src[e]];
      rel.dst[e] = perm[base.dst[e]];
    }
    compare(MetaTest::relabeling, mine(rel, live, block_bins), m);

    const std::int32_t shift = std::int32_t(1 + rng.below(std::uint64_t(1) << 30));
    Graph sh = base;
    for (auto& t : sh.ts) t += shift;
    compare(MetaTest::time_shift, mine(sh, live, block_bins), m);

    if (flipped)
      compare(MetaTest::visibility_invariance, mine(base, *flipped, block_bins), m, &*flipped);

    if (gather) {
      // Per row, sg_ordered's instances (the sum over its bins) are at most sg_gather's: ordering
      // only removes sources and intermediates (SPEC v2 §3.4). Bins are not compared.
      const Mined g = mine(base, *gather, gather_bins);
      const std::size_t nb = block_cols(gather_bins[int(Block::sg)]);
      for (std::size_t t = 0; t < want.rows.size(); ++t) {
        const auto& ord = want.rows[t][int(Block::sg_ord)];
        const auto& all = g.rows[t][int(Block::sg)];
        for (std::uint64_t e = 0; e < m; ++e) {
          const auto at = std::ptrdiff_t(e * nb);
          if (std::accumulate(ord.begin() + at, ord.begin() + at + std::ptrdiff_t(nb), std::uint64_t(0)) >
              std::accumulate(all.begin() + at, all.begin() + at + std::ptrdiff_t(nb), std::uint64_t(0))) {
            ++out.mismatched_rows[int(MetaTest::order_relaxation)][int(Block::sg_ord)];
            bad = true;
          }
        }
        out.rows_compared[int(MetaTest::order_relaxation)] += m;
      }
    }

    ++out.graphs;
    out.edges += m;
    for (std::size_t b = 0; b < limited.size(); ++b) out.limited_graphs[b] += limited[b];
    if (bad) out.bad_seeds.push_back(seed);
  };

  if (gpu) {
    for (std::uint64_t seed = s0; seed < s1; ++seed) one_graph(seed, res);
  } else {
    // Parallel over seeds; the oracle's and the drivers' own parallel loops run nested, i.e. on
    // one thread each.
#pragma omp parallel
    {
      MetamorphicResult local;
#pragma omp for schedule(dynamic, 4)
      for (std::int64_t si = std::int64_t(s0); si < std::int64_t(s1); ++si) one_graph(std::uint64_t(si), local);
#pragma omp critical
      {
        res.graphs += local.graphs;
        res.edges += local.edges;
        for (std::size_t b = 0; b < kAllBlocks.size(); ++b) res.limited_graphs[b] += local.limited_graphs[b];
        for (std::size_t t = 0; t < kMetaTests.size(); ++t) {
          res.rows_compared[t] += local.rows_compared[t];
          for (std::size_t b = 0; b < kAllBlocks.size(); ++b) res.mismatched_rows[t][b] += local.mismatched_rows[t][b];
        }
        res.bad_seeds.insert(res.bad_seeds.end(), local.bad_seeds.begin(), local.bad_seeds.end());
      }
    }
  }
  std::sort(res.bad_seeds.begin(), res.bad_seeds.end());
  return res;
}

}  // namespace blz

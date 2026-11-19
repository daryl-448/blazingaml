#include "runtime/metamorphic.hpp"

#include <algorithm>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string_view>

#include "graph/fuzz_gen.hpp"
#include "graph/index.hpp"
#include "oracle/oracle.hpp"
#include "runtime/cpu_backend.hpp"
#include "runtime/cuda_backend.hpp"

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

// rows[t][block]: the block's rows for threshold index t (one entry for oracle and cpu).
using Rows = std::vector<std::array<std::vector<std::uint32_t>, kAllBlocks.size()>>;

using BlockBins = std::array<std::vector<std::uint32_t>, kAllBlocks.size()>;

Rows mine_all(const Graph& g, const Profile& prof, Backend be, std::int32_t window, const BlockBins& bins,
              std::span<const std::uint32_t> thresholds, CudaMiner* miner) {
  const std::uint64_t m = g.m();
  const std::size_t nt = be == Backend::cuda ? thresholds.size() : 1;
  Rows rows(nt);
  if (be == Backend::oracle) {
    const Oracle o({g.src, g.dst, g.ts, g.n}, prof.visibility);
    for (const ProfileBlock& pb : prof.blocks) {
      const BlockParams p{window, bins[int(pb.block)]};
      auto& r = rows[0][int(pb.block)];
      r.assign(m * p.bins.size(), 0);
      if (pb.block == Block::sg && pb.patterns.size() == 1 && std::string_view(pb.patterns[0]->name) == "sg_gather")
        o.mine_sg_gather(p, 0, m, r.data());  // the order-relaxation bound (the oracle's sg is both roles)
      else
        o.mine(pb.block, p, prof.k_max, 0, m, r.data());
    }
    return rows;
  }
  const GraphIndex index(g.src, g.dst, g.ts, g.n);
  if (be == Backend::cpu) {
    TriggerSel all;
    all.last = m;
    for (const ProfileBlock& pb : prof.blocks) {
      const BlockParams p{window, bins[int(pb.block)]};
      auto& r = rows[0][int(pb.block)];
      r.assign(m * p.bins.size(), 0);
      mine_cpu(index.view(), pb.patterns, p, prof.visibility, all, Schedule::dynamic, r.data());
    }
    return rows;
  }
  const DeviceGraph dg(index.view());
  for (std::size_t t = 0; t < nt; ++t)
    for (const ProfileBlock& pb : prof.blocks) {
      const BlockParams p{window, bins[int(pb.block)]};
      auto& r = rows[t][int(pb.block)];
      r.assign(m * p.bins.size(), 0);
      miner->mine(dg, pb.patterns, p, prof.visibility, 0, m, thresholds[t], r.data());
    }
  return rows;
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
    case MetaTest::prefix_closure: return prof.credit == Credit::trigger;
    case MetaTest::visibility_invariance: return prof.find(Block::tcyc) || prof.find(Block::tchain);
    case MetaTest::order_relaxation: return prof.find(Block::sg_ord) != nullptr;
    default: return true;
  }
}

MetamorphicResult metamorphic(const Profile& prof, Backend be, std::uint64_t s0, std::uint64_t s1,
                              std::span<const std::uint32_t> bins, std::span<const std::uint32_t> thresholds) {
  check_profile_backend(prof, be);
  if (be == Backend::hybrid) throw std::invalid_argument("metamorphic: the hybrid backend is not supported");
  // The relations are about stream and timestamp visibility; batch mode is checked by fuzz-diff
  // against the batch oracle and by tools/gfp/fuzz_vs_gfp.py against GFP.
  if (prof.batch) throw std::invalid_argument("metamorphic: GFP batch mode profiles are not supported (SPEC v2 §8 S1)");
  if (be == Backend::cuda && thresholds.empty()) throw std::invalid_argument("metamorphic: cuda needs thresholds");
  const bool prefix_applies = meta_test_applies(MetaTest::prefix_closure, prof, be);
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
  for (const ProfileBlock& pb : prof.blocks) {
    auto& v = block_bins[int(pb.block)];
    v = bins.empty() ? pb.params.bins : std::vector<std::uint32_t>(bins.begin(), bins.end());
    if (v.empty()) throw std::invalid_argument("metamorphic: block " + std::string(block_name(pb.block)) + " has no bins");
  }
  // Order relaxation: sg_gather alone, at sg_ord's window and bins, mined by the same backend.
  std::optional<Profile> gather;
  BlockBins gather_bins;
  if (meta_test_applies(MetaTest::order_relaxation, prof, be)) {
    const PatternInfo* sg_gather = nullptr;
    for (const PatternInfo& pi : registered_patterns())
      if (std::string_view(pi.name) == "sg_gather") sg_gather = &pi;
    if (!sg_gather) throw std::runtime_error("metamorphic: order relaxation needs the pattern sg_gather");
    gather = prof;
    gather->blocks = {ProfileBlock{Block::sg, {sg_gather}, prof.find(Block::sg_ord)->params, prof.file}};
    gather_bins[int(Block::sg)] = block_bins[int(Block::sg_ord)];
  }
  MetamorphicResult res;
  std::optional<CudaMiner> miner;
  if (be == Backend::cuda) miner.emplace();

  auto one_graph = [&](std::uint64_t seed, MetamorphicResult& out) {
    const FuzzGraph fg = fuzz_graph(seed);
    Graph base{fg.src, fg.dst, fg.ts, fg.n};
    const std::uint64_t m = base.m();
    CudaMiner* mp = miner ? &*miner : nullptr;
    const Rows want = mine_all(base, prof, be, fg.window, block_bins, thresholds, mp);
    Rng rng{seed ^ 0x6d657461ULL};  // "meta"
    bool bad = false;
    // Rows of `got` (a graph whose eid e is the base's eid e, for e < rows) vs the base, for the
    // blocks of `which` (by default, every block of the profile).
    auto compare = [&](MetaTest test, const Rows& got, std::uint64_t rows, const Profile* which = nullptr) {
      for (std::size_t t = 0; t < want.size(); ++t)
        for (const ProfileBlock& pb : (which ? which : &prof)->blocks) {
          const auto& w = want[t][int(pb.block)];
          const auto& g = got[t][int(pb.block)];
          const std::size_t nb = block_bins[int(pb.block)].size();
          for (std::uint64_t e = 0; e < rows; ++e)
            if (!std::equal(g.begin() + std::ptrdiff_t(e * nb), g.begin() + std::ptrdiff_t((e + 1) * nb),
                            w.begin() + std::ptrdiff_t(e * nb))) {
              ++out.mismatched_rows[int(test)][int(pb.block)];
              bad = true;
            }
          out.rows_compared[int(test)] += rows;
        }
    };

    if (prefix_applies)
      for (int cut = 0; cut < 2; ++cut) {
        std::uint64_t k = 1 + rng.below(m - 1);  // m >= 5
        if (prof.visibility == Visibility::timestamp)
          while (k < m && base.ts[k] == base.ts[k - 1]) ++k;
        Graph pre{{base.src.begin(), base.src.begin() + std::ptrdiff_t(k)},
                  {base.dst.begin(), base.dst.begin() + std::ptrdiff_t(k)},
                  {base.ts.begin(), base.ts.begin() + std::ptrdiff_t(k)},
                  base.n};
        compare(MetaTest::prefix_closure, mine_all(pre, prof, be, fg.window, block_bins, thresholds, mp), k);
      }

    std::vector<std::uint32_t> perm(base.n);
    std::iota(perm.begin(), perm.end(), 0u);
    for (std::uint32_t i = base.n - 1; i > 0; --i) std::swap(perm[i], perm[rng.below(i + 1)]);
    Graph rel = base;
    for (std::uint64_t e = 0; e < m; ++e) {
      rel.src[e] = perm[base.src[e]];
      rel.dst[e] = perm[base.dst[e]];
    }
    compare(MetaTest::relabeling, mine_all(rel, prof, be, fg.window, block_bins, thresholds, mp), m);

    const std::int32_t shift = std::int32_t(1 + rng.below(std::uint64_t(1) << 30));
    Graph sh = base;
    for (auto& t : sh.ts) t += shift;
    compare(MetaTest::time_shift, mine_all(sh, prof, be, fg.window, block_bins, thresholds, mp), m);

    if (flipped)
      compare(MetaTest::visibility_invariance, mine_all(base, *flipped, be, fg.window, block_bins, thresholds, mp), m,
              &*flipped);

    if (gather) {
      // Per row, sg_ordered's instances (the sum over its bins) are at most sg_gather's: ordering
      // only removes sources and intermediates (SPEC v2 §3.4). Bins are not compared.
      const Rows g = mine_all(base, *gather, be, fg.window, gather_bins, thresholds, mp);
      const std::size_t nb = gather_bins[int(Block::sg)].size();
      for (std::size_t t = 0; t < want.size(); ++t) {
        const auto& ord = want[t][int(Block::sg_ord)];
        const auto& all = g[t][int(Block::sg)];
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
    if (bad) out.bad_seeds.push_back(seed);
  };

  if (be == Backend::cuda) {
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

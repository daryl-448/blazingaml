#include "runtime/fuzz_diff.hpp"

#include <algorithm>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <tuple>

#include "graph/fuzz_gen.hpp"
#include "graph/index.hpp"
#include "oracle/profile_oracle.hpp"
#include "runtime/clean_vstats.hpp"
#include "runtime/cuda_backend.hpp"
#include "runtime/fusion.hpp"
#include "runtime/hybrid.hpp"

namespace blz {

FuzzDiff fuzz_diff_cpu(std::uint64_t s0, std::uint64_t s1, std::span<const std::uint32_t> bins) {
  return fuzz_diff_cpu(registry_profile(), s0, s1, bins);
}

FuzzDiff fuzz_diff_cuda(std::uint64_t s0, std::uint64_t s1, std::span<const std::uint32_t> bins,
                        std::span<const std::uint32_t> thresholds) {
  return fuzz_diff_cuda(registry_profile(), s0, s1, bins, thresholds);
}

namespace {

std::array<std::vector<std::uint32_t>, kAllBlocks.size()> all_fuzz_bins(const Profile& prof,
                                                                        std::span<const std::uint32_t> bins) {
  std::array<std::vector<std::uint32_t>, kAllBlocks.size()> out;
  for (const ProfileBlock& pb : prof.blocks) try {
      out[int(pb.block)] = fuzz_bins(prof, pb, bins);
    } catch (const std::invalid_argument& e) {
      throw std::invalid_argument(std::string("fuzz-diff: ") + e.what());
    }
  return out;
}

// The reference rows of one block: the profile's oracle's, or the CPU backend's (dynamic schedule,
// no pruning) when `oracle` is null.
void reference_rows(const ProfileOracle* oracle, const GraphIndex& index, const ProfileBlock& pb, const BlockParams& p,
                    const Profile& prof, std::uint64_t m, std::uint32_t* out) {
  if (oracle) {
    oracle->mine(pb.block, p, 0, m, out);
    return;
  }
  TriggerSel all;
  all.last = m;
  BlockParams q = p;
  q.reach_prune = false;
  mine_cpu(index.view(), pb.patterns, q, prof.visibility, all, Schedule::dynamic, out);
}

// The reference's vstats rows of a graph (SPEC v3 R8): the profile's oracle's, or the CPU's.
std::vector<double> reference_vstats(const ProfileOracle* oracle, const GraphIndex& index,
                                     std::span<const double> amount, std::int32_t window, std::uint64_t m) {
  std::vector<double> out(m * kVstatsCols);
  TriggerSel all;
  all.last = m;
  if (oracle) oracle->vstats(window, all, out.data());
  else CleanVstats(index.view(), amount, window).rows(all, out.data());
  return out;
}

// The rows of `got` whose bits differ from `want`'s, and `want`'s rows with a nonzero column.
std::pair<std::uint64_t, std::uint64_t> vstats_diff(const std::vector<double>& got, const std::vector<double>& want) {
  std::uint64_t bad = 0, nonzero = 0;
  for (std::size_t r = 0; r < want.size() / kVstatsCols; ++r) {
    const double* w = want.data() + r * kVstatsCols;
    bad += std::memcmp(got.data() + r * kVstatsCols, w, kVstatsCols * sizeof(double)) != 0;
    nonzero += std::any_of(w, w + kVstatsCols, [](double x) { return x != 0; });
  }
  return {bad, nonzero};
}

// Runs f; returns UINT64_MAX, or the eid at which the work limit stopped it (SPEC v2 §8 S3).
template <class F>
std::uint64_t stop_eid(F&& f) {
  try {
    f();
  } catch (const WorkLimitExceeded& x) {
    return x.eid;
  }
  return UINT64_MAX;
}

}  // namespace

FuzzDiff fuzz_diff_cpu(const Profile& prof, std::uint64_t s0, std::uint64_t s1, std::span<const std::uint32_t> bins,
                       std::uint64_t work_limit, Reference ref, FuzzSetting setting) {
  const auto block_bins = all_fuzz_bins(prof, bins);
  FuzzDiff res;
  // Parallel over seeds; the oracle's and the drivers' own parallel loops run nested, i.e. on
  // one thread each.
#pragma omp parallel for schedule(dynamic, 4)
  for (std::int64_t si = std::int64_t(s0); si < std::int64_t(s1); ++si) {
    const FuzzGraph fg = fuzz_graph(std::uint64_t(si), setting);
    const std::uint64_t m = fg.src.size();
    const std::vector<double> amount = fuzz_amounts(std::uint64_t(si), m);
    std::optional<ProfileOracle> oracle;
    if (ref == Reference::oracle) oracle.emplace(EdgeStream{fg.src, fg.dst, fg.ts, fg.n}, prof, Quirks{}, amount);
    const GraphIndex index(fg.src, fg.dst, fg.ts, fg.n, prof.self_loops());
    std::array<std::uint64_t, kAllBlocks.size()> bad{}, nonzero{};
    std::uint64_t vs_bad = 0, vs_nonzero = 0, vs_sides = 0;
    const bool vs = prof.vstats && ref == Reference::oracle;
    if (vs) {
      const std::vector<double> want = reference_vstats(&*oracle, index, amount, fg.window, m);
      std::vector<double> got(m * kVstatsCols);
      TriggerSel all;
      all.last = m;
      CleanVstats cv(index.view(), amount, fg.window);
      cv.rows(all, got.data());
      vs_sides = cv.sides();
      std::tie(vs_bad, vs_nonzero) = vstats_diff(got, want);
    }
    std::array<bool, kAllBlocks.size()> limited{};
    TriggerSel all;
    all.last = m;
    for (const ProfileBlock& pb : prof.blocks) {
      const Block b = pb.block;
      BlockParams p{fg.window, block_bins[int(b)]};
      p.work_limit = work_limit;
      const std::size_t nb = block_cols(p);
      std::vector<std::uint32_t> want(m * nb, 0);
      const std::uint64_t stop =
          stop_eid([&] { reference_rows(oracle ? &*oracle : nullptr, index, pb, p, prof, m, want.data()); });
      limited[int(b)] = stop != UINT64_MAX;
      if (!limited[int(b)]) nonzero[int(b)] = nonzero_rows(want, nb);
      // A block with a pattern planned for reachability pruning (SPEC v2 §8 S6) runs both ways.
      const bool reach = std::any_of(pb.patterns.begin(), pb.patterns.end(), [](const PatternInfo* pi) { return pi->cpu_reach; });
      for (int run = 0; run < (reach ? 4 : 2); ++run) {
        if (run == 0 && ref == Reference::cpu) continue;  // the reference itself
        const Schedule s = run % 2 == 0 ? Schedule::dynamic : Schedule::sorted;
        BlockParams q = p;
        q.reach_prune = run >= 2;
        std::vector<std::uint32_t> got(m * nb, 0);
        const std::uint64_t got_stop =
            stop_eid([&] { mine_cpu(index.view(), pb.patterns, q, prof.visibility, all, s, got.data()); });
        if (stop != UINT64_MAX || got_stop != UINT64_MAX) {
          bad[int(b)] += got_stop != stop;
          continue;
        }
        for (std::uint64_t e = 0; e < m; ++e)
          bad[int(b)] += !std::equal(got.begin() + e * nb, got.begin() + (e + 1) * nb, want.begin() + e * nb);
      }
    }
#pragma omp critical
    {
      ++res.graphs;
      res.edges += m;
      res.hits += fuzz_hits(fg);
      res.vstats_graphs += vs;
      res.vstats_mismatched_rows += vs_bad;
      res.vstats_nonzero_rows += vs_nonzero;
      res.vstats_sides += vs_sides;
      bool any = vs_bad != 0, any_limited = false;
      for (std::size_t i = 0; i < bad.size(); ++i) {
        res.mismatched_rows[i] += bad[i];
        res.limited_graphs[i] += limited[i];
        res.nonzero_rows[i] += nonzero[i];
        any = any || bad[i] != 0;
        any_limited = any_limited || limited[i];
      }
      if (any) res.bad_seeds.push_back(std::uint64_t(si));
      if (any_limited) res.limited_seeds.push_back(std::uint64_t(si));
    }
  }
  std::sort(res.bad_seeds.begin(), res.bad_seeds.end());
  std::sort(res.limited_seeds.begin(), res.limited_seeds.end());
  return res;
}

FuzzDiff fuzz_diff_cuda(const Profile& prof, std::uint64_t s0, std::uint64_t s1, std::span<const std::uint32_t> bins,
                        std::span<const std::uint32_t> thresholds, std::uint64_t work_limit, CudaOptions opts,
                        Reference ref, FuzzSetting setting) {
  return fuzz_diff_hybrid(prof, s0, s1, bins, thresholds, {}, work_limit, opts, ref, setting);
}

FuzzDiff fuzz_diff_hybrid(const Profile& prof, std::uint64_t s0, std::uint64_t s1, std::span<const std::uint32_t> bins,
                          std::span<const std::uint32_t> thresholds, std::span<const std::uint32_t> cpu_thresholds,
                          std::uint64_t work_limit, CudaOptions opts, Reference ref, FuzzSetting setting) {
  const auto block_bins = all_fuzz_bins(prof, bins);
  FuzzDiff res;
  const bool hybrid = !cpu_thresholds.empty();
  if (hybrid && (opts.fuse || opts.pack)) throw std::invalid_argument("fuzz-diff: the hybrid neither fuses nor packs");
  std::vector<const ProfileBlock*> blocks;
  for (const ProfileBlock& pb : prof.blocks) blocks.push_back(&pb);
  const std::vector<CudaUnit> units = cuda_units(blocks, opts.fuse);
  CudaMiner miner(hybrid);
  // The plain CUDA runs have one pass per GPU threshold; the hybrid one per (GPU, CPU) pair.
  const std::uint32_t plain[] = {0};
  const std::span<const std::uint32_t> splits = hybrid ? cpu_thresholds : std::span<const std::uint32_t>(plain);
  // Batches: the oracle and the host index in parallel over seeds, then the GPU in seed order.
  struct Item {
    FuzzGraph fg;
    std::vector<double> amount;
    std::unique_ptr<GraphIndex> index;
    std::array<std::vector<std::uint32_t>, kAllBlocks.size()> want;
    std::array<std::uint64_t, kAllBlocks.size()> stop;  // the oracle's stop eid, or UINT64_MAX
    std::vector<double> vs_want;
  };
  constexpr std::uint64_t kBatch = 256;
  for (std::uint64_t b0 = s0; b0 < s1; b0 += kBatch) {
    const std::uint64_t b1 = std::min(b0 + kBatch, s1);
    std::vector<Item> items(b1 - b0);
#pragma omp parallel for schedule(dynamic, 4)
    for (std::int64_t i = 0; i < std::int64_t(items.size()); ++i) {
      Item& it = items[std::size_t(i)];
      it.fg = fuzz_graph(b0 + std::uint64_t(i), setting);
      const std::uint64_t m = it.fg.src.size();
      it.amount = fuzz_amounts(b0 + std::uint64_t(i), m);
      std::optional<ProfileOracle> oracle;
      if (ref == Reference::oracle)
        oracle.emplace(EdgeStream{it.fg.src, it.fg.dst, it.fg.ts, it.fg.n}, prof, Quirks{}, it.amount);
      it.index = std::make_unique<GraphIndex>(it.fg.src, it.fg.dst, it.fg.ts, it.fg.n, prof.self_loops());
      if (prof.vstats)
        it.vs_want = reference_vstats(oracle ? &*oracle : nullptr, *it.index, it.amount, it.fg.window, m);
      it.stop.fill(UINT64_MAX);
      for (const ProfileBlock& pb : prof.blocks) {
        BlockParams p{it.fg.window, block_bins[int(pb.block)]};
        p.work_limit = work_limit;
        it.want[int(pb.block)].assign(m * block_cols(p), 0);
        it.stop[int(pb.block)] = stop_eid([&] {
          reference_rows(oracle ? &*oracle : nullptr, *it.index, pb, p, prof, m, it.want[int(pb.block)].data());
        });
      }
    }
    for (std::size_t i = 0; i < items.size(); ++i) {
      const Item& it = items[i];
      const std::uint64_t m = it.fg.src.size();
      const DeviceGraph dg(it.index->view());
      bool any = false;
      // One block's rows (or stop eid) against the reference's.
      const auto check = [&](Block b, const std::vector<std::uint32_t>& got, std::uint64_t got_stop) {
        if (it.stop[int(b)] != UINT64_MAX || got_stop != UINT64_MAX) {
          if (got_stop != it.stop[int(b)]) {
            ++res.mismatched_rows[int(b)];
            any = true;
          }
          return;
        }
        const std::size_t nb = block_cols(block_bins[int(b)]);
        const auto& want = it.want[int(b)];
        for (std::uint64_t e = 0; e < m; ++e)
          if (!std::equal(got.begin() + e * nb, got.begin() + (e + 1) * nb, want.begin() + e * nb)) {
            ++res.mismatched_rows[int(b)];
            any = true;
          }
      };
      for (const CudaUnit& unit : units)
        for (std::uint32_t h : thresholds)
          for (std::uint32_t hc : splits) {
            std::vector<BlockParams> params;
            for (const ProfileBlock* pb : unit.blocks) {
              BlockParams& p = params.emplace_back(BlockParams{it.fg.window, block_bins[int(pb->block)]});
              p.work_limit = work_limit;
              p.reach_prune = hybrid;  // the hybrid's CPU side prunes where it can (SPEC v2 §8 S6)
            }
            std::vector<std::vector<std::uint32_t>> got(unit.blocks.size());
            for (std::size_t sl = 0; sl < got.size(); ++sl) got[sl].assign(m * block_cols(params[sl]), 0xdeadbeefu);
            if (unit.fusion) {  // no fused pattern counts work, so a fused run never stops
              std::vector<const BlockParams*> pp;
              std::vector<std::uint32_t*> outs;
              for (std::size_t sl = 0; sl < got.size(); ++sl) {
                pp.push_back(&params[sl]);
                outs.push_back(got[sl].data());
              }
              miner.mine_fused(dg, *unit.fusion, pp, prof.visibility, 0, m, h, outs, opts.pack);
              for (std::size_t sl = 0; sl < got.size(); ++sl) check(unit.blocks[sl]->block, got[sl], UINT64_MAX);
              continue;
            }
            const ProfileBlock& pb = *unit.blocks.front();
            const std::uint64_t got_stop = stop_eid([&] {
              if (hybrid)
                mine_hybrid(it.index->view(), dg, miner, pb.patterns, params[0], prof.visibility, 0, m, hc, h,
                            Schedule::sorted, got[0].data());
              else
                miner.mine(dg, pb.patterns, params[0], prof.visibility, 0, m, h, got[0].data(), opts.pack);
            });
            check(pb.block, got[0], got_stop);
          }
      if (prof.vstats) {  // on the GPU, for cuda and the hybrid alike
        std::vector<double> got(m * kVstatsCols);
        TriggerSel all;
        all.last = m;
        CleanVstats cv(it.index->view(), it.amount, it.fg.window, &dg);
        cv.rows(all, got.data());
        const auto [bad, nonzero] = vstats_diff(got, it.vs_want);
        ++res.vstats_graphs;
        res.vstats_mismatched_rows += bad;
        res.vstats_nonzero_rows += nonzero;
        res.vstats_sides += cv.sides();
        any = any || bad != 0;
      }
      bool limited = false;
      for (const ProfileBlock& pb : prof.blocks) {
        const bool l = it.stop[int(pb.block)] != UINT64_MAX;
        res.limited_graphs[int(pb.block)] += l;
        if (!l)
          res.nonzero_rows[int(pb.block)] += nonzero_rows(it.want[int(pb.block)], block_cols(block_bins[int(pb.block)]));
        limited = limited || l;
      }
      ++res.graphs;
      res.edges += m;
      res.hits += fuzz_hits(it.fg);
      if (any) res.bad_seeds.push_back(b0 + i);
      if (limited) res.limited_seeds.push_back(b0 + i);
    }
  }
  return res;
}

}  // namespace blz

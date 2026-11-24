#include "runtime/fuzz_diff.hpp"

#include <algorithm>
#include <memory>
#include <optional>
#include <stdexcept>

#include "graph/fuzz_gen.hpp"
#include "graph/index.hpp"
#include "oracle/profile_oracle.hpp"
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
                       std::uint64_t work_limit, Reference ref) {
  const auto block_bins = all_fuzz_bins(prof, bins);
  FuzzDiff res;
  // Parallel over seeds; the oracle's and the drivers' own parallel loops run nested, i.e. on
  // one thread each.
#pragma omp parallel for schedule(dynamic, 4)
  for (std::int64_t si = std::int64_t(s0); si < std::int64_t(s1); ++si) {
    const FuzzGraph fg = fuzz_graph(std::uint64_t(si));
    const std::uint64_t m = fg.src.size();
    std::optional<ProfileOracle> oracle;
    if (ref == Reference::oracle) oracle.emplace(EdgeStream{fg.src, fg.dst, fg.ts, fg.n}, prof);
    const GraphIndex index(fg.src, fg.dst, fg.ts, fg.n, prof.self_loops());
    std::array<std::uint64_t, kAllBlocks.size()> bad{};
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
      bool any = false, any_limited = false;
      for (std::size_t i = 0; i < bad.size(); ++i) {
        res.mismatched_rows[i] += bad[i];
        res.limited_graphs[i] += limited[i];
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
                        Reference ref) {
  return fuzz_diff_hybrid(prof, s0, s1, bins, thresholds, {}, work_limit, opts, ref);
}

FuzzDiff fuzz_diff_hybrid(const Profile& prof, std::uint64_t s0, std::uint64_t s1, std::span<const std::uint32_t> bins,
                          std::span<const std::uint32_t> thresholds, std::span<const std::uint32_t> cpu_thresholds,
                          std::uint64_t work_limit, CudaOptions opts, Reference ref) {
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
    std::unique_ptr<GraphIndex> index;
    std::array<std::vector<std::uint32_t>, kAllBlocks.size()> want;
    std::array<std::uint64_t, kAllBlocks.size()> stop;  // the oracle's stop eid, or UINT64_MAX
  };
  constexpr std::uint64_t kBatch = 256;
  for (std::uint64_t b0 = s0; b0 < s1; b0 += kBatch) {
    const std::uint64_t b1 = std::min(b0 + kBatch, s1);
    std::vector<Item> items(b1 - b0);
#pragma omp parallel for schedule(dynamic, 4)
    for (std::int64_t i = 0; i < std::int64_t(items.size()); ++i) {
      Item& it = items[std::size_t(i)];
      it.fg = fuzz_graph(b0 + std::uint64_t(i));
      const std::uint64_t m = it.fg.src.size();
      std::optional<ProfileOracle> oracle;
      if (ref == Reference::oracle) oracle.emplace(EdgeStream{it.fg.src, it.fg.dst, it.fg.ts, it.fg.n}, prof);
      it.index = std::make_unique<GraphIndex>(it.fg.src, it.fg.dst, it.fg.ts, it.fg.n, prof.self_loops());
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
      bool limited = false;
      for (const ProfileBlock& pb : prof.blocks) {
        res.limited_graphs[int(pb.block)] += it.stop[int(pb.block)] != UINT64_MAX;
        limited = limited || it.stop[int(pb.block)] != UINT64_MAX;
      }
      ++res.graphs;
      res.edges += m;
      if (any) res.bad_seeds.push_back(b0 + i);
      if (limited) res.limited_seeds.push_back(b0 + i);
    }
  }
  return res;
}

}  // namespace blz

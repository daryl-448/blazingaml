#include "runtime/fuzz_diff.hpp"

#include <algorithm>
#include <memory>
#include <optional>
#include <stdexcept>

#include "graph/fuzz_gen.hpp"
#include "graph/index.hpp"
#include "oracle/batch_oracle.hpp"
#include "oracle/oracle.hpp"
#include "runtime/batch.hpp"
#include "runtime/cuda_backend.hpp"
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

// A block's bins on fuzz graphs: `bins`, or the profile's own when `bins` is empty.
std::vector<std::uint32_t> fuzz_bins(const ProfileBlock& pb, std::span<const std::uint32_t> bins) {
  std::vector<std::uint32_t> v = bins.empty() ? pb.params.bins : std::vector<std::uint32_t>(bins.begin(), bins.end());
  if (v.empty()) throw std::invalid_argument("fuzz-diff: block " + std::string(block_name(pb.block)) + " has no bins");
  return v;
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
                       std::uint64_t work_limit) {
  const int k_max = prof.k_max;
  std::array<std::vector<std::uint32_t>, kAllBlocks.size()> block_bins;
  for (const ProfileBlock& pb : prof.blocks) block_bins[int(pb.block)] = fuzz_bins(pb, bins);
  FuzzDiff res;
  // Parallel over seeds; the oracle's and the drivers' own parallel loops run nested, i.e. on
  // one thread each.
#pragma omp parallel for schedule(dynamic, 4)
  for (std::int64_t si = std::int64_t(s0); si < std::int64_t(s1); ++si) {
    const FuzzGraph fg = fuzz_graph(std::uint64_t(si));
    const std::uint64_t m = fg.src.size();
    const Oracle oracle({fg.src, fg.dst, fg.ts, fg.n}, prof.visibility);
    const GraphIndex index(fg.src, fg.dst, fg.ts, fg.n);
    std::array<std::uint64_t, kAllBlocks.size()> bad{};
    bool limited = false;
    TriggerSel all;
    all.last = m;
    // GFP batch mode (SPEC v2 §8 S1): the batch oracle against the CPU module. Every block has the
    // graph's window, so GFP's time_window is that window; `cuts: splits` cuts at the graph's own
    // b_fv and train/test boundary.
    std::optional<BatchLayout> layout;
    std::optional<BatchOracle> batch_oracle;
    if (prof.batch) {
      layout = batch_layout(*prof.batch, m, fg.ts);
      batch_oracle.emplace(EdgeStream{fg.src, fg.dst, fg.ts, fg.n}, *layout, fg.window);
    }
    for (const ProfileBlock& pb : prof.blocks) {
      const Block b = pb.block;
      BlockParams p{fg.window, block_bins[int(b)]};
      p.work_limit = work_limit;
      const std::size_t nb = p.bins.size();
      std::vector<std::uint32_t> want(m * nb, 0);
      if (layout) {
        batch_oracle->mine(b, p, k_max, all, want.data());
        std::vector<std::uint32_t> got(m * nb, 0);
        mine_batch_cpu(index.view(), b, p, k_max, fg.window, *layout, all, got.data());
        for (std::uint64_t e = 0; e < m; ++e)
          bad[int(b)] += !std::equal(got.begin() + e * nb, got.begin() + (e + 1) * nb, want.begin() + e * nb);
        continue;
      }
      const std::uint64_t stop = stop_eid([&] { oracle.mine(b, p, k_max, 0, m, want.data()); });
      limited = limited || stop != UINT64_MAX;
      // A block with a pattern planned for reachability pruning (SPEC v2 §8 S6) runs both ways.
      const bool reach = std::any_of(pb.patterns.begin(), pb.patterns.end(), [](const PatternInfo* pi) { return pi->cpu_reach; });
      for (int run = 0; run < (reach ? 4 : 2); ++run) {
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
      bool any = false;
      for (std::size_t i = 0; i < bad.size(); ++i) {
        res.mismatched_rows[i] += bad[i];
        any = any || bad[i] != 0;
      }
      if (any) res.bad_seeds.push_back(std::uint64_t(si));
      if (limited) res.limited_seeds.push_back(std::uint64_t(si));
    }
  }
  std::sort(res.bad_seeds.begin(), res.bad_seeds.end());
  std::sort(res.limited_seeds.begin(), res.limited_seeds.end());
  return res;
}

FuzzDiff fuzz_diff_cuda(const Profile& prof, std::uint64_t s0, std::uint64_t s1, std::span<const std::uint32_t> bins,
                        std::span<const std::uint32_t> thresholds, std::uint64_t work_limit) {
  return fuzz_diff_hybrid(prof, s0, s1, bins, thresholds, {}, work_limit);
}

FuzzDiff fuzz_diff_hybrid(const Profile& prof, std::uint64_t s0, std::uint64_t s1, std::span<const std::uint32_t> bins,
                          std::span<const std::uint32_t> thresholds, std::span<const std::uint32_t> cpu_thresholds,
                          std::uint64_t work_limit) {
  const int k_max = prof.k_max;
  std::array<std::vector<std::uint32_t>, kAllBlocks.size()> block_bins;
  for (const ProfileBlock& pb : prof.blocks) block_bins[int(pb.block)] = fuzz_bins(pb, bins);
  FuzzDiff res;
  const bool hybrid = !cpu_thresholds.empty();
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
      const Oracle oracle({it.fg.src, it.fg.dst, it.fg.ts, it.fg.n}, prof.visibility);
      it.index = std::make_unique<GraphIndex>(it.fg.src, it.fg.dst, it.fg.ts, it.fg.n);
      it.stop.fill(UINT64_MAX);
      for (const ProfileBlock& pb : prof.blocks) {
        BlockParams p{it.fg.window, block_bins[int(pb.block)]};
        p.work_limit = work_limit;
        it.want[int(pb.block)].assign(m * p.bins.size(), 0);
        it.stop[int(pb.block)] =
            stop_eid([&] { oracle.mine(pb.block, p, k_max, 0, m, it.want[int(pb.block)].data()); });
      }
    }
    for (std::size_t i = 0; i < items.size(); ++i) {
      const Item& it = items[i];
      const std::uint64_t m = it.fg.src.size();
      const DeviceGraph dg(it.index->view());
      bool any = false, limited = false;
      for (const ProfileBlock& pb : prof.blocks)
        for (std::uint32_t h : thresholds)
          for (std::uint32_t hc : splits) {
            const Block b = pb.block;
            BlockParams p{it.fg.window, block_bins[int(b)]};
            p.work_limit = work_limit;
            p.reach_prune = hybrid;  // the hybrid's CPU side prunes where it can (SPEC v2 §8 S6)
            const std::size_t nb = p.bins.size();
            std::vector<std::uint32_t> got(m * nb);
            const std::uint64_t got_stop = stop_eid([&] {
              if (hybrid)
                mine_hybrid(it.index->view(), dg, miner, pb.patterns, p, prof.visibility, 0, m, hc, h, Schedule::sorted,
                            got.data());
              else
                miner.mine(dg, pb.patterns, p, prof.visibility, 0, m, h, got.data());
            });
            if (it.stop[int(b)] != UINT64_MAX || got_stop != UINT64_MAX) {
              limited = limited || it.stop[int(b)] != UINT64_MAX;
              if (got_stop != it.stop[int(b)]) {
                ++res.mismatched_rows[int(b)];
                any = true;
              }
              continue;
            }
            const auto& want = it.want[int(b)];
            for (std::uint64_t e = 0; e < m; ++e)
              if (!std::equal(got.begin() + e * nb, got.begin() + (e + 1) * nb, want.begin() + e * nb)) {
                ++res.mismatched_rows[int(b)];
                any = true;
              }
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

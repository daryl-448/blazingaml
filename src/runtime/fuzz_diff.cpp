#include "runtime/fuzz_diff.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>

#include "graph/fuzz_gen.hpp"
#include "graph/index.hpp"
#include "oracle/oracle.hpp"
#include "runtime/cuda_backend.hpp"

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

}  // namespace

FuzzDiff fuzz_diff_cpu(const Profile& prof, std::uint64_t s0, std::uint64_t s1, std::span<const std::uint32_t> bins) {
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
    TriggerSel all;
    all.last = m;
    for (const ProfileBlock& pb : prof.blocks) {
      const Block b = pb.block;
      const BlockParams p{fg.window, block_bins[int(b)]};
      const std::size_t nb = p.bins.size();
      std::vector<std::uint32_t> want(m * nb, 0);
      oracle.mine(b, p, k_max, 0, m, want.data());
      for (Schedule s : {Schedule::dynamic, Schedule::sorted}) {
        std::vector<std::uint32_t> got(m * nb, 0);
        mine_cpu(index.view(), pb.patterns, p, all, s, got.data());
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
    }
  }
  std::sort(res.bad_seeds.begin(), res.bad_seeds.end());
  return res;
}

FuzzDiff fuzz_diff_cuda(const Profile& prof, std::uint64_t s0, std::uint64_t s1, std::span<const std::uint32_t> bins,
                        std::span<const std::uint32_t> thresholds) {
  const int k_max = prof.k_max;
  std::array<std::vector<std::uint32_t>, kAllBlocks.size()> block_bins;
  for (const ProfileBlock& pb : prof.blocks) block_bins[int(pb.block)] = fuzz_bins(pb, bins);
  FuzzDiff res;
  CudaMiner miner;
  // Batches: the oracle and the host index in parallel over seeds, then the GPU in seed order.
  struct Item {
    FuzzGraph fg;
    std::unique_ptr<GraphIndex> index;
    std::array<std::vector<std::uint32_t>, kAllBlocks.size()> want;
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
      for (const ProfileBlock& pb : prof.blocks) {
        const auto& bb = block_bins[int(pb.block)];
        it.want[int(pb.block)].assign(m * bb.size(), 0);
        oracle.mine(pb.block, {it.fg.window, bb}, k_max, 0, m, it.want[int(pb.block)].data());
      }
    }
    for (std::size_t i = 0; i < items.size(); ++i) {
      const Item& it = items[i];
      const std::uint64_t m = it.fg.src.size();
      const DeviceGraph dg(it.index->view());
      bool any = false;
      for (const ProfileBlock& pb : prof.blocks)
        for (std::uint32_t h : thresholds) {
          const Block b = pb.block;
          const BlockParams p{it.fg.window, block_bins[int(b)]};
          const std::size_t nb = p.bins.size();
          std::vector<std::uint32_t> got(m * nb);
          miner.mine(dg, pb.patterns, p, 0, m, h, got.data());
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
    }
  }
  return res;
}

}  // namespace blz

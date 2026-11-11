#include "runtime/fuzz_diff.hpp"

#include <algorithm>
#include <memory>

#include "graph/fuzz_gen.hpp"
#include "graph/index.hpp"
#include "oracle/oracle.hpp"
#include "runtime/cuda_backend.hpp"

namespace blz {

FuzzDiff fuzz_diff_cpu(std::uint64_t s0, std::uint64_t s1, std::span<const std::uint32_t> bins) {
  const int k_max = compiled_k_max();
  FuzzDiff res;
  const std::vector<std::uint32_t> bin_vec(bins.begin(), bins.end());
  // Parallel over seeds; the oracle's and the drivers' own parallel loops run nested, i.e. on
  // one thread each.
#pragma omp parallel for schedule(dynamic, 4)
  for (std::int64_t si = std::int64_t(s0); si < std::int64_t(s1); ++si) {
    const FuzzGraph fg = fuzz_graph(std::uint64_t(si));
    const std::uint64_t m = fg.src.size();
    const Oracle oracle({fg.src, fg.dst, fg.ts, fg.n});
    const GraphIndex index(fg.src, fg.dst, fg.ts, fg.n);
    const BlockParams p{fg.window, bin_vec};
    std::array<std::uint64_t, kAllBlocks.size()> bad{};
    for (Block b : kAllBlocks) {
      std::vector<std::uint32_t> want(m * bins.size(), 0);
      oracle.mine(b, p, k_max, 0, m, want.data());
      for (Schedule s : {Schedule::dynamic, Schedule::sorted}) {
        std::vector<std::uint32_t> got(m * bins.size(), 0);
        mine_cpu(index.view(), b, p, 0, m, s, got.data());
        for (std::uint64_t e = 0; e < m; ++e)
          bad[int(b)] += !std::equal(got.begin() + e * bins.size(), got.begin() + (e + 1) * bins.size(),
                                     want.begin() + e * bins.size());
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

FuzzDiff fuzz_diff_cuda(std::uint64_t s0, std::uint64_t s1, std::span<const std::uint32_t> bins,
                        std::span<const std::uint32_t> thresholds) {
  const int k_max = compiled_k_max();
  const std::vector<std::uint32_t> bin_vec(bins.begin(), bins.end());
  const std::size_t nb = bins.size();
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
      const Oracle oracle({it.fg.src, it.fg.dst, it.fg.ts, it.fg.n});
      it.index = std::make_unique<GraphIndex>(it.fg.src, it.fg.dst, it.fg.ts, it.fg.n);
      for (Block b : kAllBlocks) {
        it.want[int(b)].assign(m * nb, 0);
        oracle.mine(b, {it.fg.window, bin_vec}, k_max, 0, m, it.want[int(b)].data());
      }
    }
    for (std::size_t i = 0; i < items.size(); ++i) {
      const Item& it = items[i];
      const std::uint64_t m = it.fg.src.size();
      const DeviceGraph dg(it.index->view());
      const BlockParams p{it.fg.window, bin_vec};
      std::vector<std::uint32_t> got(m * nb);
      bool any = false;
      for (Block b : kAllBlocks)
        for (std::uint32_t h : thresholds) {
          miner.mine(dg, b, p, 0, m, h, got.data());
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

#include "runtime/fuzz_diff.hpp"

#include <algorithm>

#include "graph/fuzz_gen.hpp"
#include "graph/index.hpp"
#include "oracle/oracle.hpp"

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

}  // namespace blz

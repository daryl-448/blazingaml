#include "graph/fuzz_gen.hpp"

#include <algorithm>
#include <numeric>

namespace blz {
namespace {

struct SplitMix64 {
  std::uint64_t s;
  std::uint64_t next() {
    std::uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }
  std::uint32_t below(std::uint32_t k) { return std::uint32_t(next() % k); }  // [0, k)
};

}  // namespace

FuzzGraph fuzz_graph(std::uint64_t seed) {
  SplitMix64 r{seed};
  FuzzGraph g;
  g.seed = seed;
  g.n = 3 + r.below(38);
  const std::uint32_t m = 5 + r.below(396);
  const std::uint32_t T = 2 + r.below(59);
  g.window = std::int32_t(1 + r.below(T));

  std::vector<std::uint32_t> src(m), dst(m);
  std::vector<std::int32_t> ts(m);
  for (std::uint32_t i = 0; i < m; ++i) {
    const std::uint32_t c = r.below(100);
    if (c < 20 && i > 0) {  // parallel edge: repeat an earlier pair
      const std::uint32_t j = r.below(i);
      src[i] = src[j];
      dst[i] = dst[j];
    } else if (c < 30) {  // self-loop
      src[i] = dst[i] = r.below(g.n);
    } else {
      src[i] = r.below(g.n);
      dst[i] = (src[i] + 1 + r.below(g.n - 1)) % g.n;
    }
    ts[i] = std::int32_t(r.below(T));
  }
  // Stream order: stable sort by ts (SPEC §3.3).
  std::vector<std::uint32_t> order(m);
  std::iota(order.begin(), order.end(), 0u);
  std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) { return ts[a] < ts[b]; });
  for (std::uint32_t k : order) {
    g.src.push_back(src[k]);
    g.dst.push_back(dst[k]);
    g.ts.push_back(ts[k]);
  }
  return g;
}

std::vector<double> fuzz_amounts(std::uint64_t seed, std::uint64_t m) {
  SplitMix64 r{seed ^ 0x5eed5eed5eed5eedULL};
  std::vector<double> a(m);
  for (std::uint64_t i = 0; i < m; ++i) {
    const std::uint32_t c = r.below(100);
    if (c < 40) a[i] = double(r.below(10000)) / 100;                 // 0.00 .. 99.99
    else if (c < 60) a[i] = double(1 + r.below(5));                  // ties
    else if (c < 72) a[i] = double(r.below(1000000000)) / 100;       // up to 1e7
    else if (c < 82) a[i] = double(1 + r.below(9)) / 1000;           // tiny
    else a[i] = i > 0 ? a[i - 1] : 1.0;                              // repeats
  }
  return a;
}

}  // namespace blz

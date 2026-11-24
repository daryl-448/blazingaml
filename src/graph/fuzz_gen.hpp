#pragma once
#include <cstdint>
#include <vector>

namespace blz {

// A random small multigraph for differential fuzzing (SPEC §13.3), in stream order (ts
// non-decreasing, eid = index). Fully determined by `seed`: the generator uses its own
// splitmix64 stream, not <random> distributions, so it is reproducible across compilers.
struct FuzzGraph {
  std::uint64_t seed = 0;
  std::uint32_t n = 0;       // node ids are < n, n in [3, 40]
  std::int32_t window = 0;   // delta for every block, in [1, ts range]
  std::vector<std::uint32_t> src, dst;  // m in [5, 400]
  std::vector<std::int32_t> ts;         // in [0, T), T in [2, 60]: ties are common
};

FuzzGraph fuzz_graph(std::uint64_t seed);

// One amount per edge of fuzz_graph(seed) (m values) for the vertex statistics (SPEC v2 §8 S2),
// from a splitmix64 stream of its own. It reaches cents below 100, small integers (ties), large
// and tiny values, and repeats of the previous amount, which make a window constant.
std::vector<double> fuzz_amounts(std::uint64_t seed, std::uint64_t m);

}  // namespace blz

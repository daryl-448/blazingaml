#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace blz {

// A random small multigraph for differential fuzzing (SPEC §13.3), in stream order (ts
// non-decreasing, eid = index). Fully determined by `seed` and the setting: the generator uses its
// own splitmix64 stream, not <random> distributions, so it is reproducible across compilers.
struct FuzzGraph {
  std::uint64_t seed = 0;
  std::uint32_t n = 0;       // node ids are < n; dense: n in [3, 40]; sparse: [6, 60]
  std::int32_t window = 0;   // delta for every block; dense: in [1, T]; sparse: [2, 41]
  std::vector<std::uint32_t> src, dst;  // dense: m in [5, 400]; sparse: [20, 400]
  std::vector<std::int32_t> ts;         // dense: in [0, T), T in [2, 60], ties common; sparse: below
};

// The fuzz settings (SPEC v3 §4.4, R18).
//   - dense (v1, SPEC §13.3): one component's worth of nodes, time spans of at most 60, windows up
//     to the span, so most windows hold most of the graph. Its graphs feed recorded results and
//     never change.
//   - sparse: 2 to 6 components of 3 to 10 nodes, no edge between two of them, each active in 1 to 4
//     bursts of one window that start anywhere in a span of 20 to 99 windows; and edges placed
//     against the window bounds: about 30 % of the edges share a node with an earlier edge and lie
//     exactly delta or delta - 1 after or before it, so that edge is just outside (t' = t - delta)
//     or just inside (t' = t - delta + 1) the later one's window. It keeps ties (10 %: an edge at an
//     earlier edge's time in its component), exact duplicates (10 %: an earlier edge's (u, v, t)),
//     parallel edges and self-loops (5 %).
enum class FuzzSetting { dense, sparse };
const char* fuzz_setting_name(FuzzSetting s);
FuzzSetting parse_fuzz_setting(const std::string& s);  // "dense" | "sparse"; throws otherwise

FuzzGraph fuzz_graph(std::uint64_t seed, FuzzSetting setting = FuzzSetting::dense);

// What fuzz graphs exercise (SPEC v3 §4.4 item 6, path hit), summed over graphs with +=. A trigger
// here is a non-loop edge e = (u -> v, t); a bound edge of e is another non-loop edge at u or v.
struct FuzzHits {
  std::uint64_t graphs = 0;
  std::uint64_t multi_component_graphs = 0;  // graphs whose edges make two or more weakly connected components
  std::uint64_t components = 0;              // those components, summed over the graphs
  std::uint64_t triggers = 0;
  std::uint64_t bound_out = 0;   // triggers with a bound edge at exactly t - delta: just outside (R2)
  std::uint64_t bound_in = 0;    // triggers with a bound edge at exactly t - delta + 1: just inside
  std::uint64_t tied = 0;        // edges whose timestamp another edge has (R3a)
  std::uint64_t duplicated = 0;  // edges whose (u, v, t) another edge has (R3b)
  FuzzHits& operator+=(const FuzzHits& o);
};
FuzzHits fuzz_hits(const FuzzGraph& g);

// One amount per edge of fuzz_graph(seed) (m values) for the vertex statistics (SPEC v2 §8 S2),
// from a splitmix64 stream of its own. It reaches cents below 100, small integers (ties), large
// and tiny values, and repeats of the previous amount, which make a window constant.
std::vector<double> fuzz_amounts(std::uint64_t seed, std::uint64_t m);

}  // namespace blz

#include "graph/fuzz_gen.hpp"

#include <algorithm>
#include <map>
#include <numeric>
#include <stdexcept>
#include <tuple>

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

// Stream order: a stable sort by ts (SPEC §3.3).
void to_stream(const std::vector<std::uint32_t>& src, const std::vector<std::uint32_t>& dst,
               const std::vector<std::int32_t>& ts, FuzzGraph& g) {
  std::vector<std::uint32_t> order(src.size());
  std::iota(order.begin(), order.end(), 0u);
  std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) { return ts[a] < ts[b]; });
  for (std::uint32_t k : order) {
    g.src.push_back(src[k]);
    g.dst.push_back(dst[k]);
    g.ts.push_back(ts[k]);
  }
}

// The dense setting (fuzz_gen.hpp): v1's generator, unchanged.
FuzzGraph dense_graph(std::uint64_t seed) {
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
  to_stream(src, dst, ts, g);
  return g;
}

// The sparse-time setting (fuzz_gen.hpp), from a splitmix64 stream of its own.
FuzzGraph sparse_graph(std::uint64_t seed) {
  SplitMix64 r{seed ^ 0x5ba45e5ba45e5ba4ULL};
  FuzzGraph g;
  g.seed = seed;
  struct Component {
    std::uint32_t first, size;
    std::vector<std::int32_t> bursts;  // start times; a burst lasts one window
  };
  std::vector<Component> comps(2 + r.below(5));
  for (Component& c : comps) {
    c.first = g.n;
    c.size = 3 + r.below(8);
    g.n += c.size;
  }
  g.window = std::int32_t(2 + r.below(40));
  const std::uint32_t span = std::uint32_t(g.window) * (20 + r.below(80));
  for (Component& c : comps)
    for (std::uint32_t b = 1 + r.below(4); b > 0; --b) c.bursts.push_back(std::int32_t(r.below(span)));
  const auto other = [&](const Component& c, std::uint32_t x) {  // a node of c other than x
    return c.first + (x - c.first + 1 + r.below(c.size - 1)) % c.size;
  };

  const std::uint32_t m = 20 + r.below(381);
  std::vector<std::uint32_t> src(m), dst(m), comp(m);
  std::vector<std::int32_t> ts(m);
  for (std::uint32_t i = 0; i < m; ++i) {
    const std::uint32_t c = r.below(100), j = i > 0 ? r.below(i) : 0;
    if (c < 10 && i > 0) {  // an exact duplicate (u, v, t)
      src[i] = src[j], dst[i] = dst[j], ts[i] = ts[j], comp[i] = comp[j];
    } else if (c < 40 && i > 0) {  // on a window bound of edge j, sharing a node with it
      const Component& k = comps[comp[j]];
      comp[i] = comp[j];
      switch (r.below(3)) {
        case 0: src[i] = dst[j], dst[i] = other(k, dst[j]); break;  // after j: j -> i is a path
        case 1: dst[i] = src[j], src[i] = other(k, src[j]); break;  // before j: i -> j is a path
        default: src[i] = src[j], dst[i] = dst[j];                  // parallel to j
      }
      const std::int32_t gap = g.window - std::int32_t(r.below(2));  // delta or delta - 1
      ts[i] = r.below(2) == 0 || ts[j] < gap ? ts[j] + gap : ts[j] - gap;
    } else if (c < 50 && i > 0) {  // a tie with edge j, in its component
      const Component& k = comps[comp[j]];
      comp[i] = comp[j];
      src[i] = k.first + r.below(k.size);
      dst[i] = other(k, src[i]);
      ts[i] = ts[j];
    } else {  // in a burst of some component: a self-loop (5 %) or an edge
      comp[i] = r.below(std::uint32_t(comps.size()));
      const Component& k = comps[comp[i]];
      src[i] = k.first + r.below(k.size);
      dst[i] = c < 55 ? src[i] : other(k, src[i]);
      ts[i] = k.bursts[r.below(std::uint32_t(k.bursts.size()))] + std::int32_t(r.below(std::uint32_t(g.window)));
    }
  }
  to_stream(src, dst, ts, g);
  return g;
}

}  // namespace

const char* fuzz_setting_name(FuzzSetting s) { return s == FuzzSetting::dense ? "dense" : "sparse"; }

FuzzSetting parse_fuzz_setting(const std::string& s) {
  if (s == "dense") return FuzzSetting::dense;
  if (s == "sparse") return FuzzSetting::sparse;
  throw std::invalid_argument("unknown fuzz setting '" + s + "' (dense or sparse; SPEC v3 §4.4)");
}

FuzzGraph fuzz_graph(std::uint64_t seed, FuzzSetting setting) {
  return setting == FuzzSetting::dense ? dense_graph(seed) : sparse_graph(seed);
}

FuzzHits& FuzzHits::operator+=(const FuzzHits& o) {
  graphs += o.graphs;
  multi_component_graphs += o.multi_component_graphs;
  components += o.components;
  triggers += o.triggers;
  bound_out += o.bound_out;
  bound_in += o.bound_in;
  tied += o.tied;
  duplicated += o.duplicated;
  return *this;
}

FuzzHits fuzz_hits(const FuzzGraph& g) {
  const std::uint64_t m = g.src.size();
  FuzzHits h;
  h.graphs = 1;
  std::vector<std::uint32_t> root(g.n);
  std::iota(root.begin(), root.end(), 0u);
  const auto find = [&](std::uint32_t x) {
    while (root[x] != x) x = root[x] = root[root[x]];
    return x;
  };
  std::vector<bool> touched(g.n);
  std::vector<std::vector<std::int32_t>> times(g.n);  // per node: its non-loop edges' ts, ascending
  std::map<std::int32_t, std::uint64_t> per_ts;
  std::map<std::tuple<std::uint32_t, std::uint32_t, std::int32_t>, std::uint64_t> per_uvt;
  for (std::uint64_t e = 0; e < m; ++e) {
    touched[g.src[e]] = touched[g.dst[e]] = true;
    root[find(g.src[e])] = find(g.dst[e]);
    if (g.src[e] != g.dst[e]) {
      times[g.src[e]].push_back(g.ts[e]);
      times[g.dst[e]].push_back(g.ts[e]);
    }
    ++per_ts[g.ts[e]];
    ++per_uvt[{g.src[e], g.dst[e], g.ts[e]}];
  }
  for (std::uint32_t x = 0; x < g.n; ++x) h.components += touched[x] && find(x) == x;
  h.multi_component_graphs = h.components >= 2;
  // The bound edges of trigger e at node x and time t'; e itself is one of x's edges at its own t.
  const auto at = [&](std::uint32_t x, std::int64_t t, std::int64_t own) {
    const auto [lo, hi] = std::equal_range(times[x].begin(), times[x].end(), t);
    return std::uint64_t(hi - lo) - (t == own);
  };
  for (std::uint64_t e = 0; e < m; ++e) {
    const std::uint32_t u = g.src[e], v = g.dst[e];
    const std::int64_t t = g.ts[e], out = t - g.window, in = out + 1;
    h.tied += per_ts[g.ts[e]] > 1;
    h.duplicated += per_uvt[{u, v, g.ts[e]}] > 1;
    if (u == v) continue;
    ++h.triggers;
    h.bound_out += at(u, out, t) + at(v, out, t) > 0;
    h.bound_in += at(u, in, t) + at(v, in, t) > 0;
  }
  return h;
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

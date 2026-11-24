#pragma once
// Vertex statistics (SPEC v2 §8 S2): GFP's vertex_stats for the frozen recipe's F1, as a runtime
// module (they are aggregates, not patterns). The per-side computation below is BLZ_HD, so the
// CPU driver (vstats.cpp, built with -ffp-contract=off) and the CUDA twin (vstats_cuda.cu,
// --fmad=false) run the same operations in the same order and agree bit for bit.
//
// Row e (stream mode, one GFP transform() per edge) has 4 sides: the out- and in-edges of e's
// source, then of its destination. A side's window is that node's list entries with eid <= e and
// ts >= ts[e - 1] - W (every entry for e = 0): GFP's graph at the call, which keeps the edges with
// ts >= (the largest ts before the call) - time_window. W is GFP's time_window, the largest
// pattern window of its call; GFP's vertex_stats_tw changes nothing when it is >= W (A43).
#include <cmath>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "blz/primitives.hpp"
#include "runtime/triggers.hpp"

namespace blz {

// Per side: fan, degree, ratio, then for ts and amount: avg, sum, var, skew, kurtosis.
inline constexpr int kVstatsPerSide = 13;
inline constexpr int kVstatsCols = 4 * kVstatsPerSide;  // src out, src in, dst out, dst in

// GFP's column names in GFP's order (tools/gfp/dg2_features.py vstats_names()).
std::vector<std::string> vstats_column_names();

struct VstatsParams {
  std::int32_t window = 0;  // W: GFP's time_window (> 0)
};

// GFP's cancellation guard (A44): a variance below 1e-5 * mean^2 is reported as 0, with its
// skew and kurtosis.
inline constexpr double kVstatsVarGuard = 1e-5;

// avg, sum, var, skew, kurtosis of the values a side's window gives, in eid order. Two passes:
// the sum, then the central moments about the mean (population moments; kurtosis not excess).
struct VsMoments {
  double sum = 0, m2 = 0, m3 = 0, m4 = 0;
};

BLZ_HD inline void vstats_finish(const VsMoments& a, double n, double* out) {
  if (n == 0) {
    for (int i = 0; i < 5; ++i) out[i] = 0;
    return;
  }
  const double mu = a.sum / n;
  const double m2 = a.m2 / n, m3 = a.m3 / n, m4 = a.m4 / n;
  out[0] = mu;
  out[1] = a.sum;
  if (!(m2 > 0) || m2 < kVstatsVarGuard * (mu * mu)) {
    out[2] = out[3] = out[4] = 0;
    return;
  }
  out[2] = m2;
  out[3] = m3 / (m2 * sqrt(m2));
  out[4] = m4 / (m2 * m2);
}

// One side: node x's list l, the window entries [r.lo, r.hi) (eid <= e, ts >= the lower bound),
// `amount` by eid. Writes kVstatsPerSide values.
BLZ_HD inline void vstats_side(const TCsrView& l, Range r, const double* amount, double* out) {
  std::uint32_t fan = 0;
  VsMoments t, a;
  for (std::uint32_t j = r.lo; j < r.hi; ++j) {
    fan += l.distinct_first(j, r.lo) ? 1u : 0u;
    t.sum += double(l.ts[j]);
    a.sum += amount[l.eid[j]];
  }
  const std::uint32_t deg = r.hi - r.lo;
  const double n = double(deg);
  if (deg > 0) {
    const double mt = t.sum / n, ma = a.sum / n;
    for (std::uint32_t j = r.lo; j < r.hi; ++j) {
      const double dt = double(l.ts[j]) - mt, da = amount[l.eid[j]] - ma;
      const double dt2 = dt * dt, da2 = da * da;
      t.m2 += dt2;
      t.m3 += dt2 * dt;
      t.m4 += dt2 * dt2;
      a.m2 += da2;
      a.m3 += da2 * da;
      a.m4 += da2 * da2;
    }
  }
  out[0] = double(fan);
  out[1] = n;
  out[2] = fan ? n / double(fan) : 0.0;
  vstats_finish(t, n, out + 3);
  vstats_finish(a, n, out + 8);
}

// The lower ts bound of row e's windows.
BLZ_HD inline std::int64_t vstats_lower(const std::int32_t* ts, std::uint32_t e, std::int32_t window) {
  return e == 0 ? INT64_MIN : std::int64_t(ts[e - 1]) - window;
}

// Row e's kVstatsCols values.
BLZ_HD inline void vstats_row(const GraphView& g, const double* amount, const VstatsParams& p, std::uint32_t e,
                              double* out) {
  const std::int64_t lo = vstats_lower(g.ts, e, p.window);
  const std::uint32_t u = g.src[e], v = g.dst[e];
  vstats_side(g.out, g.out.range(u, lo, kNoUpper, e), amount, out);
  vstats_side(g.in, g.in.range(u, lo, kNoUpper, e), amount, out + kVstatsPerSide);
  vstats_side(g.out, g.out.range(v, lo, kNoUpper, e), amount, out + 2 * kVstatsPerSide);
  vstats_side(g.in, g.in.range(v, lo, kNoUpper, e), amount, out + 3 * kVstatsPerSide);
}

// The rows of the triggers `sel` (sel.rows() * kVstatsCols doubles, row r = sel.eid(r)), on the
// CPU with the current OpenMP thread count. `amount` is indexed by eid (g.m values).
void vstats_cpu(const GraphView& g, std::span<const double> amount, const VstatsParams& p, const TriggerSel& sel,
                double* out);

class DeviceGraph;
// The same on the GPU, for the eid range [first, last): `dg` is the device mirror of g. Throws in
// a build without CUDA.
void vstats_cuda(const DeviceGraph& dg, std::span<const double> amount, const VstatsParams& p, std::uint64_t first,
                 std::uint64_t last, double* out);

}  // namespace blz

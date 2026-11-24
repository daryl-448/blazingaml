#pragma once
// R8's vertex statistics (SPEC v3 §3.2; the P13 brief's F7, D-57, D-58) on the loop-free index,
// on the CPU and the GPU. The per-side computation below is BLZ_HD, so the CPU driver
// (clean_vstats.cpp, built with -ffp-contract=off) and the CUDA kernel (clean_vstats_cuda.cu,
// --fmad=false) run one sequence of float64 operations, the clean oracle's (CleanOracle::vstats),
// and give its bits.
//
// A side is node x's out- or in-list in the window (t - W, t]. Under R1-R3 it is a function of
// (x, direction, t) alone, so the rows of a chunk compute each distinct side once and copy it into
// every row that has it. On HI-Small at W = 86,400 the 20.3 M sides of the rows are 17.9 M distinct
// ones with 5.9e8 window entries, against 4.4e9 entries side by side: the hubs' sides repeat.
//
// Ascending order, which R8's sums follow: the ages of a window are its entries from the newest
// back (equal ts, equal ages). The amounts come from VsOrderView: each list is cut into runs at the
// multiples of W, sorted by amount once per run, and a window (t - W, t] lies in runs
// floor(t / W) - 1 and floor(t / W), so its amounts in order are those two runs merged and filtered
// to the window: a walk over the list entries of a span of 2W at most, with no scratch memory. On
// HI-Small a prototype's sides took 1.5 s that way on 12 threads, 3.1 s with one sorted run per
// list (10 list entries per window entry) and 3.5 s with a sort per side.
#include <cmath>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "blz/primitives.hpp"
#include "runtime/triggers.hpp"
#include "runtime/vstats.hpp"

namespace blz {

// One direction's amounts in R8's order: within each W-run of each list, pos holds the run's list
// positions by ascending amount (ties by position) and amount their amounts. Both are off[n] long.
struct VsOrderView {
  const std::uint32_t* pos = nullptr;
  const double* amount = nullptr;
};

// Node x's out-list (in = 0) or in-list (in = 1) at time t.
struct VsSide {
  std::int32_t t;
  std::uint32_t x;
  std::uint32_t in;
};

BLZ_HD inline std::int64_t vs_floor_div(std::int64_t a, std::int64_t b) {
  const std::int64_t q = a / b;
  return a % b != 0 && a < 0 ? q - 1 : q;
}

// R8 for one value column: mean, sum, var, skew, kurt into out[0..4]. walk(f) calls f with the
// side's n > 0 values in ascending order. The sum, then the central moments about the mean, each
// accumulated in that order from 0.0; min = max (n = 1 included) gives var = skew = kurt = 0, and
// a computed denominator of 0 gives skew or kurt 0 (D-58).
template <class Walk>
BLZ_HD inline void vs_moments(Walk walk, double n, double* out) {
  double sum = 0.0, lo = 0.0, hi = 0.0;
  bool first = true;
  walk([&](double x) {
    if (first) lo = x;
    first = false;
    hi = x;
    sum = sum + x;
  });
  const double mean = sum / n;
  out[0] = mean;
  out[1] = sum;
  out[2] = out[3] = out[4] = 0.0;
  if (lo == hi) return;
  double m2s = 0.0, m3s = 0.0, m4s = 0.0;
  walk([&](double x) {
    const double d = x - mean, d2 = d * d;
    m2s = m2s + d2;
    m3s = m3s + d2 * d;
    m4s = m4s + d2 * d2;
  });
  const double m2 = m2s / n, m3 = m3s / n, m4 = m4s / n;
  const double d3 = m2 * sqrt(m2), d4 = m2 * m2;
  out[2] = m2;
  out[3] = d3 == 0 ? 0.0 : m3 / d3;
  out[4] = d4 == 0 ? 0.0 : m4 / d4;
}

// One side's kVstatsPerSide values (F7): fan, deg, ratio, then the age t - t' and the amount, each
// mean, sum, var, skew, kurt. `l` is a list direction of the loop-free index (R1), `o` its order.
BLZ_HD inline void clean_vs_side(const TCsrView& l, const VsOrderView& o, std::int32_t window, VsSide s,
                                 double* out) {
  for (int i = 0; i < kVstatsPerSide; ++i) out[i] = 0.0;
  const std::int64_t t = s.t;
  const std::uint32_t a = l.off[s.x], b = l.off[s.x + 1];
  const std::uint32_t lo = l.first_ts_at_least(a, b, t - window + 1);  // R2: t - W < t'
  const std::uint32_t hi = l.first_ts_at_least(lo, b, t + 1);          // R3: t' <= t
  if (lo == hi) return;  // n = 0: every column 0
  std::uint32_t fan = 0;
  for (std::uint32_t j = lo; j < hi; ++j) fan += l.distinct_first(j, lo) ? 1u : 0u;
  const double n = double(hi - lo);
  out[0] = double(fan);
  out[1] = n;
  out[2] = n / double(fan);
  vs_moments(
      [&](auto f) {
        for (std::uint32_t j = hi; j-- > lo;) f(double(t - std::int64_t(l.ts[j])));
      },
      n, out + 3);
  // The window's runs: k - 1 is [r0, r1), k is [r1, r2).
  const std::int64_t k = vs_floor_div(t, window);
  const std::uint32_t r1 = l.first_ts_at_least(a, hi, k * window);
  const std::uint32_t r0 = l.first_ts_at_least(a, r1, (k - 1) * window);
  const std::uint32_t r2 = l.first_ts_at_least(hi, b, (k + 1) * window);
  vs_moments(
      [&](auto f) {
        const auto next = [&](std::uint32_t q, std::uint32_t end) {
          while (q < end && (o.pos[q] < lo || o.pos[q] >= hi)) ++q;
          return q;
        };
        std::uint32_t i = next(r0, r1), j = next(r1, r2);
        while (i < r1 && j < r2) {
          if (o.amount[i] <= o.amount[j]) {
            f(o.amount[i]);
            i = next(i + 1, r1);
          } else {
            f(o.amount[j]);
            j = next(j + 1, r2);
          }
        }
        for (; i < r1; i = next(i + 1, r1)) f(o.amount[i]);
        for (; j < r2; j = next(j + 1, r2)) f(o.amount[j]);
      },
      n, out + 8);
}

// The column names: vs_<side>_<stat>, sides u_out, u_in, v_out, v_in (F7).
std::vector<std::string> clean_vstats_column_names();

class DeviceGraph;

// R8's rows on the CPU, or on the GPU when given the graph's device mirror. Builds the amount order
// of both directions once, for one window.
class CleanVstats {
 public:
  // `g`: the host view of an index built with SelfLoops::drop (R1). `amount`: Amount Received by
  // eid, g.m finite values. `window`: W > 0. `dg`: g's device mirror, to compute on the GPU (a build
  // without CUDA throws).
  CleanVstats(const GraphView& g, std::span<const double> amount, std::int32_t window,
              const DeviceGraph* dg = nullptr);
  CleanVstats(const CleanVstats&) = delete;
  CleanVstats& operator=(const CleanVstats&) = delete;

  // The rows of `sel` (sel.rows() * kVstatsCols doubles; row r is trigger sel.eid(r)'s), computed
  // in chunks of kChunkRows rows.
  void rows(const TriggerSel& sel, double* out);
  // The distinct sides computed so far (SPEC v3 §4.4 item 6, path hit).
  std::uint64_t sides() const { return sides_; }

  static constexpr std::uint64_t kChunkRows = 1u << 18;

 private:
  struct Device;
  struct DeviceDelete {  // defined where Device is: clean_vstats_cuda.cu
    void operator()(Device* d) const;
  };
  void init_device(const DeviceGraph& dg);
  void sides_gpu(std::span<const VsSide> sides, double* out);

  GraphView g_;
  std::int32_t window_;
  std::vector<std::uint32_t> pos_[2];
  std::vector<double> amount_[2];
  std::unique_ptr<Device, DeviceDelete> dev_;
  std::uint64_t sides_ = 0;
};

}  // namespace blz

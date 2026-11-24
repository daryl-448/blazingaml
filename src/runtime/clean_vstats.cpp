// Built with -ffp-contract=off (CMakeLists.txt): no fused multiply-add, so the CPU rows equal the
// CUDA kernel's (--fmad=false) and the clean oracle's bit for bit (SPEC v3 R8).
#include "runtime/clean_vstats.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace blz {

namespace {

// A chunk's slots (row * 4 + side) take the low 21 bits of a plan key, below (x, in).
constexpr int kSlotBits = 21;
static_assert(4 * CleanVstats::kChunkRows <= (std::uint64_t(1) << kSlotBits));

// The distinct sides of a chunk's rows and, per slot (row * 4 + side: u_out, u_in, v_out, v_in),
// the index of its side. Rows sharing a side share a timestamp, and the rows of one timestamp are
// contiguous in the chunk (eids ascend, so ts does), so each run of equal ts is deduplicated on its
// own, in parallel. The sides are then ordered by their list's length, longest first, so a GPU warp
// holds sides of similar cost: one hub side among 31 small ones keeps the warp for the hub's whole
// walk (HI-Small's kernel: 4.3 s in timestamp order, 1.5 s in this one).
struct Plan {
  std::vector<VsSide> sides;
  std::vector<std::uint32_t> side_of;
};

void plan(const GraphView& g, const TriggerSel& sel, std::uint64_t r0, std::uint64_t r1, Plan& p) {
  const std::uint64_t rows = r1 - r0;
  std::vector<std::uint64_t> keys(4 * rows);
  std::vector<std::uint64_t> runs;  // first row of each run of equal ts, then `rows`
  for (std::uint64_t r = 0; r < rows; ++r) {
    const std::uint32_t e = sel.eid(r0 + r);
    if (r == 0 || g.ts[e] != g.ts[sel.eid(r0 + r - 1)]) runs.push_back(r);
    const std::uint64_t u = g.src[e], v = g.dst[e];
    for (std::uint64_t s = 0; s < 4; ++s) {
      const std::uint64_t x = s < 2 ? u : v, in = s & 1;
      keys[4 * r + s] = ((x << 1 | in) << kSlotBits) | (4 * r + s);
    }
  }
  runs.push_back(rows);
  const std::int64_t nruns = std::int64_t(runs.size()) - 1;
  std::vector<std::uint64_t> first(runs.size(), 0);  // per run, its first side; then the total
#pragma omp parallel for schedule(dynamic, 64)
  for (std::int64_t k = 0; k < nruns; ++k) {
    const auto a = keys.begin() + std::ptrdiff_t(4 * runs[k]), b = keys.begin() + std::ptrdiff_t(4 * runs[k + 1]);
    std::sort(a, b);
    std::uint64_t distinct = 0;
    for (auto it = a; it != b; ++it) distinct += it == a || (*it >> kSlotBits) != (it[-1] >> kSlotBits);
    first[std::size_t(k) + 1] = distinct;
  }
  std::partial_sum(first.begin(), first.end(), first.begin());
  std::vector<VsSide> found(first.back());
  std::vector<std::uint8_t> cls(found.size());  // 32 - bit_width(list length): 0 for the longest
  p.side_of.resize(4 * rows);
  constexpr std::uint64_t kSlotMask = (std::uint64_t(1) << kSlotBits) - 1;
#pragma omp parallel for schedule(dynamic, 64)
  for (std::int64_t k = 0; k < nruns; ++k) {
    const std::int32_t t = g.ts[sel.eid(r0 + runs[k])];
    std::uint64_t id = first[std::size_t(k)] - 1;
    for (std::uint64_t i = 4 * runs[k]; i < 4 * runs[k + 1]; ++i) {
      const std::uint64_t xd = keys[i] >> kSlotBits;
      if (i == 4 * runs[k] || xd != keys[i - 1] >> kSlotBits) {
        const auto x = std::uint32_t(xd >> 1);
        const TCsrView& l = xd & 1 ? g.in : g.out;
        found[++id] = {t, x, std::uint32_t(xd & 1)};
        cls[id] = std::uint8_t(32 - std::bit_width(l.off[x + 1] - l.off[x]));
      }
      p.side_of[keys[i] & kSlotMask] = std::uint32_t(id);
    }
  }
  // A stable counting sort by class.
  std::array<std::uint64_t, 34> at{};
  for (std::uint8_t c : cls) ++at[c + 1];
  std::partial_sum(at.begin(), at.end(), at.begin());
  std::vector<std::uint32_t> moved(found.size());
  p.sides.resize(found.size());
  for (std::size_t i = 0; i < found.size(); ++i) {
    moved[i] = std::uint32_t(at[cls[i]]++);
    p.sides[moved[i]] = found[i];
  }
#pragma omp parallel for schedule(static)
  for (std::int64_t i = 0; i < std::int64_t(p.side_of.size()); ++i) p.side_of[std::size_t(i)] = moved[p.side_of[std::size_t(i)]];
}

}  // namespace

std::vector<std::string> clean_vstats_column_names() {
  std::vector<std::string> names;
  for (const char* side : {"u_out", "u_in", "v_out", "v_in"}) {
    const std::string p = std::string("vs_") + side + "_";
    for (const char* s : {"fan", "deg", "ratio"}) names.push_back(p + s);
    for (const char* col : {"age", "amount"})
      for (const char* s : {"mean", "sum", "var", "skew", "kurt"}) names.push_back(p + col + "_" + s);
  }
  return names;
}

CleanVstats::CleanVstats(const GraphView& g, std::span<const double> amount, std::int32_t window,
                         const DeviceGraph* dg)
    : g_(g), window_(window) {
  if (amount.size() != g.m) throw std::invalid_argument("vstats: one amount per edge is needed");
  if (window <= 0) throw std::invalid_argument("vstats: window must be > 0");
  for (double a : amount)
    if (!std::isfinite(a)) throw std::invalid_argument("vstats: an amount is not finite (SPEC v3 R8)");
  for (int d = 0; d < 2; ++d) {
    const TCsrView& l = d ? g.in : g.out;
    std::vector<std::uint32_t>& pos = pos_[d];
    std::vector<double>& am = amount_[d];
    pos.resize(l.off[g.n]);
    am.resize(l.off[g.n]);
#pragma omp parallel for schedule(dynamic, 256)
    for (std::int64_t x = 0; x < std::int64_t(g.n); ++x)
      for (std::uint32_t a = l.off[x], b = l.off[x + 1], e; a < b; a = e) {
        e = l.first_ts_at_least(a, b, (vs_floor_div(l.ts[a], window) + 1) * window);  // the W-run of a
        std::iota(pos.begin() + a, pos.begin() + e, a);
        std::sort(pos.begin() + a, pos.begin() + e, [&](std::uint32_t p, std::uint32_t q) {
          const double ap = amount[l.eid[p]], aq = amount[l.eid[q]];
          return ap < aq || (ap == aq && p < q);
        });
        for (std::uint32_t j = a; j < e; ++j) am[j] = amount[l.eid[pos[j]]];
      }
  }
  if (dg) init_device(*dg);
}

void CleanVstats::rows(const TriggerSel& sel, double* out) {
  if (sel.last > g_.m || sel.first > sel.last) throw std::invalid_argument("vstats: bad trigger range");
  Plan p;
  std::vector<double> side_out;
  for (std::uint64_t r0 = 0; r0 < sel.rows(); r0 += kChunkRows) {
    const std::uint64_t r1 = std::min(sel.rows(), r0 + kChunkRows);
    plan(g_, sel, r0, r1, p);
    side_out.resize(p.sides.size() * kVstatsPerSide);
    if (dev_) {
      sides_gpu(p.sides, side_out.data());
    } else {
      const VsOrderView o[2] = {{pos_[0].data(), amount_[0].data()}, {pos_[1].data(), amount_[1].data()}};
#pragma omp parallel for schedule(dynamic, 64)
      for (std::int64_t i = 0; i < std::int64_t(p.sides.size()); ++i) {
        const VsSide s = p.sides[std::size_t(i)];
        clean_vs_side(s.in ? g_.in : g_.out, o[s.in], window_, s, side_out.data() + std::size_t(i) * kVstatsPerSide);
      }
    }
    sides_ += p.sides.size();
#pragma omp parallel for schedule(static)
    for (std::int64_t r = 0; r < std::int64_t(r1 - r0); ++r)
      for (std::size_t s = 0; s < 4; ++s)
        std::memcpy(out + (r0 + std::uint64_t(r)) * kVstatsCols + s * kVstatsPerSide,
                    side_out.data() + std::size_t(p.side_of[4 * std::size_t(r) + s]) * kVstatsPerSide,
                    kVstatsPerSide * sizeof(double));
  }
}

#ifndef BLZ_WITH_CUDA
struct CleanVstats::Device {};
void CleanVstats::DeviceDelete::operator()(Device* d) const { delete d; }
void CleanVstats::init_device(const DeviceGraph&) { throw std::runtime_error("vstats: this build has no CUDA backend"); }
void CleanVstats::sides_gpu(std::span<const VsSide>, double*) {
  throw std::runtime_error("vstats: this build has no CUDA backend");
}
#endif

}  // namespace blz

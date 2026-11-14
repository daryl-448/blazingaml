// Built with -ffp-contract=off (CMakeLists.txt): no fused multiply-add, so the CPU rows equal the
// CUDA twin's (--fmad=false) bit for bit (SPEC v2 §8 S2).
#include "runtime/vstats.hpp"

#include <stdexcept>

namespace blz {

std::vector<std::string> vstats_column_names() {
  static const char* stats[] = {"avg", "sum", "var", "skew", "kurtosis"};
  std::vector<std::string> names;
  for (const char* vert : {"src", "dst"})
    for (const char* dir : {"out", "in"}) {
      const std::string p = std::string("vs_") + vert + "_" + dir + "_";
      for (const char* s : {"fan", "deg", "ratio"}) names.push_back(p + s);
      for (const char* col : {"ts", "amount_received"})
        for (const char* s : stats) names.push_back(p + col + "_" + s);
    }
  return names;
}

void vstats_cpu(const GraphView& g, std::span<const double> amount, const VstatsParams& p, const TriggerSel& sel,
                double* out) {
  if (amount.size() != g.m) throw std::invalid_argument("vstats: one amount per edge is needed");
  if (p.window <= 0) throw std::invalid_argument("vstats: window must be > 0");
  if (sel.last > g.m || sel.first > sel.last) throw std::invalid_argument("vstats: bad trigger range");
  const std::int64_t rows = std::int64_t(sel.rows());
#pragma omp parallel for schedule(dynamic, 1024)
  for (std::int64_t r = 0; r < rows; ++r)
    vstats_row(g, amount.data(), p, sel.eid(std::uint64_t(r)), out + std::size_t(r) * kVstatsCols);
}

#ifndef BLZ_WITH_CUDA
void vstats_cuda(const DeviceGraph&, std::span<const double>, const VstatsParams&, std::uint64_t, std::uint64_t,
                 double*) {
  throw std::runtime_error("vstats: this build has no CUDA backend");
}
#endif

}  // namespace blz

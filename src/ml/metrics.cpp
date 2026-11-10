#include "ml/metrics.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace blz {

Metrics compute_metrics(std::span<const float> score, std::span<const std::uint8_t> label,
                        double threshold) {
  if (score.size() != label.size()) throw std::invalid_argument("metrics: size mismatch");
  Metrics r;
  for (std::size_t i = 0; i < score.size(); ++i) {
    const bool pred = score[i] > threshold, pos = label[i] != 0;
    (pred ? (pos ? r.cm.tp : r.cm.fp) : (pos ? r.cm.fn : r.cm.tn))++;
  }
  const double tp = double(r.cm.tp), pp = double(r.cm.tp + r.cm.fp), ap = double(r.cm.tp + r.cm.fn);
  r.precision = pp > 0 ? tp / pp : 0;
  r.recall = ap > 0 ? tp / ap : 0;
  r.f1 = pp + ap > 0 ? 2 * tp / (pp + ap) : 0;

  if (ap > 0) {
    std::vector<std::uint32_t> order(score.size());
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(), [&](auto a, auto b) { return score[a] > score[b]; });
    double hits = 0, prev_recall = 0;
    for (std::size_t i = 0; i < order.size();) {
      std::size_t j = i;
      for (; j < order.size() && score[order[j]] == score[order[i]]; ++j) hits += label[order[j]] != 0;
      const double rec = hits / ap;
      r.pr_auc += (rec - prev_recall) * (hits / double(j));
      prev_recall = rec;
      i = j;
    }
  }
  return r;
}

}  // namespace blz

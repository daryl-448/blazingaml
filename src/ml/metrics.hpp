#pragma once
#include <cstdint>
#include <span>

namespace blz {

// Binary-classification metrics for the laundering class (SPEC §10.3).
struct Confusion {
  std::uint64_t tp = 0, fp = 0, fn = 0, tn = 0;
};

struct Metrics {
  Confusion cm;
  double precision = 0, recall = 0, f1 = 0;  // 0 when undefined (no predicted / actual positives)
  double pr_auc = 0;                         // average precision; 0 when there are no positives
};

// A row is predicted positive when score > threshold (XGBClassifier.predict's rule).
// PR-AUC is average precision, sum_k (R_k - R_{k-1}) * P_k over the distinct scores in
// decreasing order, so tied scores form one step (sklearn's average_precision_score).
Metrics compute_metrics(std::span<const float> score, std::span<const std::uint8_t> label,
                        double threshold);

}  // namespace blz

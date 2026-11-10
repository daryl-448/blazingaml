#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace blz {

// configs/xgb_*.yaml (SPEC §10.2).
struct XgbConfig {
  std::vector<std::pair<std::string, std::string>> params;  // passed verbatim to XGBoost
  int num_boost_round = 100;
  double threshold = 0.5;
  std::uint64_t batch_rows = 262144;  // rows per QuantileDMatrix callback batch
  int max_bin() const;                // params' max_bin, else XGBoost's default 256
};

// Throws std::runtime_error with the file name on any schema error.
XgbConfig load_xgb_config(const std::string& path);
XgbConfig parse_xgb_config(std::string_view yaml_text, const std::string& name);

// A dense float32 matrix with labels, produced in row batches so that it never has to be
// materialized whole (SPEC §5.4).
struct RowSource {
  std::uint64_t rows = 0, cols = 0;
  // Fills x[count * cols] (C order) and y[count] with rows [first, first + count).
  std::function<void(std::uint64_t first, std::uint64_t count, float* x, float* y)> fill;
};

struct XgbTimings {
  double train_matrix_s = 0, test_matrix_s = 0, train_s = 0, predict_s = 0;
};

// Builds a QuantileDMatrix from `train`, then one from `test` quantized with the training cut
// points (ref = the training matrix), trains `num_boost_round` rounds and returns the test
// scores (probabilities for binary:logistic). nthread <= 0 uses XGBoost's default.
std::vector<float> xgb_train_predict(const RowSource& train, const RowSource& test,
                                     const XgbConfig& cfg, int nthread, XgbTimings* timings);

std::string xgb_version();

}  // namespace blz

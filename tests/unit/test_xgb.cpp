#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <stdexcept>
#include <vector>

#include "ml/metrics.hpp"
#include "ml/xgb.hpp"

namespace {

// y = (x0 + 3 * x1) % 7 >= 4; x2 is noise. Row i of the source is global row first + i.
blz::RowSource toy(std::uint64_t first, std::uint64_t rows) {
  blz::RowSource s;
  s.rows = rows;
  s.cols = 3;
  s.fill = [first](std::uint64_t at, std::uint64_t count, float* x, float* y) {
    for (std::uint64_t r = 0; r < count; ++r) {
      const std::uint64_t i = first + at + r;
      const std::uint64_t a = i % 7, b = (i / 7) % 5;
      x[3 * r] = float(a);
      x[3 * r + 1] = float(b);
      x[3 * r + 2] = float((i * 2654435761u) % 1000);
      y[r] = float((a + 3 * b) % 7 >= 4);
    }
  };
  return s;
}

}  // namespace

TEST_CASE("xgb config: defaults, max_bin and schema errors") {
  const auto cfg = blz::parse_xgb_config(
      "num_boost_round: 7\nparams: {objective: 'binary:logistic', max_bin: 64}\n", "t");
  CHECK(cfg.num_boost_round == 7);
  CHECK(cfg.threshold == 0.5);
  CHECK(cfg.batch_rows == 262144);
  CHECK(cfg.max_bin() == 64);
  CHECK(cfg.params.size() == 2);
  CHECK(blz::parse_xgb_config("params: {}\n", "t").max_bin() == 256);
  CHECK_THROWS_AS(blz::parse_xgb_config("params: {}\nrounds: 3\n", "t"), std::runtime_error);
  CHECK_THROWS_AS(blz::parse_xgb_config("num_boost_round: 3\n", "t"), std::runtime_error);
  CHECK_THROWS_AS(blz::parse_xgb_config("params: {nthread: 4}\n", "t"), std::runtime_error);
  CHECK_THROWS_AS(blz::parse_xgb_config("params: {max_bin: x}\n", "t"), std::runtime_error);
}

TEST_CASE("xgb: batched QuantileDMatrix, train and predict on a learnable toy problem") {
  auto cfg = blz::parse_xgb_config(
      "num_boost_round: 30\nbatch_rows: 97\n"
      "params: {objective: 'binary:logistic', tree_method: hist, max_depth: 4, seed: 0}\n",
      "t");
  const auto train = toy(0, 2000), test = toy(2000, 700);
  blz::XgbTimings t;
  const auto score = blz::xgb_train_predict(train, test, cfg, 2, &t);
  REQUIRE(score.size() == 700);
  std::vector<float> xs(700 * 3), ys(700);
  test.fill(0, 700, xs.data(), ys.data());
  std::vector<std::uint8_t> y(ys.begin(), ys.end());
  const auto m = blz::compute_metrics(score, y, cfg.threshold);
  CHECK(m.f1 == 1.0);

  // Same data in one batch: identical scores.
  cfg.batch_rows = 1 << 20;
  CHECK(blz::xgb_train_predict(train, test, cfg, 2, nullptr) == score);
}

TEST_CASE("xgb: an exception in the row source propagates") {
  const auto cfg = blz::parse_xgb_config("batch_rows: 10\nparams: {objective: 'binary:logistic'}\n", "t");
  auto bad = toy(0, 100);
  bad.fill = [](std::uint64_t at, std::uint64_t, float*, float*) {
    if (at >= 30) throw std::runtime_error("boom");
  };
  CHECK_THROWS_WITH(blz::xgb_train_predict(bad, toy(0, 10), cfg, 1, nullptr), "boom");
}

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <vector>

#include "ml/metrics.hpp"

using Catch::Approx;

TEST_CASE("metrics: confusion, F1 and the strict > threshold") {
  const std::vector<float> s = {0.9f, 0.5f, 0.7f, 0.2f, 0.6f, 0.1f};
  const std::vector<std::uint8_t> y = {1, 1, 0, 1, 1, 0};
  const auto m = blz::compute_metrics(s, y, 0.5);
  CHECK(m.cm.tp == 2);  // 0.9, 0.6 (0.5 is not > 0.5)
  CHECK(m.cm.fp == 1);
  CHECK(m.cm.fn == 2);
  CHECK(m.cm.tn == 1);
  CHECK(m.precision == Approx(2.0 / 3));
  CHECK(m.recall == Approx(0.5));
  CHECK(m.f1 == Approx(4.0 / 7));
}

TEST_CASE("metrics: average precision, with tied scores as one step") {
  // Ranked: 0.9(+) 0.8(-) 0.7(+) 0.7(-) 0.1(+). Steps: R=1/3 P=1; R=2/3 P=2/4; R=1 P=3/5.
  // sklearn.metrics.average_precision_score gives 0.6333...
  const std::vector<float> s = {0.7f, 0.9f, 0.1f, 0.8f, 0.7f};
  const std::vector<std::uint8_t> y = {1, 1, 1, 0, 0};
  const auto m = blz::compute_metrics(s, y, 0.5);
  CHECK(m.pr_auc == Approx(1.0 / 3 + (1.0 / 3) * 0.5 + (1.0 / 3) * 0.6));
}

TEST_CASE("metrics: undefined ratios are 0") {
  const std::vector<float> s = {0.1f, 0.2f};
  const std::vector<std::uint8_t> none = {0, 0}, some = {1, 0};
  auto m = blz::compute_metrics(s, none, 0.5);
  CHECK(m.precision == 0);
  CHECK(m.recall == 0);
  CHECK(m.f1 == 0);
  CHECK(m.pr_auc == 0);
  m = blz::compute_metrics(s, some, 0.5);
  CHECK(m.cm.fn == 1);
  CHECK(m.precision == 0);
  CHECK(m.f1 == 0);
  CHECK(m.pr_auc == Approx(0.5));
}

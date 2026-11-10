#include "ml/xgb.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>

#include <xgboost/c_api.h>
#include <yaml-cpp/yaml.h>

namespace blz {

int XgbConfig::max_bin() const {
  for (const auto& [k, v] : params)
    if (k == "max_bin") return std::stoi(v);
  return 256;
}

XgbConfig parse_xgb_config(std::string_view yaml_text, const std::string& name) {
  auto fail = [&](const std::string& msg) { throw std::runtime_error(name + ": " + msg); };
  XgbConfig cfg;
  try {
    YAML::Node root = YAML::Load(std::string(yaml_text));
    if (!root.IsMap()) fail("top level must be a map");
    for (auto kv : root) {
      const auto key = kv.first.as<std::string>();
      if (key != "params" && key != "num_boost_round" && key != "threshold" && key != "batch_rows")
        fail("unknown key '" + key + "'");
    }
    if (root["num_boost_round"]) cfg.num_boost_round = root["num_boost_round"].as<int>();
    if (root["threshold"]) cfg.threshold = root["threshold"].as<double>();
    if (root["batch_rows"]) cfg.batch_rows = root["batch_rows"].as<std::uint64_t>();
    if (cfg.num_boost_round < 1) fail("num_boost_round must be >= 1");
    if (cfg.batch_rows < 1) fail("batch_rows must be >= 1");
    const YAML::Node params = root["params"];
    if (!params || !params.IsMap()) fail("'params' must be a map");
    for (auto kv : params) {
      if (!kv.second.IsScalar()) fail("params: '" + kv.first.as<std::string>() + "' must be a scalar");
      const auto k = kv.first.as<std::string>();
      if (k == "nthread") fail("params: nthread comes from --threads");
      cfg.params.emplace_back(k, kv.second.as<std::string>());
    }
    if (cfg.max_bin() < 2) fail("params: max_bin must be >= 2");
  } catch (const YAML::Exception& e) {
    throw std::runtime_error(name + ": " + e.what());
  } catch (const std::logic_error& e) {  // std::stoi in max_bin()
    throw std::runtime_error(name + ": params: bad max_bin");
  }
  return cfg;
}

XgbConfig load_xgb_config(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::stringstream ss;
  ss << in.rdbuf();
  return parse_xgb_config(ss.str(), path);
}

namespace {

void check(int rc) {
  if (rc != 0) throw std::runtime_error(std::string("xgboost: ") + XGBGetLastError());
}

struct DMatrixFree {
  void operator()(void* h) const { XGDMatrixFree(h); }
};
struct BoosterFree {
  void operator()(void* h) const { XGBoosterFree(h); }
};
using DMatrix = std::unique_ptr<void, DMatrixFree>;
using Booster = std::unique_ptr<void, BoosterFree>;

std::string array_interface(const float* p, std::uint64_t rows, std::uint64_t cols) {
  std::string shape = std::to_string(rows);
  if (cols > 0) shape += ", " + std::to_string(cols);
  return R"({"data": [)" + std::to_string(reinterpret_cast<std::uintptr_t>(p)) +
         R"(, true], "shape": [)" + shape + R"(], "typestr": "<f4", "version": 3})";
}

// The QuantileDMatrix data iterator. XGBoost passes over the data more than once (sketch, then
// index), calling reset in between; each batch buffer stays valid until the next call.
struct BatchIter {
  const RowSource* src = nullptr;
  std::uint64_t batch = 0, pos = 0;
  void* proxy = nullptr;
  std::vector<float> x, y;
  std::exception_ptr error;

  static void reset(DataIterHandle h) { static_cast<BatchIter*>(h)->pos = 0; }

  static int next(DataIterHandle h) {
    auto* it = static_cast<BatchIter*>(h);
    if (it->error || it->pos >= it->src->rows) return 0;
    try {
      const std::uint64_t count = std::min(it->batch, it->src->rows - it->pos);
      it->x.resize(count * it->src->cols);
      it->y.resize(count);
      it->src->fill(it->pos, count, it->x.data(), it->y.data());
      check(XGProxyDMatrixSetDataDense(
          it->proxy, array_interface(it->x.data(), count, it->src->cols).c_str()));
      check(XGDMatrixSetInfoFromInterface(it->proxy, "label",
                                          array_interface(it->y.data(), count, 0).c_str()));
      it->pos += count;
      return 1;
    } catch (...) {
      it->error = std::current_exception();
      return 0;
    }
  }
};

DMatrix quantile_dmatrix(const RowSource& src, void* ref, const XgbConfig& cfg, int nthread) {
  if (src.rows == 0 || src.cols == 0) throw std::runtime_error("xgboost: empty matrix");
  void* proxy = nullptr;
  check(XGProxyDMatrixCreate(&proxy));
  DMatrix proxy_owner(proxy);
  BatchIter it{&src, cfg.batch_rows, 0, proxy, {}, {}, nullptr};
  std::string conf = R"({"missing": NaN, "max_bin": )" + std::to_string(cfg.max_bin());
  if (nthread > 0) conf += R"(, "nthread": )" + std::to_string(nthread);
  conf += "}";
  void* out = nullptr;
  const int rc = XGQuantileDMatrixCreateFromCallback(&it, proxy, ref, &BatchIter::reset,
                                                     &BatchIter::next, conf.c_str(), &out);
  DMatrix dm(out);
  if (it.error) std::rethrow_exception(it.error);
  check(rc);
  bst_ulong rows = 0, cols = 0;
  check(XGDMatrixNumRow(dm.get(), &rows));
  check(XGDMatrixNumCol(dm.get(), &cols));
  if (rows != src.rows || cols != src.cols)
    throw std::runtime_error("xgboost: QuantileDMatrix shape does not match the source");
  return dm;
}

double secs_since(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

}  // namespace

std::vector<float> xgb_train_predict(const RowSource& train, const RowSource& test,
                                     const XgbConfig& cfg, int nthread, XgbTimings* timings) {
  if (train.cols != test.cols) throw std::runtime_error("xgboost: train/test column mismatch");
  XgbTimings t;
  auto t0 = std::chrono::steady_clock::now();
  DMatrix dtrain = quantile_dmatrix(train, nullptr, cfg, nthread);
  t.train_matrix_s = secs_since(t0);

  t0 = std::chrono::steady_clock::now();
  void* dm = dtrain.get();
  void* b = nullptr;
  check(XGBoosterCreate(&dm, 1, &b));
  Booster booster(b);
  if (nthread > 0) check(XGBoosterSetParam(b, "nthread", std::to_string(nthread).c_str()));
  for (const auto& [k, v] : cfg.params) check(XGBoosterSetParam(b, k.c_str(), v.c_str()));
  for (int i = 0; i < cfg.num_boost_round; ++i) check(XGBoosterUpdateOneIter(b, i, dm));
  t.train_s = secs_since(t0);

  t0 = std::chrono::steady_clock::now();
  DMatrix dtest = quantile_dmatrix(test, dm, cfg, nthread);
  t.test_matrix_s = secs_since(t0);

  t0 = std::chrono::steady_clock::now();
  const bst_ulong* shape = nullptr;
  bst_ulong dim = 0;
  const float* res = nullptr;
  check(XGBoosterPredictFromDMatrix(
      b, dtest.get(),
      R"({"type": 0, "training": false, "iteration_begin": 0, "iteration_end": 0, "strict_shape": false})",
      &shape, &dim, &res));
  if (dim != 1 || shape[0] != test.rows) throw std::runtime_error("xgboost: unexpected prediction shape");
  std::vector<float> out(res, res + test.rows);
  t.predict_s = secs_since(t0);
  if (timings) *timings = t;
  return out;
}

std::string xgb_version() {
  int a = 0, b = 0, c = 0;
  XGBoostVersion(&a, &b, &c);
  return std::to_string(a) + "." + std::to_string(b) + "." + std::to_string(c);
}

}  // namespace blz

// blz — BlazingAML runtime CLI (SPEC §14). Commands so far: prepare, export-edges, stats (P0);
// mine (oracle backend), compare, fuzz (P1); train (P2).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <tuple>
#include <vector>

#include <sys/resource.h>

#include <CLI/CLI.hpp>
#include <omp.h>
#include <nlohmann/json.hpp>

#include "common/mmap_file.hpp"
#include "common/npy.hpp"
#include "common/sha256.hpp"
#include "graph/fuzz_gen.hpp"
#include "graph/prepare.hpp"
#include "graph/stats.hpp"
#include "io/blzg.hpp"
#include "io/csv.hpp"
#include "ml/metrics.hpp"
#include "ml/xgb.hpp"
#include "oracle/oracle.hpp"
#include "runtime/features.hpp"

namespace {

using Clock = std::chrono::steady_clock;
double secs(Clock::time_point a) { return std::chrono::duration<double>(Clock::now() - a).count(); }

void ensure_parent(const std::string& path) {
  auto parent = std::filesystem::path(path).parent_path();
  if (!parent.empty()) std::filesystem::create_directories(parent);
}

int prepare(const std::string& csv, const std::string& out) {
  auto t0 = Clock::now();
  blz::MappedFile file(csv);
  std::string_view text(file.data(), file.size());
  blz::RawTable raw = blz::parse_csv(text);
  std::fprintf(stderr, "parsed %zu rows in %.1f s\n", raw.size(), secs(t0));
  blz::EdgeTable t = blz::build_edge_table(raw);
  t.csv_sha256 = blz::sha256(file.data(), file.size());
  std::fprintf(stderr, "built stream: m=%zu n=%zu (%.1f s)\n", t.m(), t.n(), secs(t0));

  auto bad = blz::check_hi_small_invariants(t);
  for (auto& b : bad) std::fprintf(stderr, "invariant failed: %s\n", b.c_str());
  if (!bad.empty()) return 1;
  std::fprintf(stderr, "SPEC §3.5 invariants: pass\n");

  ensure_parent(out);
  blz::write_blzg(out, t);
  std::fprintf(stderr, "wrote %s (%.1f s total), csv sha256 %s\n", out.c_str(), secs(t0),
               blz::to_hex(t.csv_sha256).c_str());
  return 0;
}

int export_edges(const std::string& graph, const std::string& out) {
  blz::BlzgView g(graph);
  std::vector<std::int64_t> rows(g.m() * 4);
  for (std::uint64_t e = 0; e < g.m(); ++e) {
    rows[4 * e] = std::int64_t(e);
    rows[4 * e + 1] = g.src[e];
    rows[4 * e + 2] = g.dst[e];
    rows[4 * e + 3] = g.ts[e];
  }
  ensure_parent(out);
  blz::write_npy_i64(out, rows, g.m(), 4);
  std::fprintf(stderr, "wrote %s: int64[%llu, 4] (eid, src, dst, ts)\n", out.c_str(),
               (unsigned long long)g.m());
  return 0;
}

int stats(const std::string& graph, const std::string& out) {
  auto t0 = Clock::now();
  blz::BlzgView g(graph);
  nlohmann::json profile = blz::profile_stream({g.src, g.dst, g.ts, g.label, g.n()});
  profile["source"] = {{"graph", graph}, {"csv_sha256", blz::to_hex(g.csv_sha256())}};

  // Keep sections other tools merged in (e.g. gfp_tcyc from tools/gfp/e0_tcyc.py).
  nlohmann::json doc = nlohmann::json::object();
  if (std::filesystem::exists(out)) doc = nlohmann::json::parse(std::ifstream(out));
  doc.update(profile);
  ensure_parent(out);
  std::ofstream(out) << doc.dump(1) << '\n';
  std::fprintf(stderr, "wrote %s (%.1f s)\n", out.c_str(), secs(t0));
  return 0;
}


// "a:b" -> [a, b)
std::pair<std::uint64_t, std::uint64_t> parse_range(const std::string& s) {
  const auto c = s.find(':');
  if (c == std::string::npos) throw std::runtime_error("expected a:b, got '" + s + "'");
  const std::uint64_t a = std::stoull(s.substr(0, c)), b = std::stoull(s.substr(c + 1));
  if (a > b) throw std::runtime_error("empty range '" + s + "'");
  return {a, b};
}

// "2..33" or "2,3,5"
std::vector<std::uint32_t> parse_bins(const std::string& s) {
  std::vector<std::uint32_t> bins;
  if (auto d = s.find(".."); d != std::string::npos) {
    for (auto b = std::stoul(s.substr(0, d)), e = std::stoul(s.substr(d + 2)); b <= e; ++b)
      bins.push_back(std::uint32_t(b));
  } else {
    for (std::size_t p = 0; p <= s.size();) {
      auto q = s.find(',', p);
      if (q == std::string::npos) q = s.size();
      bins.push_back(std::uint32_t(std::stoul(s.substr(p, q - p))));
      p = q + 1;
    }
  }
  return bins;
}

nlohmann::json git_info() {
  auto run = [](const std::string& cmd) {
    std::string out;
    if (FILE* f = popen(cmd.c_str(), "r")) {
      char buf[256];
      while (std::fgets(buf, sizeof buf, f)) out += buf;
      if (pclose(f) != 0) return std::string();
    }
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
    return out;
  };
  const std::string git = "git -C '" BLZ_SOURCE_DIR "' ";
  const std::string sha = run(git + "rev-parse HEAD 2>/dev/null");
  if (sha.empty()) return nullptr;
  return {{"sha", sha}, {"dirty", !run(git + "status --porcelain 2>/dev/null").empty()}};
}

nlohmann::json block_params_json(const blz::BlockParams& p, blz::Block b) {
  return {{"window", p.window},
          {"lower", blz::closed_lower_bound(b) ? "closed" : "open"},
          {"bins", p.bins}};
}

int mine(const std::string& graph, const std::string& config, const std::string& backend,
         int threads, const std::string& blocks_opt, const std::string& triggers,
         const std::string& out) {
  if (backend != "oracle")
    throw std::runtime_error("backend '" + backend + "' is not implemented yet (P4/P5)");
  const blz::FeatureConfig cfg = blz::load_feature_config(config);
  std::vector<blz::Block> blocks;
  if (blocks_opt.empty()) {
    for (blz::Block b : blz::kAllBlocks)
      if (cfg[b]) blocks.push_back(b);
  } else {
    for (std::size_t p = 0; p <= blocks_opt.size();) {
      auto q = std::min(blocks_opt.find(',', p), blocks_opt.size());
      const std::string name = blocks_opt.substr(p, q - p);
      const auto b = blz::parse_block(name);
      if (!b) throw std::runtime_error("unknown block '" + name + "'");
      if (!cfg[*b]) throw std::runtime_error("block '" + name + "' is not in " + config);
      blocks.push_back(*b);
      p = q + 1;
    }
  }

  auto t0 = Clock::now();
  blz::BlzgView g(graph);
  std::uint64_t first = 0, last = g.m();
  if (triggers == "train") {
    last = blz::train_size(g.m());
  } else if (triggers == "test") {
    first = blz::train_size(g.m());
  } else if (triggers.rfind("range:", 0) == 0) {
    std::tie(first, last) = parse_range(triggers.substr(6));
    if (last > g.m()) throw std::runtime_error("trigger range past m");
  } else if (triggers != "all") {
    throw std::runtime_error("--triggers must be all, train, test or range:a:b");
  }
  if (threads > 0) omp_set_num_threads(threads);
  const blz::Oracle oracle({g.src, g.dst, g.ts, g.n()});
  const double build_s = secs(t0);
  std::fprintf(stderr, "oracle index built in %.1f s (m=%llu, n=%llu)\n", build_s,
               (unsigned long long)g.m(), (unsigned long long)g.n());

  std::filesystem::create_directories(out);
  const std::string meta_path = out + "/features.json";
  const nlohmann::json run_key = {{"backend", backend},
                                  {"csv_sha256", blz::to_hex(g.csv_sha256())},
                                  {"k_max", cfg.k_max},
                                  {"triggers", {{"spec", triggers}, {"first", first}, {"last", last}}}};
  nlohmann::json meta = nlohmann::json::object();
  if (std::filesystem::exists(meta_path)) {
    auto old = nlohmann::json::parse(std::ifstream(meta_path));
    if (old.value("run", nlohmann::json()) == run_key) meta = old;  // same run: keep its blocks
  }
  meta["run"] = run_key;
  meta["graph"] = graph;
  meta["config"] = config;
  meta["git"] = git_info();
  meta["threads"] = omp_get_max_threads();
  meta["index_build_seconds"] = build_s;
  if (!meta.contains("blocks")) meta["blocks"] = nlohmann::json::object();

  const std::uint64_t rows = last - first;
  std::uint64_t total_saturated = 0;
  for (blz::Block b : blocks) {
    const blz::BlockParams& p = *cfg[b];
    std::vector<std::uint32_t> feats(rows * p.bins.size(), 0);
    auto t1 = Clock::now();
    const std::uint64_t sat = oracle.mine(b, p, cfg.k_max, first, last, feats.data());
    const double mine_s = secs(t1);
    const std::string file = std::string(blz::block_name(b)) + ".npy";
    blz::write_npy_u32(out + "/" + file, feats, rows, p.bins.size());
    auto j = block_params_json(p, b);
    j["file"] = file;
    j["columns"] = blz::column_names(b, p.bins);
    j["mine_seconds"] = mine_s;
    j["saturated"] = sat;
    meta["blocks"][std::string(blz::block_name(b))] = j;
    total_saturated += sat;
    std::fprintf(stderr, "%-8s %8.1f s  saturated=%llu\n", std::string(blz::block_name(b)).c_str(),
                 mine_s, (unsigned long long)sat);
    std::ofstream(meta_path) << meta.dump(1) << '\n';
  }
  if (total_saturated > 0) std::fprintf(stderr, "warning: %llu saturated counts\n",
                                        (unsigned long long)total_saturated);
  return 0;
}

// Mismatch report between two uint32 feature blocks (SPEC §11.1). Exit code 0 if equal, 2 if not.
int compare(const std::string& a_path, const std::string& b_path, const std::string& graph,
            std::int64_t tw, std::uint64_t first_eid, const std::string& report,
            const std::string& name) {
  blz::NpyU32View a(a_path), b(b_path);
  if (a.rows() != b.rows() || a.cols() != b.cols())
    throw std::runtime_error("shapes differ: (" + std::to_string(a.rows()) + ", " +
                             std::to_string(a.cols()) + ") vs (" + std::to_string(b.rows()) +
                             ", " + std::to_string(b.cols()) + ")");
  const std::uint64_t rows = a.rows(), cols = a.cols();
  std::vector<std::uint64_t> col_mismatch(cols, 0);
  std::vector<std::uint64_t> bad_rows;
  for (std::uint64_t r = 0; r < rows; ++r) {
    auto x = a.row(r), y = b.row(r);
    bool bad = false;
    for (std::uint64_t c = 0; c < cols; ++c)
      if (x[c] != y[c]) {
        ++col_mismatch[c];
        bad = true;
      }
    if (bad) bad_rows.push_back(r);
  }

  nlohmann::json res = {{"a", a_path}, {"b", b_path}, {"rows", rows}, {"cols", cols},
                        {"first_eid", first_eid}, {"mismatched_rows", bad_rows.size()}};
  nlohmann::json per_col = nlohmann::json::object();
  for (std::uint64_t c = 0; c < cols; ++c)
    if (col_mismatch[c]) per_col[std::to_string(c)] = col_mismatch[c];
  res["mismatches_by_column"] = per_col;
  auto fmt_row = [&](std::span<const std::uint32_t> r) {
    return std::vector<std::uint32_t>(r.begin(), r.end());
  };
  nlohmann::json examples = nlohmann::json::array();
  for (std::size_t i = 0; i < std::min<std::size_t>(20, bad_rows.size()); ++i) {
    const auto r = bad_rows[i];
    examples.push_back({{"eid", first_eid + r}, {"a", fmt_row(a.row(r))}, {"b", fmt_row(b.row(r))}});
  }
  res["first_mismatches"] = examples;

  if (!graph.empty() && !bad_rows.empty()) {
    // Trigger properties of the mismatching rows: a visible self-loop at u or v, a ts tie with
    // the previous edge, and (with --tw) an edge at u or v at exactly t - tw.
    blz::BlzgView g(graph);
    if (first_eid + rows > g.m()) throw std::runtime_error("rows past the graph's m");
    std::vector<std::uint64_t> first_loop(g.n(), UINT64_MAX);
    std::vector<std::vector<std::uint32_t>> inc(tw > 0 ? g.n() : 0);
    for (std::uint64_t e = 0; e < g.m(); ++e) {
      if (g.src[e] == g.dst[e]) first_loop[g.src[e]] = std::min<std::uint64_t>(first_loop[g.src[e]], e);
      if (tw > 0) {
        inc[g.src[e]].push_back(std::uint32_t(e));
        if (g.dst[e] != g.src[e]) inc[g.dst[e]].push_back(std::uint32_t(e));
      }
    }
    std::uint64_t loop = 0, tie = 0, boundary = 0;
    for (auto r : bad_rows) {
      const std::uint64_t e = first_eid + r;
      const auto u = g.src[e], v = g.dst[e];
      if (first_loop[u] <= e || first_loop[v] <= e) ++loop;
      if (e > 0 && g.ts[e] == g.ts[e - 1]) ++tie;
      if (tw > 0) {
        const std::int64_t edge_ts = std::int64_t(g.ts[e]) - tw;
        auto at_boundary = [&](std::uint32_t x) {
          return std::any_of(inc[x].begin(), inc[x].end(),
                             [&](std::uint32_t f) { return f <= e && g.ts[f] == edge_ts; });
        };
        if (at_boundary(u) || at_boundary(v)) ++boundary;
      }
    }
    res["mismatch_properties"] = {{"self_loop_endpoint", loop}, {"tie_with_previous", tie}};
    if (tw > 0) res["mismatch_properties"]["window_boundary_edge"] = boundary;
  }

  std::printf("%s vs %s: %llu x %llu, %llu mismatching rows\n", a_path.c_str(), b_path.c_str(),
              (unsigned long long)rows, (unsigned long long)cols,
              (unsigned long long)bad_rows.size());
  if (!bad_rows.empty()) std::printf("%s\n", res.dump(1).c_str());
  if (!report.empty()) {
    nlohmann::json doc = nlohmann::json::object();
    if (std::filesystem::exists(report)) doc = nlohmann::json::parse(std::ifstream(report));
    doc[nlohmann::json::json_pointer("/" + name)] = res;
    ensure_parent(report);
    std::ofstream(report) << doc.dump(1) << '\n';
  }
  return bad_rows.empty() ? 0 : 2;
}

// Fuzz graphs (SPEC §13.3) plus their oracle features, for tools/gfp/fuzz_vs_gfp.py. Writes
// graphs.npy (int64[total, 6]: seed, eid, src, dst, ts, window) and <block>.npy (uint32 rows
// aligned with graphs.npy) for every v1 block, with the graph's window for every block.
int fuzz(const std::string& seeds, const std::string& bins_opt, int k_max, const std::string& out) {
  const auto [s0, s1] = parse_range(seeds);
  const auto bins = parse_bins(bins_opt);
  std::vector<std::int64_t> rows;
  std::map<blz::Block, std::vector<std::uint32_t>> feats;
  for (std::uint64_t seed = s0; seed < s1; ++seed) {
    const blz::FuzzGraph fg = blz::fuzz_graph(seed);
    const std::uint64_t m = fg.src.size();
    for (std::uint64_t e = 0; e < m; ++e)
      rows.insert(rows.end(), {std::int64_t(seed), std::int64_t(e), fg.src[e], fg.dst[e],
                               fg.ts[e], fg.window});
    const blz::Oracle oracle({fg.src, fg.dst, fg.ts, fg.n});
    for (blz::Block b : blz::kAllBlocks) {
      auto& f = feats[b];
      const std::size_t at = f.size();
      f.resize(at + m * bins.size(), 0);
      oracle.mine(b, {fg.window, bins}, k_max, 0, m, f.data() + at);
    }
  }
  std::filesystem::create_directories(out);
  const std::uint64_t total = rows.size() / 6;
  blz::write_npy_i64(out + "/graphs.npy", rows, total, 6);
  for (auto& [b, f] : feats)
    blz::write_npy_u32(out + "/" + std::string(blz::block_name(b)) + ".npy", f, total, bins.size());
  std::fprintf(stderr, "seeds %s: %llu edges -> %s\n", seeds.c_str(), (unsigned long long)total,
               out.c_str());
  return 0;
}

// Table 2 feature sets (SPEC §10.1): src_id and dst_id, then these blocks in the order the sets
// add them, so each set's columns are a prefix of the next set's.
std::vector<blz::Block> set_blocks(const std::string& set) {
  using B = blz::Block;
  const std::vector<std::pair<std::string, std::vector<B>>> sets = {
      {"xgb_only", {}},
      {"fan", {B::fan_in, B::fan_out}},
      {"fan_deg", {B::fan_in, B::fan_out, B::deg_in, B::deg_out}},
      {"fan_deg_cyc", {B::fan_in, B::fan_out, B::deg_in, B::deg_out, B::tcyc}},
      {"all", {B::fan_in, B::fan_out, B::deg_in, B::deg_out, B::tcyc, B::sg}}};
  for (const auto& [name, blocks] : sets)
    if (name == set) return blocks;
  throw std::runtime_error("unknown feature set '" + set + "'");
}

// Trains XGBoost on the train split and evaluates on the test split (SPEC §10). Merges the
// result into `out` under `name`.
int train(const std::string& graph, const std::string& features, const std::string& set,
          const std::string& xgb_path, const std::string& out, const std::string& name,
          int threads) {
  const std::vector<blz::Block> blocks = set_blocks(set);
  std::ifstream xgb_in(xgb_path);
  if (!xgb_in) throw std::runtime_error("cannot open " + xgb_path);
  std::stringstream xgb_text;
  xgb_text << xgb_in.rdbuf();
  const blz::XgbConfig cfg = blz::parse_xgb_config(xgb_text.str(), xgb_path);
  const std::string xgb_str = xgb_text.str();

  blz::BlzgView g(graph);
  const std::uint64_t m = g.m(), ntrain = blz::train_size(m);
  std::vector<std::string> columns = {"src_id", "dst_id"};
  std::vector<std::unique_ptr<blz::NpyU32View>> views;
  nlohmann::json feat = nullptr;
  if (!blocks.empty()) {
    const std::string meta_path = features + "/features.json";
    std::ifstream meta_in(meta_path);
    if (!meta_in) throw std::runtime_error("cannot open " + meta_path);
    const auto meta = nlohmann::json::parse(meta_in);
    const auto& run = meta.at("run");
    if (run.at("csv_sha256") != blz::to_hex(g.csv_sha256()))
      throw std::runtime_error(meta_path + ": mined from a different CSV than " + graph);
    if (run.at("triggers").at("first") != 0 || run.at("triggers").at("last") != m)
      throw std::runtime_error(meta_path + ": features must cover every eid (--triggers all)");
    for (blz::Block b : blocks) {
      const std::string bname(blz::block_name(b));
      if (!meta.at("blocks").contains(bname))
        throw std::runtime_error(meta_path + ": no block '" + bname + "'");
      const auto& j = meta["blocks"][bname];
      auto v = std::make_unique<blz::NpyU32View>(features + "/" + j.at("file").get<std::string>());
      const auto names = j.at("columns").get<std::vector<std::string>>();
      if (v->rows() != m || v->cols() != names.size())
        throw std::runtime_error(bname + ".npy: shape does not match the graph and features.json");
      columns.insert(columns.end(), names.begin(), names.end());
      views.push_back(std::move(v));
    }
    feat = {{"dir", features}, {"run", run}, {"git", meta.value("git", nlohmann::json())},
            {"config", meta.value("config", "")}};
  }

  const std::uint64_t ncols = columns.size();
  auto rows_from = [&](std::uint64_t e0, std::uint64_t rows) {
    blz::RowSource s;
    s.rows = rows;
    s.cols = ncols;
    s.fill = [&, e0](std::uint64_t first, std::uint64_t count, float* x, float* y) {
#pragma omp parallel for schedule(static)
      for (std::int64_t r = 0; r < std::int64_t(count); ++r) {
        const std::uint64_t e = e0 + first + std::uint64_t(r);
        float* row = x + std::uint64_t(r) * ncols;
        std::uint64_t c = 0;
        row[c++] = float(g.src[e]);  // node ids < 2^24, exact in float
        row[c++] = float(g.dst[e]);
        for (const auto& v : views)
          for (std::uint32_t f : v->row(e)) row[c++] = float(f);
        y[r] = float(g.label[e]);
      }
    };
    return s;
  };
  if (threads > 0) omp_set_num_threads(threads);
  const int nthread = omp_get_max_threads();

  std::fprintf(stderr, "train %s: %llu columns, %llu train rows, %llu test rows, %d threads\n",
               set.c_str(), (unsigned long long)ncols, (unsigned long long)ntrain,
               (unsigned long long)(m - ntrain), nthread);
  blz::XgbTimings t;
  const std::vector<float> score =
      blz::xgb_train_predict(rows_from(0, ntrain), rows_from(ntrain, m - ntrain), cfg, nthread, &t);
  const auto test_labels = g.label.subspan(ntrain);
  const blz::Metrics mt = blz::compute_metrics(score, test_labels, cfg.threshold);
  std::uint64_t train_pos = 0, test_pos = 0;
  for (std::uint64_t e = 0; e < m; ++e) (e < ntrain ? train_pos : test_pos) += g.label[e] != 0;

  rusage ru{};
  getrusage(RUSAGE_SELF, &ru);
  nlohmann::json params = nlohmann::json::object();
  for (const auto& [k, v] : cfg.params) params[k] = v;
  nlohmann::json res = {
      {"set", set},
      {"columns", columns},
      {"graph", graph},
      {"csv_sha256", blz::to_hex(g.csv_sha256())},
      {"features", feat},
      {"xgb", {{"config", xgb_path},
               {"config_sha256", blz::to_hex(blz::sha256(xgb_str.data(), xgb_str.size()))},
               {"params", params},
               {"num_boost_round", cfg.num_boost_round},
               {"threshold", cfg.threshold},
               {"batch_rows", cfg.batch_rows},
               {"nthread", nthread},
               {"version", blz::xgb_version()}}},
      {"split", {{"train_rows", ntrain}, {"train_positives", train_pos},
                 {"test_rows", m - ntrain}, {"test_positives", test_pos}}},
      {"metrics", {{"f1_x100", 100 * mt.f1},
                   {"precision", mt.precision},
                   {"recall", mt.recall},
                   {"pr_auc", mt.pr_auc},
                   {"confusion", {{"tp", mt.cm.tp}, {"fn", mt.cm.fn},
                                  {"fp", mt.cm.fp}, {"tn", mt.cm.tn}}}}},
      {"seconds", {{"train_matrix", t.train_matrix_s}, {"train", t.train_s},
                   {"test_matrix", t.test_matrix_s}, {"predict", t.predict_s}}},
      {"peak_rss_mb", double(ru.ru_maxrss) / 1024.0},
      {"git", git_info()},
      {"compiler", __VERSION__}};
  std::fprintf(stderr, "F1 %.1f  precision %.3f  recall %.3f  PR-AUC %.3f  TP %llu FP %llu FN %llu\n",
               100 * mt.f1, mt.precision, mt.recall, mt.pr_auc, (unsigned long long)mt.cm.tp,
               (unsigned long long)mt.cm.fp, (unsigned long long)mt.cm.fn);

  nlohmann::json doc = nlohmann::json::object();
  if (std::filesystem::exists(out)) doc = nlohmann::json::parse(std::ifstream(out));
  doc[nlohmann::json::json_pointer("/" + name)] = res;
  ensure_parent(out);
  std::ofstream(out) << doc.dump(1) << '\n';
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  CLI::App app{"blz — BlazingAML runtime"};
  app.require_subcommand(1);

  std::string csv, out, graph;
  auto* prep = app.add_subcommand("prepare", "CSV -> .blzg, asserting the HI-Small invariants");
  prep->add_option("--csv", csv)->required();
  prep->add_option("--out", out)->required();

  auto* exp = app.add_subcommand("export-edges", ".blzg -> edges.npy (int64[m,4]) for tools/gfp");
  exp->add_option("--graph", graph)->required();
  exp->add_option("--out", out)->required();

  auto* st = app.add_subcommand("stats", "E0 dataset profile (without the GFP part)");
  st->add_option("--graph", graph)->required();
  st->add_option("--out", out)->required();

  std::string config, backend = "oracle", blocks, triggers = "all";
  int threads = 0;
  auto* mn = app.add_subcommand("mine", "Mine feature blocks -> <out>/<block>.npy + features.json");
  mn->add_option("--graph", graph)->required();
  mn->add_option("--config", config)->required();
  mn->add_option("--backend", backend)->check(CLI::IsMember({"oracle", "cpu", "cuda"}));
  mn->add_option("--threads", threads);
  mn->add_option("--blocks", blocks, "comma-separated; default: every block in the config");
  mn->add_option("--triggers", triggers, "all | train | test | range:a:b");
  mn->add_option("--out", out)->required();

  std::string a_npy, b_npy, report, name;
  std::int64_t tw = 0;
  std::uint64_t first_eid = 0;
  auto* cmp = app.add_subcommand("compare", "Diff two uint32 feature blocks (exit 2 on mismatch)");
  cmp->add_option("a", a_npy)->required();
  cmp->add_option("b", b_npy)->required();
  cmp->add_option("--graph", graph, "for the mismatch-by-trigger-property histogram");
  cmp->add_option("--tw", tw, "window, for the window-boundary property");
  cmp->add_option("--first-eid", first_eid, "eid of row 0");
  cmp->add_option("--report", report, "merge the summary into this JSON file");
  cmp->add_option("--name", name, "key (JSON pointer path) in --report")->needs("--report");

  std::string seeds, bins = "2..33";
  int k_max = 64;
  auto* fz = app.add_subcommand("fuzz", "Fuzz graphs + oracle features (SPEC §13.3)");
  fz->add_option("--seeds", seeds, "a:b")->required();
  fz->add_option("--bins", bins);
  fz->add_option("--k-max", k_max, "tcyc cap; the default exceeds every fuzz graph's n");
  fz->add_option("--out", out)->required();

  std::string features, set, xgb = "configs/xgb_default.yaml";
  auto* tr = app.add_subcommand("train", "XGBoost on the train split, metrics on the test split");
  tr->add_option("--graph", graph)->required();
  tr->add_option("--features", features, "blz mine output dir (not read for xgb_only)")->required();
  tr->add_option("--set", set)->required()->check(
      CLI::IsMember({"xgb_only", "fan", "fan_deg", "fan_deg_cyc", "all"}));
  tr->add_option("--xgb", xgb);
  tr->add_option("--threads", threads);
  tr->add_option("--out", out, "merge the result into this JSON file")->required();
  tr->add_option("--name", name, "key (JSON pointer path) in --out; default: the set");

  CLI11_PARSE(app, argc, argv);
  try {
    if (*prep) return prepare(csv, out);
    if (*exp) return export_edges(graph, out);
    if (*st) return stats(graph, out);
    if (*mn) return mine(graph, config, backend, threads, blocks, triggers, out);
    if (*cmp) {
      if (!report.empty() && name.empty()) throw std::runtime_error("--report needs --name");
      return compare(a_npy, b_npy, graph, tw, first_eid, report, name);
    }
    if (*fz) return fuzz(seeds, bins, k_max, out);
    if (*tr) return train(graph, features, set, xgb, out, name.empty() ? set : name, threads);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 1;
}

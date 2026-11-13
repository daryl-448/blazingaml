// blz — BlazingAML runtime CLI (SPEC §14). Commands so far: prepare, export-edges, stats (P0);
// mine (oracle backend), compare, fuzz (P1); train (P2); mine --backend cpu, bench, fuzz-diff (P4);
// the cuda backend of mine, bench and fuzz-diff (P5); profiles, manifests, trigger samples,
// metamorphic and gt-join (SPEC v2 P7); coverage (SPEC v2 P9).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
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
#include "graph/gt_join.hpp"
#include "graph/index.hpp"
#include "graph/prepare.hpp"
#include "graph/stats.hpp"
#include "io/blzg.hpp"
#include "io/csv.hpp"
#include "ml/metrics.hpp"
#if BLZ_WITH_XGBOOST
#include "ml/xgb.hpp"
#endif
#include "oracle/oracle.hpp"
#include "runtime/coverage.hpp"
#include "runtime/cpu_backend.hpp"
#include "runtime/cuda_backend.hpp"
#include "runtime/features.hpp"
#include "runtime/fuzz_diff.hpp"
#include "runtime/metamorphic.hpp"
#include "runtime/profile.hpp"
#include "runtime/triggers.hpp"

namespace {

using Clock = std::chrono::steady_clock;
double secs(Clock::time_point a) { return std::chrono::duration<double>(Clock::now() - a).count(); }

void ensure_parent(const std::string& path) {
  auto parent = std::filesystem::path(path).parent_path();
  if (!parent.empty()) std::filesystem::create_directories(parent);
}

std::string read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
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
  const blz::StreamCounts c = blz::count_stream(t);
  std::fprintf(stderr,
               "SPEC §3.5 and v2 §3.3 invariants: pass (b_fv %llu: fit %llu positives, validation %llu, test %llu; "
               "%llu concatenated node keys; file_row reproduces the stable sort)\n",
               (unsigned long long)c.fit_val_boundary, (unsigned long long)c.fit_positives,
               (unsigned long long)c.val_positives, (unsigned long long)c.test_positives,
               (unsigned long long)c.concatenated_keys);

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
  if (sha.empty()) {
    // A source tree shipped without .git (bench/a30.sh) names its commit in the environment.
    const char* env_sha = std::getenv("BLZ_GIT_SHA");
    const char* env_dirty = std::getenv("BLZ_GIT_DIRTY");
    if (!env_sha || !*env_sha) return nullptr;
    return {{"sha", env_sha}, {"dirty", env_dirty && std::string(env_dirty) == "1"}, {"from", "BLZ_GIT_SHA"}};
  }
  return {{"sha", sha}, {"dirty", !run(git + "status --porcelain 2>/dev/null").empty()}};
}

nlohmann::json block_params_json(const blz::BlockParams& p, blz::Block b) {
  return {{"window", p.window},
          {"lower", blz::closed_lower_bound(b) ? "closed" : "open"},
          {"bins", p.bins}};
}

// --blocks: a comma-separated list (in that order), or every block of the profile.
std::vector<const blz::ProfileBlock*> select_blocks(const blz::Profile& prof, const std::string& blocks_opt) {
  std::vector<const blz::ProfileBlock*> blocks;
  if (blocks_opt.empty()) {
    for (const blz::ProfileBlock& pb : prof.blocks) blocks.push_back(&pb);
    return blocks;
  }
  for (std::size_t p = 0; p <= blocks_opt.size();) {
    auto q = std::min(blocks_opt.find(',', p), blocks_opt.size());
    const std::string name = blocks_opt.substr(p, q - p);
    const auto b = blz::parse_block(name);
    if (!b) throw std::runtime_error("unknown block '" + name + "'");
    const blz::ProfileBlock* pb = prof.find(*b);
    if (!pb) throw std::runtime_error("block '" + name + "' is not in " + prof.file);
    blocks.push_back(pb);
    p = q + 1;
  }
  return blocks;
}

std::string file_sha256(const std::string& path) {
  const blz::MappedFile f(path);
  return blz::to_hex(blz::sha256(f.data(), f.size()));
}

std::uint64_t fnv1a(std::span<const std::uint32_t> data) {
  std::uint64_t h = 1469598103934665603ULL;
  const auto* b = reinterpret_cast<const unsigned char*>(data.data());
  for (std::size_t i = 0; i < data.size_bytes(); ++i) h = (h ^ b[i]) * 1099511628211ULL;
  return h;
}

double peak_rss_mb() {
  rusage ru{};
  getrusage(RUSAGE_SELF, &ru);
  return double(ru.ru_maxrss) / 1024.0;
}

// The profile part of a manifest (SPEC v2 §3.2).
nlohmann::json profile_json(const blz::Profile& prof) {
  return {{"name", prof.name},
          {"file", prof.file},
          {"format", prof.format == 1 ? "v1 feature config" : "profile"},
          {"id", prof.id()},
          {"causal", prof.causal()},
          {"row_sees", prof.row_sees()},
          {"resolved", prof.resolved()}};
}

// Every pattern the profile lists, with its spec and the spec's SHA-256.
nlohmann::json patterns_json(const blz::Profile& prof) {
  nlohmann::json j = nlohmann::json::object();
  for (const blz::ProfileBlock& pb : prof.blocks)
    for (const blz::PatternInfo* pi : pb.patterns)
      j[pi->name] = {{"block", pi->block}, {"spec", pi->file}, {"spec_sha256", pi->spec_sha256},
                     {"visibility_pin", *pi->visibility ? nlohmann::json(pi->visibility) : nlohmann::json(nullptr)}};
  return j;
}

// The CUDA device and build, for results files (SPEC §12.1).
nlohmann::json cuda_json() {
  const blz::CudaDevice d = blz::cuda_device();
  return {{"device", d.name},
          {"compute_capability", std::to_string(d.cc_major) + "." + std::to_string(d.cc_minor)},
          {"sm_count", d.sm_count},
          {"memory_mb", {{"total", d.total_bytes >> 20}, {"free", d.free_bytes >> 20}}},
          {"driver_version", d.driver_version},
          {"runtime_version", d.runtime_version},
          {"build", BLZ_CUDA_BUILD}};
}

// "0,16,32,all" -> light/heavy thresholds; "all" (every trigger light) is kAllLight.
std::vector<std::uint32_t> parse_thresholds(const std::string& s) {
  std::vector<std::uint32_t> out;
  for (std::size_t p = 0; p <= s.size();) {
    auto q = std::min(s.find(',', p), s.size());
    const std::string v = s.substr(p, q - p);
    out.push_back(v == "all" ? blz::kAllLight : std::uint32_t(std::stoul(v)));
    p = q + 1;
  }
  return out;
}
std::string threshold_name(std::uint32_t h) { return h == blz::kAllLight ? "all" : std::to_string(h); }

int mine(const std::string& graph, const std::string& profile_path, const std::string& backend, int threads,
         const std::string& schedule_opt, std::uint32_t heavy, const std::string& blocks_opt,
         const std::string& triggers, std::uint64_t edges, const std::string& out) {
  const blz::Backend be = blz::parse_backend(backend);
  const bool cpu = be == blz::Backend::cpu, gpu = be == blz::Backend::cuda;
  if (gpu && !blz::cuda_built()) blz::cuda_device();  // throws: no CUDA backend in this build
  const blz::Schedule schedule = blz::parse_schedule(schedule_opt);
  const blz::Profile prof = blz::load_profile(profile_path);  // E403
  blz::check_profile_backend(prof, be);                       // E404 (v1 configs: E402, K_MAX)
  const std::vector<const blz::ProfileBlock*> blocks = select_blocks(prof, blocks_opt);

  auto t0 = Clock::now();
  blz::BlzgView g(graph);
  // --edges k: the graph is the stream prefix [0, k); later edges do not exist for this run, under
  // either visibility (SPEC v2 §4.4 R5). The named splits are defined on the whole stream.
  if (edges > g.m()) throw std::runtime_error("--edges " + std::to_string(edges) + " > m = " + std::to_string(g.m()));
  if (edges && !(triggers == "all" || triggers.rfind("range:", 0) == 0 || triggers.rfind("sample:", 0) == 0))
    throw std::runtime_error("--edges: --triggers must be all, range:a:b or sample:<pct>:<seed>");
  const std::uint64_t m = edges ? edges : g.m();
  const auto src = g.src.first(m), dst = g.dst.first(m);
  const auto ts = g.ts.first(m);
  const blz::TriggerSel sel = blz::select_triggers(triggers, ts);
  if (gpu && sel.sampled)
    throw std::runtime_error("--triggers sample: runs on the oracle and cpu backends (the preflight of SPEC v2 §2.5 "
                             "uses cpu); the CUDA kernels take an eid range");
  if (threads > 0) omp_set_num_threads(threads);
  std::optional<blz::Oracle> oracle;
  std::optional<blz::GraphIndex> index;
  if (cpu || gpu) index.emplace(src, dst, ts, g.n());
  else oracle.emplace(blz::EdgeStream{src, dst, ts, g.n()}, prof.visibility);
  const double build_s = secs(t0);
  std::fprintf(stderr, "%s index built in %.1f s (m=%llu, n=%llu); profile %s (%s); triggers %s: %llu rows\n",
               backend.c_str(), build_s, (unsigned long long)m, (unsigned long long)g.n(), prof.name.c_str(),
               prof.id().c_str(), sel.spec.c_str(), (unsigned long long)sel.rows());
  std::optional<blz::DeviceGraph> dgraph;
  std::optional<blz::CudaMiner> miner;
  if (gpu) {
    dgraph.emplace(index->view());
    miner.emplace();
    std::fprintf(stderr, "graph uploaded in %.3f s (%.0f MB)\n", dgraph->upload_seconds(), double(dgraph->bytes()) / 1e6);
  }

  std::filesystem::create_directories(out);
  const std::string meta_path = out + "/features.json";
  const std::string blzg_sha = file_sha256(graph), csv_sha = blz::to_hex(g.csv_sha256());
  // The run a block belongs to: blocks of an earlier run with the same key are kept, so blocks
  // can be mined one invocation at a time into one directory.
  const nlohmann::json run_key = {{"backend", backend},     {"csv_sha256", csv_sha},
                                  {"blzg_sha256", blzg_sha}, {"profile_id", prof.id()},
                                  {"k_max", prof.k_max},     {"triggers", sel.json()},
                                  {"edges", m}};
  nlohmann::json meta = nlohmann::json::object();
  if (std::filesystem::exists(meta_path)) {
    auto old = nlohmann::json::parse(std::ifstream(meta_path));
    if (old.value("run", nlohmann::json()) == run_key) meta = old;  // same run: keep its blocks
  }
  // The manifest (SPEC v2 §3.2).
  meta["manifest_version"] = 2;
  meta["run"] = run_key;
  meta["profile"] = profile_json(prof);
  meta["patterns"] = patterns_json(prof);
  nlohmann::json jbe = {{"name", backend}, {"threads", omp_get_max_threads()},
                        {"disabled_optimizations", nlohmann::json::array()}};  // blzc --disable: SPEC v2 §7.2 (P11)
  if (cpu) jbe["schedule"] = blz::schedule_name(schedule);
  if (gpu) jbe["heavy_threshold"] = threshold_name(heavy);
  meta["backend"] = jbe;
  meta["triggers"] = sel.json();
  meta["graph"] = graph;
  meta["edges"] = m;  // the graph is eids [0, edges) (--edges; m otherwise)
  meta["blzg_sha256"] = blzg_sha;
  meta["csv_sha256"] = csv_sha;
  meta["config"] = profile_path;
  meta["git"] = git_info();
  meta["threads"] = omp_get_max_threads();
  if (cpu) meta["schedule"] = blz::schedule_name(schedule);
  if (gpu)
    meta["cuda"] = {{"env", cuda_json()},
                    {"heavy_threshold", threshold_name(heavy)},
                    {"graph_upload_seconds", dgraph->upload_seconds()},
                    {"device_graph_bytes", dgraph->bytes()}};
  meta["index_build_seconds"] = build_s;
  if (!meta.contains("blocks")) meta["blocks"] = nlohmann::json::object();
  if (sel.sampled) {
    blz::write_npy_u32(out + "/triggers.npy", sel.eids, sel.eids.size(), 1);
    meta["triggers"]["file"] = "triggers.npy";
  }

  const std::uint64_t rows = sel.rows();
  std::uint64_t total_saturated = 0;
  for (const blz::ProfileBlock* pb : blocks) {
    const blz::Block b = pb->block;
    const blz::BlockParams& p = pb->params;
    const std::uint64_t cells = rows * p.bins.size();
    std::vector<std::uint32_t> feats;
    std::optional<blz::PinnedBuffer> pinned;  // cuda: the rows, copied straight from the device
    if (gpu) pinned.emplace(std::max<std::uint64_t>(cells, 1));
    else feats.assign(cells, 0);
    auto t1 = Clock::now();
    std::uint64_t sat = 0;
    blz::CpuBlockTiming tm;
    blz::CudaBlockTiming ctm;
    if (cpu) {
      tm = blz::mine_cpu(index->view(), pb->patterns, p, prof.visibility, sel, schedule, feats.data());
      sat = tm.saturated;
    } else if (gpu) {
      ctm = miner->mine(*dgraph, pb->patterns, p, prof.visibility, sel.first, sel.last, heavy, pinned->data());
      sat = ctm.saturated;
    } else if (sel.sampled) {
      sat = oracle->mine(b, p, prof.k_max, sel.eids, feats.data());
    } else {
      sat = oracle->mine(b, p, prof.k_max, sel.first, sel.last, feats.data());
    }
    const double mine_s = secs(t1);
    const std::string file = std::string(blz::block_name(b)) + ".npy";
    const std::span<const std::uint32_t> data =
        gpu ? std::span<const std::uint32_t>(pinned->data(), cells) : std::span<const std::uint32_t>(feats);
    blz::write_npy_u32(out + "/" + file, data, rows, p.bins.size());
    auto j = block_params_json(p, b);
    j["file"] = file;
    j["columns"] = blz::column_names(b, p.bins);
    j["mine_seconds"] = mine_s;
    if (sel.sampled && rows > 0) j["extrapolated_mine_seconds"] = mine_s * double(m) / double(rows);
    if (cpu) {
      j["schedule_seconds"] = tm.schedule_seconds;
      j["loop_seconds"] = tm.loop_seconds;
    }
    if (gpu)
      j["cuda"] = {{"kernel_seconds", ctm.kernel_seconds}, {"bucket_seconds", ctm.bucket_seconds},
                   {"light_seconds", ctm.light_seconds},   {"heavy_seconds", ctm.heavy_seconds},
                   {"d2h_seconds", ctm.d2h_seconds},       {"light_triggers", ctm.light},
                   {"heavy_triggers", ctm.heavy}};
    nlohmann::json pats = nlohmann::json::array();
    for (const blz::PatternInfo* pi : pb->patterns) pats.push_back(pi->name);
    j["patterns"] = pats;
    j["output_fnv1a"] = fnv1a(data);
    j["saturated"] = sat;
    meta["blocks"][std::string(blz::block_name(b))] = j;
    meta["peak_rss_mb"] = peak_rss_mb();
    total_saturated += sat;
    std::fprintf(stderr, "%-8s %8.1f s  saturated=%llu  fnv1a=%llu\n", std::string(blz::block_name(b)).c_str(), mine_s,
                 (unsigned long long)sat, (unsigned long long)j["output_fnv1a"].get<std::uint64_t>());
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

#if BLZ_WITH_XGBOOST
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
    if (run.at("triggers").value("sampled", false) || run.at("triggers").at("first") != 0 ||
        run.at("triggers").at("last") != m)
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
    // SPEC v2 §3.2: every results JSON that consumes features records the profile id.
    const nlohmann::json prof = meta.value("profile", nlohmann::json::object());
    feat = {{"dir", features}, {"run", run}, {"git", meta.value("git", nlohmann::json())},
            {"config", meta.value("config", "")},
            {"profile", prof.value("name", nlohmann::json())},
            {"profile_id", prof.value("id", nlohmann::json())}};
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
#else
int train(const std::string&, const std::string&, const std::string&, const std::string&, const std::string&,
          const std::string&, int) {
  throw std::runtime_error("this build has no XGBoost (BLZ_WITH_XGBOOST=OFF); use the rel or rel-cuda preset");
}
#endif

// A shell command's stdout ("" if it cannot run).
std::string run_command(const std::string& cmd) {
  std::string out;
  if (FILE* f = popen(cmd.c_str(), "r")) {
    char buf[512];
    while (std::fgets(buf, sizeof buf, f)) out += buf;
    pclose(f);
  }
  return out;
}

// The machine and build a timing was taken on (SPEC §12.1).
nlohmann::json environment(const std::string& config) {
  auto run = run_command;
  auto env = [](const char* k) -> nlohmann::json {
    const char* v = std::getenv(k);
    return v ? nlohmann::json(v) : nlohmann::json(nullptr);
  };
  nlohmann::json mem = nlohmann::json::object();
  {
    std::ifstream in("/proc/meminfo");
    std::string key;
    std::uint64_t kb = 0;
    std::string unit;
    while (in >> key >> kb) {
      std::getline(in, unit);
      if (key == "MemTotal:" || key == "MemAvailable:" || key == "MemFree:")
        mem[key.substr(0, key.size() - 1) + "_mb"] = kb / 1024;
    }
  }
  const std::string cfg_text = read_file(config);
  return {{"git", git_info()},
          {"compiler", __VERSION__},
          {"build", {{"type", BLZ_BUILD_TYPE}, {"cxx_flags", BLZ_CXX_FLAGS}, {"native", BLZ_NATIVE_FLAG}}},
          {"lscpu", run("lscpu 2>/dev/null")},
          {"memory", mem},
          {"omp", {{"OMP_PROC_BIND", env("OMP_PROC_BIND")}, {"OMP_PLACES", env("OMP_PLACES")},
                   {"max_threads", omp_get_max_threads()}}},
          {"config", config},
          {"config_sha256", blz::to_hex(blz::sha256(cfg_text.data(), cfg_text.size()))}};
}

std::vector<int> parse_int_list(const std::string& s) {
  std::vector<int> out;
  for (std::size_t p = 0; p <= s.size();) {
    auto q = std::min(s.find(',', p), s.size());
    out.push_back(std::stoi(s.substr(p, q - p)));
    if (out.back() <= 0) throw std::runtime_error("expected positive integers, got '" + s + "'");
    p = q + 1;
  }
  return out;
}

// E4 (SPEC §12): per block and thread count, 1 warm-up and `reps` timed runs of the CPU backend
// over every trigger; each run's output must equal the first (determinism, SPEC §8).
int bench(const std::string& graph, const std::string& config, const std::string& backend,
          const std::string& threads_opt, int reps, const std::string& schedules_opt,
          const std::string& blocks_opt, const std::string& out, const std::string& name) {
  if (backend != "cpu") throw std::runtime_error("bench: backend '" + backend + "' is not the cpu backend");
  if (reps < 1) throw std::runtime_error("--reps must be >= 1");
  const blz::Profile prof = blz::load_profile(config);
  blz::check_profile_backend(prof, blz::Backend::cpu);
  const std::vector<const blz::ProfileBlock*> blocks = select_blocks(prof, blocks_opt);
  const std::vector<int> threads = parse_int_list(threads_opt);
  std::vector<blz::Schedule> schedules;
  for (std::size_t p = 0; p <= schedules_opt.size();) {
    auto q = std::min(schedules_opt.find(',', p), schedules_opt.size());
    schedules.push_back(blz::parse_schedule(schedules_opt.substr(p, q - p)));
    p = q + 1;
  }
  nlohmann::json res = {{"env", environment(config)}, {"backend", backend}, {"graph", graph},
                        {"reps", reps}, {"warmup", 1}, {"threads", threads},
                        {"profile", {{"name", prof.name}, {"id", prof.id()}}}};

  blz::BlzgView g(graph);
  auto t0 = Clock::now();
  const blz::GraphIndex index(g.src, g.dst, g.ts, g.n());
  res["index_build_seconds"] = secs(t0);
  res["index_bytes"] = index.bytes();
  res["csv_sha256"] = blz::to_hex(g.csv_sha256());
  std::fprintf(stderr, "index built in %.2f s (%.0f MB)\n", secs(t0), double(index.bytes()) / 1e6);

  const std::uint64_t m = g.m();
  blz::TriggerSel all;
  all.spec = "all";
  all.last = m;
  nlohmann::json runs = nlohmann::json::object();
  for (const blz::ProfileBlock* pb : blocks) {
    const blz::BlockParams& p = pb->params;
    const std::string bname(blz::block_name(pb->block));
    std::vector<std::uint32_t> feats(m * p.bins.size()), ref;
    nlohmann::json jb = {{"patterns", nlohmann::json::array()}};
    for (const blz::PatternInfo* pi : pb->patterns) jb["patterns"].push_back(pi->name);
    for (blz::Schedule sched : schedules) {
      nlohmann::json js = nlohmann::json::object();
      for (int nt : threads) {
        omp_set_num_threads(nt);
        std::vector<double> total, sched_s, loop_s;
        for (int r = 0; r <= reps; ++r) {  // r == 0 is the warm-up
          std::fill(feats.begin(), feats.end(), 0u);
          const blz::CpuBlockTiming tm =
              blz::mine_cpu(index.view(), pb->patterns, p, prof.visibility, all, sched, feats.data());
          if (ref.empty()) {
            ref = feats;
            jb["output_fnv1a"] = fnv1a(ref);
            jb["saturated"] = tm.saturated;
          } else if (feats != ref) {
            throw std::runtime_error("bench: " + bname + " output differs at " + std::to_string(nt) +
                                     " threads, schedule " + blz::schedule_name(sched) + " (SPEC §8)");
          }
          if (r == 0) continue;
          total.push_back(tm.schedule_seconds + tm.loop_seconds);
          sched_s.push_back(tm.schedule_seconds);
          loop_s.push_back(tm.loop_seconds);
        }
        auto median = [](std::vector<double> v) {
          std::sort(v.begin(), v.end());
          return v.size() % 2 ? v[v.size() / 2] : (v[v.size() / 2 - 1] + v[v.size() / 2]) / 2;
        };
        js[std::to_string(nt)] = {{"median", median(total)},
                                  {"min", *std::min_element(total.begin(), total.end())},
                                  {"max", *std::max_element(total.begin(), total.end())},
                                  {"seconds", total},
                                  {"schedule_median", median(sched_s)},
                                  {"loop_median", median(loop_s)}};
        std::fprintf(stderr, "%-8s %-7s %2d threads: median %8.3f s (min %.3f, max %.3f)\n", bname.c_str(),
                     blz::schedule_name(sched), nt, median(total), js[std::to_string(nt)]["min"].get<double>(),
                     js[std::to_string(nt)]["max"].get<double>());
      }
      jb[blz::schedule_name(sched)] = js;
    }
    jb["window"] = p.window;
    jb["bins"] = p.bins;
    runs[bname] = jb;
  }
  res["blocks"] = runs;
  res["peak_rss_mb"] = peak_rss_mb();

  nlohmann::json doc = nlohmann::json::object();
  if (std::filesystem::exists(out)) doc = nlohmann::json::parse(std::ifstream(out));
  doc[nlohmann::json::json_pointer("/" + name)] = res;
  ensure_parent(out);
  std::ofstream(out) << doc.dump(1) << '\n';
  std::fprintf(stderr, "wrote %s (/%s)\n", out.c_str(), name.c_str());
  return 0;
}

nlohmann::json summary(const std::vector<double>& v) {
  std::vector<double> s = v;
  std::sort(s.begin(), s.end());
  const double med = s.size() % 2 ? s[s.size() / 2] : (s[s.size() / 2 - 1] + s[s.size() / 2]) / 2;
  return {{"median", med}, {"min", s.front()}, {"max", s.back()}, {"seconds", v}};
}

// E5 (SPEC §12.2): per block and light/heavy threshold, 1 warm-up and `reps` timed runs of the
// CUDA backend over every trigger. Every run's rows, at every threshold, must equal the block's
// first run (determinism, SPEC §8). The graph upload is timed separately, as `reps` fresh uploads
// after one warm-up upload, and T_BlazingAML(block) = median upload + kernels + D2H (SPEC §11.2).
int bench_cuda(const std::string& graph, const std::string& config, int reps, const std::string& heavy_opt,
               const std::string& blocks_opt, const std::string& out, const std::string& name) {
  if (reps < 1) throw std::runtime_error("--reps must be >= 1");
  const blz::Profile prof = blz::load_profile(config);
  blz::check_profile_backend(prof, blz::Backend::cuda);
  const std::vector<const blz::ProfileBlock*> blocks = select_blocks(prof, blocks_opt);
  const std::vector<std::uint32_t> thresholds = parse_thresholds(heavy_opt);
  nlohmann::json hs = nlohmann::json::array();
  for (std::uint32_t h : thresholds) hs.push_back(threshold_name(h));
  nlohmann::json env = environment(config);
  env["nvidia_smi"] = run_command("nvidia-smi -q 2>/dev/null");
  nlohmann::json res = {{"env", env}, {"cuda", cuda_json()}, {"backend", "cuda"},
                        {"graph", graph}, {"reps", reps}, {"warmup", 1}, {"heavy_thresholds", hs},
                        {"profile", {{"name", prof.name}, {"id", prof.id()}}}};

  blz::BlzgView g(graph);
  auto t0 = Clock::now();
  const blz::GraphIndex index(g.src, g.dst, g.ts, g.n());
  res["index_build_seconds"] = secs(t0);
  res["index_bytes"] = index.bytes();
  res["csv_sha256"] = blz::to_hex(g.csv_sha256());
  std::vector<double> uploads;
  for (int r = 0; r <= reps; ++r) {  // r == 0 is the warm-up
    const blz::DeviceGraph dg(index.view());
    if (r > 0) uploads.push_back(dg.upload_seconds());
    res["device_graph_bytes"] = dg.bytes();
  }
  res["graph_upload"] = summary(uploads);
  const double upload_med = res["graph_upload"]["median"].get<double>();
  std::fprintf(stderr, "index built in %.2f s; graph upload median %.3f s (%.0f MB)\n",
               res["index_build_seconds"].get<double>(), upload_med, res["device_graph_bytes"].get<double>() / 1e6);

  const blz::DeviceGraph dg(index.view());
  blz::CudaMiner miner;
  const std::uint64_t m = g.m();
  nlohmann::json runs = nlohmann::json::object();
  for (const blz::ProfileBlock* pb : blocks) {
    const blz::BlockParams& p = pb->params;
    const std::string bname(blz::block_name(pb->block));
    const std::uint64_t cells = m * p.bins.size();
    blz::PinnedBuffer buf(cells);
    std::vector<std::uint32_t> ref;
    nlohmann::json jb = {{"patterns", nlohmann::json::array()}, {"window", p.window}, {"bins", p.bins}};
    for (const blz::PatternInfo* pi : pb->patterns) jb["patterns"].push_back(pi->name);
    for (std::uint32_t h : thresholds) {
      std::vector<double> kernel, d2h, total, bucket, light, heavy, wall;
      blz::CudaBlockTiming tm;
      for (int r = 0; r <= reps; ++r) {  // r == 0 is the warm-up
        tm = miner.mine(dg, pb->patterns, p, prof.visibility, 0, m, h, buf.data());
        const std::span<const std::uint32_t> got(buf.data(), cells);
        if (ref.empty()) {
          ref.assign(got.begin(), got.end());
          jb["output_fnv1a"] = fnv1a(ref);
          jb["saturated"] = tm.saturated;
        } else if (!std::equal(got.begin(), got.end(), ref.begin())) {
          throw std::runtime_error("bench: " + bname + " output differs at heavy threshold " + threshold_name(h) +
                                   " (SPEC §8)");
        }
        if (r == 0) continue;
        kernel.push_back(tm.kernel_seconds);
        d2h.push_back(tm.d2h_seconds);
        total.push_back(upload_med + tm.kernel_seconds + tm.d2h_seconds);
        bucket.push_back(tm.bucket_seconds);
        light.push_back(tm.light_seconds);
        heavy.push_back(tm.heavy_seconds);
        wall.push_back(tm.wall_seconds);
      }
      nlohmann::json jh = {{"total", summary(total)},     {"kernel", summary(kernel)},
                           {"d2h", summary(d2h)},         {"bucket_median", summary(bucket)["median"]},
                           {"light_median", summary(light)["median"]},
                           {"heavy_median", summary(heavy)["median"]},
                           {"wall_median", summary(wall)["median"]},
                           {"light_triggers", tm.light},  {"heavy_triggers", tm.heavy}};
      std::fprintf(stderr, "%-8s H=%-4s kernels median %8.4f s (light %.4f, heavy %.4f, bucket %.4f), d2h %.4f s, "
                   "%llu heavy triggers\n",
                   bname.c_str(), threshold_name(h).c_str(), jh["kernel"]["median"].get<double>(),
                   jh["light_median"].get<double>(), jh["heavy_median"].get<double>(),
                   jh["bucket_median"].get<double>(), jh["d2h"]["median"].get<double>(),
                   (unsigned long long)tm.heavy);
      jb["by_threshold"][threshold_name(h)] = jh;
    }
    runs[bname] = jb;
  }
  res["blocks"] = runs;
  res["peak_rss_mb"] = peak_rss_mb();

  nlohmann::json doc = nlohmann::json::object();
  if (std::filesystem::exists(out)) doc = nlohmann::json::parse(std::ifstream(out));
  doc[nlohmann::json::json_pointer("/" + name)] = res;
  ensure_parent(out);
  std::ofstream(out) << doc.dump(1) << '\n';
  std::fprintf(stderr, "wrote %s (/%s)\n", out.c_str(), name.c_str());
  return 0;
}

// The JSON that names a profile in a results file.
nlohmann::json profile_ref(const blz::Profile& prof) {
  return {{"name", prof.name}, {"file", prof.file}, {"id", prof.id()}};
}

// The bins of fuzz-diff and metamorphic: --bins if given; else, with a profile, each block's own
// (an empty list); else 2..33, the v1 default.
std::vector<std::uint32_t> fuzz_bins(const std::string& bins_opt, bool have_profile) {
  if (!bins_opt.empty()) return parse_bins(bins_opt);
  return have_profile ? std::vector<std::uint32_t>{} : parse_bins("2..33");
}

void merge_report(const std::string& report, const std::string& name, const nlohmann::json& res) {
  if (report.empty()) return;
  nlohmann::json doc = nlohmann::json::object();
  if (std::filesystem::exists(report)) doc = nlohmann::json::parse(std::ifstream(report));
  doc[nlohmann::json::json_pointer("/" + name)] = res;
  ensure_parent(report);
  std::ofstream(report) << doc.dump(1) << '\n';
}

// AC1 on fuzz graphs (SPEC §13.3): a generated backend vs the oracle. Exit code 0 if equal, 2 if
// not. cuda runs every graph once per light/heavy threshold. With --profile (SPEC v2 §10), the
// profile's blocks, patterns and k_max; without, every registered pattern (v1).
int fuzz_diff(const std::string& seeds, const std::string& bins_opt, const std::string& backend,
              const std::string& heavy_opt, const std::string& profile_path, const std::string& report,
              const std::string& name) {
  const auto [s0, s1] = parse_range(seeds);
  const bool gpu = backend == "cuda";
  const blz::Profile prof = profile_path.empty() ? blz::registry_profile() : blz::load_profile(profile_path);
  blz::check_profile_backend(prof, blz::Backend::oracle);
  blz::check_profile_backend(prof, blz::parse_backend(backend));
  const auto bins = fuzz_bins(bins_opt, !profile_path.empty());
  const std::vector<std::uint32_t> thresholds = parse_thresholds(heavy_opt);
  auto t0 = Clock::now();
  const blz::FuzzDiff d =
      gpu ? blz::fuzz_diff_cuda(prof, s0, s1, bins, thresholds) : blz::fuzz_diff_cpu(prof, s0, s1, bins);
  nlohmann::json per_block = nlohmann::json::object();
  std::uint64_t bad = 0;
  for (const blz::ProfileBlock& pb : prof.blocks) {
    per_block[std::string(blz::block_name(pb.block))] = d.mismatched_rows[int(pb.block)];
    bad += d.mismatched_rows[int(pb.block)];
  }
  std::vector<std::uint64_t> first_bad(d.bad_seeds.begin(),
                                       d.bad_seeds.begin() + std::ptrdiff_t(std::min<std::size_t>(20, d.bad_seeds.size())));
  nlohmann::json res = {{"backend", backend},
                        {"seeds", seeds},
                        {"graphs", d.graphs},
                        {"edges", d.edges},
                        {"bins", bins.empty() ? nlohmann::json("profile") : nlohmann::json(bins)},
                        {"k_max", prof.k_max},
                        {"mismatched_rows_by_block", per_block},
                        {"mismatched_seeds", d.bad_seeds.size()},
                        {"first_mismatched_seeds", first_bad},
                        {"seconds", secs(t0)},
                        {"git", git_info()}};
  if (!profile_path.empty()) res["profile"] = profile_ref(prof);
  if (gpu) {
    nlohmann::json hs = nlohmann::json::array();
    for (std::uint32_t h : thresholds) hs.push_back(threshold_name(h));
    res["heavy_thresholds"] = hs;
    res["cuda"] = cuda_json();
  } else {
    res["schedules"] = {"dynamic", "sorted"};
  }
  std::printf("fuzz-diff %s vs oracle, seeds %s: %llu graphs, %llu edges, %llu mismatching rows (%.1f s)\n",
              backend.c_str(), seeds.c_str(), (unsigned long long)d.graphs, (unsigned long long)d.edges,
              (unsigned long long)bad, secs(t0));
  if (bad) std::printf("%s\n", res.dump(1).c_str());
  merge_report(report, name, res);
  return bad == 0 ? 0 : 2;
}

// SPEC v2 §3.4: the metamorphic tests on fuzz graphs, for one backend. Exit code 0 if every row
// is unchanged, 2 if not.
int metamorphic(const std::string& seeds, const std::string& bins_opt, const std::string& backend,
                const std::string& heavy_opt, const std::string& profile_path, const std::string& report,
                const std::string& name) {
  const auto [s0, s1] = parse_range(seeds);
  const blz::Backend be = blz::parse_backend(backend);
  const blz::Profile prof = profile_path.empty() ? blz::registry_profile() : blz::load_profile(profile_path);
  const auto bins = fuzz_bins(bins_opt, !profile_path.empty());
  const std::vector<std::uint32_t> thresholds = parse_thresholds(heavy_opt);
  auto t0 = Clock::now();
  const blz::MetamorphicResult r = blz::metamorphic(prof, be, s0, s1, bins, thresholds);
  std::uint64_t bad = 0;
  nlohmann::json tests = nlohmann::json::object();
  for (blz::MetaTest t : blz::kMetaTests) {
    nlohmann::json per_block = nlohmann::json::object();
    std::uint64_t tb = 0;
    for (const blz::ProfileBlock& pb : prof.blocks) {
      per_block[std::string(blz::block_name(pb.block))] = r.mismatched_rows[int(t)][int(pb.block)];
      tb += r.mismatched_rows[int(t)][int(pb.block)];
    }
    bad += tb;
    tests[blz::meta_test_name(t)] = {{"applies", blz::meta_test_applies(t, prof, be)},
                                     {"rows_compared", r.rows_compared[int(t)]},
                                     {"mismatched_rows", tb},
                                     {"mismatched_rows_by_block", per_block}};
  }
  std::vector<std::uint64_t> first_bad(r.bad_seeds.begin(),
                                       r.bad_seeds.begin() + std::ptrdiff_t(std::min<std::size_t>(20, r.bad_seeds.size())));
  nlohmann::json res = {{"backend", backend},
                        {"profile", profile_path.empty() ? nlohmann::json(nullptr) : profile_ref(prof)},
                        {"seeds", seeds},
                        {"graphs", r.graphs},
                        {"edges", r.edges},
                        {"bins", bins.empty() ? nlohmann::json("profile") : nlohmann::json(bins)},
                        {"k_max", prof.k_max},
                        {"tests", tests},
                        {"mismatched_seeds", r.bad_seeds.size()},
                        {"first_mismatched_seeds", first_bad},
                        {"seconds", secs(t0)},
                        {"git", git_info()}};
  if (be == blz::Backend::cuda) {
    nlohmann::json hs = nlohmann::json::array();
    for (std::uint32_t h : thresholds) hs.push_back(threshold_name(h));
    res["heavy_thresholds"] = hs;
    res["cuda"] = cuda_json();
  }
  std::printf("metamorphic %s, %s, seeds %s: %llu graphs, %llu edges, %llu mismatching rows (%.1f s)\n",
              backend.c_str(), prof.name.c_str(), seeds.c_str(), (unsigned long long)r.graphs,
              (unsigned long long)r.edges, (unsigned long long)bad, secs(t0));
  if (bad) std::printf("%s\n", res.dump(1).c_str());
  merge_report(report, name, res);
  return bad == 0 ? 0 : 2;
}

// SPEC v2 §3.5: the ground-truth join, written as results/gt_join.json.
int gt_join(const std::string& graph, const std::string& csv, const std::string& patterns, const std::string& out) {
  auto t0 = Clock::now();
  const blz::BlzgView g(graph);
  const blz::MappedFile csv_file(csv);
  const blz::Sha256Digest csv_sha = blz::sha256(csv_file.data(), csv_file.size());
  if (csv_sha != g.csv_sha256())
    throw std::runtime_error(csv + " is not the CSV " + graph + " was prepared from (SHA-256 differs)");
  const std::string pat_text = read_file(patterns);
  nlohmann::ordered_json j = blz::gt_join(std::string_view(csv_file.data(), csv_file.size()), pat_text,
                                          {g.file_row, g.ts, g.label});
  nlohmann::ordered_json head = {
      {"inputs",
       {{"graph", graph},
        {"csv", csv},
        {"csv_sha256", blz::to_hex(csv_sha)},
        {"patterns", patterns},
        {"patterns_sha256", blz::to_hex(blz::sha256(pat_text.data(), pat_text.size()))},
        {"m", g.m()}}},
      {"git", git_info()}};
  for (auto it = j.begin(); it != j.end(); ++it) head[it.key()] = it.value();
  ensure_parent(out);
  std::ofstream(out) << blz::gt_join_dump(head);
  const auto& sm = head["summary"];
  std::fprintf(stderr, "gt-join: %s attempts, %s rows; rows %s; attempts %s; %s unlisted positives (%.1f s) -> %s\n",
               sm["attempts"].dump().c_str(), sm["listed_rows"].dump().c_str(), sm["rows_by_status"].dump().c_str(),
               sm["attempts_by_status"].dump().c_str(), sm["unlisted_positives"].dump().c_str(), secs(t0), out.c_str());
  return 0;
}

// SPEC v2 §5 (E9): per listed attempt and profile block, structural eligibility (the oracle on the
// attempt's edges alone, or --eligibility for GFP-produced features), activation (the full
// mining in --features) and detection (--pred, u8[m] codes of runtime/coverage.hpp).
int coverage(const std::string& graph, const std::string& gt_path, const std::string& profile_path,
             const std::string& features, const std::string& pred_path, const std::string& elig_path,
             const std::string& out) {
  auto t0 = Clock::now();
  const blz::BlzgView g(graph);
  const std::uint64_t m = g.m();
  const std::string blzg_sha = file_sha256(graph), csv_sha = blz::to_hex(g.csv_sha256());
  const nlohmann::json gt = nlohmann::json::parse(std::ifstream(gt_path));
  if (gt.at("inputs").at("csv_sha256") != csv_sha || gt.at("inputs").at("m") != m)
    throw std::runtime_error(gt_path + " is not a gt-join of " + graph + " (CSV SHA-256 or m differs)");
  const std::uint64_t b_fv = blz::fit_val_boundary(g.ts), b_test = blz::train_size(m);
  const nlohmann::json want_splits = {{"fit", {0, b_fv}}, {"val", {b_fv, b_test}}, {"test", {b_test, m}}};
  if (gt.at("splits") != want_splits) throw std::runtime_error(gt_path + ": its splits differ from SPEC v2 §3.3's");

  const blz::Profile prof = blz::load_profile(profile_path);   // E403
  blz::check_profile_backend(prof, blz::Backend::oracle);     // E404: eligibility runs the oracle

  // The full mining (activation): a blz manifest of this profile over every edge, or a
  // GFP-produced pass (tools/gfp/dg2_features.py) at this profile's windows and bins.
  const nlohmann::json fm = nlohmann::json::parse(std::ifstream(features + "/features.json"));
  if (fm.at("blzg_sha256") != blzg_sha) throw std::runtime_error(features + " was mined from another .blzg");
  const bool gfp = fm.value("producer", "") == "tools/gfp/dg2_features.py";
  nlohmann::ordered_json jfeat = {{"dir", features}, {"source", gfp ? "gfp" : "blz"}};
  if (gfp) {
    if (fm.at("rows").at("first") != 0 || fm.at("rows").at("last") != m)
      throw std::runtime_error(features + ": not every edge's row");
    for (const blz::ProfileBlock& pb : prof.blocks) {
      const std::string b(blz::block_name(pb.block));
      const auto& fb = fm.at("blocks").at(b);
      if (fb.at("window") != pb.params.window || fb.at("bins") != nlohmann::json(pb.params.bins))
        throw std::runtime_error(features + ": block " + b + " has another window or bins than " + profile_path);
    }
    if (elig_path.empty())
      throw std::runtime_error(features + " is GFP-produced: its eligibility runs the same GFP recipe (SPEC v2 §5); "
                               "give --eligibility from tools/gfp/coverage_gfp.py");
    jfeat["gfp_recipe"] = fm.at("gfp_recipe").at("id");
    jfeat["causal"] = fm.at("causal");
    jfeat["row_sees"] = fm.at("row_sees");
  } else {
    if (fm.at("profile").at("id") != prof.id())
      throw std::runtime_error(features + ": profile " + fm.at("profile").at("id").get<std::string>() + ", not " +
                               prof.id() + " (" + profile_path + ")");
    const auto& tr = fm.at("triggers");
    if (tr.value("sampled", false) || tr.at("first") != 0 || tr.at("last") != m || fm.value("edges", m) != m)
      throw std::runtime_error(features + ": not a mining of every edge");
    if (!elig_path.empty())
      throw std::runtime_error("--eligibility is for GFP-produced features; blz features use the oracle (SPEC v2 §5)");
    jfeat["profile_id"] = prof.id();
    jfeat["backend"] = fm.at("backend").at("name");
  }
  std::deque<blz::NpyU32View> views;  // not movable: a deque never relocates them
  nlohmann::ordered_json jfiles;
  for (const blz::ProfileBlock& pb : prof.blocks) {
    const std::string b(blz::block_name(pb.block));
    const auto& fb = fm.at("blocks").at(b);
    views.emplace_back(features + "/" + fb.at("file").get<std::string>());
    if (views.back().rows() != m || views.back().cols() != pb.params.bins.size())
      throw std::runtime_error(features + ": block " + b + " is not m x bins");
    jfiles[b] = fb.contains("output_fnv1a") ? fb.at("output_fnv1a") : fb.at("output_sha256");
  }
  jfeat["blocks"] = jfiles;

  blz::CoverageInputs in;
  in.graph = {g.src, g.dst, g.ts, g.label};
  in.b_fv = b_fv;
  in.b_test = b_test;
  in.profile = &prof;
  in.attempts = blz::coverage_attempts(gt, m);
  for (const auto& v : views) in.blocks.push_back(v.data());

  std::optional<blz::NpyU8View> pred;
  nlohmann::ordered_json jpred;
  if (!pred_path.empty()) {
    pred.emplace(pred_path);
    if (pred->cols() != 1 || pred->rows() != m) throw std::runtime_error(pred_path + ": not u8[m]");
    in.pred = pred->data();
    jpred = {{"file", pred_path}, {"sha256", file_sha256(pred_path)}};
  }

  nlohmann::ordered_json jelig = {{"source", gfp ? "external" : "oracle"}};
  if (!elig_path.empty()) {
    const nlohmann::json el = nlohmann::json::parse(std::ifstream(elig_path));
    if (el.at("producer") != "tools/gfp/coverage_gfp.py") throw std::runtime_error(elig_path + ": not written by tools/gfp/coverage_gfp.py");
    if (el.at("gfp_recipe") != fm.at("gfp_recipe"))
      throw std::runtime_error(elig_path + ": its GFP recipe differs from " + features + "'s");
    if (el.at("gt_join_sha256") != file_sha256(gt_path)) throw std::runtime_error(elig_path + ": computed from another gt-join");
    std::vector<std::size_t> col;  // profile block -> column of the file's rows
    const auto names = el.at("blocks").get<std::vector<std::string>>();
    for (const blz::ProfileBlock& pb : prof.blocks) {
      auto it = std::find(names.begin(), names.end(), std::string(blz::block_name(pb.block)));
      if (it == names.end()) throw std::runtime_error(elig_path + ": no block " + std::string(blz::block_name(pb.block)));
      col.push_back(std::size_t(it - names.begin()));
    }
    for (const auto& a : el.at("attempts")) {
      const auto r = a.at("rows").get<std::vector<std::uint64_t>>();
      if (r.size() != names.size()) throw std::runtime_error(elig_path + ": a row of the wrong width");
      std::vector<std::uint64_t> v;
      for (std::size_t c : col) v.push_back(r[c]);
      in.external_eligible[a.at("id").get<std::int64_t>()] = v;
    }
    jelig = {{"source", "gfp"},
             {"file", elig_path},
             {"sha256", file_sha256(elig_path)},
             {"gfp_recipe", el.at("gfp_recipe").at("id")},
             {"call", el.at("call")}};
  }

  nlohmann::ordered_json res = blz::coverage(in);
  nlohmann::ordered_json head = {
      {"experiment", "E9"},
      {"spec", "SPEC v2 §5"},
      {"inputs",
       {{"graph", graph}, {"blzg_sha256", blzg_sha}, {"gt_join", gt_path}, {"gt_join_sha256", file_sha256(gt_path)}}},
      {"profile", profile_json(prof)},
      {"features", jfeat},
      {"eligibility_source", jelig},
      {"pred", jpred},
      {"git", git_info()},
      {"threads", omp_get_max_threads()}};
  for (auto it = res.begin(); it != res.end(); ++it) head[it.key()] = it.value();
  head["seconds"] = secs(t0);
  head["peak_rss_mb"] = peak_rss_mb();
  ensure_parent(out);
  std::ofstream(out) << blz::coverage_dump(head);
  const auto& all = head["by_typology"]["ALL"];
  std::fprintf(stderr, "coverage %s: %s complete attempts; eligible (any) %s, activated (any) %s; (%.1f s) -> %s\n",
               prof.name.c_str(), all["complete"].dump().c_str(), all["eligible"]["any"].dump().c_str(),
               all["activated"]["any"].dump().c_str(), secs(t0), out.c_str());
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

  std::string config, profile, backend = "oracle", blocks, triggers = "all", schedule = "dynamic";
  int threads = 0;
  std::uint64_t edges = 0;  // mine --edges (0: every edge)
  std::uint32_t heavy = blz::kDefaultHeavyThreshold;
  auto* mn = app.add_subcommand("mine", "Mine feature blocks -> <out>/<block>.npy + features.json (the manifest)");
  mn->add_option("--graph", graph)->required();
  auto* mn_prof = mn->add_option("--profile", profile, "semantic profile (configs/profiles/*.yaml, SPEC v2 §3.1)");
  auto* mn_cfg = mn->add_option("--config", config, "a v1 feature config (or a profile)");
  mn_prof->excludes(mn_cfg);
  mn->add_option("--backend", backend)->check(CLI::IsMember({"oracle", "cpu", "cuda"}));
  mn->add_option("--threads", threads);
  mn->add_option("--schedule", schedule, "cpu backend: dynamic | sorted")->check(CLI::IsMember({"dynamic", "sorted"}));
  mn->add_option("--heavy-threshold", heavy, "cuda backend: work estimate from which a trigger is heavy");
  mn->add_option("--blocks", blocks, "comma-separated; default: every block of the profile");
  mn->add_option("--triggers", triggers, "all | train | test | fit | val | range:a:b | sample:<pct>:<seed>");
  mn->add_option("--edges", edges, "mine on the stream prefix [0, k) only: later edges do not exist (SPEC v2 §4.4 R5)");
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

  std::string threads_list = "1,2,4,6,8,12", schedules = "dynamic", bench_name, heavy_list;
  int reps = 5;
  auto* bn = app.add_subcommand("bench", "E4/E5: time the CPU or CUDA backend per block (SPEC §12)");
  bn->add_option("--graph", graph)->required();
  auto* bn_prof = bn->add_option("--profile", profile, "semantic profile (SPEC v2 §3.1)");
  auto* bn_cfg = bn->add_option("--config", config, "a v1 feature config (or a profile)");
  bn_prof->excludes(bn_cfg);
  bn->add_option("--backend", backend)->check(CLI::IsMember({"cpu", "cuda"}))->required();
  bn->add_option("--threads", threads_list, "comma-separated thread counts");
  bn->add_option("--reps", reps, "timed runs after one warm-up");
  bn->add_option("--schedule", schedules, "cpu: comma-separated: dynamic, sorted");
  bn->add_option("--heavy", heavy_list, "cuda: comma-separated light/heavy thresholds; 'all' = every trigger light "
                                        "(default 32)");
  bn->add_option("--blocks", blocks, "comma-separated; default: every block in the config");
  bn->add_option("--out", out, "merge the result into this JSON file")->required();
  bn->add_option("--name", bench_name, "key (JSON pointer path) in --out; default: the backend");

  std::string fd_backend = "cpu", fd_heavy = "0,4,all", fd_bins;
  auto* fd = app.add_subcommand("fuzz-diff", "AC1: a generated backend vs the oracle on fuzz graphs (SPEC §13.3)");
  fd->add_option("--seeds", seeds, "a:b")->required();
  fd->add_option("--bins", fd_bins, "default: the profile's bins per block, or 2..33 without --profile");
  fd->add_option("--backend", fd_backend)->check(CLI::IsMember({"cpu", "cuda"}));
  fd->add_option("--heavy", fd_heavy, "cuda: light/heavy thresholds, each run on every graph");
  fd->add_option("--profile", profile, "blocks, patterns and k_max (SPEC v2 §10); windows come from the graphs");
  fd->add_option("--report", report, "merge the summary into this JSON file");
  fd->add_option("--name", name, "key (JSON pointer path) in --report")->needs("--report");

  std::string mt_backend = "cpu";
  auto* mt = app.add_subcommand("metamorphic", "SPEC v2 §3.4: prefix closure, relabeling, time shift, tcyc visibility invariance on fuzz graphs");
  mt->add_option("--seeds", seeds, "a:b")->required();
  mt->add_option("--bins", fd_bins, "default: the profile's bins per block, or 2..33 without --profile");
  mt->add_option("--backend", mt_backend)->check(CLI::IsMember({"oracle", "cpu", "cuda"}));
  mt->add_option("--heavy", fd_heavy, "cuda: light/heavy thresholds, each run on every graph");
  mt->add_option("--profile", profile, "default: every registered pattern (v1)");
  mt->add_option("--report", report, "merge the summary into this JSON file");
  mt->add_option("--name", name, "key (JSON pointer path) in --report")->needs("--report");

  std::string gt_csv, gt_patterns;
  auto* gj = app.add_subcommand("gt-join", "SPEC v2 §3.5: join Patterns.txt to the stream's eids");
  gj->add_option("--graph", graph)->required();
  gj->add_option("--csv", gt_csv)->required();
  gj->add_option("--patterns", gt_patterns)->required();
  gj->add_option("--out", out)->required();

  std::string gt_path = "results/gt_join.json", pred_path, elig_path;
  auto* cv = app.add_subcommand("coverage", "SPEC v2 §5 (E9): eligibility, activation and detection per listed attempt");
  cv->add_option("--graph", graph)->required();
  cv->add_option("--gt", gt_path, "blz gt-join output");
  cv->add_option("--profile", profile, "the profile whose blocks are covered")->required();
  cv->add_option("--features", features, "the full mining under the profile (blz mine, or tools/gfp/dg2_features.py)")->required();
  cv->add_option("--pred", pred_path, "u8[m] prediction codes: 0, 1, 255 = not scored");
  cv->add_option("--eligibility", elig_path, "GFP-produced features: tools/gfp/coverage_gfp.py output");
  cv->add_option("--threads", threads);
  cv->add_option("--out", out)->required();

  CLI11_PARSE(app, argc, argv);
  try {
    if (*prep) return prepare(csv, out);
    if (*exp) return export_edges(graph, out);
    if (*st) return stats(graph, out);
    if (*mn || *bn) {
      if (profile.empty() == config.empty()) throw std::runtime_error("give one of --profile and --config");
      if (profile.empty()) profile = config;
    }
    if (*mn) return mine(graph, profile, backend, threads, schedule, heavy, blocks, triggers, edges, out);
    if (*bn) {
      if (bench_name.empty()) bench_name = backend;
      if (backend == "cuda") {
        if (*bn->get_option("--threads") || *bn->get_option("--schedule"))
          throw std::runtime_error("bench --backend cuda takes --heavy, not --threads or --schedule");
        return bench_cuda(graph, profile, reps, heavy_list.empty() ? std::to_string(blz::kDefaultHeavyThreshold) : heavy_list, blocks, out, bench_name);
      }
      if (!heavy_list.empty()) throw std::runtime_error("--heavy is for --backend cuda");
      return bench(graph, profile, backend, threads_list, reps, schedules, blocks, out, bench_name);
    }
    if (*fd) {
      if (!report.empty() && name.empty()) throw std::runtime_error("--report needs --name");
      return fuzz_diff(seeds, fd_bins, fd_backend, fd_heavy, profile, report, name);
    }
    if (*mt) {
      if (!report.empty() && name.empty()) throw std::runtime_error("--report needs --name");
      return metamorphic(seeds, fd_bins, mt_backend, fd_heavy, profile, report, name);
    }
    if (*gj) return gt_join(graph, gt_csv, gt_patterns, out);
    if (*cv) {
      if (threads > 0) omp_set_num_threads(threads);
      return coverage(graph, gt_path, profile, features, pred_path, elig_path, out);
    }
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

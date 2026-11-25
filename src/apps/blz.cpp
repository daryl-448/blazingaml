// blz — BlazingAML runtime CLI (SPEC §14). Commands so far: prepare, export-edges, stats (P0);
// mine (oracle backend), compare, fuzz (P1); train (P2); mine --backend cpu, bench, fuzz-diff (P4);
// the cuda backend of mine, bench and fuzz-diff (P5); profiles, manifests, trigger samples,
// metamorphic and gt-join (SPEC v2 P7); coverage (SPEC v2 P9).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
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
#include "graph/index_contracts.hpp"
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
#include "oracle/profile_oracle.hpp"
#include "runtime/vstats.hpp"
#include "runtime/clean_vstats.hpp"
#include "runtime/coverage.hpp"
#include "runtime/cpu_backend.hpp"
#include "runtime/cuda_backend.hpp"
#include "runtime/fusion.hpp"
#include "runtime/features.hpp"
#include "runtime/fuzz_diff.hpp"
#include "runtime/hybrid.hpp"
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

// Every clean window is (t - delta, t] (SPEC v3 R2); a legacy block's lower bound is its block's.
nlohmann::json block_params_json(const blz::BlockParams& p, blz::Block b, blz::Family f) {
  return {{"window", p.window},
          {"lower", f == blz::Family::legacy && blz::closed_lower_bound(b) ? "closed" : "open"},
          {"bins", p.bins}};
}

// --blocks: a comma-separated list (in that order), or every block of the profile. `vstats` (a
// clean profile's, SPEC v3 R8) is not a pattern block: select_vstats says whether it is chosen.
std::vector<const blz::ProfileBlock*> select_blocks(const blz::Profile& prof, const std::string& blocks_opt) {
  std::vector<const blz::ProfileBlock*> blocks;
  if (blocks_opt.empty()) {
    for (const blz::ProfileBlock& pb : prof.blocks) blocks.push_back(&pb);
    return blocks;
  }
  for (std::size_t p = 0; p <= blocks_opt.size();) {
    auto q = std::min(blocks_opt.find(',', p), blocks_opt.size());
    const std::string name = blocks_opt.substr(p, q - p);
    if (name == "vstats") {
      if (!prof.vstats) throw std::runtime_error("block 'vstats' is not in " + prof.file);
      p = q + 1;
      continue;
    }
    const auto b = blz::parse_block(name);
    if (!b) throw std::runtime_error("unknown block '" + name + "'");
    const blz::ProfileBlock* pb = prof.find(*b);
    if (!pb) throw std::runtime_error("block '" + name + "' is not in " + prof.file);
    blocks.push_back(pb);
    p = q + 1;
  }
  return blocks;
}

bool select_vstats(const blz::Profile& prof, const std::string& blocks_opt) {
  if (blocks_opt.empty()) return prof.vstats.has_value();
  return ("," + blocks_opt + ",").find(",vstats,") != std::string::npos;
}

std::string file_sha256(const std::string& path) {
  const blz::MappedFile f(path);
  return blz::to_hex(blz::sha256(f.data(), f.size()));
}

// FNV-1a over bytes; `h` continues an earlier call's hash.
std::uint64_t fnv1a_bytes(const void* data, std::size_t bytes, std::uint64_t h = 1469598103934665603ULL) {
  const auto* b = static_cast<const unsigned char*>(data);
  for (std::size_t i = 0; i < bytes; ++i) h = (h ^ b[i]) * 1099511628211ULL;
  return h;
}

std::uint64_t fnv1a(std::span<const std::uint32_t> data) { return fnv1a_bytes(data.data(), data.size_bytes()); }

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

// The optimizations this build's generated code was compiled without (blzc --disable, SPEC v2 §7.2).
nlohmann::json disabled_json() {
  nlohmann::json j = nlohmann::json::array();
  for (const char* d : blz::disabled_optimizations()) j.push_back(d);
  return j;
}

// The CPU plan of each pattern of a block: "sweep" (SPEC v2 §7.1) or "per_trigger".
nlohmann::json cpu_plans_json(const blz::ProfileBlock& pb) {
  nlohmann::json j = nlohmann::json::object();
  for (const blz::PatternInfo* pi : pb.patterns) j[pi->name] = pi->cpu_sweep ? "sweep" : "per_trigger";
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

// SPEC v3 R8 -> <out>/vstats.npy (float64[rows, kVstatsCols], F7's columns), written in chunks of
// CleanVstats::kChunkRows rows, so the rows never sit in memory whole. rows_of(part, buf) computes
// the rows of the selection `part`. The manifest entry's fnv1a is over the data bytes.
template <class RowsOf>
nlohmann::json write_vstats(const std::string& out, const blz::TriggerSel& sel, RowsOf rows_of) {
  const std::string path = out + "/vstats.npy", tmp = path + ".tmp";
  std::ofstream f(tmp, std::ios::binary);
  if (!f) throw std::runtime_error("cannot write " + tmp);
  const std::uint64_t rows = sel.rows();
  const std::string header = blz::npy_header("<f8", rows, blz::kVstatsCols);
  f.write(header.data(), std::streamsize(header.size()));
  std::uint64_t h = fnv1a_bytes(nullptr, 0);
  std::vector<double> buf;
  double mine_s = 0, write_s = 0;
  for (std::uint64_t a = 0; a < rows; a += blz::CleanVstats::kChunkRows) {
    const std::uint64_t b = std::min(rows, a + blz::CleanVstats::kChunkRows);
    blz::TriggerSel part;
    if (sel.sampled) {
      part.sampled = true;
      part.eids.assign(sel.eids.begin() + std::ptrdiff_t(a), sel.eids.begin() + std::ptrdiff_t(b));
      part.first = part.eids.front();
      part.last = std::uint64_t(part.eids.back()) + 1;
    } else {
      part.first = sel.first + a;
      part.last = sel.first + b;
    }
    buf.resize((b - a) * blz::kVstatsCols);
    auto t0 = Clock::now();
    rows_of(part, buf.data());
    mine_s += secs(t0);
    t0 = Clock::now();
    f.write(reinterpret_cast<const char*>(buf.data()), std::streamsize(buf.size() * sizeof(double)));
    h = fnv1a_bytes(buf.data(), buf.size() * sizeof(double), h);
    write_s += secs(t0);
  }
  f.close();
  if (!f) throw std::runtime_error("write failed: " + tmp);
  std::filesystem::rename(tmp, path);
  return {{"file", "vstats.npy"},
          {"dtype", "float64"},
          {"columns", blz::clean_vstats_column_names()},
          {"rule", "SPEC v3 R8: per side (u_out, u_in, v_out, v_in) the node's loop-free edges with t - window < ts' "
                   "<= t; fan, deg, deg / fan, then the age t - ts' and the amount received, each mean, sum, "
                   "population var, skew and kurtosis, accumulated in ascending order"},
          {"output_fnv1a", h},
          {"mine_seconds", mine_s},
          {"write_seconds", write_s}};
}

int mine(const std::string& graph, const std::string& profile_path, const std::string& backend, int threads,
         const std::string& schedule_opt, std::uint32_t heavy, std::uint32_t cpu_heavy, bool reach_prune,
         const std::string& blocks_opt, const std::string& triggers, std::uint64_t edges, std::uint64_t work_limit,
         bool fuse, bool pack, const std::string& quirks_opt, const std::string& out) {
  const blz::Backend be = blz::parse_backend(backend);
  if ((fuse || pack) && be != blz::Backend::cuda) throw std::runtime_error("--fuse and --pack are for the cuda backend");
  // hybrid (SPEC v2 §8 S6): the cuda backend's device and rows, plus the cpu backend's index.
  const bool hybrid = be == blz::Backend::hybrid;
  const bool cpu = be == blz::Backend::cpu, gpu = be == blz::Backend::cuda || hybrid;
  if (gpu && !blz::cuda_built()) blz::cuda_device();  // throws: no CUDA backend in this build
  if (reach_prune && !(cpu || hybrid)) throw std::runtime_error("--reach-prune is for the cpu and hybrid backends");
  const blz::Schedule schedule = blz::parse_schedule(schedule_opt);
  const blz::Profile prof = blz::load_profile(profile_path);  // E403
  blz::check_profile_backend(prof, be);                       // E404 (v1 configs: E402, K_MAX)
  const std::vector<const blz::ProfileBlock*> blocks = select_blocks(prof, blocks_opt);
  // E16 (SPEC v3 §3.6): GFP / v2 behaviours switched back on in the clean oracle.
  const blz::Quirks quirks = blz::parse_quirks(quirks_opt);
  const std::vector<std::string> quirk_ids = blz::quirk_ids(quirks);
  if (!quirk_ids.empty() && (be != blz::Backend::oracle || prof.family != blz::Family::clean))
    throw std::runtime_error("--quirks is for the oracle backend and a clean profile (SPEC v3 §3.6)");

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
  const auto amount = g.amount_received.first(m);  // vstats only
  const bool want_vstats = select_vstats(prof, blocks_opt);
  std::optional<blz::ProfileOracle> oracle;  // a clean profile's is the clean oracle, on its own lists
  std::optional<blz::GraphIndex> index;
  if (cpu || gpu) index.emplace(src, dst, ts, g.n(), prof.self_loops());  // hybrid: gpu
  else oracle.emplace(blz::EdgeStream{src, dst, ts, g.n()}, prof, quirks, amount);
  const double build_s = secs(t0);
  std::fprintf(stderr, "%s index built in %.1f s (m=%llu, n=%llu); profile %s (%s); triggers %s: %llu rows\n",
               backend.c_str(), build_s, (unsigned long long)m, (unsigned long long)g.n(), prof.name.c_str(),
               prof.id().c_str(), sel.spec.c_str(), (unsigned long long)sel.rows());
  std::optional<blz::DeviceGraph> dgraph;
  std::optional<blz::CudaMiner> miner;
  if (gpu) {
    dgraph.emplace(index->view());
    miner.emplace(hybrid);  // hybrid: the GPU's host thread sleeps while the CPU team works
    std::fprintf(stderr, "graph uploaded in %.3f s (%.0f MB)\n", dgraph->upload_seconds(), double(dgraph->bytes()) / 1e6);
  }

  std::filesystem::create_directories(out);
  const std::string meta_path = out + "/features.json";
  const std::string blzg_sha = file_sha256(graph), csv_sha = blz::to_hex(g.csv_sha256());
  // The run a block belongs to: blocks of an earlier run with the same key are kept, so blocks
  // can be mined one invocation at a time into one directory.
  nlohmann::json run_key = {{"backend", backend},     {"csv_sha256", csv_sha},
                            {"blzg_sha256", blzg_sha}, {"profile_id", prof.id()},
                            {"k_max", prof.k_max},     {"triggers", sel.json()},
                            {"edges", m}};
  // Quirk rows are not the profile's rows: a quirk run never shares a directory's blocks with a
  // clean one, or with another quirk set.
  if (!quirk_ids.empty()) run_key["quirks"] = quirk_ids;
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
  // blzc --disable (SPEC v2 §7.2) changes only the generated CPU code.
  nlohmann::json jbe = {{"name", backend}, {"threads", omp_get_max_threads()},
                        {"disabled_optimizations", cpu || hybrid ? disabled_json() : nlohmann::json::array()}};
  if (cpu || hybrid) jbe["schedule"] = blz::schedule_name(schedule);
  if (cpu || hybrid) jbe["reach_prune"] = reach_prune;
  if (gpu) jbe["heavy_threshold"] = threshold_name(heavy);
  if (be == blz::Backend::cuda) {  // SPEC v2 §8 S7: the same rows either way
    jbe["fuse"] = fuse;
    jbe["pack"] = pack;
  }
  if (hybrid) jbe["cpu_threshold"] = threshold_name(cpu_heavy);
  if (be == blz::Backend::oracle && prof.family == blz::Family::clean) jbe["quirks"] = quirk_ids;
  meta["backend"] = jbe;
  meta["triggers"] = sel.json();
  meta["graph"] = graph;
  meta["edges"] = m;  // the graph is eids [0, edges) (--edges; m otherwise)
  meta["blzg_sha256"] = blzg_sha;
  meta["csv_sha256"] = csv_sha;
  meta["config"] = profile_path;
  meta["git"] = git_info();
  meta["threads"] = omp_get_max_threads();
  if (cpu || hybrid) meta["schedule"] = blz::schedule_name(schedule);
  if (gpu)
    meta["cuda"] = {{"env", cuda_json()},
                    {"heavy_threshold", threshold_name(heavy)},
                    {"graph_upload_seconds", dgraph->upload_seconds()},
                    {"device_graph_bytes", dgraph->bytes()}};
  meta["index_build_seconds"] = build_s;
  if (!meta.contains("blocks")) meta["blocks"] = nlohmann::json::object();
  if (sel.sampled) meta["triggers"]["file"] = "triggers.npy";
  // The sample's eids, written with the first block (SPEC v3 R11: after lcyc, so a run that aborts
  // leaves `out` as it was).
  bool triggers_written = !sel.sampled;
  const auto write_triggers = [&] {
    if (!triggers_written) blz::write_npy_u32(out + "/triggers.npy", sel.eids, sel.eids.size(), 1);
    triggers_written = true;
  };

  const std::uint64_t rows = sel.rows();
  std::uint64_t total_saturated = 0;
  // A block's file and manifest entry, once its rows are in `data`.
  const auto finish = [&](const blz::ProfileBlock& pb, const blz::BlockParams& p, std::span<const std::uint32_t> data,
                          double mine_s, std::uint64_t sat, nlohmann::json j) {
    write_triggers();
    const blz::Block b = pb.block;
    const std::string file = std::string(blz::block_name(b)) + ".npy";
    blz::write_npy_u32(out + "/" + file, data, rows, blz::block_cols(p));
    j.update(block_params_json(p, b, prof.family));
    if (b == blz::Block::lcyc) j["work_limit"] = p.work_limit;  // bindings per trigger; no row reached it
    j["file"] = file;
    j["columns"] = blz::column_names(b, p.bins);
    j["mine_seconds"] = mine_s;
    nlohmann::json pats = nlohmann::json::array();
    for (const blz::PatternInfo* pi : pb.patterns) pats.push_back(pi->name);
    j["patterns"] = pats;
    j["output_fnv1a"] = fnv1a(data);
    j["saturated"] = sat;
    meta["blocks"][std::string(blz::block_name(b))] = j;
    meta["peak_rss_mb"] = peak_rss_mb();
    total_saturated += sat;
    std::fprintf(stderr, "%-8s %8.1f s  saturated=%llu  fnv1a=%llu\n", std::string(blz::block_name(b)).c_str(), mine_s,
                 (unsigned long long)sat, (unsigned long long)j["output_fnv1a"].get<std::uint64_t>());
    std::ofstream(meta_path) << meta.dump(1) << '\n';
  };
  const auto cuda_json_of = [&](const blz::CudaBlockTiming& ctm, const blz::CudaBlockTiming& d2h) {
    nlohmann::json c = {{"kernel_seconds", ctm.kernel_seconds}, {"bucket_seconds", ctm.bucket_seconds},
                        {"light_seconds", ctm.light_seconds},   {"heavy_seconds", ctm.heavy_seconds},
                        {"d2h_seconds", d2h.d2h_seconds},       {"light_triggers", ctm.light},
                        {"heavy_triggers", ctm.heavy}};
    if (pack)
      c["pack"] = {{"pack_seconds", d2h.pack_seconds}, {"widen_seconds", d2h.widen_seconds},
                   {"width", d2h.d2h_width},           {"bytes", d2h.d2h_bytes}};
    return c;
  };
  // The CUDA backend mines a fusion's blocks together (SPEC v2 §8 S7); the others one by one.
  std::vector<blz::CudaUnit> units = blz::cuda_units(blocks, fuse && be == blz::Backend::cuda);
  // SPEC v3 R11: a clean run that reaches the work limit aborts with an error and no rows. lcyc is
  // the one block that counts work, so it runs first: an abort comes before any block is written.
  if (prof.family == blz::Family::clean)
    std::stable_partition(units.begin(), units.end(),
                          [](const blz::CudaUnit& u) { return u.blocks.front()->block == blz::Block::lcyc; });
  for (const blz::CudaUnit& unit : units) {
    if (unit.fusion) {
      const std::size_t nslots = unit.blocks.size();
      std::vector<blz::BlockParams> params;
      std::vector<const blz::BlockParams*> param_ptrs;
      std::vector<std::unique_ptr<blz::PinnedBuffer>> bufs;
      std::vector<std::uint32_t*> outs;
      params.reserve(nslots);
      for (const blz::ProfileBlock* pb : unit.blocks) {
        params.push_back(pb->params);
        params.back().work_limit = work_limit;
        bufs.push_back(std::make_unique<blz::PinnedBuffer>(std::max<std::uint64_t>(rows * blz::block_cols(pb->params), 1)));
        outs.push_back(bufs.back()->data());
      }
      for (const blz::BlockParams& bp : params) param_ptrs.push_back(&bp);
      auto t1 = Clock::now();
      const blz::CudaMiner::FusedTiming ftm =
          miner->mine_fused(*dgraph, *unit.fusion, param_ptrs, prof.visibility, sel.first, sel.last, heavy, outs, pack);
      const double mine_s = secs(t1);
      for (std::size_t sl = 0; sl < nslots; ++sl) {
        nlohmann::json j;
        j["cuda"] = cuda_json_of(ftm.group, ftm.slots[sl]);
        // The kernel times and trigger counts are the fusion's, shared by its blocks.
        j["cuda"]["fused"] = unit.fusion->name;
        finish(*unit.blocks[sl], params[sl], std::span<const std::uint32_t>(outs[sl], rows * blz::block_cols(params[sl])),
               mine_s, ftm.slots[sl].saturated, std::move(j));
      }
      continue;
    }
    const blz::ProfileBlock* pb = unit.blocks.front();
    const blz::Block b = pb->block;
    blz::BlockParams p = pb->params;
    p.work_limit = work_limit;  // a run over it stops with an error (SPEC v2 §8 S3)
    p.reach_prune = reach_prune;  // SPEC v2 §8 S6: the same rows, faster where a pattern is planned for it
    const std::uint64_t cells = rows * blz::block_cols(p);
    std::vector<std::uint32_t> feats;
    std::optional<blz::PinnedBuffer> pinned;  // cuda: the rows, copied straight from the device
    if (gpu) pinned.emplace(std::max<std::uint64_t>(cells, 1));
    else feats.assign(cells, 0);
    auto t1 = Clock::now();
    std::uint64_t sat = 0;
    blz::CpuBlockTiming tm;
    blz::CudaBlockTiming ctm;
    blz::HybridTiming htm;
    if (cpu) {
      tm = blz::mine_cpu(index->view(), pb->patterns, p, prof.visibility, sel, schedule, feats.data());
      sat = tm.saturated;
    } else if (hybrid) {
      htm = blz::mine_hybrid(index->view(), *dgraph, *miner, pb->patterns, p, prof.visibility, sel.first, sel.last,
                             cpu_heavy, heavy, schedule, pinned->data());
      sat = htm.saturated;
    } else if (gpu) {
      ctm = miner->mine(*dgraph, pb->patterns, p, prof.visibility, sel.first, sel.last, heavy, pinned->data(), pack);
      sat = ctm.saturated;
    } else if (sel.sampled) {
      sat = oracle->mine(b, p, sel.eids, feats.data());
    } else {
      sat = oracle->mine(b, p, sel.first, sel.last, feats.data());
    }
    const double mine_s = secs(t1);
    const std::span<const std::uint32_t> data =
        gpu ? std::span<const std::uint32_t>(pinned->data(), cells) : std::span<const std::uint32_t>(feats);
    nlohmann::json j = nlohmann::json::object();
    // A sweep (SPEC v2 §7.1) walks every list entry whatever the sample, so its time is not scaled.
    if (sel.sampled && rows > 0)
      j["extrapolated_mine_seconds"] = tm.sweep_seconds + (mine_s - tm.sweep_seconds) * double(m) / double(rows);
    if (cpu) {
      j["schedule_seconds"] = tm.schedule_seconds;
      j["loop_seconds"] = tm.loop_seconds;
      j["cpu_plans"] = cpu_plans_json(*pb);
    }
    if (hybrid) {
      ctm = htm.gpu;
      j["hybrid"] = {{"split_seconds", htm.split_seconds},     {"cpu_seconds", htm.cpu_seconds},
                     {"gpu_seconds", htm.gpu_seconds},         {"overlap_seconds", htm.overlap_seconds},
                     {"merge_seconds", htm.merge_seconds},     {"total_seconds", htm.total_seconds},
                     {"cpu_triggers", htm.cpu_triggers},       {"gpu_triggers", htm.gpu_triggers},
                     {"gpu_split_seconds", htm.gpu.split_seconds}, {"cpu_plans", cpu_plans_json(*pb)}};
    }
    if (gpu) j["cuda"] = cuda_json_of(ctm, ctm);
    finish(*pb, p, data, mine_s, sat, std::move(j));
  }
  if (want_vstats) {
    // SPEC v3 R8. The cuda backend and the hybrid compute it on the GPU.
    write_triggers();
    const std::int32_t window = prof.vstats->window;
    auto t1 = Clock::now();
    std::optional<blz::CleanVstats> vs;
    if (!oracle) vs.emplace(index->view(), amount, window, gpu ? &*dgraph : nullptr);
    const double order_s = secs(t1);
    nlohmann::json j = write_vstats(out, sel, [&](const blz::TriggerSel& part, double* buf) {
      if (vs) vs->rows(part, buf);
      else oracle->vstats(window, part, buf);
    });
    j["mine_seconds"] = j["mine_seconds"].get<double>() + order_s;
    j["window"] = window;
    j["lower"] = "open";
    j["device"] = oracle ? "oracle" : gpu ? "cuda" : "cpu";
    if (vs) j["sides"] = vs->sides();
    meta["blocks"]["vstats"] = j;
    meta["peak_rss_mb"] = peak_rss_mb();
    std::fprintf(stderr, "%-8s %8.1f s  fnv1a=%llu\n", "vstats", j["mine_seconds"].get<double>(),
                 (unsigned long long)j["output_fnv1a"].get<std::uint64_t>());
    std::ofstream(meta_path) << meta.dump(1) << '\n';
  }
  if (total_saturated > 0) std::fprintf(stderr, "warning: %llu saturated counts\n",
                                        (unsigned long long)total_saturated);
  return 0;
}

// Vertex statistics (SPEC v2 §8 S2) -> <out>/vstats.npy (float64[rows, 52], GFP's columns) +
// features.json. Rows are computed and written in chunks, so the output never sits in memory.
int vstats(const std::string& graph, std::int32_t window, const std::string& backend, int threads,
           const std::string& triggers, const std::string& out) {
  const bool gpu = backend == "cuda";
  if (backend != "cpu" && !gpu) throw std::runtime_error("vstats: --backend is cpu or cuda");
  if (window <= 0) throw std::runtime_error("vstats: --window must be > 0");
  if (threads > 0) omp_set_num_threads(threads);
  auto t0 = Clock::now();
  blz::BlzgView g(graph);
  const blz::TriggerSel sel = blz::select_triggers(triggers, g.ts);
  if (gpu && sel.sampled) throw std::runtime_error("vstats --backend cuda takes an eid range, not a sample");
  const blz::GraphIndex index(g.src, g.dst, g.ts, g.n());
  std::optional<blz::DeviceGraph> dgraph;
  if (gpu) dgraph.emplace(index.view());
  const double build_s = secs(t0);
  const blz::VstatsParams p{window};
  std::filesystem::create_directories(out);
  const std::string path = out + "/vstats.npy", tmp = path + ".tmp";
  std::ofstream f(tmp, std::ios::binary);
  if (!f) throw std::runtime_error("cannot write " + tmp);
  const std::uint64_t rows = sel.rows();
  const std::string header = blz::npy_header("<f8", rows, blz::kVstatsCols);
  f.write(header.data(), std::streamsize(header.size()));
  blz::Sha256 sha;  // of the data bytes, without the header (as dg2_features.py's output_sha256)
  constexpr std::uint64_t kChunk = 1u << 18;
  std::vector<double> buf;
  auto t1 = Clock::now();
  double compute_s = 0;  // the statistics alone, without the write and the hash
  for (std::uint64_t a = 0; a < rows; a += kChunk) {
    const std::uint64_t b = std::min(rows, a + kChunk);
    buf.assign((b - a) * blz::kVstatsCols, 0.0);
    blz::TriggerSel part;
    if (sel.sampled) {
      part.sampled = true;
      part.eids.assign(sel.eids.begin() + std::ptrdiff_t(a), sel.eids.begin() + std::ptrdiff_t(b));
      part.first = part.eids.front();
      part.last = std::uint64_t(part.eids.back()) + 1;
    } else {
      part.first = sel.first + a;
      part.last = sel.first + b;
    }
    const auto tc = Clock::now();
    if (gpu) blz::vstats_cuda(*dgraph, g.amount_received, p, part.first, part.last, buf.data());
    else blz::vstats_cpu(index.view(), g.amount_received, p, part, buf.data());
    compute_s += secs(tc);
    const auto* bytes = reinterpret_cast<const char*>(buf.data());
    f.write(bytes, std::streamsize(buf.size() * sizeof(double)));
    sha.update(bytes, buf.size() * sizeof(double));
  }
  f.close();
  if (!f) throw std::runtime_error("write failed: " + tmp);
  std::filesystem::rename(tmp, path);
  const double vs_s = secs(t1);
  const std::string data_sha = blz::to_hex(sha.finish());
  if (sel.sampled) blz::write_npy_u32(out + "/triggers.npy", sel.eids, sel.eids.size(), 1);
  nlohmann::json meta = {
      {"manifest_version", 2},
      {"producer", "blz vstats"},
      {"graph", graph},
      {"blzg_sha256", file_sha256(graph)},
      {"csv_sha256", blz::to_hex(g.csv_sha256())},
      {"triggers", sel.json()},
      {"backend", {{"name", backend}, {"threads", omp_get_max_threads()}}},
      {"index_build_seconds", build_s},
      {"blocks",
       {{"vstats",
         {{"file", "vstats.npy"},
          {"dtype", "float64"},
          {"columns", blz::vstats_column_names()},
          {"columns_of_input", {"ts", "amount_received"}},
          {"window", window},
          {"rule", "GFP stream-mode vertex statistics (SPEC v2 §8 S2): per side, the node's edges with eid <= e "
                   "and ts >= ts[e - 1] - window; two-pass population moments in eid order; variance < 1e-5 * "
                   "mean^2 is reported as 0 with its skew and kurtosis"},
          {"output_sha256", data_sha},
          {"mine_seconds", compute_s},
          {"write_seconds", vs_s - compute_s}}}}},
      {"peak_rss_mb", peak_rss_mb()},
      {"git", git_info()}};
  if (gpu) meta["cuda"] = {{"env", cuda_json()}};
  std::ofstream(out + "/features.json") << meta.dump(1) << '\n';
  std::fprintf(stderr, "vstats %s: %llu rows in %.1f s (index %.1f s, write and hash %.1f s), sha256 %s\n",
               backend.c_str(), (unsigned long long)rows, compute_s, build_s, vs_s - compute_s, data_sha.c_str());
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
// aligned with graphs.npy) for every v1 block, with the graph's window for every block. With
// lcyc (SPEC v2 §8 S3), also lcyc.npy: the oracle's lcyc rows, length <= the compiled K_MAX, at
// `work_limit`; every row of a graph that reaches the limit is UINT32_MAX (tools/gfp/lcyc_vs_gfp.py
// skips those graphs).
int fuzz(const std::string& seeds, const std::string& bins_opt, int k_max, bool vstats, bool lcyc,
         std::uint64_t work_limit, const std::string& setting_opt, const std::string& out) {
  const auto [s0, s1] = parse_range(seeds);
  const blz::FuzzSetting setting = blz::parse_fuzz_setting(setting_opt);
  const auto bins = parse_bins(bins_opt);
  std::vector<std::int64_t> rows;
  std::map<blz::Block, std::vector<std::uint32_t>> feats;
  std::vector<double> amounts, vs;  // --vstats (SPEC v2 §8 S2)
  std::uint64_t limited = 0;        // --lcyc: graphs that reached the work limit
  for (std::uint64_t seed = s0; seed < s1; ++seed) {
    const blz::FuzzGraph fg = blz::fuzz_graph(seed, setting);
    const std::uint64_t m = fg.src.size();
    if (vstats) {
      const std::vector<double> a = blz::fuzz_amounts(seed, m);
      const blz::GraphIndex idx(fg.src, fg.dst, fg.ts, fg.n);
      blz::TriggerSel all;
      all.last = m;
      const std::size_t at = vs.size();
      vs.resize(at + m * blz::kVstatsCols);
      blz::vstats_cpu(idx.view(), a, {fg.window}, all, vs.data() + at);
      amounts.insert(amounts.end(), a.begin(), a.end());
    }
    for (std::uint64_t e = 0; e < m; ++e)
      rows.insert(rows.end(), {std::int64_t(seed), std::int64_t(e), fg.src[e], fg.dst[e],
                               fg.ts[e], fg.window});
    const blz::Oracle oracle({fg.src, fg.dst, fg.ts, fg.n});
    for (blz::Block b : blz::kV1Blocks) {  // GFP's blocks: sg_ord has no GFP counterpart
      auto& f = feats[b];
      const std::size_t at = f.size();
      f.resize(at + m * bins.size(), 0);
      oracle.mine(b, {fg.window, bins}, k_max, 0, m, f.data() + at);
    }
    if (lcyc) {
      auto& f = feats[blz::Block::lcyc];
      const std::size_t at = f.size();
      f.resize(at + m * bins.size(), 0);
      blz::BlockParams p{fg.window, bins};
      p.work_limit = work_limit;
      try {
        oracle.mine(blz::Block::lcyc, p, blz::compiled_k_max(), 0, m, f.data() + at);
      } catch (const blz::WorkLimitExceeded&) {
        std::fill(f.begin() + std::ptrdiff_t(at), f.end(), UINT32_MAX);
        ++limited;
      }
    }
  }
  std::filesystem::create_directories(out);
  const std::uint64_t total = rows.size() / 6;
  blz::write_npy_i64(out + "/graphs.npy", rows, total, 6);
  for (auto& [b, f] : feats)
    blz::write_npy_u32(out + "/" + std::string(blz::block_name(b)) + ".npy", f, total, bins.size());
  if (vstats) {
    blz::write_npy_f64(out + "/amounts.npy", amounts, total, 1);
    blz::write_npy_f64(out + "/vstats.npy", vs, total, blz::kVstatsCols);
  }
  std::fprintf(stderr, "seeds %s (%s): %llu edges -> %s\n", seeds.c_str(), blz::fuzz_setting_name(setting),
               (unsigned long long)total, out.c_str());
  if (lcyc)
    std::fprintf(stderr, "lcyc: %llu graphs reached the work limit of %llu bindings (rows UINT32_MAX)\n",
                 (unsigned long long)limited, (unsigned long long)work_limit);
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
                        {"profile", {{"name", prof.name}, {"id", prof.id()}}},
                        {"disabled_optimizations", disabled_json()}};

  blz::BlzgView g(graph);
  auto t0 = Clock::now();
  const blz::GraphIndex index(g.src, g.dst, g.ts, g.n(), prof.self_loops());
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
    std::vector<std::uint32_t> feats(m * blz::block_cols(p)), ref;
    nlohmann::json jb = {{"patterns", nlohmann::json::array()}, {"cpu_plans", cpu_plans_json(*pb)}};
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
// SPEC v2 §8 S7: with `fuse`, a fusion's blocks are one unit, timed together (its kernels run once
// for all of them); with `pack`, D2H is packed and T adds the pack kernels. Units are keyed by the
// fusion's name or the block's.
int bench_cuda(const std::string& graph, const std::string& config, int reps, const std::string& heavy_opt,
               const std::string& blocks_opt, bool fuse, bool pack, const std::string& out, const std::string& name) {
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
                        {"fuse", fuse}, {"pack", pack},
                        {"profile", {{"name", prof.name}, {"id", prof.id()}}}};

  blz::BlzgView g(graph);
  auto t0 = Clock::now();
  const blz::GraphIndex index(g.src, g.dst, g.ts, g.n(), prof.self_loops());
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
  for (const blz::CudaUnit& unit : blz::cuda_units(blocks, fuse)) {
    const std::string uname = unit.name();
    const std::size_t nslots = unit.blocks.size();
    std::vector<std::unique_ptr<blz::PinnedBuffer>> bufs;
    std::vector<std::uint32_t*> outs;
    std::vector<const blz::BlockParams*> params;
    std::vector<std::vector<std::uint32_t>> ref(nslots);
    nlohmann::json jb = {{"patterns", nlohmann::json::array()}, {"blocks", nlohmann::json::array()},
                         {"fusion", unit.fusion ? nlohmann::json(unit.fusion->name) : nlohmann::json(nullptr)}};
    for (const blz::ProfileBlock* pb : unit.blocks) {
      bufs.push_back(std::make_unique<blz::PinnedBuffer>(std::max<std::uint64_t>(m * blz::block_cols(pb->params), 1)));
      outs.push_back(bufs.back()->data());
      params.push_back(&pb->params);
      jb["blocks"].push_back(std::string(blz::block_name(pb->block)));
      for (const blz::PatternInfo* pi : pb->patterns) jb["patterns"].push_back(pi->name);
    }
    if (!unit.fusion) {
      jb["window"] = params[0]->window;
      jb["bins"] = params[0]->bins;
    }
    for (std::uint32_t h : thresholds) {
      std::vector<double> kernel, d2h, packs, widen, total, bucket, light, heavy, wall;
      blz::CudaBlockTiming tm;
      std::vector<blz::CudaBlockTiming> slot_tm;
      for (int r = 0; r <= reps; ++r) {  // r == 0 is the warm-up
        if (unit.fusion) {
          blz::CudaMiner::FusedTiming ft =
              miner.mine_fused(dg, *unit.fusion, params, prof.visibility, 0, m, h, outs, pack);
          tm = ft.group;
          slot_tm = std::move(ft.slots);
        } else {
          tm = miner.mine(dg, unit.blocks[0]->patterns, *params[0], prof.visibility, 0, m, h, outs[0], pack);
          slot_tm = {tm};
        }
        for (std::size_t sl = 0; sl < nslots; ++sl) {
          const std::string bname(blz::block_name(unit.blocks[sl]->block));
          const std::span<const std::uint32_t> got(outs[sl], m * blz::block_cols(*params[sl]));
          if (ref[sl].empty()) {
            ref[sl].assign(got.begin(), got.end());
            jb["outputs"][bname] = {{"output_fnv1a", fnv1a(ref[sl])}, {"saturated", slot_tm[sl].saturated}};
          } else if (!std::equal(got.begin(), got.end(), ref[sl].begin())) {
            throw std::runtime_error("bench: " + bname + " output differs at heavy threshold " + threshold_name(h) +
                                     " (SPEC §8)");
          }
        }
        if (r == 0) continue;
        kernel.push_back(tm.kernel_seconds);
        d2h.push_back(tm.d2h_seconds);
        packs.push_back(tm.pack_seconds);
        widen.push_back(tm.widen_seconds);
        total.push_back(upload_med + tm.kernel_seconds + tm.pack_seconds + tm.d2h_seconds);
        bucket.push_back(tm.bucket_seconds);
        light.push_back(tm.light_seconds);
        heavy.push_back(tm.heavy_seconds);
        wall.push_back(tm.wall_seconds);
      }
      if (!unit.fusion) {  // v1's keys
        jb["output_fnv1a"] = jb["outputs"].begin().value()["output_fnv1a"];
        jb["saturated"] = jb["outputs"].begin().value()["saturated"];
      }
      nlohmann::json widths = nlohmann::json::object();
      for (std::size_t sl = 0; sl < nslots; ++sl)
        widths[std::string(blz::block_name(unit.blocks[sl]->block))] = slot_tm[sl].d2h_width;
      nlohmann::json jh = {{"total", summary(total)},     {"kernel", summary(kernel)},
                           {"d2h", summary(d2h)},         {"bucket_median", summary(bucket)["median"]},
                           {"light_median", summary(light)["median"]},
                           {"heavy_median", summary(heavy)["median"]},
                           {"wall_median", summary(wall)["median"]},
                           {"light_triggers", tm.light},  {"heavy_triggers", tm.heavy}};
      if (pack) {
        jh["pack"] = summary(packs);
        jh["widen_median"] = summary(widen)["median"];
        jh["d2h_bytes"] = tm.d2h_bytes;
        jh["d2h_width"] = widths;
      }
      std::fprintf(stderr, "%-8s H=%-4s kernels median %8.4f s (light %.4f, heavy %.4f, bucket %.4f), d2h %.4f s, "
                   "pack %.4f s, %llu heavy triggers\n",
                   uname.c_str(), threshold_name(h).c_str(), jh["kernel"]["median"].get<double>(),
                   jh["light_median"].get<double>(), jh["heavy_median"].get<double>(),
                   jh["bucket_median"].get<double>(), jh["d2h"]["median"].get<double>(),
                   summary(packs)["median"].get<double>(), (unsigned long long)tm.heavy);
      jb["by_threshold"][threshold_name(h)] = jh;
    }
    runs[uname] = jb;
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

// E13 (SPEC v2 §8 S6): the hybrid against GPU-only and CPU-only, per block, with `reps` timed runs
// after one warm-up each. Every run's rows must equal the block's first run (SPEC §8).
//   gpu: CudaMiner::mine over every trigger at --heavy (E5's form); T = upload + kernels + D2H.
//   cpu: mine_cpu over every trigger, per --prune setting; T = schedule + loop.
//   hybrid: mine_hybrid per CPU threshold and --prune setting; T = upload + the call's wall clock
//     (the GPU's split, then both sides at once, then the merge; the GPU side holds every kernel
//     and the D2H). T_formula = upload + max(gpu, split + cpu) + merge, from each side's own wall
//     clock: the two paths from the call to the merge.
// The CPU uses `threads` (0: the OpenMP default) and the schedule `sched_opt`.
int bench_hybrid(const std::string& graph, const std::string& config, int reps, int threads, const std::string& sched_opt,
                 const std::string& heavy_opt, const std::string& cpu_heavy_opt, const std::string& prune_opt,
                 const std::string& blocks_opt, const std::string& out, const std::string& name) {
  if (reps < 1) throw std::runtime_error("--reps must be >= 1");
  const blz::Profile prof = blz::load_profile(config);
  blz::check_profile_backend(prof, blz::Backend::hybrid);
  const std::vector<const blz::ProfileBlock*> blocks = select_blocks(prof, blocks_opt);
  const blz::Schedule schedule = blz::parse_schedule(sched_opt);
  const std::vector<std::uint32_t> gpu_h = parse_thresholds(heavy_opt), cpu_h = parse_thresholds(cpu_heavy_opt);
  if (gpu_h.size() != 1) throw std::runtime_error("bench --backend hybrid takes one --heavy threshold");
  std::vector<bool> prunes;
  for (std::size_t q = 0; q <= prune_opt.size();) {
    const std::size_t r = std::min(prune_opt.find(',', q), prune_opt.size());
    const std::string v = prune_opt.substr(q, r - q);
    if (v != "off" && v != "on") throw std::runtime_error("--prune takes off and on, got '" + v + "'");
    prunes.push_back(v == "on");
    q = r + 1;
  }
  if (threads > 0) omp_set_num_threads(threads);
  nlohmann::json hs = nlohmann::json::array();
  for (std::uint32_t h : cpu_h) hs.push_back(threshold_name(h));
  nlohmann::json env = environment(config);
  env["nvidia_smi"] = run_command("nvidia-smi -q 2>/dev/null");
  nlohmann::json res = {{"env", env},
                        {"cuda", cuda_json()},
                        {"backend", "hybrid"},
                        {"graph", graph},
                        {"reps", reps},
                        {"warmup", 1},
                        {"threads", omp_get_max_threads()},
                        {"schedule", blz::schedule_name(schedule)},
                        {"heavy_threshold", threshold_name(gpu_h[0])},
                        {"cpu_thresholds", hs},
                        {"prune", prune_opt},
                        {"profile", {{"name", prof.name}, {"id", prof.id()}}},
                        {"disabled_optimizations", disabled_json()}};

  blz::BlzgView g(graph);
  auto t0 = Clock::now();
  const blz::GraphIndex index(g.src, g.dst, g.ts, g.n(), prof.self_loops());
  res["index_build_seconds"] = secs(t0);
  res["csv_sha256"] = blz::to_hex(g.csv_sha256());
  std::vector<double> uploads;
  for (int r = 0; r <= reps; ++r) {  // r == 0 is the warm-up
    const blz::DeviceGraph dg(index.view());
    if (r > 0) uploads.push_back(dg.upload_seconds());
  }
  res["graph_upload"] = summary(uploads);
  const double upload = res["graph_upload"]["median"].get<double>();
  const blz::DeviceGraph dg(index.view());
  blz::CudaMiner gpu_miner;           // E5's miner, for the GPU-only runs
  blz::CudaMiner hybrid_miner(true);  // blocking sync, next to the CPU team
  const std::uint64_t m = g.m();
  blz::TriggerSel all;
  all.spec = "all";
  all.last = m;

  nlohmann::json runs = nlohmann::json::object();
  for (const blz::ProfileBlock* pb : blocks) {
    const std::string bname(blz::block_name(pb->block));
    const std::uint64_t cells = m * blz::block_cols(pb->params);
    blz::PinnedBuffer buf(cells);
    std::vector<std::uint32_t> ref;
    nlohmann::json jb = {{"patterns", nlohmann::json::array()},
                         {"window", pb->params.window},
                         {"bins", pb->params.bins},
                         {"cpu_plans", cpu_plans_json(*pb)}};
    for (const blz::PatternInfo* pi : pb->patterns) jb["patterns"].push_back(pi->name);
    const auto check = [&](const std::string& what) {
      const std::span<const std::uint32_t> got(buf.data(), cells);
      if (ref.empty()) {
        ref.assign(got.begin(), got.end());
        jb["output_fnv1a"] = fnv1a(ref);
      } else if (!std::equal(got.begin(), got.end(), ref.begin())) {
        throw std::runtime_error("bench: " + bname + " output differs in " + what + " (SPEC §8)");
      }
    };
    {  // GPU only
      std::vector<double> total, kernel, d2h;
      blz::CudaBlockTiming tm;
      for (int r = 0; r <= reps; ++r) {
        tm = gpu_miner.mine(dg, pb->patterns, pb->params, prof.visibility, 0, m, gpu_h[0], buf.data());
        check("gpu");
        if (r == 0) continue;
        total.push_back(upload + tm.kernel_seconds + tm.d2h_seconds);
        kernel.push_back(tm.kernel_seconds);
        d2h.push_back(tm.d2h_seconds);
      }
      jb["gpu"] = {{"total", summary(total)},
                   {"kernel", summary(kernel)},
                   {"d2h", summary(d2h)},
                   {"light_triggers", tm.light},
                   {"heavy_triggers", tm.heavy}};
      std::fprintf(stderr, "%-8s gpu    H=%-4s total median %8.4f s\n", bname.c_str(), threshold_name(gpu_h[0]).c_str(),
                   jb["gpu"]["total"]["median"].get<double>());
    }
    for (const bool prune : prunes) {
      blz::BlockParams p = pb->params;
      p.reach_prune = prune;
      const std::string pk = prune ? "prune_on" : "prune_off";
      {  // CPU only
        std::vector<double> total;
        std::vector<std::uint32_t> feats(cells);
        for (int r = 0; r <= reps; ++r) {
          std::fill(feats.begin(), feats.end(), 0u);
          const blz::CpuBlockTiming tm =
              blz::mine_cpu(index.view(), pb->patterns, p, prof.visibility, all, schedule, feats.data());
          std::copy(feats.begin(), feats.end(), buf.data());
          check("cpu, " + pk);
          if (r > 0) total.push_back(tm.schedule_seconds + tm.loop_seconds);
        }
        jb["cpu"][pk] = {{"total", summary(total)}};
        std::fprintf(stderr, "%-8s cpu    %-9s total median %8.4f s\n", bname.c_str(), pk.c_str(),
                     jb["cpu"][pk]["total"]["median"].get<double>());
      }
      for (const std::uint32_t hc : cpu_h) {  // hybrid, per CPU threshold
        std::vector<double> total, formula, split, gpu_split, cpu_s, gpu_s, overlap, merge, kernel, d2h;
        blz::HybridTiming tm;
        for (int r = 0; r <= reps; ++r) {
          std::fill(buf.data(), buf.data() + cells, 0xdeadbeefu);  // every row must be written
          tm = blz::mine_hybrid(index.view(), dg, hybrid_miner, pb->patterns, p, prof.visibility, 0, m, hc, gpu_h[0],
                                schedule, buf.data());
          check("hybrid, " + pk + ", cpu threshold " + threshold_name(hc));
          if (r == 0) continue;
          total.push_back(upload + tm.total_seconds);
          formula.push_back(upload + std::max(tm.gpu_seconds, tm.split_seconds + tm.cpu_seconds) + tm.merge_seconds);
          split.push_back(tm.split_seconds);
          cpu_s.push_back(tm.cpu_seconds);
          gpu_s.push_back(tm.gpu_seconds);
          overlap.push_back(tm.overlap_seconds);
          merge.push_back(tm.merge_seconds);
          gpu_split.push_back(tm.gpu.split_seconds);
          kernel.push_back(tm.gpu.kernel_seconds);
          d2h.push_back(tm.gpu.d2h_seconds);
        }
        nlohmann::json jh = {{"total", summary(total)}, {"total_formula", summary(formula)},
                             {"split", summary(split)}, {"cpu", summary(cpu_s)},
                             {"gpu", summary(gpu_s)},   {"overlap", summary(overlap)},
                             {"merge", summary(merge)}, {"gpu_split", summary(gpu_split)},
                             {"kernel", summary(kernel)}, {"d2h", summary(d2h)},
                             {"cpu_triggers", tm.cpu_triggers}, {"gpu_triggers", tm.gpu_triggers}};
        std::fprintf(stderr,
                     "%-8s hybrid %-9s Hc=%-5s total median %8.4f s (cpu %.4f s on %llu, gpu %.4f s on %llu, split "
                     "%.4f, merge %.4f)\n",
                     bname.c_str(), pk.c_str(), threshold_name(hc).c_str(), jh["total"]["median"].get<double>(),
                     jh["cpu"]["median"].get<double>(), (unsigned long long)tm.cpu_triggers,
                     jh["gpu"]["median"].get<double>(), (unsigned long long)tm.gpu_triggers,
                     jh["split"]["median"].get<double>(), jh["merge"]["median"].get<double>());
        jb["hybrid"][pk][threshold_name(hc)] = jh;
      }
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

// Path hit (SPEC v3 §4.4 item 6): what a fuzz run's graphs exercised, as JSON, and one line on stdout.
// vs_nonzero, vs_sides: a profile with vstats (SPEC v3 R8), its reference's nonzero rows and the
// distinct sides the backend computed. `graphs`: whether h describes fuzz graphs (an exhaustive run
// has every graph of its bound instead).
nlohmann::json path_hits(const blz::FuzzHits& h, std::span<const std::uint64_t> nonzero, const blz::Profile& prof,
                         std::uint64_t vs_nonzero, std::uint64_t vs_sides, bool graphs = true) {
  nlohmann::json nz = nlohmann::json::object();
  std::string line;
  for (const blz::ProfileBlock& pb : prof.blocks) {
    nz[std::string(blz::block_name(pb.block))] = nonzero[int(pb.block)];
    line += " " + std::string(blz::block_name(pb.block)) + "=" + std::to_string(nonzero[int(pb.block)]);
  }
  if (prof.vstats) {
    nz["vstats"] = vs_nonzero;
    line += " vstats=" + std::to_string(vs_nonzero) + " (" + std::to_string(vs_sides) + " sides computed)";
  }
  if (!graphs) {
    std::printf("  nonzero rows (the reference's, over every window):%s\n", line.c_str());
    return {{"nonzero_rows_by_block", nz}, {"vstats_sides", vs_sides}};
  }
  std::printf("  path hits: %llu of %llu graphs with 2+ components (%llu in all); of %llu non-loop triggers, %llu "
              "with a bound edge at t - delta, %llu at t - delta + 1; %llu tied, %llu duplicate (u, v, t) edges\n"
              "  nonzero rows:%s\n",
              (unsigned long long)h.multi_component_graphs, (unsigned long long)h.graphs,
              (unsigned long long)h.components, (unsigned long long)h.triggers, (unsigned long long)h.bound_out,
              (unsigned long long)h.bound_in, (unsigned long long)h.tied, (unsigned long long)h.duplicated,
              line.c_str());
  return {{"graphs_with_two_or_more_components", h.multi_component_graphs},
          {"components", h.components},
          {"non_loop_triggers", h.triggers},
          {"triggers_with_bound_edge_at_t_minus_delta", h.bound_out},
          {"triggers_with_bound_edge_at_t_minus_delta_plus_1", h.bound_in},
          {"tied_edges", h.tied},
          {"duplicate_uvt_edges", h.duplicated},
          {"nonzero_rows_by_block", nz},
          {"vstats_sides", vs_sides}};
}

// The rows that each optimised path computed in a fuzz-diff run (SPEC v3 §4.1 T2, §4.4 item 6), as
// JSON, and one line on stdout with the paths the run took.
nlohmann::json optimised_paths(const blz::FuzzDiff& d, const std::string& backend, bool fuse, bool pack) {
  const auto n = [](std::uint64_t x) { return std::to_string(x); };
  std::string line;
  if (backend == "cpu") line = "sweep rows " + n(d.sweep_rows) + ", reach-pruned rows " + n(d.reach_rows);
  else {
    line = "light-kernel triggers " + n(d.light_triggers) + ", heavy-kernel triggers " + n(d.heavy_triggers);
    if (fuse) line += ", fused rows " + n(d.fused_rows);
    if (pack)
      line += ", packed rows at 1/2/4 bytes " + n(d.packed_rows[0]) + "/" + n(d.packed_rows[1]) + "/" + n(d.packed_rows[2]);
    if (backend == "hybrid")
      line += ", CPU-side triggers " + n(d.hybrid_cpu_triggers) + " (reach-pruned rows " + n(d.reach_rows) +
              "), GPU-side triggers " + n(d.hybrid_gpu_triggers);
  }
  std::printf("  optimised paths: %s\n", line.c_str());
  return {{"sweep_rows", d.sweep_rows},
          {"reach_pruned_rows", d.reach_rows},
          {"fused_rows", d.fused_rows},
          {"packed_rows_by_width", {{"1", d.packed_rows[0]}, {"2", d.packed_rows[1]}, {"4", d.packed_rows[2]}}},
          {"light_kernel_triggers", d.light_triggers},
          {"heavy_kernel_triggers", d.heavy_triggers},
          {"hybrid_cpu_triggers", d.hybrid_cpu_triggers},
          {"hybrid_gpu_triggers", d.hybrid_gpu_triggers}};
}

// A fuzz-diff counterexample (SPEC v3 §4.3) in the code's terms, on stdout and as JSON: the graph's
// edges in eid order as u->v@t, the block, window and path, and the trigger's got and want rows.
nlohmann::json print_counterexample(const blz::Counterexample& cx, bool exhaustive) {
  const blz::Mismatch& f = cx.first;
  const blz::EdgeList& g = cx.g;
  const std::uint64_t m = g.src.size();
  const auto edge = [&](std::uint64_t e) {
    return std::to_string(g.src[e]) + "->" + std::to_string(g.dst[e]) + "@" + std::to_string(g.ts[e]);
  };
  std::string what = exhaustive ? "graph " + std::to_string(cx.graph) + " of the bound (batch " +
                                      std::to_string(cx.case_id) + ")"
                                : "seed " + std::to_string(cx.graph);
  std::printf("  counterexample: %s, m = %llu, n = %u%s\n    block %s, window %d, path %s\n", what.c_str(),
              (unsigned long long)m, g.n,
              cx.alone ? "" : " (fails only inside its batch: no graph of the batch fails alone)", f.block.c_str(),
              f.window, f.path.c_str());
  std::string edges;
  nlohmann::json jedges = nlohmann::json::array();
  for (std::uint64_t e = 0; e < m; ++e) {
    edges += (e ? "  " : "") + std::to_string(e) + ": " + edge(e);
    jedges.push_back(edge(e));
  }
  if (m <= 64) std::printf("    edges (eid: u->v@t): %s\n", edges.c_str());
  nlohmann::json j = {{"case", cx.case_id}, {"graph", cx.graph}, {"alone", cx.alone}, {"n", g.n},
                      {"edges", jedges},    {"block", f.block}, {"window", f.window},  {"path", f.path}};
  const auto stop = [](std::uint64_t x) { return x == UINT64_MAX ? std::string("completed") : std::to_string(x); };
  if (f.got_stop != UINT64_MAX || f.want_stop != UINT64_MAX) {
    std::printf("    work limit: got %s, want %s (stop eids)\n", stop(f.got_stop).c_str(), stop(f.want_stop).c_str());
    j["got_stop"] = stop(f.got_stop);
    j["want_stop"] = stop(f.want_stop);
    return j;
  }
  const std::string trig = f.eid < m ? " (" + edge(f.eid) + ")" : "";
  j["trigger_eid"] = f.eid;
  if (f.block == "vstats") {
    const std::vector<std::string> names = blz::clean_vstats_column_names();
    std::printf("    trigger eid %llu%s, the columns that differ:\n", (unsigned long long)f.eid, trig.c_str());
    nlohmann::json cols = nlohmann::json::object();
    for (std::size_t k = 0; k < f.got_vstats.size(); ++k)
      if (std::memcmp(&f.got_vstats[k], &f.want_vstats[k], sizeof(double)) != 0) {
        std::printf("      %s: got %.17g, want %.17g\n", names[k].c_str(), f.got_vstats[k], f.want_vstats[k]);
        cols[names[k]] = {{"got", f.got_vstats[k]}, {"want", f.want_vstats[k]}};
      }
    j["columns"] = cols;
    return j;
  }
  const auto row = [](const std::vector<std::uint32_t>& r) {
    std::string t = "[";
    for (std::size_t k = 0; k < r.size(); ++k) t += (k ? " " : "") + std::to_string(r[k]);
    return t + "]";
  };
  std::printf("    trigger eid %llu%s: got %s, want %s\n", (unsigned long long)f.eid, trig.c_str(), row(f.got).c_str(),
              row(f.want).c_str());
  j["got"] = f.got;
  j["want"] = f.want;
  return j;
}

// AC1 on fuzz graphs (SPEC §13.3), or with --bound L2's bounded code check (SPEC v3 §4.3): a
// generated backend vs the profile's oracle (the clean oracle for a clean profile, SPEC v3 §3.2),
// or with --reference cpu vs the cpu backend's dynamic schedule (backend against backend).
// Exit code 0 if equal, 2 if not. cuda runs every graph once per light/heavy threshold. With
// --profile (SPEC v2 §10), the profile's blocks, patterns and k_max; without, every registered
// legacy pattern (v1).
int fuzz_diff(const std::string& seeds, const std::string& bound_opt, std::uint64_t batch_edges,
              const std::string& bins_opt, const std::string& backend, const std::string& heavy_opt,
              const std::string& cpu_heavy_opt, const std::string& profile_path, std::uint64_t work_limit, bool fuse,
              bool pack, const std::string& reference, const std::string& setting_opt, const std::string& report,
              const std::string& name) {
  const bool exhaustive = !bound_opt.empty();
  const blz::FuzzSetting setting = blz::parse_fuzz_setting(setting_opt);
  const blz::DiffSource src = [&] {
    if (exhaustive) return blz::DiffSource::exhaustive(blz::parse_exhaustive_bound(bound_opt), batch_edges);
    const auto [s0, s1] = parse_range(seeds);
    return blz::DiffSource::fuzz(s0, s1, setting);
  }();
  const bool hybrid = backend == "hybrid";
  const bool gpu = backend == "cuda" || hybrid;
  if ((fuse || pack) && backend != "cuda") throw std::runtime_error("--fuse and --pack are for --backend cuda");
  const blz::Reference ref = reference == "cpu" ? blz::Reference::cpu : blz::Reference::oracle;
  const blz::Profile prof = profile_path.empty() ? blz::registry_profile() : blz::load_profile(profile_path);
  if (ref == blz::Reference::oracle) blz::check_profile_backend(prof, blz::Backend::oracle);
  else blz::check_profile_backend(prof, blz::Backend::cpu);
  blz::check_profile_backend(prof, blz::parse_backend(backend));
  const auto bins = fuzz_bins(bins_opt, !profile_path.empty());
  const std::vector<std::uint32_t> thresholds = parse_thresholds(heavy_opt);
  const std::vector<std::uint32_t> cpu_thresholds = hybrid ? parse_thresholds(cpu_heavy_opt) : std::vector<std::uint32_t>{};
  auto t0 = Clock::now();
  const blz::FuzzDiff d =
      hybrid ? blz::fuzz_diff_hybrid(prof, src, bins, thresholds, cpu_thresholds, work_limit, {}, ref)
      : gpu  ? blz::fuzz_diff_cuda(prof, src, bins, thresholds, work_limit, {fuse, pack}, ref)
             : blz::fuzz_diff_cpu(prof, src, bins, work_limit, ref);
  const double seconds = secs(t0);
  // SPEC v3 R11: a block of a graph where the reference hit the work limit is inconclusive; a gate
  // counts the completed ones.
  nlohmann::json per_block = nlohmann::json::object(), completed = nlohmann::json::object(),
                 limited = nlohmann::json::object();
  std::uint64_t bad = 0;
  for (const blz::ProfileBlock& pb : prof.blocks) {
    const std::string bn(blz::block_name(pb.block));
    per_block[bn] = d.mismatched_rows[int(pb.block)];
    completed[bn] = d.cases - d.limited_graphs[int(pb.block)];
    limited[bn] = d.limited_graphs[int(pb.block)];
    bad += d.mismatched_rows[int(pb.block)];
  }
  if (prof.vstats) {  // SPEC v3 R8: never limited; compared on vstats_graphs graphs (none for --reference cpu on cpu)
    per_block["vstats"] = d.vstats_mismatched_rows;
    completed["vstats"] = d.vstats_graphs;
    limited["vstats"] = 0;
    bad += d.vstats_mismatched_rows;
  }
  std::vector<std::uint64_t> first_bad(d.bad_seeds.begin(),
                                       d.bad_seeds.begin() + std::ptrdiff_t(std::min<std::size_t>(20, d.bad_seeds.size())));
  const std::string ids = exhaustive ? "batches" : "seeds";
  nlohmann::json res = {{"backend", backend},
                        {"graphs", d.graphs},
                        {"edges", d.edges},
                        {"trigger_checks", d.trigger_checks},
                        {"row_checks", d.row_checks},
                        {"bins", bins.empty() ? nlohmann::json("profile") : nlohmann::json(bins)},
                        {"k_max", prof.k_max},
                        {"mismatched_rows_by_block", per_block},
                        {"mismatched_" + ids, d.bad_seeds.size()},
                        {"first_mismatched_" + ids, first_bad},
                        {"work_limit", work_limit},
                        {"completed_" + std::string(exhaustive ? "batches" : "graphs") + "_by_block", completed},
                        {"limited_" + std::string(exhaustive ? "batches" : "graphs") + "_by_block", limited},
                        {"limited_" + std::string(exhaustive ? "batches" : "graphs"), d.limited_seeds.size()},
                        {"first_limited_" + ids, std::vector<std::uint64_t>(d.limited_seeds.begin(),
                                                    d.limited_seeds.begin() + std::ptrdiff_t(std::min<std::size_t>(
                                                                                  20, d.limited_seeds.size())))},
                        {"seconds", seconds},
                        {"threads", omp_get_max_threads()},
                        {"git", git_info()}};
  if (exhaustive) {
    const blz::ExhaustiveBound& b = src.graphs()->bound();
    res["bound"] = {{"m_max", b.m_max}, {"n_max", b.n_max}, {"t", b.t}, {"windows", "1.." + std::to_string(b.t)}};
    res["batches"] = d.cases;
    res["batch_edges"] = src.batch_edges();
  } else {
    res["seeds"] = seeds;
    res["setting"] = blz::fuzz_setting_name(setting);
  }
  if (!profile_path.empty()) res["profile"] = profile_ref(prof);
  if (ref != blz::Reference::oracle) res["reference"] = {{"backend", "cpu"}, {"schedule", "dynamic"}};
  if (gpu) {
    nlohmann::json hs = nlohmann::json::array();
    for (std::uint32_t h : thresholds) hs.push_back(threshold_name(h));
    res["heavy_thresholds"] = hs;
    if (!hybrid) {  // SPEC v2 §8 S7
      res["fuse"] = fuse;
      res["pack"] = pack;
    }
    if (hybrid) {
      nlohmann::json cs = nlohmann::json::array();
      for (std::uint32_t h : cpu_thresholds) cs.push_back(threshold_name(h));
      res["cpu_thresholds"] = cs;
      res["hybrid_cpu"] = {{"schedule", "sorted"}, {"reach_prune", true}};
    }
    res["cuda"] = cuda_json();
  } else {
    res["schedules"] = ref == blz::Reference::oracle ? nlohmann::json{"dynamic", "sorted"} : nlohmann::json{"sorted"};
  }
  const char* vs = ref == blz::Reference::oracle ? "oracle" : "cpu (dynamic)";
  if (exhaustive) {
    const blz::ExhaustiveBound& b = src.graphs()->bound();
    const std::string batches = d.cases == d.graphs ? "each alone"
                                                    : "in " + std::to_string(d.cases) + " batches of up to " +
                                                          std::to_string(src.batch_edges()) + " edges";
    std::printf("fuzz-diff %s vs %s, bound %s (windows 1..%d), %s: %llu graphs %s, "
                "%llu edges, %llu trigger checks, %llu row checks, %llu mismatching rows (%.1f s, %d threads)\n",
                backend.c_str(), vs, blz::bound_name(b).c_str(), b.t, prof.name.c_str(), (unsigned long long)d.graphs,
                batches.c_str(), (unsigned long long)d.edges,
                (unsigned long long)d.trigger_checks, (unsigned long long)d.row_checks, (unsigned long long)bad, seconds,
                omp_get_max_threads());
  } else {
    std::printf("fuzz-diff %s vs %s, seeds %s (%s): %llu graphs, %llu edges, %llu mismatching rows (%.1f s)\n",
                backend.c_str(), vs, seeds.c_str(), blz::fuzz_setting_name(setting), (unsigned long long)d.graphs,
                (unsigned long long)d.edges, (unsigned long long)bad, seconds);
  }
  for (const blz::ProfileBlock& pb : prof.blocks)
    if (const std::uint64_t l = d.limited_graphs[int(pb.block)])
      std::printf("  %s: %llu %s completed, %llu reached the work limit of %llu bindings (inconclusive: their "
                  "stop eids compared, not their rows)\n",
                  std::string(blz::block_name(pb.block)).c_str(), (unsigned long long)(d.cases - l),
                  exhaustive ? "batches" : "graphs", (unsigned long long)l, (unsigned long long)work_limit);
  res["path_hits"] = path_hits(d.hits, d.nonzero_rows, prof, d.vstats_nonzero_rows, d.vstats_sides, !exhaustive);
  res["path_hits"]["optimised_paths"] = optimised_paths(d, backend, fuse, pack);
  if (d.counterexample) res["counterexample"] = print_counterexample(*d.counterexample, exhaustive);
  if (bad) std::printf("  mismatching rows by block: %s\n", per_block.dump().c_str());
  merge_report(report, name, res);
  return bad == 0 ? 0 : 2;
}

// T4 of SPEC v3 §4.1 in L2 (§4.3): the index primitives' contracts against naive definitions on
// every graph of a bound, both self-loop modes, both directions. Exit code 0 if every query agrees,
// 2 if not.
int index_contracts(const std::string& bound_opt, const std::string& report, const std::string& name) {
  const blz::ExhaustiveGraphs gs(blz::parse_exhaustive_bound(bound_opt));
  auto t0 = Clock::now();
  const blz::IndexContracts r = blz::check_index_contracts(gs);
  const double seconds = secs(t0);
  std::uint64_t checks = 0, bad = 0;
  std::string line;
  nlohmann::json by = nlohmann::json::object();
  for (std::size_t c = 0; c < blz::kContracts; ++c) {
    const char* cn = blz::contract_name(blz::Contract(c));
    checks += r.checks[c];
    bad += r.violations[c];
    line += std::string(c ? ", " : "") + cn + " " + std::to_string(r.checks[c]);
    by[cn] = {{"checks", r.checks[c]}, {"violations", r.violations[c]}};
  }
  std::printf("index-contracts, bound %s, self-loops kept and dropped, out- and in-lists: %llu graphs, %llu queries, "
              "%llu violations (%.1f s, %d threads)\n  queries: %s\n",
              blz::bound_name(gs.bound()).c_str(), (unsigned long long)r.graphs, (unsigned long long)checks,
              (unsigned long long)bad, seconds, omp_get_max_threads(), line.c_str());
  nlohmann::json res = {{"bound", blz::bound_name(gs.bound())}, {"graphs", r.graphs}, {"queries", checks},
                        {"violations", bad}, {"by_contract", by}, {"seconds", seconds},
                        {"threads", omp_get_max_threads()}, {"git", git_info()}};
  if (bad) {
    std::printf("  first violation: %s\n", r.first.c_str());
    res["first_violation"] = {{"graph", r.first_graph}, {"text", r.first}};
  }
  merge_report(report, name, res);
  return bad == 0 ? 0 : 2;
}

// SPEC v2 §3.4: the metamorphic tests on fuzz graphs, for one backend. Exit code 0 if every row
// is unchanged, 2 if not.
int metamorphic(const std::string& seeds, const std::string& bins_opt, const std::string& backend,
                const std::string& heavy_opt, const std::string& cpu_heavy_opt, const std::string& profile_path,
                std::uint64_t work_limit, const std::string& setting_opt, const std::string& report,
                const std::string& name) {
  const auto [s0, s1] = parse_range(seeds);
  const blz::FuzzSetting setting = blz::parse_fuzz_setting(setting_opt);
  const blz::Backend be = blz::parse_backend(backend);
  const blz::Profile prof = profile_path.empty() ? blz::registry_profile() : blz::load_profile(profile_path);
  const auto bins = fuzz_bins(bins_opt, !profile_path.empty());
  const std::vector<std::uint32_t> thresholds = parse_thresholds(heavy_opt);
  const std::vector<std::uint32_t> cpu_thresholds =
      be == blz::Backend::hybrid ? parse_thresholds(cpu_heavy_opt) : std::vector<std::uint32_t>{};
  auto t0 = Clock::now();
  const blz::MetamorphicResult r =
      blz::metamorphic(prof, be, s0, s1, bins, thresholds, cpu_thresholds, work_limit, setting);
  std::uint64_t bad = 0;
  nlohmann::json tests = nlohmann::json::object();
  for (blz::MetaTest t : blz::kMetaTests) {
    nlohmann::json per_block = nlohmann::json::object();
    std::uint64_t tb = 0;
    for (const blz::ProfileBlock& pb : prof.blocks) {
      per_block[std::string(blz::block_name(pb.block))] = r.mismatched_rows[int(t)][int(pb.block)];
      tb += r.mismatched_rows[int(t)][int(pb.block)];
    }
    if (prof.vstats) {
      per_block["vstats"] = r.vstats_mismatched_rows[int(t)];
      tb += r.vstats_mismatched_rows[int(t)];
    }
    bad += tb;
    tests[blz::meta_test_name(t)] = {{"applies", blz::meta_test_applies(t, prof, be)},
                                     {"rows_compared", r.rows_compared[int(t)]},
                                     {"mismatched_rows", tb},
                                     {"mismatched_rows_by_block", per_block}};
  }
  // SPEC v3 R11: a block that reached the work limit on a graph is inconclusive there.
  nlohmann::json completed = nlohmann::json::object(), limited = nlohmann::json::object();
  for (const blz::ProfileBlock& pb : prof.blocks) {
    completed[std::string(blz::block_name(pb.block))] = r.graphs - r.limited_graphs[int(pb.block)];
    limited[std::string(blz::block_name(pb.block))] = r.limited_graphs[int(pb.block)];
  }
  if (prof.vstats) {  // SPEC v3 R8: never limited
    completed["vstats"] = r.graphs;
    limited["vstats"] = 0;
  }
  std::vector<std::uint64_t> first_bad(r.bad_seeds.begin(),
                                       r.bad_seeds.begin() + std::ptrdiff_t(std::min<std::size_t>(20, r.bad_seeds.size())));
  nlohmann::json res = {{"backend", backend},
                        {"profile", profile_path.empty() ? nlohmann::json(nullptr) : profile_ref(prof)},
                        {"seeds", seeds},
                        {"setting", blz::fuzz_setting_name(setting)},
                        {"graphs", r.graphs},
                        {"edges", r.edges},
                        {"bins", bins.empty() ? nlohmann::json("profile") : nlohmann::json(bins)},
                        {"k_max", prof.k_max},
                        {"tests", tests},
                        {"work_limit", work_limit},
                        {"completed_graphs_by_block", completed},
                        {"limited_graphs_by_block", limited},
                        {"mismatched_seeds", r.bad_seeds.size()},
                        {"first_mismatched_seeds", first_bad},
                        {"seconds", secs(t0)},
                        {"git", git_info()}};
  if (be == blz::Backend::cuda || be == blz::Backend::hybrid) {
    nlohmann::json hs = nlohmann::json::array();
    for (std::uint32_t h : thresholds) hs.push_back(threshold_name(h));
    res["heavy_thresholds"] = hs;
    if (be == blz::Backend::hybrid) {
      nlohmann::json cs = nlohmann::json::array();
      for (std::uint32_t h : cpu_thresholds) cs.push_back(threshold_name(h));
      res["cpu_thresholds"] = cs;
      res["hybrid_cpu"] = {{"schedule", "sorted"}, {"reach_prune", true}};
    }
    res["cuda"] = cuda_json();
  }
  std::printf("metamorphic %s, %s, seeds %s (%s): %llu graphs, %llu edges, %llu mismatching rows (%.1f s)\n",
              backend.c_str(), prof.name.c_str(), seeds.c_str(), blz::fuzz_setting_name(setting),
              (unsigned long long)r.graphs, (unsigned long long)r.edges, (unsigned long long)bad, secs(t0));
  for (const blz::ProfileBlock& pb : prof.blocks)
    if (const std::uint64_t l = r.limited_graphs[int(pb.block)])
      std::printf("  %s: %llu graphs completed, %llu reached the work limit of %llu bindings (inconclusive: not "
                  "compared)\n",
                  std::string(blz::block_name(pb.block)).c_str(), (unsigned long long)(r.graphs - l),
                  (unsigned long long)l, (unsigned long long)work_limit);
  res["path_hits"] = path_hits(r.hits, r.nonzero_rows, prof, r.vstats_nonzero_rows, r.vstats_sides);
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
// attempt's edges alone), activation (the full mining in --features) and detection (--pred, u8[m]
// codes of runtime/coverage.hpp).
int coverage(const std::string& graph, const std::string& gt_path, const std::string& profile_path,
             const std::string& features, const std::string& pred_path, const std::string& out) {
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

  const nlohmann::json fm = nlohmann::json::parse(std::ifstream(features + "/features.json"));
  if (fm.at("blzg_sha256") != blzg_sha) throw std::runtime_error(features + " was mined from another .blzg");
  if (fm.at("profile").at("id") != prof.id())
    throw std::runtime_error(features + ": profile " + fm.at("profile").at("id").get<std::string>() + ", not " +
                             prof.id() + " (" + profile_path + ")");
  const auto& tr = fm.at("triggers");
  if (tr.value("sampled", false) || tr.at("first") != 0 || tr.at("last") != m || fm.value("edges", m) != m)
    throw std::runtime_error(features + ": not a mining of every edge");
  nlohmann::ordered_json jfeat = {
      {"dir", features}, {"source", "blz"}, {"profile_id", prof.id()}, {"backend", fm.at("backend").at("name")}};
  std::deque<blz::NpyU32View> views;  // not movable: a deque never relocates them
  nlohmann::ordered_json jfiles;
  for (const blz::ProfileBlock& pb : prof.blocks) {
    const std::string b(blz::block_name(pb.block));
    const auto& fb = fm.at("blocks").at(b);
    views.emplace_back(features + "/" + fb.at("file").get<std::string>());
    if (views.back().rows() != m || views.back().cols() != blz::block_cols(pb.params))
      throw std::runtime_error(features + ": block " + b + " is not m x bins");
    jfiles[b] = fb.at("output_fnv1a");
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

  const nlohmann::ordered_json jelig = {{"source", "oracle"}};

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
  mn->add_option("--backend", backend)->check(CLI::IsMember({"oracle", "cpu", "cuda", "hybrid"}));
  mn->add_option("--threads", threads);
  mn->add_option("--schedule", schedule, "cpu backend: dynamic | sorted")->check(CLI::IsMember({"dynamic", "sorted"}));
  mn->add_option("--heavy-threshold", heavy, "cuda backend: work estimate from which a trigger is heavy");
  std::string cpu_heavy_opt = std::to_string(blz::kDefaultCpuThreshold);
  bool reach_prune = false;
  mn->add_option("--cpu-threshold", cpu_heavy_opt,
                 "hybrid backend: work estimate from which a trigger runs on the CPU ('all': none; SPEC v2 §8 S6)");
  mn->add_flag("--reach-prune", reach_prune, "cpu and hybrid backends: reachability pruning (SPEC v2 §8 S6), same rows");
  mn->add_option("--blocks", blocks, "comma-separated, vstats included; default: every block of the profile");
  mn->add_option("--triggers", triggers, "all | train | test | fit | val | range:a:b | sample:<pct>:<seed>");
  mn->add_option("--edges", edges, "mine on the stream prefix [0, k) only: later edges do not exist (SPEC v2 §4.4 R5)");
  std::uint64_t work_limit = blz::kDefaultWorkLimit;
  mn->add_option("--work-limit", work_limit, "lcyc: bindings per trigger; past it the run stops with an error (SPEC v2 §8 S3)");
  bool fuse = false, pack = false;
  mn->add_flag("--fuse", fuse, "cuda backend: mine each compiled fusion's blocks in one kernel (SPEC v2 §8 S7), same rows");
  mn->add_flag("--pack", pack, "cuda backend: copy the rows as uint8 or uint16 where they fit (SPEC v2 §8 S7), same rows");
  std::string quirks;
  mn->add_option("--quirks", quirks,
                 "oracle backend, clean profile: GFP / v2 behaviours to switch back on for E16, e.g. Q1,Q3 or all "
                 "(SPEC v3 §3.1, §3.6)");
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
  bool fz_vstats = false;
  fz->add_flag("--vstats", fz_vstats, "also amounts.npy and vstats.npy (SPEC v2 §8 S2): window = the graph's");
  bool fz_lcyc = false;
  fz->add_flag("--lcyc", fz_lcyc, "also lcyc.npy (SPEC v2 §8 S3): length <= the compiled K_MAX; limited graphs' rows UINT32_MAX");
  fz->add_option("--work-limit", work_limit, "--lcyc: bindings per trigger");
  std::string setting = "dense";
  const auto setting_check = CLI::IsMember({"dense", "sparse"});
  const char* setting_help = "fuzz graphs: dense (v1) or sparse (time spans of many windows, edges on the window "
                             "bounds, several components; SPEC v3 §4.4)";
  fz->add_option("--setting", setting, setting_help)->check(setting_check);
  fz->add_option("--out", out)->required();

  std::int32_t vs_window = 0;
  auto* vsc = app.add_subcommand("vstats", "Vertex statistics (SPEC v2 §8 S2) -> <out>/vstats.npy (float64) + features.json");
  vsc->add_option("--graph", graph)->required();
  vsc->add_option("--window", vs_window, "GFP's time_window: the largest pattern window of its call (F1: 86400)")->required();
  vsc->add_option("--backend", backend, "cpu | cuda")->check(CLI::IsMember({"cpu", "cuda"}));
  vsc->add_option("--threads", threads);
  vsc->add_option("--triggers", triggers, "all | train | test | fit | val | range:a:b | sample:<pct>:<seed>");
  vsc->add_option("--out", out)->required();

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
  bn->add_option("--backend", backend)->check(CLI::IsMember({"cpu", "cuda", "hybrid"}))->required();
  bn->add_option("--threads", threads_list, "comma-separated thread counts");
  bn->add_option("--reps", reps, "timed runs after one warm-up");
  bn->add_option("--schedule", schedules, "cpu: comma-separated: dynamic, sorted");
  bn->add_option("--heavy", heavy_list, "cuda: comma-separated light/heavy thresholds; 'all' = every trigger light "
                                        "(default 32)");
  bn->add_option("--blocks", blocks, "comma-separated; default: every block in the config");
  std::string bn_cpu_heavy = "0,64,128,256,512,1024,all", bn_prune = "off,on";
  bn->add_option("--cpu-threshold", bn_cpu_heavy, "hybrid: comma-separated CPU thresholds (SPEC v2 §8 S6)");
  bn->add_option("--prune", bn_prune, "hybrid: reachability pruning on the CPU side: off, on or off,on");
  bn->add_flag("--fuse", fuse, "cuda: time each compiled fusion's blocks as one unit (SPEC v2 §8 S7)");
  bn->add_flag("--pack", pack, "cuda: packed D2H (SPEC v2 §8 S7)");
  bn->add_option("--out", out, "merge the result into this JSON file")->required();
  bn->add_option("--name", bench_name, "key (JSON pointer path) in --out; default: the backend");

  std::string fd_backend = "cpu", fd_heavy = "0,4,all", fd_bins;
  auto* fd = app.add_subcommand("fuzz-diff", "AC1: a generated backend vs the oracle on fuzz graphs (SPEC §13.3), or "
                                               "with --bound on every graph within a bound (L2, SPEC v3 §4.3)");
  std::string fd_bound;
  std::uint64_t batch_edges = 16384;
  auto* fd_seeds = fd->add_option("--seeds", seeds, "a:b");
  auto* fd_bound_opt = fd->add_option("--bound", fd_bound,
                                      "m=<edges>,n=<nodes>,t=<timestamps>: every edge list of 1..m edges over at most n "
                                      "nodes with ts in [0, t), once per node relabelling, at every window 1..t");
  fd_seeds->excludes(fd_bound_opt);
  fd->add_option("--batch-edges", batch_edges,
                 "--bound: graphs per disjoint-union batch, as edges (1: each graph alone)")->needs(fd_bound_opt);
  fd->add_option("--bins", fd_bins, "default: the profile's bins per block, or 2..33 without --profile");
  fd->add_option("--backend", fd_backend)->check(CLI::IsMember({"cpu", "cuda", "hybrid"}));
  fd->add_option("--heavy", fd_heavy, "cuda, hybrid: light/heavy thresholds, each run on every graph");
  std::string fd_cpu_heavy = "0,4,all";
  fd->add_option("--cpu-threshold", fd_cpu_heavy, "hybrid: CPU thresholds, each run with every --heavy (SPEC v2 §8 S6)");
  fd->add_option("--profile", profile, "blocks, patterns and k_max (SPEC v2 §10); windows come from the graphs");
  fd->add_option("--work-limit", work_limit, "lcyc: bindings per trigger; a graph past it is reported, its stop eid compared");
  fd->add_flag("--fuse", fuse, "cuda: run each compiled fusion's blocks in one kernel (SPEC v2 §8 S7)");
  fd->add_flag("--pack", pack, "cuda: packed D2H (SPEC v2 §8 S7)");
  std::string fd_reference = "oracle";
  fd->add_option("--reference", fd_reference,
                 "the rows compared against: the profile's oracle (the clean oracle for a clean profile), or the cpu "
                 "backend's dynamic schedule (backend against backend; --backend cpu then checks the sorted schedule)")
      ->check(CLI::IsMember({"oracle", "cpu"}));
  fd->add_option("--setting", setting, setting_help)->check(setting_check);
  fd->add_option("--report", report, "merge the summary into this JSON file");
  fd->add_option("--name", name, "key (JSON pointer path) in --report")->needs("--report");

  std::string ic_bound;
  auto* ic = app.add_subcommand("index-contracts", "T4 in L2 (SPEC v3 §4.1, §4.3): the index primitives against naive "
                                                   "definitions on every graph within a bound");
  ic->add_option("--bound", ic_bound, "m=<edges>,n=<nodes>,t=<timestamps>, as fuzz-diff --bound")->required();
  ic->add_option("--report", report, "merge the summary into this JSON file");
  ic->add_option("--name", name, "key (JSON pointer path) in --report")->needs("--report");

  std::string mt_backend = "cpu";
  auto* mt = app.add_subcommand("metamorphic", "SPEC v2 §3.4: prefix closure, relabeling, time shift, tcyc and tchain visibility invariance, order relaxation on fuzz graphs");
  mt->add_option("--seeds", seeds, "a:b")->required();
  mt->add_option("--bins", fd_bins, "default: the profile's bins per block, or 2..33 without --profile");
  mt->add_option("--backend", mt_backend)->check(CLI::IsMember({"oracle", "cpu", "cuda", "hybrid"}));
  mt->add_option("--heavy", fd_heavy, "cuda, hybrid: light/heavy thresholds, each run on every graph");
  mt->add_option("--cpu-threshold", fd_cpu_heavy, "hybrid: CPU thresholds, each run with every --heavy (SPEC v2 §8 S6)");
  mt->add_option("--profile", profile, "default: every registered pattern (v1)");
  mt->add_option("--work-limit", work_limit, "lcyc: bindings per trigger; a graph past it is reported, not compared");
  mt->add_option("--setting", setting, setting_help)->check(setting_check);
  mt->add_option("--report", report, "merge the summary into this JSON file");
  mt->add_option("--name", name, "key (JSON pointer path) in --report")->needs("--report");

  std::string gt_csv, gt_patterns;
  auto* gj = app.add_subcommand("gt-join", "SPEC v2 §3.5: join Patterns.txt to the stream's eids");
  gj->add_option("--graph", graph)->required();
  gj->add_option("--csv", gt_csv)->required();
  gj->add_option("--patterns", gt_patterns)->required();
  gj->add_option("--out", out)->required();

  std::string gt_path = "results/gt_join.json", pred_path;
  auto* cv = app.add_subcommand("coverage", "SPEC v2 §5 (E9): eligibility, activation and detection per listed attempt");
  cv->add_option("--graph", graph)->required();
  cv->add_option("--gt", gt_path, "blz gt-join output");
  cv->add_option("--profile", profile, "the profile whose blocks are covered")->required();
  cv->add_option("--features", features, "the full mining under the profile (blz mine)")->required();
  cv->add_option("--pred", pred_path, "u8[m] prediction codes: 0, 1, 255 = not scored");
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
    if (*mn) {
      // The hybrid's CPU side runs the sorted schedule unless --schedule says otherwise (SPEC v2 §8 S6).
      if (backend == "hybrid" && !*mn->get_option("--schedule")) schedule = "sorted";
      const std::vector<std::uint32_t> cpu_h = parse_thresholds(cpu_heavy_opt);
      if (cpu_h.size() != 1) throw std::runtime_error("mine --cpu-threshold takes one value");
      return mine(graph, profile, backend, threads, schedule, heavy, cpu_h[0], reach_prune, blocks, triggers, edges,
                  work_limit, fuse, pack, quirks, out);
    }
    if (*bn) {
      if (bench_name.empty()) bench_name = backend;
      if ((fuse || pack) && backend != "cuda") throw std::runtime_error("--fuse and --pack are for --backend cuda");
      if (backend == "hybrid") {
        const std::string nt = *bn->get_option("--threads") ? threads_list : "0";
        if (nt.find(',') != std::string::npos) throw std::runtime_error("bench --backend hybrid takes one --threads value");
        if (schedules.find(',') != std::string::npos) throw std::runtime_error("bench --backend hybrid takes one --schedule");
        if (!*bn->get_option("--schedule")) schedules = "sorted";
        return bench_hybrid(graph, profile, reps, std::stoi(nt), schedules,
                            heavy_list.empty() ? std::to_string(blz::kDefaultHeavyThreshold) : heavy_list, bn_cpu_heavy,
                            bn_prune, blocks, out, bench_name);
      }
      if (backend == "cuda") {
        if (*bn->get_option("--threads") || *bn->get_option("--schedule"))
          throw std::runtime_error("bench --backend cuda takes --heavy, not --threads or --schedule");
        return bench_cuda(graph, profile, reps, heavy_list.empty() ? std::to_string(blz::kDefaultHeavyThreshold) : heavy_list,
                          blocks, fuse, pack, out, bench_name);
      }
      if (!heavy_list.empty()) throw std::runtime_error("--heavy is for --backend cuda");
      return bench(graph, profile, backend, threads_list, reps, schedules, blocks, out, bench_name);
    }
    if (*fd) {
      if (!report.empty() && name.empty()) throw std::runtime_error("--report needs --name");
      if (seeds.empty() == fd_bound.empty()) throw std::runtime_error("fuzz-diff needs --seeds or --bound");
      if (!fd_bound.empty() && *fd->get_option("--setting"))
        throw std::runtime_error("--setting is for fuzz graphs (--seeds), not --bound");
      return fuzz_diff(seeds, fd_bound, batch_edges, fd_bins, fd_backend, fd_heavy, fd_cpu_heavy, profile, work_limit,
                       fuse, pack, fd_reference, setting, report, name);
    }
    if (*ic) {
      if (!report.empty() && name.empty()) throw std::runtime_error("--report needs --name");
      return index_contracts(ic_bound, report, name);
    }
    if (*mt) {
      if (!report.empty() && name.empty()) throw std::runtime_error("--report needs --name");
      return metamorphic(seeds, fd_bins, mt_backend, fd_heavy, fd_cpu_heavy, profile, work_limit, setting, report,
                         name);
    }
    if (*gj) return gt_join(graph, gt_csv, gt_patterns, out);
    if (*cv) {
      if (threads > 0) omp_set_num_threads(threads);
      return coverage(graph, gt_path, profile, features, pred_path, out);
    }
    if (*cmp) {
      if (!report.empty() && name.empty()) throw std::runtime_error("--report needs --name");
      return compare(a_npy, b_npy, graph, tw, first_eid, report, name);
    }
    if (*fz) return fuzz(seeds, bins, k_max, fz_vstats, fz_lcyc, work_limit, setting, out);
    if (*vsc) return vstats(graph, vs_window, backend == "oracle" ? "cpu" : backend, threads, triggers, out);
    if (*tr) return train(graph, features, set, xgb, out, name.empty() ? set : name, threads);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 1;
}

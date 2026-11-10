// blz — BlazingAML runtime CLI (SPEC §14). P0 commands: prepare, export-edges, stats.
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>

#include "common/mmap_file.hpp"
#include "common/npy.hpp"
#include "common/sha256.hpp"
#include "graph/prepare.hpp"
#include "graph/stats.hpp"
#include "io/blzg.hpp"
#include "io/csv.hpp"

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

  CLI11_PARSE(app, argc, argv);
  try {
    if (*prep) return prepare(csv, out);
    if (*exp) return export_edges(graph, out);
    if (*st) return stats(graph, out);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 1;
}

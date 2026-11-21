// blzc — the BlazingAML pattern compiler (SPEC §7, §14).
//
//   blzc <spec.yaml>... --out d               generate code: d/<stem>.hpp, d/<stem>_cpu.cpp and
//                                             d/<stem>_cuda.cu for each spec, and d/registry.cpp
//                                             (files whose text is unchanged are not rewritten)
//   blzc <spec.yaml>... --snapshots d         compare the generated code with the snapshots in d
//                                             (exit 1 on any difference) ...
//                       [--update-snapshots]  ... or rewrite them (SPEC §13.4)
//   blzc <spec.yaml>... --check-only          diagnostics only
//   blzc <spec.yaml>... --dump-ir [--out d]   the planned IR (stdout, or d/<pattern>.ir.txt)
//   blzc <spec.yaml>    --expand  [--out d]   the spec after repeat expansion
//   --const NAME=VALUE overrides a const of the spec (e.g. --const K_MAX=5).
//   --disable OPT[,OPT...] plans and generates without these optimizations (SPEC v2 §7.2): find_start,
//   distinct_first, degree_order, count_only, sweep. The E11 ablation variants; CPU code only.
//   --fuse P1,P2[,...] (repeatable) also generates d/fused_<P1>__<P2>_cuda.cu, one CUDA kernel body
//   that counts the patterns with one scan of their trigger list (SPEC v2 §8 S7), and lists it in
//   the registry.
#include <algorithm>
#include <charconv>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>

#include <CLI/CLI.hpp>

#include "codegen/codegen.hpp"
#include "ir/ir.hpp"

namespace {

void write_file(const std::string& path, const std::string& text) {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("cannot write " + path);
  out << text;
  if (!out) throw std::runtime_error("write failed: " + path);
}

// Leaves an identical file untouched, so the build does not recompile it.
void write_if_changed(const std::string& path, const std::string& text) {
  if (std::filesystem::exists(path) && blz::ir::read_text_file(path) == text) return;
  write_file(path, text);
}

}  // namespace

int main(int argc, char** argv) {
  CLI::App app{"blzc: pattern compiler"};
  std::vector<std::string> specs;
  std::string out_dir, snap_dir;
  std::vector<std::string> const_args;
  std::string disable;
  std::vector<std::string> fuse_args;
  bool check_only = false, dump_ir = false, expand = false, update = false;
  app.add_option("specs", specs, "pattern specs (patterns/*.yaml)")->required()->check(CLI::ExistingFile);
  app.add_option("--out", out_dir, "output directory");
  auto* s = app.add_option("--snapshots", snap_dir, "compare the generated code with this directory");
  app.add_flag("--update-snapshots", update, "with --snapshots: rewrite the snapshots")->needs(s);
  app.add_option("--const", const_args, "override a const: NAME=VALUE");
  app.add_option("--disable", disable,
                 "switch optimizations off (comma-separated): find_start, distinct_first, degree_order, count_only, sweep");
  app.add_option("--fuse", fuse_args, "fuse these patterns into one CUDA kernel (comma-separated; repeatable)")
      ->expected(1)
      ->allow_extra_args(false)  // one group per --fuse: the spec paths after it stay positional
      ->multi_option_policy(CLI::MultiOptionPolicy::TakeAll);
  auto* c = app.add_flag("--check-only", check_only, "run the front end and report diagnostics");
  auto* d = app.add_flag("--dump-ir", dump_ir, "write the planned IR");
  auto* e = app.add_flag("--expand", expand, "write the spec after repeat expansion");
  c->excludes(d)->excludes(e)->excludes(s);
  d->excludes(e)->excludes(s);
  e->excludes(s);
  CLI11_PARSE(app, argc, argv);

  blz::ir::IntEnv overrides;
  for (const std::string& a : const_args) {
    const std::size_t eq = a.find('=');
    std::int64_t v = 0;
    const char* b = a.data() + (eq == std::string::npos ? 0 : eq + 1);
    const auto [p, ec] = std::from_chars(b, a.data() + a.size(), v);
    if (eq == std::string::npos || eq == 0 || ec != std::errc() || p != a.data() + a.size()) {
      std::fprintf(stderr, "blzc: --const expects NAME=INTEGER, got '%s'\n", a.c_str());
      return 2;
    }
    overrides.emplace_back(a.substr(0, eq), v);
  }
  blz::ir::Opts opts;
  try {
    opts = blz::ir::parse_disable(disable);
  } catch (const std::invalid_argument& err) {
    std::fprintf(stderr, "blzc: --disable: %s\n", err.what());
    return 2;
  }
  const bool codegen = !check_only && !dump_ir && !expand;
  if (codegen && out_dir.empty() == snap_dir.empty()) {
    std::fprintf(stderr, "blzc: code generation needs exactly one of --out and --snapshots\n");
    return 2;
  }
  if (!codegen && !fuse_args.empty()) {
    std::fprintf(stderr, "blzc: --fuse is checked when generating code (--out or --snapshots)\n");
    return 2;
  }
  if (!out_dir.empty()) std::filesystem::create_directories(out_dir);

  int failures = 0;
  std::vector<blz::ir::Pattern> patterns;
  std::vector<std::string> stems;
  for (const std::string& path : specs) {
    try {
      if (expand) {
        const std::string text = blz::dsl::emit_yaml(blz::ir::expand_text(blz::ir::read_text_file(path), path, overrides));
        if (out_dir.empty()) std::cout << text;
        else write_file(out_dir + "/" + std::filesystem::path(path).stem().string() + ".expanded.yaml", text);
        continue;
      }
      blz::ir::Pattern p = blz::ir::compile_file(path, overrides, opts);
      if (dump_ir) {
        const std::string text = blz::ir::dump(p);
        if (out_dir.empty()) std::cout << text;
        else write_file(out_dir + "/" + p.name + ".ir.txt", text);
      } else if (check_only) {
        std::fprintf(stderr, "%s: ok: pattern %s, block %s, %d binding level(s)\n", path.c_str(),
                     p.name.c_str(), p.block.c_str(), p.plan.depth);
      } else {
        stems.push_back(std::filesystem::path(path).stem().string());
        patterns.push_back(std::move(p));
      }
    } catch (const blz::dsl::CompileError& err) {
      std::fprintf(stderr, "%s\n", err.what());
      ++failures;
    } catch (const std::exception& err) {
      std::fprintf(stderr, "%s: %s\n", path.c_str(), err.what());
      ++failures;
    }
  }
  if (!codegen || failures > 0) return failures == 0 ? 0 : 1;

  // Code generation (SPEC §7.1 steps 8–9).
  std::vector<blz::codegen::GeneratedFile> files;
  try {
    std::set<std::string> seen_stems;
    for (std::size_t i = 0; i < patterns.size(); ++i) {
      if (!seen_stems.insert(stems[i]).second)
        throw std::runtime_error("two specs have the file name " + stems[i] + ".yaml; generated files are named after it");
      for (auto& f : blz::codegen::emit_cpu(patterns[i], stems[i])) files.push_back(std::move(f));
      files.push_back(blz::codegen::emit_cuda(patterns[i], stems[i]));
    }
    std::vector<std::vector<std::string>> groups;
    for (const std::string& a : fuse_args) {
      std::vector<std::string>& g = groups.emplace_back();
      for (std::size_t q = 0; q <= a.size();) {
        const std::size_t r = std::min(a.find(',', q), a.size());
        g.push_back(a.substr(q, r - q));
        q = r + 1;
      }
    }
    const std::vector<blz::codegen::Fusion> fusions = blz::codegen::plan_fusions(patterns, groups);
    for (const auto& f : fusions) files.push_back(blz::codegen::emit_fused_cuda(f, patterns));
    files.push_back(blz::codegen::emit_registry(patterns, stems, fusions));
  } catch (const blz::dsl::CompileError& err) {
    std::fprintf(stderr, "%s\n", err.what());
    return 1;
  } catch (const std::exception& err) {
    std::fprintf(stderr, "blzc: %s\n", err.what());
    return 1;
  }

  if (!out_dir.empty()) {
    for (const auto& f : files) write_if_changed(out_dir + "/" + f.name, f.text);
    return 0;
  }
  // Snapshots: every generated file must equal its snapshot, and the directory must hold no other
  // file. --update-snapshots rewrites the generated ones and only reports the others.
  namespace fs = std::filesystem;
  if (update) {
    fs::create_directories(snap_dir);
    std::set<std::string> names;
    for (const auto& f : files) names.insert(f.name);
    for (const auto& entry : fs::directory_iterator(snap_dir))
      if (!names.count(entry.path().filename().string()))
        std::fprintf(stderr, "stale snapshot (not generated; remove it by hand): %s\n", entry.path().c_str());
    for (const auto& f : files) write_if_changed(snap_dir + "/" + f.name, f.text);
    std::fprintf(stderr, "updated %zu snapshot(s) in %s\n", files.size(), snap_dir.c_str());
    return 0;
  }
  int diffs = 0;
  std::set<std::string> names;
  for (const auto& f : files) {
    names.insert(f.name);
    const std::string path = snap_dir + "/" + f.name;
    if (!fs::exists(path)) {
      std::fprintf(stderr, "snapshot missing: %s\n", path.c_str());
      ++diffs;
    } else if (blz::ir::read_text_file(path) != f.text) {
      std::fprintf(stderr, "snapshot differs: %s\n", path.c_str());
      ++diffs;
    }
  }
  if (fs::exists(snap_dir))
    for (const auto& entry : fs::directory_iterator(snap_dir))
      if (!names.count(entry.path().filename().string())) {
        std::fprintf(stderr, "stale snapshot: %s\n", entry.path().c_str());
        ++diffs;
      }
  if (diffs > 0) {
    std::fprintf(stderr,
                 "%d snapshot difference(s); review the generated code, then run the same command "
                 "with --update-snapshots\n",
                 diffs);
    return 1;
  }
  std::fprintf(stderr, "%zu generated file(s) match %s\n", files.size(), snap_dir.c_str());
  return 0;
}

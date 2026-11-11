// blzc — the BlazingAML pattern compiler (SPEC §7, §14). P3 provides the front end: parse,
// expand, resolve, type-check, classify and plan. Code generation arrives in P4.
//
//   blzc <spec.yaml>... --check-only          diagnostics only
//   blzc <spec.yaml>... --dump-ir [--out d]   the planned IR (stdout, or d/<pattern>.ir.txt)
//   blzc <spec.yaml>    --expand  [--out d]   the spec after repeat expansion
//   --const NAME=VALUE overrides a const of the spec (e.g. --const K_MAX=5).
#include <charconv>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>

#include <CLI/CLI.hpp>

#include "ir/ir.hpp"

namespace {

void write_file(const std::string& path, const std::string& text) {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("cannot write " + path);
  out << text;
  if (!out) throw std::runtime_error("write failed: " + path);
}

}  // namespace

int main(int argc, char** argv) {
  CLI::App app{"blzc: pattern compiler (front end)"};
  std::vector<std::string> specs;
  std::string out_dir;
  std::vector<std::string> const_args;
  bool check_only = false, dump_ir = false, expand = false;
  app.add_option("specs", specs, "pattern specs (patterns/*.yaml)")->required()->check(CLI::ExistingFile);
  app.add_option("--out", out_dir, "output directory");
  app.add_option("--const", const_args, "override a const: NAME=VALUE");
  auto* c = app.add_flag("--check-only", check_only, "run the front end and report diagnostics");
  auto* d = app.add_flag("--dump-ir", dump_ir, "write the planned IR");
  auto* e = app.add_flag("--expand", expand, "write the spec after repeat expansion");
  c->excludes(d)->excludes(e);
  d->excludes(e);
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
  if (!check_only && !dump_ir && !expand) {
    std::fprintf(stderr,
                 "blzc: code generation is not implemented yet (P4); use --check-only, --dump-ir or --expand\n");
    return 2;
  }
  if (!out_dir.empty()) std::filesystem::create_directories(out_dir);

  int failures = 0;
  for (const std::string& path : specs) {
    try {
      if (expand) {
        const std::string text = blz::dsl::emit_yaml(blz::ir::expand_text(blz::ir::read_text_file(path), path, overrides));
        if (out_dir.empty()) std::cout << text;
        else write_file(out_dir + "/" + std::filesystem::path(path).stem().string() + ".expanded.yaml", text);
        continue;
      }
      const blz::ir::Pattern p = blz::ir::compile_file(path, overrides);
      if (dump_ir) {
        const std::string text = blz::ir::dump(p);
        if (out_dir.empty()) std::cout << text;
        else write_file(out_dir + "/" + p.name + ".ir.txt", text);
      } else {
        std::fprintf(stderr, "%s: ok: pattern %s, block %s, %d binding level(s)\n", path.c_str(),
                     p.name.c_str(), p.block.c_str(), p.plan.depth);
      }
    } catch (const blz::dsl::CompileError& err) {
      std::fprintf(stderr, "%s\n", err.what());
      ++failures;
    } catch (const std::exception& err) {
      std::fprintf(stderr, "%s: %s\n", path.c_str(), err.what());
      ++failures;
    }
  }
  return failures == 0 ? 0 : 1;
}

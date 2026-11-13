// Code generation (SPEC §6.7): runtime visibility (SPEC v2 §6.1), and duplicate pattern names
// across specs (E102). The generated text itself is covered by the blzc_snapshots test.
#include <catch2/catch_test_macros.hpp>
#include <string>
#include <vector>

#include "codegen/codegen.hpp"

using namespace blz;

namespace {

const std::string kFan =
    "version: 1\n"
    "pattern: fan_in\n"
    "block: fan_in\n"
    "window: {span: delta_fan, lower: open}\n"
    "VIS"
    "trigger: {edge: e0, src: U, dst: V}\n"
    "constraints: [{skip_if: \"U == V\"}]\n"
    "stages:\n"
    "  - {id: in_nbrs, op: count, src: V.in, edge: e1, element: X, output: n,\n"
    "     constraints: [{skip_if: \"X == V\"}], emit: {size: n}}\n";

std::string with(const std::string& vis) { return std::string(kFan).replace(kFan.find("VIS"), 3, vis); }

}  // namespace

TEST_CASE("codegen: visibility is a runtime parameter; a spec's visibility: key only pins it (SPEC v2 §6.1)") {
  // The same text with or without the key: the generated code reads Params::ts_visibility.
  const auto body = [](const std::vector<codegen::GeneratedFile>& fs) {
    std::string h = fs[0].text;
    return h.substr(h.find("#pragma once"));
  };
  const auto none = codegen::emit_cpu(ir::compile_text(with(""), "a.yaml"), "a");
  for (const char* vis : {"visibility: stream\n", "visibility: timestamp\n"}) {
    const auto pinned = codegen::emit_cpu(ir::compile_text(with(vis), "a.yaml"), "a");
    CHECK(pinned.size() == 2);
    CHECK(body(pinned) == body(none));
  }
  CHECK(none[0].text.find("const std::uint32_t vis_e = p.ts_visibility ? kAnyEid : e;") != std::string::npos);
  CHECK(none[0].text.find("const std::int64_t vis_t = p.ts_visibility ? t : kNoUpper;") != std::string::npos);
  // The pin goes to the registry, which E403 checks against the profile (test_profile.cpp).
  const ir::Pattern ts = ir::compile_text(with("visibility: timestamp\n"), "a.yaml");
  const std::string reg = codegen::emit_registry({ts}, {"a"}).text;
  CHECK(reg.find("\"timestamp\", \"a.yaml:5:13\"") != std::string::npos);
}

TEST_CASE("codegen: two specs defining one pattern name are E102") {
  const ir::Pattern a = ir::compile_text(with(""), "a.yaml");
  const ir::Pattern b = ir::compile_text(with(""), "b.yaml");
  try {
    codegen::emit_registry({a, b}, {"a", "b"});
    FAIL("accepted");
  } catch (const dsl::CompileError& e) {
    CHECK(e.code() == "E102");
    CHECK(e.loc().file == "b.yaml");
    CHECK(e.loc().line == 2);
  }
}

TEST_CASE("codegen: blzc --disable variants are CPU-only and listed in the registry (SPEC v2 §7.2)") {
  const ir::Pattern full = ir::compile_text(with(""), "a.yaml");
  const ir::Pattern off = ir::compile_text(with(""), "a.yaml", {}, ir::parse_disable("find_start,sweep"));
  CHECK(codegen::emit_cuda(full, "a").text.find("#error") == std::string::npos);
  CHECK(codegen::emit_cuda(off, "a").text.find("#error \"blzc --disable variants are CPU-only") != std::string::npos);
  CHECK(codegen::emit_registry({full}, {"a"}).text.find("disabled_optimizations() { return {}; }") != std::string::npos);
  const std::string reg = codegen::emit_registry({off}, {"a"}).text;
  CHECK(reg.find("constexpr const char* kDisabled[] = {\"find_start\", \"sweep\"};") != std::string::npos);
  // The sweep plan replaces the OpenMP driver; --disable sweep restores v1's.
  CHECK(codegen::emit_cpu(full, "a")[1].text.find("sweep_count<true>(g, g.in, p.ts_visibility") != std::string::npos);
  CHECK(codegen::emit_cpu(off, "a")[1].text.find("#pragma omp parallel for schedule(dynamic, chunk)") != std::string::npos);
}

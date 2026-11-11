// Code generation errors (SPEC §6.7): what the v1 emitter refuses (E501) and duplicate pattern
// names across specs (E102). The generated text itself is covered by the blzc_snapshots test.
#include <catch2/catch_test_macros.hpp>
#include <string>

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

TEST_CASE("codegen: timestamp visibility is E501 at its value") {
  const ir::Pattern p = ir::compile_text(with("visibility: timestamp\n"), "a.yaml");
  try {
    codegen::emit_cpu(p, "a");
    FAIL("accepted");
  } catch (const dsl::CompileError& e) {
    CHECK(e.code() == "E501");
    CHECK(e.loc().line == 5);
    CHECK(e.loc().col == 13);
  }
  CHECK(codegen::emit_cpu(ir::compile_text(with("visibility: stream\n"), "a.yaml"), "a").size() == 2);
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

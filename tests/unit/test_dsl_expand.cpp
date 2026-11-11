#include <catch2/catch_test_macros.hpp>
#include <stdexcept>
#include <string>

#include "dsl/expand.hpp"
#include "ir/ir.hpp"

using namespace blz::dsl;

namespace {

const Loc kLoc{"x.yaml", 1, 1};

template <class F>
std::string code_of(F&& f) {
  try {
    f();
  } catch (const CompileError& e) {
    return e.code();
  }
  return "ok";
}

std::string src(const std::string& rel) { return std::string(BLZ_SOURCE_DIR) + "/" + rel; }

const char* kHead =
    "version: 1\npattern: p\nblock: fan_in\nwindow: {span: d, lower: open}\n"
    "trigger: {edge: e0, src: U, dst: V}\n";

YNode expand(const std::string& body, const IntEnv& overrides = {}) {
  return blz::ir::expand_text(std::string(kHead) + body, "x.yaml", overrides);
}

std::vector<std::string> ids(const YNode& root) {
  std::vector<std::string> out;
  for (const YNode& st : root.get("stages")->seq) out.push_back(st.get("id")->scalar);
  return out;
}

}  // namespace

TEST_CASE("eval_int") {
  const IntEnv env = {{"K_MAX", 10}, {"k", 3}};
  CHECK(eval_int("2*3+1", env, kLoc) == 7);
  CHECK(eval_int("-(2-5)", env, kLoc) == 3);
  CHECK(eval_int(" K_MAX - 2 ", env, kLoc) == 8);
  CHECK(eval_int("k*k - k", env, kLoc) == 6);
  CHECK(code_of([&] { eval_int("j + 1", env, kLoc); }) == "E102");
  CHECK(code_of([&] { eval_int("1 +", env, kLoc); }) == "E101");
  CHECK(code_of([&] { eval_int("1 2", env, kLoc); }) == "E101");
  CHECK(code_of([&] { eval_int("9223372036854775807 + 1", env, kLoc); }) == "E101");
}

TEST_CASE("substitute: arithmetic and prefixed ranges (SPEC §6.5)") {
  const IntEnv k3 = {{"k", 3}};
  const IntEnv k1 = {{"k", 1}};
  CHECK(substitute("hop_{k}", k3, kLoc) == "hop_3");
  CHECK(substitute("C{k-1}.out", k3, kLoc) == "C2.out");
  bool whole = false;
  CHECK(substitute("{k+2}", k3, kLoc, &whole) == "5");
  CHECK(whole);
  CHECK(substitute("x{k+2}", k3, kLoc, &whole) == "x5");
  CHECK_FALSE(whole);
  CHECK(substitute("C{k} in [U, V, C{1..k-1}]", k3, kLoc) == "C3 in [U, V, C1, C2]");
  CHECK(substitute("C{k} in [U, V, C{1..k-1}]", k1, kLoc) == "C1 in [U, V]");
  CHECK(substitute("[C{1..0}, U]", k1, kLoc) == "[U]");
  CHECK(substitute("[C{1..0}]", k1, kLoc) == "[]");
  CHECK(substitute("[U, C{1..k}, V]", k1, kLoc) == "[U, C1, V]");
  CHECK(substitute("{1..3}", k1, kLoc) == "1, 2, 3");
  CHECK(substitute("no braces", k1, kLoc) == "no braces");
  CHECK(code_of([&] { substitute("C{k", k1, kLoc); }) == "E101");
  CHECK(code_of([&] { substitute("C}", k1, kLoc); }) == "E101");
  CHECK(code_of([&] { substitute("C{{k}}", k1, kLoc); }) == "E101");
  CHECK(code_of([&] { substitute("C{q}", k1, kLoc); }) == "E102");
}

TEST_CASE("expand_spec: nested repeats, sequence ranges, zero iterations") {
  const YNode r = expand(
      "consts: {N: 2}\n"
      "stages:\n"
      "  - repeat: {var: i, from: 1, to: N}\n"
      "    stages:\n"
      "      - repeat: {var: j, from: \"{i}\", to: N}\n"
      "        stages:\n"
      "          - {id: \"s{i}_{j}\", op: count, edges: [\"e{1..i}\", x]}\n"
      "  - repeat: {var: z, from: 3, to: 2}\n"
      "    stages: [{id: never}]\n"
      "  - {id: last}\n");
  CHECK(ids(r) == std::vector<std::string>{"s1_1", "s1_2", "s2_2", "last"});
  const YNode& edges = *r.get("stages")->seq[2].get("edges");
  REQUIRE(edges.seq.size() == 3);
  CHECK(edges.seq[0].scalar == "e1");
  CHECK(edges.seq[1].scalar == "e2");
  CHECK(edges.seq[2].scalar == "x");
  // Expanded stages keep the template's position.
  CHECK(r.get("stages")->seq[0].loc.line == 12);
}

TEST_CASE("expand_spec: errors") {
  CHECK(code_of([] { expand("consts: {k: 1}\nstages:\n  - repeat: {var: k, from: 1, to: 2}\n    stages: []\n"); }) ==
        "E102");
  CHECK(code_of([] { expand("stages:\n  - repeat: {var: k, from: 1}\n    stages: []\n"); }) == "E101");
  CHECK(code_of([] { expand("stages:\n  - repeat: {var: k, from: 1, to: 2}\n"); }) == "E101");
  CHECK(code_of([] { expand("consts: {K: x}\nstages: []\n"); }) == "E101");
  CHECK(code_of([] { expand("consts: {t: 1}\nstages: []\n"); }) == "E102");
  CHECK_THROWS_AS(expand("consts: {K: 1}\nstages: []\n", {{"Q", 2}}), std::invalid_argument);
}

TEST_CASE("expand_spec: const override and emit_yaml round trip") {
  const YNode a = expand("consts: {K: 1}\nstages:\n  - {id: \"s{K}\"}\n", {{"K", 7}});
  CHECK(ids(a) == std::vector<std::string>{"s7"});
  CHECK(a.get("consts")->get("K")->scalar == "7");
  const YNode b = load_yaml(emit_yaml(a), "y.yaml");
  CHECK(same_tree(a, b));
}

TEST_CASE("tcyc.yaml expanded with K_MAX = 5 equals the fixture (SPEC §13.4)") {
  const std::string path = src("patterns/tcyc.yaml");
  const YNode got = blz::ir::expand_text(blz::ir::read_text_file(path), path, {{"K_MAX", 5}});
  const std::string fix = src("tests/specs/expanded/tcyc_k5.yaml");
  const YNode want = load_yaml(blz::ir::read_text_file(fix), fix);
  CHECK(same_tree(got, want));
  // And the emitted text parses back to the same tree.
  CHECK(same_tree(load_yaml(emit_yaml(got), "e.yaml"), want));
}

TEST_CASE("tcyc.yaml at the default K_MAX = 10") {
  const std::string path = src("patterns/tcyc.yaml");
  const YNode got = blz::ir::expand_text(blz::ir::read_text_file(path), path);
  const auto v = ids(got);
  REQUIRE(v.size() == 17);  // close_2, hop_1, close_3, then (hop_k, close_k+2) for k = 2..8
  CHECK(v.front() == "close_2");
  CHECK(v[15] == "hop_8");
  CHECK(v.back() == "close_10");
}

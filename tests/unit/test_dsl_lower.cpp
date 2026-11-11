#include <catch2/catch_test_macros.hpp>
#include <string>

#include "ir/ir.hpp"

using namespace blz;
using blz::dsl::CompileError;

namespace {

std::string src(const std::string& rel) { return std::string(BLZ_SOURCE_DIR) + "/" + rel; }

const char* kHead =
    "version: 1\npattern: p\nblock: fan_in\nwindow: {span: d, lower: open}\n"
    "trigger: {edge: e0, src: U, dst: V}\n";

ir::Pattern compile(const std::string& body, const ir::IntEnv& overrides = {}) {
  return ir::compile_text(std::string(kHead) + body, "x.yaml", overrides);
}

std::string code_of(const std::string& body) {
  try {
    compile(body);
  } catch (const CompileError& e) {
    return e.code();
  }
  return "ok";
}

// A count stage over V.in through e1 with one constraint.
std::string count_with(const std::string& kw, const std::string& pred) {
  return "stages:\n  - id: s\n    op: count\n    src: V.in\n    edge: e1\n    element: X\n    output: n\n"
         "    constraints: [{" + kw + ": \"" + pred + "\"}]\n";
}

const ir::Stage& root_check(const ir::Pattern& p, std::size_t i = 0) { return p.levels.at(0).checks.at(i); }

std::string texts(const std::vector<ir::Expr>& xs) {
  std::string s;
  for (const auto& x : xs) s += (s.empty() ? "" : " | ") + to_string(x);
  return s;
}

}  // namespace

TEST_CASE("lower: time bounds from skip_if / break_if (SPEC §6.4)") {
  struct Case {
    const char* kw;
    const char* pred;
    bool upper;
    bool strict;
  };
  const Case cases[] = {
      {"skip_if", "e1.t <= t - 5", false, true},  {"skip_if", "e1.t < t - 5", false, false},
      {"skip_if", "e1.t >= t", true, true},       {"skip_if", "e1.t > t", true, false},
      {"break_if", "e1.t > t", true, false},      {"break_if", "e1.t >= t", true, true},
      {"skip_if", "t - 5 >= e1.t", false, true},  {"skip_if", "t - 5 > e1.t", false, false},
      {"break_if", "t <= e1.t", true, true},      {"break_if", "t < e1.t", true, false},
  };
  for (const Case& c : cases) {
    INFO(c.kw << ": " << c.pred);
    const ir::Pattern p = compile(count_with(c.kw, c.pred));
    const ir::Operand& o = root_check(p).operands.at(0);
    REQUIRE(o.lo.size() == (c.upper ? 1u : 2u));  // the window bound comes first
    CHECK(o.lo[0].from_window);
    CHECK(o.lo[0].strict);  // open window: ts > t - d
    const ir::TimeBound& b = c.upper ? o.hi.at(0) : o.lo.at(1);
    CHECK(b.strict == c.strict);
    CHECK_FALSE(b.from_window);
    CHECK(root_check(p).filters.empty());
  }
}

TEST_CASE("lower: node filters, or-splitting and the trigger time in filters") {
  const ir::Pattern p = compile(count_with("skip_if", "X == V or e1.t >= t or t > e0.t + 1"));
  const ir::Stage& s = root_check(p);
  CHECK(texts(s.filters) == "X == V | t > e0.t + 1");
  REQUIRE(s.operands[0].hi.size() == 1);
  CHECK(to_string(s.operands[0].hi[0].expr) == "t");
  CHECK(s.operands[0].hi[0].strict);
  CHECK(s.output_var == "n");
  CHECK(s.exports == std::vector<std::string>{"n"});
}

TEST_CASE("plan: inclusive bounds, deduplicated") {
  const ir::Pattern p = compile(
      "stages:\n  - id: s\n    op: count\n    src: V.in\n    edge: e1\n    output: n\n"
      "    constraints: [{skip_if: \"e1.t <= t - d\"}, {break_if: \"e1.t >= t\"}, {skip_if: \"e1.t > t - 1\"}]\n");
  const ir::Operand& o = root_check(p).operands[0];
  // window (open) ts > t - d and the constraint ts > t - d are one bound: ts >= t - d + 1.
  CHECK(texts(o.lo_incl) == "t - d + 1");
  CHECK(texts(o.hi_incl) == "t - 1");
}

TEST_CASE("lower: gated exists exports its witness; constants resolve") {
  const ir::Pattern p = compile(
      "consts: {K: 3}\n"
      "stages:\n"
      "  - {id: c, op: exists, src: V.out, node: U, edge: c2, gate: true}\n"
      "  - id: h\n    op: for_all\n    src: V.out\n    edge: h1\n    dst_var: C1\n"
      "    constraints: [{skip_if: \"h1.t <= c2.t\"}]\n"
      "    emit: {size: \"K + 1\", min_size: 0}\n");
  CHECK(root_check(p).gate);
  CHECK(root_check(p).exports == std::vector<std::string>{"c2"});
  const ir::Stage& h = *p.levels.at(1).binding;
  CHECK(texts(h.operands[0].lo_incl) == "t - d + 1 | c2.t + 1");
  REQUIRE(h.emit);
  CHECK(h.emit->size.kids[0].sym == dsl::Sym::constant);
  CHECK(h.emit->size.kids[0].ival == 3);
  CHECK(h.emit->min_size == 2);  // max(min_size, 2)
  CHECK(h.exports == std::vector<std::string>{"C1", "h1"});
}

TEST_CASE("lower: visibility timestamp and indexed vars are accepted") {
  const ir::Pattern p = ir::compile_text(
      "version: 1\npattern: p\nblock: sg\nwindow: {span: d, lower: closed}\nvisibility: timestamp\n"
      "trigger: {edge: e0, src: U, dst: V}\n"
      "stages:\n  - {id: h, op: for_all, src: V.out, edge: h1, dst_var: \"C[1]\"}\n"
      "  - {id: c, op: exists, src: \"C[1].out\", node: U, edge: c3, constraints: [{skip_if: \"C[1] == V\"}]}\n",
      "y.yaml");
  CHECK(p.visibility == ir::Visibility::timestamp);
  CHECK(p.levels.at(1).binding->bind_var == "C[1]");
  CHECK(p.levels.at(1).checks.at(0).operands[0].node == "C[1]");
  CHECK_FALSE(p.levels.at(1).binding->operands[0].lo[0].strict);  // closed window
}

TEST_CASE("lower: the depth limit is 16 binding levels (E301)") {
  const std::string path = src("patterns/tcyc.yaml");
  const std::string text = ir::read_text_file(path);
  CHECK(ir::compile_text(text, path, {{"K_MAX", 18}}).plan.depth == 16);
  try {
    ir::compile_text(text, path, {{"K_MAX", 19}});
    FAIL("K_MAX = 19 must be rejected");
  } catch (const CompileError& e) {
    CHECK(e.code() == "E301");
  }
  CHECK(ir::compile_text(text, path, {{"K_MAX", 3}}).plan.depth == 1);
}

TEST_CASE("lower: classification errors") {
  CHECK(code_of(count_with("break_if", "e1.t <= t")) == "E201");
  CHECK(code_of(count_with("break_if", "X == V")) == "E202");
  CHECK(code_of(count_with("break_if", "t > e0.t")) == "E202");
  CHECK(code_of(count_with("skip_if", "not e1.t > t")) == "E203");
  CHECK(code_of(count_with("skip_if", "e1.t != t")) == "E203");
  CHECK(code_of(count_with("skip_if", "e1.t <= e1.t")) == "E203");
  CHECK(code_of(count_with("skip_if", "e0.t > t")) == "ok");  // e0.t is the trigger's own time
  CHECK(code_of(count_with("skip_if", "X == n")) == "E102");  // the output is not visible yet
}

TEST_CASE("library: fan, degree, sg and tcyc lower and plan as specified") {
  const ir::Pattern fan = ir::compile_file(src("patterns/fan_in.yaml"));
  CHECK(fan.block == "fan_in");
  CHECK(fan.plan.depth == 0);
  CHECK(fan.plan.split_raw_count);
  CHECK(fan.plan.split_stage == "in_nbrs");
  CHECK(texts(fan.trigger_filters) == "U == V");
  const ir::Stage& c = root_check(fan);
  CHECK(c.op == ir::Op::count);
  CHECK(c.operands[0].access == ir::Access::scan_distinct);
  CHECK(texts(c.operands[0].lo_incl) == "t - delta_fan + 1");
  CHECK(c.operands[0].hi_incl.empty());
  CHECK(texts(c.filters) == "X == V");

  const ir::Pattern deg = ir::compile_file(src("patterns/deg_out.yaml"));
  CHECK(root_check(deg).operands[0].access == ir::Access::scan_edges);
  CHECK(root_check(deg).operands[0].node == "U");
  CHECK(root_check(deg).operands[0].dir == ir::Dir::out);

  const ir::Pattern sg = ir::compile_file(src("patterns/sg_scatter.yaml"));
  CHECK(sg.block == "sg");
  REQUIRE(sg.levels.size() == 2);
  const ir::Stage& sinks = *sg.levels[1].binding;
  CHECK(sinks.bind_var == "W");
  CHECK(texts(sinks.filters) == "W in [S, X]");
  const ir::Stage& mid = sg.levels[1].checks.at(0);
  CHECK(mid.op == ir::Op::intersection);
  CHECK_FALSE(mid.binding);
  CHECK_FALSE(mid.materialize);
  CHECK(mid.operands[0].access == ir::Access::intersect);
  CHECK(texts(mid.operands[1].lo_incl) == "t - delta_sg");
  CHECK(texts(mid.filters) == "Y == S");
  CHECK(sg.plan.split_stage == "sinks");

  const ir::Pattern cyc = ir::compile_file(src("patterns/tcyc.yaml"));
  CHECK(cyc.plan.depth == 8);
  CHECK(cyc.plan.work_stage == "hop_1");
  std::vector<std::int64_t> sizes;
  for (const auto& lv : cyc.levels)
    for (const auto& s : lv.checks)
      if (s.emit) sizes.push_back(s.emit->size.ival);
  CHECK(sizes == std::vector<std::int64_t>{2, 3, 4, 5, 6, 7, 8, 9, 10});
  const ir::Stage& hop8 = *cyc.levels[8].binding;
  CHECK(texts(hop8.operands[0].lo_incl) == "t - delta_cyc | h7.t + 1");
  CHECK(texts(hop8.operands[0].hi_incl) == "t - 1");
  CHECK(texts(hop8.filters) == "C8 in [U, V, C1, C2, C3, C4, C5, C6, C7]");
  const ir::Stage& close10 = cyc.levels[8].checks.at(0);
  CHECK(close10.probe_node == "U");
  CHECK(close10.operands[0].access == ir::Access::probe);
  CHECK(texts(close10.operands[0].lo_incl) == "t - delta_cyc | h8.t + 1");
}

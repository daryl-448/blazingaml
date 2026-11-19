#include <catch2/catch_test_macros.hpp>
#include <algorithm>
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

// SPEC v2 §8 S5: a distinct for_all with `witness: latest` exports a latest witness, which later
// scans may use as an upper bound only; min() is an int.
TEST_CASE("lower: witness: latest bounds later scans from above; min() of ints") {
  const ir::Pattern p = compile(
      "stages:\n"
      "  - id: h\n    op: for_all\n    src: U.in\n    edge: h1\n    dst_var: C1\n    witness: latest\n"
      "    constraints: [{break_if: \"h1.t >= t\"}]\n"
      "  - id: g\n    op: for_all\n    src: C1.in\n    edge: h2\n    dst_var: C2\n"
      "    constraints: [{break_if: \"h2.t >= h1.t\"}]\n"
      "  - id: n\n    op: count\n    src: C2.out\n    edge: e3\n    output: k\n"
      "    emit: {size: \"min(k, 2 + 1)\"}\n");
  const ir::Stage& h = *p.levels.at(1).binding;
  const ir::Stage& g = *p.levels.at(2).binding;
  CHECK(h.latest_witness);
  CHECK_FALSE(g.latest_witness);
  CHECK(texts(g.operands[0].hi_incl) == "h1.t - 1");
  const ir::Stage& n = p.levels.at(2).checks.at(0);
  REQUIRE(n.emit);
  CHECK(n.emit->size.k == dsl::Expr::K::Min);
  CHECK(n.emit->size.ty == dsl::Ty::int_);
  // The earliest witness (the default) may not bound from above, the latest not from below.
  CHECK(code_of("stages:\n"
                     "  - {id: h, op: for_all, src: U.in, edge: h1, dst_var: C1, witness: earliest}\n"
                     "  - id: g\n    op: for_all\n    src: C1.in\n    edge: h2\n    dst_var: C2\n"
                     "    constraints: [{break_if: \"h2.t >= h1.t\"}]\n") == "E204");
  CHECK(code_of("stages:\n"
                     "  - {id: h, op: for_all, src: U.in, edge: h1, dst_var: C1, witness: latest}\n"
                     "  - id: g\n    op: exists\n    src: C1.in\n    node: V\n    edge: h2\n"
                     "    constraints: [{skip_if: \"h2.t < h1.t\"}]\n") == "E204");
  CHECK(code_of("stages:\n"
                     "  - {id: h, op: for_all, src: U.in, edge: h1, dst_var: C1, witness: latest}\n"
                     "  - id: g\n    op: exists\n    src: C1.in\n    node: V\n    edge: h2\n"
                     "    constraints: [{break_if: \"h2.t >= h1.t\"}]\n") == "ok");
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
  CHECK(code_of(count_with("skip_if", "e1.t <= e1.t")) == "E205");  // not two scanned edges of an intersection
  CHECK(code_of(count_with("skip_if", "e0.t > t")) == "ok");  // e0.t is the trigger's own time
  CHECK(code_of(count_with("skip_if", "X == n")) == "E102");  // the output is not visible yet
}

namespace {

// An intersection check stage over S.out (e2) and V.in (e3), with one constraint.
std::string intersection_with(const std::string& kw, const std::string& pred) {
  return "stages:\n  - {id: src, op: for_all, src: U.in, edge: e1, dst_var: S}\n"
         "  - id: mid\n    op: intersection\n    src: S.out\n    dst: V.in\n    edges: [e2, e3]\n"
         "    output: I\n    element: Y\n    constraints: [{" + kw + ": \"" + pred + "\"}]\n";
}

bool holds(ir::Expr::Cmp c, std::int64_t a, std::int64_t b) {
  switch (c) {
    case ir::Expr::Cmp::lt: return a < b;
    case ir::Expr::Cmp::le: return a <= b;
    case ir::Expr::Cmp::gt: return a > b;
    case ir::Expr::Cmp::ge: return a >= b;
    case ir::Expr::Cmp::eq: return a == b;
    case ir::Expr::Cmp::ne: return a != b;
  }
  return false;
}

}  // namespace

// SPEC v2 §6.2 (D-27): the order constraint keeps an element exactly when some witness pair
// fails the skip; the IR's earliest/latest lowering is checked against every pair.
TEST_CASE("lower: order constraints, lowered through earliest and latest witnesses, equal brute force") {
  std::uint64_t state = 11;
  auto next = [&](std::uint64_t k) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (state >> 33) % k;
  };
  const std::pair<const char*, ir::Expr::Cmp> ops[] = {
      {"<", ir::Expr::Cmp::lt}, {"<=", ir::Expr::Cmp::le}, {">", ir::Expr::Cmp::gt}, {">=", ir::Expr::Cmp::ge}};
  for (const auto& [op, written] : ops)
    for (bool reversed : {false, true}) {
      const std::string pred = reversed ? std::string("e3.t ") + op + " e2.t" : std::string("e2.t ") + op + " e3.t";
      INFO(pred);
      const ir::Pattern p = compile(intersection_with("skip_if", pred));
      const ir::Stage& mid = p.levels.at(1).checks.at(0);
      REQUIRE(mid.order);
      const ir::Order& o = *mid.order;
      CHECK(o.a == "e2");
      CHECK(o.b == "e3");
      CHECK(mid.filters.empty());
      CHECK(mid.operands[0].lo.size() == 1);  // the window bound only: no time bound came from it
      CHECK(mid.operands[1].lo.size() == 1);
      CHECK(mid.operands[0].hi.empty());
      for (int trial = 0; trial < 2000; ++trial) {
        std::vector<std::int64_t> A(1 + next(4)), B(1 + next(4));  // ties and duplicates: values in [0, 5)
        for (auto& x : A) x = std::int64_t(next(5));
        for (auto& x : B) x = std::int64_t(next(5));
        bool want = false;  // some pair (a, b) for which the skip predicate is false
        for (std::int64_t a : A)
          for (std::int64_t b : B) want = want || !(reversed ? holds(written, b, a) : holds(written, a, b));
        const std::int64_t ea = o.a_latest ? *std::max_element(A.begin(), A.end()) : *std::min_element(A.begin(), A.end());
        const std::int64_t eb = o.b_latest ? *std::max_element(B.begin(), B.end()) : *std::min_element(B.begin(), B.end());
        CHECK(holds(o.keep, ea, eb) == want);
      }
    }
  // sg_ordered: skip_if e2.t >= e3.t keeps y when min(e2) < max(e3).
  const ir::Pattern so = ir::compile_file(src("patterns/sg_ordered.yaml"));
  const ir::Stage& mid = so.levels.at(1).checks.at(0);
  REQUIRE(mid.order);
  CHECK(mid.order->skip == ir::Expr::Cmp::ge);
  CHECK_FALSE(mid.order->a_latest);
  CHECK(mid.order->b_latest);
  CHECK(mid.order->keep == ir::Expr::Cmp::lt);
  CHECK(texts(mid.filters) == "Y == W");
  CHECK(texts(so.levels.at(1).binding->operands[0].hi_incl) == "t - 1");  // sources: e1.t < t
  // E205 and E203 around it.
  CHECK(code_of(intersection_with("skip_if", "e2.t >= e3.t and Y == V")) == "E203");
  CHECK(code_of(intersection_with("skip_if", "not e2.t >= e3.t")) == "E203");
  CHECK(code_of(intersection_with("skip_if", "e2.t != e3.t")) == "E205");
  CHECK(code_of(intersection_with("skip_if", "e2.t - 1 >= e3.t")) == "E205");
  CHECK(code_of(intersection_with("break_if", "e2.t > e3.t")) == "E205");
  CHECK(code_of(intersection_with("skip_if", "e2.t >= e3.t or e2.t <= t - 5")) == "E205");
  CHECK(code_of(intersection_with("skip_if", "e2.t <= t - 5 or Y == V")) == "ok");  // no order constraint
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

namespace {

// A count over `port` with element X, constraints `cons` (YAML list) and emit size `emit`.
std::string count_stage(const std::string& port, const std::string& cons, const std::string& emit = "n",
                        const std::string& extra = "") {
  return "constraints: [{skip_if: \"U == V\"}]\nstages:\n  - id: s\n    op: count\n    src: " + port +
         "\n    edge: e1\n    element: X\n    output: n\n    constraints: " + cons + "\n    emit: {size: \"" + emit +
         "\"}\n" + extra;
}

bool sweeps(const std::string& body, const ir::Opts& opts = {}) {
  return ir::compile_text(std::string(kHead) + body, "x.yaml", {}, opts).plan.sweep;
}

}  // namespace

TEST_CASE("plan: the sweep lowering's recognition (SPEC v2 §7.1)") {
  for (const char* f : {"fan_in", "fan_out", "deg_in", "deg_out"}) {
    const ir::Pattern p = ir::compile_file(src(std::string("patterns/") + f + ".yaml"));
    INFO(f);
    CHECK(p.plan.sweep);
    CHECK(p.plan.sweep_stage == root_check(p).id);
  }
  for (const char* f : {"sg_gather", "sg_scatter", "sg_ordered", "tcyc"})
    CHECK_FALSE(ir::compile_file(src(std::string("patterns/") + f + ".yaml")).plan.sweep);

  CHECK(sweeps(count_stage("V.in", "[{skip_if: \"X == V\"}]")));
  CHECK(sweeps(count_stage("U.out", "[{skip_if: \"X == U\"}]", "n - 1")));
  CHECK(sweeps(count_stage("V.in", "[]")));
  // Condition 1: the trigger's own list, and a single stage.
  CHECK_FALSE(sweeps(count_stage("V.out", "[]")));
  CHECK_FALSE(sweeps(count_stage("U.in", "[]")));
  CHECK_FALSE(sweeps(count_stage("V.in", "[]", "n",
                                 "  - {id: s2, op: count, src: V.in, edge: e2, element: Y, output: k, emit: {size: k}}\n")));
  // Condition 2: the element filters see only the element, the list's node and constants; no
  // time bound but the window's.
  CHECK_FALSE(sweeps(count_stage("V.in", "[{skip_if: \"X == U\"}]")));
  CHECK_FALSE(sweeps(count_stage("V.in", "[{break_if: \"e1.t >= t\"}]")));
  CHECK_FALSE(sweeps(count_stage("V.in", "[{skip_if: \"e1.t <= t - 5\"}]")));
  // Condition 3: the emit sees only the count.
  CHECK_FALSE(sweeps(count_stage("V.in", "[]", "d")));
  // --disable sweep keeps the per-trigger plan.
  CHECK_FALSE(sweeps(count_stage("V.in", "[]"), ir::parse_disable("sweep")));
}

TEST_CASE("plan: blzc --disable names (SPEC v2 §7.2)") {
  CHECK(ir::disabled_names(ir::parse_disable("")).empty());
  const ir::Opts all = ir::parse_disable("sweep,find_start,count_only,degree_order,distinct_first");
  CHECK(ir::disabled_names(all) ==
        std::vector<std::string>{"find_start", "distinct_first", "degree_order", "count_only", "sweep"});
  CHECK_THROWS_AS(ir::parse_disable("find_start,bogus"), std::invalid_argument);
  CHECK_THROWS_AS(ir::parse_disable("find_start,"), std::invalid_argument);
  // count_only off materializes a check intersection's set.
  const ir::Pattern sg = ir::compile_file(src("patterns/sg_gather.yaml"), {}, ir::parse_disable("count_only"));
  CHECK(sg.levels.at(1).checks.at(0).materialize);
}

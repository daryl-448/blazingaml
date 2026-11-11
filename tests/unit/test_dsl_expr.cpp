#include <catch2/catch_test_macros.hpp>
#include <string>

#include "dsl/expr.hpp"

using namespace blz::dsl;

namespace {

Expr parse(const std::string& s) { return parse_expr(s, {"x.yaml", 3, 10}, true); }

std::string code_of(const std::string& s) {
  try {
    parse(s);
  } catch (const CompileError& e) {
    return e.code();
  }
  return "ok";
}

}  // namespace

TEST_CASE("expr: canonical text round-trips") {
  for (const char* s : {"U == V", "S in [X, W]", "C3 not in [U, V, C1, C2]", "h2.t <= h1.t",
                        "c4.t >= t", "len(I) >= 2", "t - delta + 1", "X == V or e1.t >= t",
                        "not X == V", "(X == V or Y == V) and e1.t < t", "C[3] in []", "n != 0"}) {
    const Expr e = parse(s);
    CHECK(to_string(e) == s);
    CHECK(to_string(parse(to_string(e))) == to_string(e));
  }
}

TEST_CASE("expr: precedence and associativity") {
  const Expr a = parse("A == B or C == D and E == F");
  REQUIRE(a.k == Expr::K::Or);
  CHECK(a.kids[1].k == Expr::K::And);

  const Expr b = parse("not X in [U]");
  REQUIRE(b.k == Expr::K::Not);
  CHECK(b.kids[0].k == Expr::K::In);
  CHECK_FALSE(b.kids[0].negated);

  const Expr c = parse("X not in [U, V]");
  REQUIRE(c.k == Expr::K::In);
  CHECK(c.negated);
  CHECK(c.kids.size() == 3);

  const Expr d = parse("t - d + 1");  // (t - d) + 1
  REQUIRE(d.k == Expr::K::Add);
  CHECK(d.kids[0].k == Expr::K::Sub);

  const Expr e = parse("e1.t");
  CHECK(e.k == Expr::K::EdgeT);
  CHECK(e.name == "e1");
  CHECK(parse("C[12]").name == "C[12]");
}

TEST_CASE("expr: node positions are column offsets into the scalar") {
  const Expr e = parse("U == V");
  CHECK(e.loc.line == 3);
  CHECK(e.loc.col == 10);
  CHECK(e.kids[1].loc.col == 15);
  const Expr f = parse_expr("U == V", {"x.yaml", 3, 10}, false);
  CHECK(f.kids[1].loc.col == 10);  // rewritten text: the scalar's own position
}

TEST_CASE("expr: syntax errors are E101") {
  for (const char* s : {"", "U ==", "a < b < c", "U.x", "len(3)", "len I", "1x", "U @ V", "(U == V",
                        "in", "X in U", "X in [U,", "U == V V", "U * 2", "99999999999999999999"}) {
    INFO(s);
    CHECK(code_of(s) == "E101");
  }
}

TEST_CASE("expr: flip") {
  CHECK(flip(Expr::Cmp::lt) == Expr::Cmp::gt);
  CHECK(flip(Expr::Cmp::ge) == Expr::Cmp::le);
  CHECK(flip(Expr::Cmp::eq) == Expr::Cmp::eq);
}

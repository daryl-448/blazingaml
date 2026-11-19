#pragma once
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "dsl/diag.hpp"

namespace blz::dsl {

// Kinds of values (SPEC §6.2, E103: node, time, int and set do not mix).
enum class Ty { none, node, edge, time, int_, set, bool_ };

// What a name refers to; filled in by lowering (ir/lower.cpp).
enum class Sym {
  unresolved,
  node,       // a node variable (trigger src/dst, dst_var, element)
  edge,       // an edge variable (only valid as `e.t`)
  int_var,    // the output of a `count` stage
  set_var,    // the output of an `intersection` check stage (only valid in len())
  param,      // a runtime int parameter (the window span)
  constant,   // a compile-time const; ival holds its value
  trig_time,  // `t` (= e0.t)
};

// An expression of the predicate language (SPEC §6.2).
struct Expr {
  enum class K {
    Int,    // ival
    Name,   // name: a node var, int var, param, const or `t`
    EdgeT,  // name.t, name an edge var
    Len,    // len(name)
    Min,    // min(kids[0], kids[1]), ints (SPEC v2 §8 S5)
    Add,    // kids[0] + kids[1]
    Sub,    // kids[0] - kids[1]
    Cmp,    // kids[0] <cmp> kids[1]
    In,     // kids[0] (not) in [kids[1..]]
    And,
    Or,
    Not,    // kids[0]
  };
  enum class Cmp { eq, ne, lt, le, gt, ge };

  K k = K::Int;
  std::int64_t ival = 0;
  std::string name;
  Cmp cmp = Cmp::eq;
  bool negated = false;  // In: `not in`
  std::vector<Expr> kids;
  Loc loc;  // the expression's first token

  Sym sym = Sym::unresolved;  // Name / EdgeT / Len, after lowering
  Ty ty = Ty::none;           // after lowering
};

// Parses a predicate or integer expression. `base` is the position of the expression's first
// character; when `exact_cols` is false (a multi-line or rewritten scalar) every node gets
// `base` itself instead of a column offset. E101 on a syntax error.
Expr parse_expr(std::string_view text, const Loc& base, bool exact_cols);

// Canonical text, with parentheses only where precedence needs them.
std::string to_string(const Expr& e);

const char* cmp_text(Expr::Cmp c);
// a OP b  <=>  b flip(OP) a
Expr::Cmp flip(Expr::Cmp c);

}  // namespace blz::dsl

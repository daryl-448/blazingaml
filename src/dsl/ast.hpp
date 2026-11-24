#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "dsl/expand.hpp"
#include "dsl/expr.hpp"
#include "dsl/ynode.hpp"

namespace blz::dsl {

// The schema-checked form of one expanded spec file (SPEC §6.1, §6.3). Names are not resolved
// yet; that is lowering's job (ir/lower.cpp).

enum class Dir { in, out };
// union_ and difference: the set ops of SPEC v2 §8 S4 (check stages over two operands).
enum class Op { for_all, intersection, exists, count, union_, difference };
enum class Mult { distinct, edges };
enum class Visibility { stream, timestamp };
// `family: clean` (SPEC v3 §3.2): the spec follows the clean rules R1–R11; absent, legacy.
enum class Family { legacy, clean };

struct Name {
  std::string s;
  Loc loc;
};

// `X.in` / `X.out`
struct Port {
  Name var;
  Dir dir = Dir::out;
  Loc loc;
};

struct Constraint {
  bool is_break = false;  // break_if (else skip_if)
  Expr pred;
  Loc loc;
};

struct Emit {
  Expr size;
  std::int64_t min_size = 2;
  Loc loc;
};

struct Stage {
  Loc loc;
  Name id;
  Op op = Op::for_all;
  Loc op_loc;
  std::optional<Port> src, dst;
  std::optional<Name> edge, dst_var, node, output, element;
  std::vector<Name> edges;  // intersection, union, difference: [src-side edge, dst-side edge]
  std::optional<Mult> multiplicity;
  Loc multiplicity_loc;
  // `witness: latest` (SPEC v2 §8 S5): a distinct for_all binds each neighbour with its latest
  // witness edge instead of its earliest; later stages may then use that edge's time only as an
  // upper bound. `witness: earliest` is the default.
  bool latest_witness = false;
  std::optional<bool> gate;
  Loc gate_loc;
  std::vector<Constraint> constraints;
  std::optional<Emit> emit;
};

struct Spec {
  std::string file;
  Name pattern, block;
  IntEnv consts;
  std::vector<Loc> const_locs;  // the position of each const's name
  Family family = Family::legacy;
  Loc family_loc;  // the `family:` value, if given
  Name span;  // the window's span parameter
  bool lower_open = false;
  Loc lower_loc;  // the window's `lower:` value
  Visibility visibility = Visibility::stream;
  Loc visibility_loc;  // the `visibility:` value, if given
  bool work_limit = false;  // `work_limit: true` (SPEC v2 §8 S3): count bindings per trigger
  Name trig_edge, trig_src, trig_dst;
  std::vector<Constraint> constraints;  // trigger level, skip_if only
  std::vector<Stage> stages;
};

// YAML tree of an *expanded* document (expand_spec) -> Spec. E101 on schema errors, E401 on the
// reserved op `differentiate`, E101 on expression syntax errors.
Spec parse_spec(const YNode& expanded, const std::string& file);

}  // namespace blz::dsl

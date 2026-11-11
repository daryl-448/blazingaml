#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "dsl/ast.hpp"

// The pattern IR (SPEC §7.2): resolved, type-checked, constraints classified (§6.4) and planned
// (§7.3). The backends (P4, P5) read only this.
namespace blz::ir {

using dsl::Dir;
using dsl::Expr;
using dsl::IntEnv;
using dsl::Loc;
using dsl::Mult;
using dsl::Op;
using dsl::Visibility;

inline constexpr int kMaxDepth = 16;  // binding levels after expansion (E301)

// One bound on a scanned edge's timestamp. In `Operand::lo`: ts > expr (strict) or ts >= expr.
// In `Operand::hi`: ts < expr (strict) or ts <= expr.
struct TimeBound {
  Expr expr;
  bool strict = false;
  bool from_window = false;  // the window's lower bound, added to every scan (§6.4)
  Loc loc;
};

// How an operand's neighbour list is read (planner, §7.3).
enum class Access {
  scan_distinct,  // T-CSR scan, distinct-first: each neighbour once, with its earliest witness
  scan_edges,     // T-CSR scan, every edge
  probe,          // P-CSR probe for one node: the earliest witness within the bounds, or none
  intersect,      // one side of an intersection: scanned or probed, chosen per trigger at runtime
};

// A neighbour list X.dir read through edge variable `edge`.
struct Operand {
  std::string node;  // X
  Dir dir = Dir::out;
  std::string edge;
  std::vector<TimeBound> lo, hi;  // as written plus the window bound
  // Planner: the same bounds normalized to inclusive form (strict lower x -> x + 1, strict upper
  // x -> x - 1, timestamps being integers) and deduplicated. The scan range is
  // ts >= max(lo_incl), ts <= min(hi_incl), plus the visibility bound (Pattern::visibility).
  std::vector<Expr> lo_incl, hi_incl;
  Access access = Access::scan_distinct;
};

struct Emit {
  Expr size;
  std::int64_t min_size = 2;  // already max(min_size, 2) (§6.6)
  Loc loc;
};

struct Stage {
  std::string id;
  Op op = Op::for_all;
  Loc loc;
  bool binding = false;            // opens a loop level (for_all, intersection with dst_var)
  std::vector<Operand> operands;   // intersection: {src side, dst side}; otherwise one
  std::string bind_var;            // binding stages: the node variable bound per iteration
  std::string output_var;          // count: the int; check intersection: the set
  std::string element_var;         // check stages: the per-element node (own constraints only)
  std::string probe_node;          // exists: the node looked for
  Mult multiplicity = Mult::distinct;  // for_all, count
  std::vector<Expr> filters;       // node filters (§6.4): skip the element / binding if any holds
  bool gate = false;               // check stages: later stages run only if the check passed
  std::optional<Emit> emit;
  std::vector<std::string> exports;  // names later stages may use
  // Planner.
  bool materialize = false;        // check intersection: false when the set is used only via len()
  std::vector<std::string> notes;  // planner decisions, emitted as comments by the backends
};

// A loop level: the binding stage that opens it (none for the root) and the check stages that
// run once per binding, in spec order. Level i + 1 is nested inside level i.
struct Level {
  std::optional<Stage> binding;
  std::vector<Stage> checks;
};

struct Plan {
  int depth = 0;               // number of binding levels
  std::string work_stage;      // the first scan; its window size is the work estimate (§7.4)
  int work_operand = 0;
  std::string split_stage;     // CUDA heavy kernel: the scan split across the warp (§7.5)
  bool split_raw_count = false;  // no binding stage: lanes sum raw counts, lane 0 emits once
  std::vector<std::string> notes;
};

struct Pattern {
  std::string name, block, file;
  Loc block_loc;
  std::string span_param;      // runtime int parameter: the window span (the block's window)
  bool lower_open = false;
  Visibility visibility = Visibility::stream;
  std::string e0, src_var, dst_var;
  IntEnv consts;
  std::vector<Expr> trigger_filters;
  std::vector<Level> levels;   // levels[0] is the root
  Plan plan;

  const Stage* find_stage(std::string_view id) const;
};

// Resolve, type-check and classify (§7.1 steps 3–5, 7): E102, E103, E201–E204, E301.
Pattern lower(const dsl::Spec& spec);

// Planner (§7.3): access paths, effective bounds, work estimate, warp split, notes.
void plan(Pattern& p);

// Human-readable IR, deterministic (tests/specs/ir/ and `blzc --dump-ir`).
std::string dump(const Pattern& p);

// The front end, steps 1–7: parse, expand (with const overrides), schema-check, lower, plan.
dsl::YNode expand_text(std::string_view text, const std::string& file, const IntEnv& overrides = {});
Pattern compile_text(std::string_view text, const std::string& file, const IntEnv& overrides = {});
Pattern compile_file(const std::string& path, const IntEnv& overrides = {});
std::string read_text_file(const std::string& path);

}  // namespace blz::ir

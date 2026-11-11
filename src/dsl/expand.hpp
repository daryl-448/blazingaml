#pragma once
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "dsl/ynode.hpp"

namespace blz::dsl {

// Compile-time integer bindings: consts and the repeat variables in scope.
using IntEnv = std::vector<std::pair<std::string, std::int64_t>>;

// Limits on expansion. The binding depth limit (16, E301) is checked after lowering, where the
// stage kinds are known; these guard against a runaway repeat before that.
inline constexpr std::int64_t kMaxRepeatCount = 1024;
inline constexpr std::size_t kMaxExpandedStages = 1024;

// Evaluates an integer expression over the names in `env`: decimal literals, + - *, unary minus
// and parentheses. E102 on an unknown name, E101 on a syntax error or overflow.
std::int64_t eval_int(std::string_view text, const IntEnv& env, const Loc& loc);

// Substitutes every `{a}` in `s` by the decimal value of `a`, and every `P{a..b}` by the item
// list "Pa, Pa+1, ..., Pb" (SPEC §6.5). An empty range removes the item and one separating comma.
// `whole` is set when `s` consisted of exactly one `{a}` group (the value then becomes an integer).
std::string substitute(std::string_view s, const IntEnv& env, const Loc& loc, bool* whole = nullptr);

// Reads the top-level `consts` map (applying `overrides`, which must name declared consts),
// unrolls every `repeat` in `stages` (nested repeats included), and substitutes `{...}` in every
// scalar outside `consts`, with the consts and the enclosing repeat variables in scope. The
// result has `consts` rewritten with the overridden values and no `repeat` items left.
// std::invalid_argument when an override names an undeclared const.
YNode expand_spec(const YNode& root, const IntEnv& overrides);

// The consts of an expanded (or raw) document, in file order. E101 on a malformed map.
IntEnv read_consts(const YNode& root);

}  // namespace blz::dsl

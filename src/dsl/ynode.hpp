#pragma once
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "dsl/diag.hpp"

namespace blz::dsl {

// A YAML document as a plain tree that keeps every node's position. The front end converts
// yaml-cpp's tree into this once, so that repeat expansion (SPEC §6.5) can rewrite it freely
// and every later diagnostic still points into the original file.
struct YNode {
  enum class Kind { Null, Scalar, Seq, Map };
  Kind kind = Kind::Null;
  std::string scalar;                         // Kind::Scalar
  bool quoted = false;                        // the scalar was written in quotes
  bool rewritten = false;                     // repeat expansion changed the scalar's text
  bool verbatim = false;                      // the scalar's text appears unchanged in the source
                                              // at loc (no folding, escapes or line breaks)
  std::vector<YNode> seq;                     // Kind::Seq
  std::vector<std::pair<YNode, YNode>> map;   // Kind::Map, in file order; keys are scalars
  Loc loc;

  bool is_scalar() const { return kind == Kind::Scalar; }
  bool is_seq() const { return kind == Kind::Seq; }
  bool is_map() const { return kind == Kind::Map; }
  // The value under `key` in a map, or nullptr.
  const YNode* get(std::string_view key) const;
  YNode* get(std::string_view key);
};

// Parses YAML text. E101 on a YAML syntax error, a non-scalar or duplicate map key, or a
// multi-document stream.
YNode load_yaml(std::string_view text, const std::string& file);

// Block-style YAML for a tree (used by `blzc --expand`). Scalars are quoted when they were
// quoted in the source, or when YAML would otherwise read them differently.
std::string emit_yaml(const YNode& root);

// Structural equality: same kinds, same scalar text, same map keys in the same order. Positions
// and quoting are ignored.
bool same_tree(const YNode& a, const YNode& b);

}  // namespace blz::dsl

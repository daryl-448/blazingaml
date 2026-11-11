#pragma once
#include <array>
#include <string_view>

namespace blz::dsl {

inline bool is_ident_start(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}
inline bool is_ident_char(char c) { return is_ident_start(c) || (c >= '0' && c <= '9'); }

// [A-Za-z_][A-Za-z0-9_]*, optionally followed by an index "[<digits>]" (SPEC §6.2: indexed node
// vars). The index is part of the name.
inline bool is_ident(std::string_view s) {
  if (s.empty() || !is_ident_start(s[0])) return false;
  std::size_t i = 1;
  while (i < s.size() && is_ident_char(s[i])) ++i;
  if (i == s.size()) return true;
  if (s[i] != '[' || s.back() != ']' || i + 2 >= s.size()) return false;
  for (std::size_t j = i + 1; j + 1 < s.size(); ++j)
    if (s[j] < '0' || s[j] > '9') return false;
  return true;
}

// Words of the expression language; they cannot name anything.
inline bool is_reserved(std::string_view s) {
  static constexpr std::array<std::string_view, 6> kWords = {"t", "and", "or", "not", "in", "len"};
  for (auto w : kWords)
    if (s == w) return true;
  return false;
}

}  // namespace blz::dsl

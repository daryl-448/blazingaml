#include "dsl/expand.hpp"

#include <charconv>
#include <stdexcept>

#include "dsl/names.hpp"

namespace blz::dsl {

namespace {

const std::int64_t* lookup(const IntEnv& env, std::string_view name) {
  for (const auto& [k, v] : env)
    if (k == name) return &v;
  return nullptr;
}

class IntParser {
 public:
  IntParser(std::string_view s, const IntEnv& env, const Loc& loc) : s_(s), env_(env), loc_(loc) {}

  std::int64_t parse() {
    const std::int64_t v = sum();
    skip_ws();
    if (i_ != s_.size()) bad("unexpected '" + std::string(s_.substr(i_)) + "'");
    return v;
  }

 private:
  [[noreturn]] void bad(const std::string& why) {
    fail("E101", loc_, "integer expression '" + std::string(s_) + "': " + why);
  }
  void skip_ws() {
    while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\t')) ++i_;
  }
  bool eat(char c) {
    skip_ws();
    if (i_ < s_.size() && s_[i_] == c) {
      ++i_;
      return true;
    }
    return false;
  }
  void check(bool overflow) {
    if (overflow) bad("overflow");
  }
  std::int64_t sum() {
    std::int64_t v = product();
    for (;;) {
      std::int64_t r = 0;
      if (eat('+')) {
        const std::int64_t b = product();
        check(__builtin_add_overflow(v, b, &r));
        v = r;
      } else if (eat('-')) {
        const std::int64_t b = product();
        check(__builtin_sub_overflow(v, b, &r));
        v = r;
      } else {
        return v;
      }
    }
  }
  std::int64_t product() {
    std::int64_t v = unary();
    while (eat('*')) {
      std::int64_t r = 0;
      const std::int64_t b = unary();
      check(__builtin_mul_overflow(v, b, &r));
      v = r;
    }
    return v;
  }
  std::int64_t unary() {
    if (eat('-')) {
      std::int64_t r = 0;
      const std::int64_t b = unary();
      check(__builtin_sub_overflow(std::int64_t{0}, b, &r));
      return r;
    }
    return primary();
  }
  std::int64_t primary() {
    skip_ws();
    if (i_ >= s_.size()) bad("unexpected end");
    if (eat('(')) {
      const std::int64_t v = sum();
      if (!eat(')')) bad("missing ')'");
      return v;
    }
    const char c = s_[i_];
    if (c >= '0' && c <= '9') {
      std::int64_t v = 0;
      const char* b = s_.data() + i_;
      const auto [p, ec] = std::from_chars(b, s_.data() + s_.size(), v);
      if (ec != std::errc()) bad("integer literal out of range");
      i_ += std::size_t(p - b);
      return v;
    }
    if (is_ident_start(c)) {
      const std::size_t b = i_;
      while (i_ < s_.size() && is_ident_char(s_[i_])) ++i_;
      const std::string_view name = s_.substr(b, i_ - b);
      if (const std::int64_t* v = lookup(env_, name)) return *v;
      fail("E102", loc_,
           "unknown name '" + std::string(name) + "' in integer expression '" + std::string(s_) +
               "' (only consts and repeat variables are in scope)");
    }
    bad(std::string("unexpected '") + c + "'");
  }

  std::string_view s_;
  const IntEnv& env_;
  const Loc& loc_;
  std::size_t i_ = 0;
};

// A scalar that is exactly `P{a..b}` (a YAML sequence item that expands to several items).
bool split_range_item(const YNode& item, const IntEnv& env, std::vector<YNode>& out) {
  if (!item.is_scalar()) return false;
  const std::string& s = item.scalar;
  const std::size_t open = s.find('{');
  if (open == std::string::npos || open == 0 || s.back() != '}') return false;
  if (s.find_first_of("{}", open + 1) != s.size() - 1) return false;
  for (std::size_t i = 0; i < open; ++i)
    if (!is_ident_char(s[i]) || !is_ident_start(s[0])) return false;
  const std::string_view inner = std::string_view(s).substr(open + 1, s.size() - open - 2);
  const std::size_t dd = inner.find("..");
  if (dd == std::string_view::npos) return false;
  const std::int64_t a = eval_int(inner.substr(0, dd), env, item.loc);
  const std::int64_t b = eval_int(inner.substr(dd + 2), env, item.loc);
  if (b >= a && b - a >= kMaxRepeatCount) fail("E301", item.loc, "range is too long");
  for (std::int64_t k = a; k <= b; ++k) {
    YNode x = item;
    x.scalar = s.substr(0, open) + std::to_string(k);
    x.quoted = false;
    x.rewritten = true;
    out.push_back(std::move(x));
  }
  return true;
}

YNode subst_tree(const YNode& n, const IntEnv& env) {
  YNode out = n;
  switch (n.kind) {
    case YNode::Kind::Null:
      break;
    case YNode::Kind::Scalar:
      if (n.scalar.find_first_of("{}") != std::string::npos) {
        bool whole = false;
        out.scalar = substitute(n.scalar, env, n.loc, &whole);
        out.rewritten = true;
        if (whole) out.quoted = false;
      }
      break;
    case YNode::Kind::Seq:
      out.seq.clear();
      for (const auto& item : n.seq)
        if (!split_range_item(item, env, out.seq)) out.seq.push_back(subst_tree(item, env));
      break;
    case YNode::Kind::Map:
      for (auto& [k, v] : out.map) v = subst_tree(v, env);
      break;
  }
  return out;
}

void expand_stages(const YNode& seq, IntEnv& env, std::vector<YNode>& out) {
  for (const YNode& item : seq.seq) {
    const YNode* rep = item.is_map() ? item.get("repeat") : nullptr;
    if (!rep) {
      out.push_back(subst_tree(item, env));
      if (out.size() > kMaxExpandedStages)
        fail("E301", item.loc,
             "more than " + std::to_string(kMaxExpandedStages) + " stages after expansion");
      continue;
    }
    for (const auto& [k, v] : item.map)
      if (k.scalar != "repeat" && k.scalar != "stages")
        fail("E101", k.loc, "unknown key '" + k.scalar + "' in a repeat item (expected repeat, stages)");
    const YNode* body = item.get("stages");
    if (!body) fail("E101", item.loc, "repeat item needs 'stages'");
    if (!body->is_seq()) fail("E101", body->loc, "'stages' of a repeat must be a list");
    if (!rep->is_map()) fail("E101", rep->loc, "'repeat' must be a map {var, from, to}");
    for (const auto& [k, v] : rep->map)
      if (k.scalar != "var" && k.scalar != "from" && k.scalar != "to")
        fail("E101", k.loc, "unknown key '" + k.scalar + "' in repeat (expected var, from, to)");
    const YNode* var = rep->get("var");
    const YNode* from = rep->get("from");
    const YNode* to = rep->get("to");
    if (!var || !from || !to) fail("E101", rep->loc, "repeat needs 'var', 'from' and 'to'");
    for (const YNode* x : {var, from, to})
      if (!x->is_scalar()) fail("E101", x->loc, "repeat 'var', 'from' and 'to' must be scalars");
    const std::string& name = var->scalar;
    if (!is_ident(name) || name.find('[') != std::string::npos)
      fail("E101", var->loc, "repeat variable '" + name + "' is not an identifier");
    if (is_reserved(name)) fail("E102", var->loc, "'" + name + "' is a reserved word");
    if (lookup(env, name))
      fail("E102", var->loc, "repeat variable '" + name + "' shadows a const or an enclosing repeat variable");
    const std::int64_t a = eval_int(substitute(from->scalar, env, from->loc), env, from->loc);
    const std::int64_t b = eval_int(substitute(to->scalar, env, to->loc), env, to->loc);
    if (b >= a && b - a >= kMaxRepeatCount)
      fail("E301", rep->loc, "repeat runs more than " + std::to_string(kMaxRepeatCount) + " times");
    for (std::int64_t k = a; k <= b; ++k) {
      env.emplace_back(name, k);
      expand_stages(*body, env, out);
      env.pop_back();
    }
  }
}

}  // namespace

std::int64_t eval_int(std::string_view text, const IntEnv& env, const Loc& loc) {
  return IntParser(text, env, loc).parse();
}

std::string substitute(std::string_view s, const IntEnv& env, const Loc& loc, bool* whole) {
  if (whole) *whole = false;
  std::string out;
  std::size_t i = 0;
  while (i < s.size()) {
    const char c = s[i];
    if (c == '}') fail("E101", loc, "unmatched '}' in '" + std::string(s) + "'");
    if (c != '{') {
      out += c;
      ++i;
      continue;
    }
    std::size_t j = s.find_first_of("{}", i + 1);
    if (j == std::string_view::npos || s[j] == '{')
      fail("E101", loc, "unmatched or nested '{' in '" + std::string(s) + "'");
    const std::string_view inner = s.substr(i + 1, j - i - 1);
    const std::size_t dd = inner.find("..");
    if (dd == std::string_view::npos) {
      out += std::to_string(eval_int(inner, env, loc));
      if (whole && i == 0 && j == s.size() - 1) *whole = true;
      i = j + 1;
      continue;
    }
    // P{a..b}: the identifier characters just before '{' are the prefix of every item.
    std::size_t p = out.size();
    while (p > 0 && is_ident_char(out[p - 1])) --p;
    const std::string prefix = out.substr(p);
    out.resize(p);
    const std::int64_t a = eval_int(inner.substr(0, dd), env, loc);
    const std::int64_t b = eval_int(inner.substr(dd + 2), env, loc);
    if (b >= a) {
      if (b - a >= kMaxRepeatCount) fail("E301", loc, "range is too long");
      for (std::int64_t k = a; k <= b; ++k) {
        if (k > a) out += ", ";
        out += prefix + std::to_string(k);
      }
    } else {
      // Empty: drop the item and one separating comma (the preceding one if there is one).
      std::size_t q = out.size();
      while (q > 0 && out[q - 1] == ' ') --q;
      if (q > 0 && out[q - 1] == ',') {
        out.resize(q - 1);
      } else {
        std::size_t k = j + 1;
        while (k < s.size() && s[k] == ' ') ++k;
        if (k < s.size() && s[k] == ',') {
          ++k;
          while (k < s.size() && s[k] == ' ') ++k;
          j = k - 1;
        }
      }
    }
    i = j + 1;
  }
  return out;
}

IntEnv read_consts(const YNode& root) {
  IntEnv env;
  const YNode* c = root.is_map() ? root.get("consts") : nullptr;
  if (!c || c->kind == YNode::Kind::Null) return env;
  if (!c->is_map()) fail("E101", c->loc, "'consts' must be a map of name: integer");
  for (const auto& [k, v] : c->map) {
    if (!is_ident(k.scalar) || k.scalar.find('[') != std::string::npos)
      fail("E101", k.loc, "const name '" + k.scalar + "' is not an identifier");
    if (is_reserved(k.scalar)) fail("E102", k.loc, "'" + k.scalar + "' is a reserved word");
    std::int64_t x = 0;
    const std::string& s = v.scalar;
    const auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), x);
    if (!v.is_scalar() || v.quoted || s.empty() || ec != std::errc() || p != s.data() + s.size())
      fail("E101", v.loc, "const '" + k.scalar + "' must be an integer");
    env.emplace_back(k.scalar, x);
  }
  return env;
}

YNode expand_spec(const YNode& root, const IntEnv& overrides) {
  if (!root.is_map()) fail("E101", root.loc, "a spec must be a YAML map");
  IntEnv consts = read_consts(root);
  for (const auto& [name, value] : overrides) {
    bool found = false;
    for (auto& [k, v] : consts)
      if (k == name) {
        v = value;
        found = true;
      }
    if (!found) throw std::invalid_argument("const override '" + name + "' is not declared by the spec");
  }
  YNode out = root;
  for (auto& [key, value] : out.map) {
    if (key.scalar == "consts") {
      if (value.is_map())
        for (std::size_t i = 0; i < value.map.size(); ++i) {
          value.map[i].second.scalar = std::to_string(consts[i].second);
          value.map[i].second.quoted = false;
        }
    } else if (key.scalar == "stages" && value.is_seq()) {
      std::vector<YNode> stages;
      IntEnv env = consts;
      expand_stages(value, env, stages);
      value.seq = std::move(stages);
    } else {
      value = subst_tree(value, consts);
    }
  }
  return out;
}

}  // namespace blz::dsl

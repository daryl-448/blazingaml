#include "dsl/expr.hpp"

#include <charconv>

#include "dsl/names.hpp"

namespace blz::dsl {

namespace {

struct Tok {
  enum class T { end, int_, ident, dot, lparen, rparen, lbrack, rbrack, comma, plus, minus, op };
  T t = T::end;
  std::string text;
  std::size_t pos = 0;
};

class Parser {
 public:
  Parser(std::string_view s, const Loc& base, bool exact) : s_(s), base_(base), exact_(exact) {
    lex();
  }

  Expr parse() {
    Expr e = parse_or();
    if (peek().t != Tok::T::end) bad(peek(), "unexpected '" + peek().text + "'");
    return e;
  }

 private:
  Loc at(std::size_t pos) const {
    Loc l = base_;
    if (exact_ && l.col > 0) l.col += int(pos);
    return l;
  }
  [[noreturn]] void bad(const Tok& t, const std::string& why) const {
    fail("E101", at(t.pos), "in '" + std::string(s_) + "': " + why);
  }

  void lex() {
    std::size_t i = 0;
    while (i < s_.size()) {
      const char c = s_[i];
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
        ++i;
        continue;
      }
      Tok t;
      t.pos = i;
      if (c >= '0' && c <= '9') {
        std::size_t j = i;
        while (j < s_.size() && s_[j] >= '0' && s_[j] <= '9') ++j;
        if (j < s_.size() && is_ident_start(s_[j])) {
          t.text = std::string(s_.substr(i, j - i + 1));
          bad(t, "malformed number");
        }
        t.t = Tok::T::int_;
        t.text = std::string(s_.substr(i, j - i));
        i = j;
      } else if (is_ident_start(c)) {
        std::size_t j = i;
        while (j < s_.size() && is_ident_char(s_[j])) ++j;
        // An index directly attached to the name, `C[3]`, is part of it (SPEC §6.2).
        if (j < s_.size() && s_[j] == '[') {
          std::size_t k = j + 1;
          while (k < s_.size() && s_[k] >= '0' && s_[k] <= '9') ++k;
          if (k > j + 1 && k < s_.size() && s_[k] == ']') j = k + 1;
        }
        t.t = Tok::T::ident;
        t.text = std::string(s_.substr(i, j - i));
        i = j;
      } else {
        const std::string_view two = s_.substr(i, 2);
        if (two == "==" || two == "!=" || two == "<=" || two == ">=") {
          t.t = Tok::T::op;
          t.text = std::string(two);
          i += 2;
        } else {
          t.text = std::string(1, c);
          switch (c) {
            case '.': t.t = Tok::T::dot; break;
            case '(': t.t = Tok::T::lparen; break;
            case ')': t.t = Tok::T::rparen; break;
            case '[': t.t = Tok::T::lbrack; break;
            case ']': t.t = Tok::T::rbrack; break;
            case ',': t.t = Tok::T::comma; break;
            case '+': t.t = Tok::T::plus; break;
            case '-': t.t = Tok::T::minus; break;
            case '<':
            case '>': t.t = Tok::T::op; break;
            default: bad(t, "unexpected character '" + t.text + "'");
          }
          ++i;
        }
      }
      toks_.push_back(std::move(t));
    }
    Tok end;
    end.pos = s_.size();
    end.text = "end of expression";
    toks_.push_back(end);
  }

  const Tok& peek(std::size_t ahead = 0) const {
    return toks_[std::min(p_ + ahead, toks_.size() - 1)];
  }
  const Tok& next() {
    const Tok& t = peek();
    if (p_ + 1 < toks_.size()) ++p_;
    return t;
  }
  bool is_word(const Tok& t, std::string_view w) const { return t.t == Tok::T::ident && t.text == w; }
  void expect(Tok::T type, const char* what) {
    if (peek().t != type) bad(peek(), std::string("expected ") + what + ", found '" + peek().text + "'");
    next();
  }

  Expr node(Expr::K k, std::size_t pos) const {
    Expr e;
    e.k = k;
    e.loc = at(pos);
    return e;
  }
  Expr binary(Expr::K k, Expr a, Expr b) const {
    Expr e;
    e.k = k;
    e.loc = a.loc;
    e.kids.push_back(std::move(a));
    e.kids.push_back(std::move(b));
    return e;
  }

  Expr parse_or() {
    Expr e = parse_and();
    while (is_word(peek(), "or")) {
      next();
      e = binary(Expr::K::Or, std::move(e), parse_and());
    }
    return e;
  }
  Expr parse_and() {
    Expr e = parse_not();
    while (is_word(peek(), "and")) {
      next();
      e = binary(Expr::K::And, std::move(e), parse_not());
    }
    return e;
  }
  Expr parse_not() {
    if (is_word(peek(), "not")) {
      const Tok& t = next();
      Expr e = node(Expr::K::Not, t.pos);
      e.kids.push_back(parse_not());
      return e;
    }
    return parse_cmp();
  }
  Expr parse_cmp() {
    Expr lhs = parse_sum();
    const Tok& t = peek();
    if (t.t == Tok::T::op) {
      const std::string op = next().text;
      Expr e = binary(Expr::K::Cmp, std::move(lhs), parse_sum());
      e.cmp = op == "==" ? Expr::Cmp::eq
              : op == "!=" ? Expr::Cmp::ne
              : op == "<"  ? Expr::Cmp::lt
              : op == "<=" ? Expr::Cmp::le
              : op == ">"  ? Expr::Cmp::gt
                           : Expr::Cmp::ge;
      if (peek().t == Tok::T::op) bad(peek(), "comparisons do not chain");
      return e;
    }
    bool negated = false;
    if (is_word(t, "not") && is_word(peek(1), "in")) {
      next();
      negated = true;
    }
    if (is_word(peek(), "in")) {
      next();
      Expr e = node(Expr::K::In, 0);
      e.loc = lhs.loc;
      e.negated = negated;
      e.kids.push_back(std::move(lhs));
      expect(Tok::T::lbrack, "'[' after 'in'");
      if (peek().t != Tok::T::rbrack) {
        e.kids.push_back(parse_sum());
        while (peek().t == Tok::T::comma) {
          next();
          e.kids.push_back(parse_sum());
        }
      }
      expect(Tok::T::rbrack, "']' closing the list");
      return e;
    }
    return lhs;
  }
  Expr parse_sum() {
    Expr e = parse_atom();
    while (peek().t == Tok::T::plus || peek().t == Tok::T::minus) {
      const bool plus = next().t == Tok::T::plus;
      e = binary(plus ? Expr::K::Add : Expr::K::Sub, std::move(e), parse_atom());
    }
    return e;
  }
  Expr parse_atom() {
    const Tok& t = peek();
    switch (t.t) {
      case Tok::T::int_: {
        next();
        Expr e = node(Expr::K::Int, t.pos);
        const auto [p, ec] = std::from_chars(t.text.data(), t.text.data() + t.text.size(), e.ival);
        if (ec != std::errc() || e.ival > (std::int64_t{1} << 40)) bad(t, "integer literal out of range");
        return e;
      }
      case Tok::T::lparen: {
        next();
        Expr e = parse_or();
        expect(Tok::T::rparen, "')'");
        return e;
      }
      case Tok::T::ident: {
        if (t.text == "and" || t.text == "or" || t.text == "not" || t.text == "in")
          bad(t, "unexpected '" + t.text + "'");
        next();
        if (t.text == "len") {
          expect(Tok::T::lparen, "'(' after len");
          const Tok& v = peek();
          if (v.t != Tok::T::ident || is_reserved(v.text)) bad(v, "len() takes a set variable");
          next();
          expect(Tok::T::rparen, "')' closing len(");
          Expr e = node(Expr::K::Len, t.pos);
          e.name = v.text;
          return e;
        }
        if (t.text == "min") {  // SPEC v2 §8 S5: min(iexpr, iexpr)
          expect(Tok::T::lparen, "'(' after min");
          Expr a = parse_sum();
          expect(Tok::T::comma, "',' between the two arguments of min(");
          Expr b = parse_sum();
          expect(Tok::T::rparen, "')' closing min(");
          Expr e = binary(Expr::K::Min, std::move(a), std::move(b));
          e.loc = at(t.pos);
          return e;
        }
        if (peek().t == Tok::T::dot) {
          next();
          const Tok& f = peek();
          if (!is_word(f, "t")) bad(f, "the only edge field is '.t'");
          next();
          Expr e = node(Expr::K::EdgeT, t.pos);
          e.name = t.text;
          return e;
        }
        Expr e = node(Expr::K::Name, t.pos);
        e.name = t.text;
        return e;
      }
      default:
        bad(t, "expected a value, found '" + t.text + "'");
    }
  }

  std::string_view s_;
  Loc base_;
  bool exact_;
  std::vector<Tok> toks_;
  std::size_t p_ = 0;
};

int prec(const Expr& e) {
  switch (e.k) {
    case Expr::K::Or: return 1;
    case Expr::K::And: return 2;
    case Expr::K::Not: return 3;
    case Expr::K::Cmp:
    case Expr::K::In: return 4;
    case Expr::K::Add:
    case Expr::K::Sub: return 5;
    default: return 6;
  }
}

std::string wrap(const Expr& e, int min_prec) {
  std::string s = to_string(e);
  return prec(e) < min_prec ? "(" + s + ")" : s;
}

}  // namespace

Expr parse_expr(std::string_view text, const Loc& base, bool exact_cols) {
  return Parser(text, base, exact_cols).parse();
}

const char* cmp_text(Expr::Cmp c) {
  switch (c) {
    case Expr::Cmp::eq: return "==";
    case Expr::Cmp::ne: return "!=";
    case Expr::Cmp::lt: return "<";
    case Expr::Cmp::le: return "<=";
    case Expr::Cmp::gt: return ">";
    case Expr::Cmp::ge: return ">=";
  }
  return "?";
}

Expr::Cmp flip(Expr::Cmp c) {
  switch (c) {
    case Expr::Cmp::lt: return Expr::Cmp::gt;
    case Expr::Cmp::le: return Expr::Cmp::ge;
    case Expr::Cmp::gt: return Expr::Cmp::lt;
    case Expr::Cmp::ge: return Expr::Cmp::le;
    default: return c;
  }
}

std::string to_string(const Expr& e) {
  switch (e.k) {
    case Expr::K::Int: return std::to_string(e.ival);
    case Expr::K::Name: return e.name;
    case Expr::K::EdgeT: return e.name + ".t";
    case Expr::K::Len: return "len(" + e.name + ")";
    case Expr::K::Min: return "min(" + to_string(e.kids[0]) + ", " + to_string(e.kids[1]) + ")";
    case Expr::K::Add: return wrap(e.kids[0], 5) + " + " + wrap(e.kids[1], 6);
    case Expr::K::Sub: return wrap(e.kids[0], 5) + " - " + wrap(e.kids[1], 6);
    case Expr::K::Cmp:
      return wrap(e.kids[0], 5) + " " + cmp_text(e.cmp) + " " + wrap(e.kids[1], 5);
    case Expr::K::In: {
      std::string s = wrap(e.kids[0], 5) + (e.negated ? " not in [" : " in [");
      for (std::size_t i = 1; i < e.kids.size(); ++i) s += (i > 1 ? ", " : "") + wrap(e.kids[i], 5);
      return s + "]";
    }
    case Expr::K::And: return wrap(e.kids[0], 2) + " and " + wrap(e.kids[1], 3);
    case Expr::K::Or: return wrap(e.kids[0], 1) + " or " + wrap(e.kids[1], 2);
    case Expr::K::Not: return "not " + wrap(e.kids[0], 3);
  }
  return "?";
}

}  // namespace blz::dsl

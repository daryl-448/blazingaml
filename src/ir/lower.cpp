// Lowering (SPEC §7.1 steps 3–5 and 7): resolve names, type-check, classify constraints (§6.4)
// and build the level structure of the IR (§7.2).
#include <map>
#include <set>

#include "ir/ir.hpp"

namespace blz::ir {

namespace {

using dsl::CompileError;
using dsl::fail;
using dsl::format_loc;
using dsl::Sym;
using dsl::Ty;

enum class EdgeRole { none, trigger, witness, scanned };

struct Symbol {
  Sym sym = Sym::unresolved;
  Ty ty = Ty::none;
  Loc loc;
  std::int64_t value = 0;  // consts
  EdgeRole role = EdgeRole::none;
  std::string owner;       // the stage that bound a witness edge
  bool latest = false;     // a latest witness (`witness: latest`, SPEC v2 §8 S5), else the earliest
};

// "the earliest witness of stage 'x'" or "the latest witness of stage 'x'".
std::string witness_text(const Symbol& s) {
  return std::string(s.latest ? "the latest" : "the earliest") + " witness of stage '" + s.owner + "'";
}

Symbol symbol(Sym sym, Ty ty, const Loc& loc, EdgeRole role = EdgeRole::none, std::string owner = {}) {
  Symbol s;
  s.sym = sym;
  s.ty = ty;
  s.loc = loc;
  s.role = role;
  s.owner = std::move(owner);
  return s;
}

using Scope = std::map<std::string, Symbol>;

const char* ty_name(Ty t) {
  switch (t) {
    case Ty::none: return "nothing";
    case Ty::node: return "a node";
    case Ty::edge: return "an edge";
    case Ty::time: return "a time";
    case Ty::int_: return "an int";
    case Ty::set: return "a set";
    case Ty::bool_: return "a predicate";
  }
  return "?";
}

void edge_refs(const Expr& e, std::vector<const Expr*>& out) {
  if (e.k == Expr::K::EdgeT) out.push_back(&e);
  for (const Expr& k : e.kids) edge_refs(k, out);
}

class Lowerer {
 public:
  explicit Lowerer(const dsl::Spec& s) : s_(s) {}

  Pattern run() {
    collect_declarations();

    Pattern p;
    p.name = s_.pattern.s;
    p.block = s_.block.s;
    p.name_loc = s_.pattern.loc;
    p.block_loc = s_.block.loc;
    p.visibility_loc = s_.visibility_loc;
    p.file = s_.file;
    p.span_param = s_.span.s;
    p.lower_open = s_.lower_open;
    p.visibility = s_.visibility;
    p.work_limit = s_.work_limit;
    p.e0 = s_.trig_edge.s;
    p.src_var = s_.trig_src.s;
    p.dst_var = s_.trig_dst.s;
    p.consts = s_.consts;

    Scope scope;
    for (const auto& [name, value] : s_.consts) {
      Symbol c = symbol(Sym::constant, Ty::int_, const_loc(name));
      c.value = value;
      scope[name] = c;
    }
    scope[s_.span.s] = symbol(Sym::param, Ty::int_, s_.span.loc);
    scope[s_.trig_edge.s] = symbol(Sym::edge, Ty::edge, s_.trig_edge.loc, EdgeRole::trigger);
    scope[s_.trig_src.s] = symbol(Sym::node, Ty::node, s_.trig_src.loc);
    scope[s_.trig_dst.s] = symbol(Sym::node, Ty::node, s_.trig_dst.loc);

    for (const dsl::Constraint& c : s_.constraints) {
      Expr pred = c.pred;
      expect_ty(pred, scope, Ty::bool_, "a constraint");
      std::vector<const Expr*> refs;
      edge_refs(pred, refs);
      for (const Expr* r : refs)
        if (scope.at(r->name).role != EdgeRole::trigger)
          fail("E203", r->loc, "trigger-level constraints may only use the trigger's own time");
      p.trigger_filters.push_back(std::move(pred));
    }

    p.levels.emplace_back();  // root
    int depth = 0;
    for (const dsl::Stage& st : s_.stages) {
      Stage out = lower_stage(st, scope, p);
      if (out.binding) {
        if (++depth > kMaxDepth)
          fail("E301", st.loc,
               "stage '" + st.id.s + "' is binding level " + std::to_string(depth) +
                   "; at most " + std::to_string(kMaxDepth) + " are allowed after expansion");
        Level lv;
        lv.binding = std::move(out);
        p.levels.push_back(std::move(lv));
      } else {
        p.levels.back().checks.push_back(std::move(out));
      }
    }
    return p;
  }

 private:
  Loc const_loc(const std::string& name) const {
    auto it = decls_.find(name);
    return it == decls_.end() ? Loc{s_.file, 0, 0} : it->second.first;
  }

  void declare(const dsl::Name& n, const std::string& what) {
    auto [it, fresh] = decls_.emplace(n.s, std::make_pair(n.loc, what));
    if (!fresh)
      fail("E102", n.loc,
           "'" + n.s + "' is already declared at " + format_loc(it->second.first) + " (" +
               it->second.second + "); every name is declared once per spec");
  }

  // Every declaration in the file, so that duplicates are caught and an out-of-scope use can say
  // where the name does live.
  void collect_declarations() {
    for (std::size_t i = 0; i < s_.consts.size(); ++i)
      declare({s_.consts[i].first, s_.const_locs[i]}, "a const");
    declare(s_.span, "the window span parameter");
    declare(s_.trig_edge, "the trigger edge");
    declare(s_.trig_src, "the trigger source");
    declare(s_.trig_dst, "the trigger destination");
    std::map<std::string, Loc> ids;
    for (const dsl::Stage& st : s_.stages) {
      auto [it, fresh] = ids.emplace(st.id.s, st.id.loc);
      if (!fresh)
        fail("E102", st.id.loc,
             "stage id '" + st.id.s + "' is already used at " + format_loc(it->second));
      const std::string in = " of stage '" + st.id.s + "'";
      if (st.edge) declare(*st.edge, "the edge" + in);
      for (const auto& e : st.edges) declare(e, "an edge" + in);
      if (st.dst_var) declare(*st.dst_var, "the dst_var" + in);
      if (st.element) declare(*st.element, "the element" + in + ", visible only in its constraints");
      if (st.output) declare(*st.output, "the output" + in);
    }
  }

  const Symbol& lookup(const std::string& name, const Loc& loc, const Scope& sc) const {
    auto it = sc.find(name);
    if (it != sc.end()) return it->second;
    auto d = decls_.find(name);
    if (d != decls_.end())
      fail("E102", loc,
           "'" + name + "' is not in scope here (declared at " + format_loc(d->second.first) +
               " as " + d->second.second + ")");
    fail("E102", loc, "unknown name '" + name + "'");
  }

  // Resolves names in `e` and returns its type (E102 / E103).
  Ty check(Expr& e, const Scope& sc) const {
    auto mismatch = [&](const std::string& why) -> void { fail("E103", e.loc, why + " in '" + to_string(e) + "'"); };
    switch (e.k) {
      case Expr::K::Int:
        e.ty = Ty::int_;
        break;
      case Expr::K::Name: {
        if (e.name == "t") {
          e.sym = Sym::trig_time;
          e.ty = Ty::time;
          break;
        }
        const Symbol& s = lookup(e.name, e.loc, sc);
        e.sym = s.sym;
        if (s.sym == Sym::edge)
          fail("E103", e.loc, "'" + e.name + "' is an edge; only its time '" + e.name + ".t' is a value");
        if (s.sym == Sym::set_var)
          fail("E103", e.loc, "'" + e.name + "' is a set; it can only be used as len(" + e.name + ")");
        if (s.sym == Sym::constant) e.ival = s.value;
        e.ty = s.ty;
        break;
      }
      case Expr::K::EdgeT: {
        const Symbol& s = lookup(e.name, e.loc, sc);
        if (s.sym != Sym::edge)
          fail("E103", e.loc, "'" + e.name + "' is " + ty_name(s.ty) + ", not an edge; '.t' needs an edge variable");
        e.sym = Sym::edge;
        e.ty = Ty::time;
        break;
      }
      case Expr::K::Len: {
        const Symbol& s = lookup(e.name, e.loc, sc);
        if (s.sym != Sym::set_var)
          fail("E103", e.loc, "len() takes a set (an intersection output); '" + e.name + "' is " + ty_name(s.ty));
        e.sym = Sym::set_var;
        e.ty = Ty::int_;
        break;
      }
      case Expr::K::Add:
      case Expr::K::Sub: {
        const Ty a = check(e.kids[0], sc), b = check(e.kids[1], sc);
        if (a == Ty::int_ && b == Ty::int_) e.ty = Ty::int_;
        else if (a == Ty::time && b == Ty::int_) e.ty = Ty::time;
        else if (e.k == Expr::K::Add && a == Ty::int_ && b == Ty::time) e.ty = Ty::time;
        else mismatch(std::string("cannot ") + (e.k == Expr::K::Add ? "add " : "subtract ") + ty_name(b) +
                      (e.k == Expr::K::Add ? " to " : " from ") + ty_name(a));
        break;
      }
      case Expr::K::Min: {  // SPEC v2 §8 S5
        const Ty a = check(e.kids[0], sc), b = check(e.kids[1], sc);
        if (a != Ty::int_ || b != Ty::int_)
          mismatch(std::string("min() takes two ints, not ") + ty_name(a) + " and " + ty_name(b));
        e.ty = Ty::int_;
        break;
      }
      case Expr::K::Cmp: {
        const Ty a = check(e.kids[0], sc), b = check(e.kids[1], sc);
        const bool eq = e.cmp == Expr::Cmp::eq || e.cmp == Expr::Cmp::ne;
        if (a != b) mismatch(std::string("cannot compare ") + ty_name(a) + " with " + ty_name(b));
        if (a == Ty::node && !eq) mismatch("nodes only compare with == and !=");
        if (a != Ty::node && a != Ty::time && a != Ty::int_) mismatch(std::string("cannot compare ") + ty_name(a));
        e.ty = Ty::bool_;
        break;
      }
      case Expr::K::In: {
        for (Expr& k : e.kids)
          if (check(k, sc) != Ty::node)
            fail("E103", k.loc, "'in' compares nodes; '" + to_string(k) + "' is " + ty_name(k.ty));
        e.ty = Ty::bool_;
        break;
      }
      case Expr::K::And:
      case Expr::K::Or:
      case Expr::K::Not:
        for (Expr& k : e.kids)
          if (check(k, sc) != Ty::bool_)
            fail("E103", k.loc, "'" + to_string(k) + "' is " + ty_name(k.ty) + ", not a predicate");
        e.ty = Ty::bool_;
        break;
    }
    return e.ty;
  }

  void expect_ty(Expr& e, const Scope& sc, Ty want, const std::string& what) const {
    const Ty got = check(e, sc);
    if (got != want)
      fail("E103", e.loc, what + " must be " + ty_name(want) + "; '" + to_string(e) + "' is " + ty_name(got));
  }

  const Symbol& use_node(const dsl::Name& n, const Scope& sc, const std::string& what) const {
    const Symbol& s = lookup(n.s, n.loc, sc);
    if (s.sym != Sym::node)
      fail("E103", n.loc, what + " must be a node variable; '" + n.s + "' is " + ty_name(s.ty));
    return s;
  }

  static void split_or(const Expr& e, std::vector<Expr>& out) {
    if (e.k == Expr::K::Or) {
      split_or(e.kids[0], out);
      split_or(e.kids[1], out);
    } else {
      out.push_back(e);
    }
  }

  static bool mentions_scanned(const Expr& e, const Scope& sc) {
    std::vector<const Expr*> r;
    edge_refs(e, r);
    for (const Expr* x : r)
      if (sc.at(x->name).role == EdgeRole::scanned) return true;
    return false;
  }

  // The comparisons in `e` whose two sides both read an edge the stage scans (SPEC v2 §6.2), each
  // with whether it sits inside an `or`.
  static void cross_cmps(const Expr& e, const Scope& sc, bool in_or, std::vector<std::pair<const Expr*, bool>>& out) {
    if (e.k == Expr::K::Cmp && mentions_scanned(e.kids[0], sc) && mentions_scanned(e.kids[1], sc))
      out.emplace_back(&e, in_or);
    for (const Expr& k : e.kids) cross_cmps(k, sc, in_or || e.k == Expr::K::Or, out);
  }

  // `ea.t OP eb.t` between the two scanned edges of an intersection (SPEC v2 §6.2): the stage's
  // order constraint, lowered through the earliest and latest witnesses (D-27). E205 otherwise.
  static void order_constraint(const Expr& d, Stage& st) {
    const std::string what = "'" + to_string(d) + "' compares the times of edges this stage scans";
    if (d.cmp == Expr::Cmp::eq || d.cmp == Expr::Cmp::ne)
      fail("E205", d.loc,
           what + " with " + dsl::cmp_text(d.cmp) + "; an order constraint uses <, <=, > or >=");
    const Expr& l = d.kids[0];
    const Expr& r = d.kids[1];
    if (l.k != Expr::K::EdgeT || r.k != Expr::K::EdgeT)
      fail("E205", d.loc, what + " with an offset; an order constraint compares the bare times: ea.t OP eb.t");
    if (st.op != Op::intersection || l.name == r.name)
      fail("E205", d.loc,
           what + "; an order constraint compares the two scanned edges of an intersection with each other");
    if (st.order)
      fail("E205", d.loc,
           what + "; stage '" + st.id + "' already has the order constraint at " + format_loc(st.order->loc) +
               ", and an intersection takes at most one");
    Order o;
    o.a = st.operands[0].edge;
    o.b = st.operands[1].edge;
    o.skip = l.name == o.a ? d.cmp : dsl::flip(d.cmp);  // now: a.t skip b.t
    // Keep y when some witness pair fails the skip (SPEC v2 §6.2, the lowering table).
    switch (o.skip) {
      case Expr::Cmp::ge: o.keep = Expr::Cmp::lt; o.b_latest = true; break;  // min(a) < max(b)
      case Expr::Cmp::gt: o.keep = Expr::Cmp::le; o.b_latest = true; break;  // min(a) <= max(b)
      case Expr::Cmp::le: o.keep = Expr::Cmp::gt; o.a_latest = true; break;  // max(a) > min(b)
      case Expr::Cmp::lt: o.keep = Expr::Cmp::ge; o.a_latest = true; break;  // max(a) >= min(b)
      default: break;
    }
    o.loc = d.loc;
    st.order = std::move(o);
  }

  // Classifies one constraint of a stage (§6.4) into time bounds on its operands, node filters
  // or the order constraint of an intersection (SPEC v2 §6.2).
  void classify(const dsl::Constraint& c, const Scope& sc, Stage& st) const {
    Expr pred = c.pred;
    expect_ty(pred, sc, Ty::bool_, "a constraint");
    // An order constraint must be a skip_if of its own: not in a break_if, not inside `or`.
    std::vector<std::pair<const Expr*, bool>> cross;
    cross_cmps(pred, sc, false, cross);
    for (const auto& [x, in_or] : cross) {
      if (c.is_break)
        fail("E205", x->loc,
             "break_if '" + to_string(*x) + "' compares the times of edges this stage scans; an order "
             "constraint is a skip_if");
      if (in_or)
        fail("E205", x->loc,
             "'" + to_string(*x) + "' compares the times of edges this stage scans inside 'or'; an order "
             "constraint must be a skip_if of its own");
    }
    // skip_if (a or b) == skip_if a; skip_if b (and the same for break_if).
    std::vector<Expr> parts;
    split_or(pred, parts);
    for (Expr& d : parts) {
      std::vector<const Expr*> refs;
      edge_refs(d, refs);
      std::vector<const Expr*> scanned, witness;
      for (const Expr* r : refs) {
        const EdgeRole role = sc.at(r->name).role;
        if (role == EdgeRole::scanned) scanned.push_back(r);
        if (role == EdgeRole::witness) witness.push_back(r);
      }
      if (!scanned.empty() && d.k == Expr::K::Cmp && mentions_scanned(d.kids[0], sc) &&
          mentions_scanned(d.kids[1], sc)) {
        order_constraint(d, st);
        continue;
      }
      if (scanned.empty()) {
        if (c.is_break)
          fail("E202", d.loc,
               "break_if '" + to_string(d) + "' bounds no edge this stage scans; a break_if must be a "
               "time bound (ex.t > τ or ex.t >= τ) on one of them, and node tests belong in skip_if");
        if (!witness.empty()) {
          const Symbol& ws = sc.at(witness[0]->name);
          fail("E204", witness[0]->loc,
               "'" + witness[0]->name + ".t' is " + witness_text(ws) + "; an earlier stage's witness time may only be " +
                   (ws.latest ? "an upper" : "a lower") + " bound of a later scan");
        }
        st.filters.push_back(std::move(d));
        continue;
      }
      // A time bound: exactly `ex.t OP τ` or `τ OP ex.t`, ex scanned by this stage, τ free of ex.
      if (d.k != Expr::K::Cmp || d.kids[0].ty != Ty::time)
        fail("E203", d.loc,
             "'" + to_string(d) + "' mixes the scanned edge '" + scanned[0]->name +
                 "' with other tests; a constraint on a scanned edge must be one comparison ex.t OP τ");
      const Expr& lhs = d.kids[0];
      const Expr& rhs = d.kids[1];
      const bool l = mentions_scanned(lhs, sc);  // only one side does: both is an order constraint
      const Expr& ex = l ? lhs : rhs;
      const Expr& tau = l ? rhs : lhs;
      if (ex.k != Expr::K::EdgeT)
        fail("E203", ex.loc,
             "'" + to_string(ex) + "': a scanned edge's time must stand alone on its side of a bound");
      Expr::Cmp op = l ? d.cmp : dsl::flip(d.cmp);  // now: ex.t OP τ
      if (op == Expr::Cmp::eq || op == Expr::Cmp::ne)
        fail("E203", d.loc, "'" + to_string(d) + "': == and != on a scanned edge's time are not bounds");
      bool upper = false, strict = false;
      if (c.is_break) {
        if (op == Expr::Cmp::lt || op == Expr::Cmp::le)
          fail("E201", d.loc,
               "break_if '" + to_string(d) + "' is not monotone: scans run in increasing time, so a "
               "break_if may only stop at an upper bound (ex.t > τ or ex.t >= τ)");
        upper = true;
        strict = op == Expr::Cmp::ge;  // stop at ts >= τ: keep ts < τ
      } else {
        switch (op) {
          case Expr::Cmp::le: strict = true; break;     // skip ts <= τ: keep ts > τ
          case Expr::Cmp::lt: strict = false; break;    // skip ts < τ: keep ts >= τ
          case Expr::Cmp::ge: upper = true; strict = true; break;   // keep ts < τ
          case Expr::Cmp::gt: upper = true; strict = false; break;  // keep ts <= τ
          default: break;
        }
      }
      // The earliest witness may only bound a later scan from below, the latest (SPEC v2 §8 S5) only
      // from above: each is then the best witness for every later use, so the rule is exact.
      for (const Expr* x : witness) {
        const Symbol& ws = sc.at(x->name);
        if (upper && !ws.latest)
          fail("E204", x->loc,
               "'" + x->name + ".t' (" + witness_text(ws) +
                   ") is used in an upper bound; an earlier stage's witness time may only be a lower "
                   "bound of a later scan, or the earliest-witness rule is not exact (a latest witness, "
                   "witness: latest, may bound from above)");
        if (!upper && ws.latest)
          fail("E204", x->loc,
               "'" + x->name + ".t' (" + witness_text(ws) +
                   ") is used in a lower bound; a latest witness's time may only be an upper bound of a "
                   "later scan, or the latest-witness rule is not exact");
      }
      // A witness lower bound is exact only on a scan whose result is an existence: exists, a
      // distinct for_all, a binding intersection. On an aggregate (count, a check intersection,
      // union or difference, for_all over every edge) the result would depend on which witness
      // was taken.
      const bool aggregate = (st.op == Op::count) || (st.op == Op::intersection && !st.binding) ||
                             st.op == Op::union_ || st.op == Op::difference ||
                             (st.op == Op::for_all && st.multiplicity == Mult::edges);
      if (!witness.empty() && aggregate)
        fail("E204", witness[0]->loc,
             "'" + witness[0]->name + ".t' (" + witness_text(sc.at(witness[0]->name)) +
                 ") bounds an aggregating scan; its result would depend on which witness was taken, "
                 "so witness times may only bound exists, distinct for_all and binding intersection scans");
      Operand* target = nullptr;
      for (Operand& o : st.operands)
        if (o.edge == ex.name) target = &o;
      TimeBound b{tau, strict, false, d.loc};
      (upper ? target->hi : target->lo).push_back(std::move(b));
    }
  }

  Operand operand(const dsl::Port& port, const dsl::Name& edge, const Scope& sc, const std::string& what) const {
    use_node(port.var, sc, what);
    Operand o;
    o.node = port.var.s;
    o.dir = port.dir;
    o.edge = edge.s;
    return o;
  }

  Stage lower_stage(const dsl::Stage& st, Scope& scope, const Pattern& p) const {
    Stage out;
    out.id = st.id.s;
    out.op = st.op;
    out.loc = st.loc;
    const std::string where = "stage '" + st.id.s + "'";

    // Operands and the names this stage introduces for its own constraints.
    Scope local = scope;
    auto add_scanned = [&](const dsl::Name& e) {
      local[e.s] = symbol(Sym::edge, Ty::edge, e.loc, EdgeRole::scanned, st.id.s);
    };
    switch (st.op) {
      case Op::for_all:
        out.binding = true;
        out.operands.push_back(operand(*st.src, *st.edge, scope, "'src' of " + where));
        out.bind_var = st.dst_var->s;
        out.multiplicity = st.multiplicity.value_or(Mult::distinct);
        out.latest_witness = st.latest_witness;
        add_scanned(*st.edge);
        local[out.bind_var] = symbol(Sym::node, Ty::node, st.dst_var->loc);
        break;
      case Op::intersection:
        out.binding = st.dst_var.has_value();
        out.operands.push_back(operand(*st.src, st.edges[0], scope, "'src' of " + where));
        out.operands.push_back(operand(*st.dst, st.edges[1], scope, "'dst' of " + where));
        add_scanned(st.edges[0]);
        add_scanned(st.edges[1]);
        if (out.binding) {
          out.bind_var = st.dst_var->s;
          local[out.bind_var] = symbol(Sym::node, Ty::node, st.dst_var->loc);
        } else {
          out.output_var = st.output->s;
          if (st.element) {
            out.element_var = st.element->s;
            local[out.element_var] = symbol(Sym::node, Ty::node, st.element->loc);
          }
        }
        break;
      case Op::union_:
      case Op::difference:  // SPEC v2 §8 S4: a check over two operands, like a check intersection
        out.operands.push_back(operand(*st.src, st.edges[0], scope, "'src' of " + where));
        out.operands.push_back(operand(*st.dst, st.edges[1], scope, "'dst' of " + where));
        add_scanned(st.edges[0]);
        add_scanned(st.edges[1]);
        out.output_var = st.output->s;
        if (st.element) {
          out.element_var = st.element->s;
          local[out.element_var] = symbol(Sym::node, Ty::node, st.element->loc);
        }
        break;
      case Op::exists:
        out.operands.push_back(operand(*st.src, *st.edge, scope, "'src' of " + where));
        use_node(*st.node, scope, "'node' of " + where);
        out.probe_node = st.node->s;
        add_scanned(*st.edge);
        break;
      case Op::count:
        out.operands.push_back(operand(*st.src, *st.edge, scope, "'src' of " + where));
        out.output_var = st.output->s;
        out.multiplicity = st.multiplicity.value_or(Mult::distinct);
        add_scanned(*st.edge);
        if (st.element) {
          out.element_var = st.element->s;
          local[out.element_var] = symbol(Sym::node, Ty::node, st.element->loc);
        }
        break;
    }
    out.gate = st.gate.value_or(false);

    // The window's lower bound on every scanned edge (§6.4): ts > t - span (open) or >= (closed).
    for (Operand& o : out.operands) {
      Expr t;
      t.k = Expr::K::Name;
      t.name = "t";
      t.sym = Sym::trig_time;
      t.ty = Ty::time;
      t.loc = st.loc;
      Expr span;
      span.k = Expr::K::Name;
      span.name = p.span_param;
      span.sym = Sym::param;
      span.ty = Ty::int_;
      span.loc = st.loc;
      Expr lo;
      lo.k = Expr::K::Sub;
      lo.ty = Ty::time;
      lo.loc = st.loc;
      lo.kids = {std::move(t), std::move(span)};
      o.lo.push_back({std::move(lo), p.lower_open, true, st.loc});
    }

    for (const dsl::Constraint& c : st.constraints) classify(c, local, out);

    // Emit sees the bound variable (binding stages) or the output (check stages), not the element.
    if (st.emit) {
      Scope esc = scope;
      if (out.binding) esc[out.bind_var] = local[out.bind_var];
      if (!out.output_var.empty())
        esc[out.output_var] = symbol(st.op == Op::count ? Sym::int_var : Sym::set_var,
                                     st.op == Op::count ? Ty::int_ : Ty::set, st.output->loc);
      Emit e;
      e.size = st.emit->size;
      e.loc = st.emit->loc;
      e.min_size = std::max<std::int64_t>(st.emit->min_size, 2);
      expect_ty(e.size, esc, Ty::int_, "emit size");
      out.emit = std::move(e);
    }

    // What later stages see.
    auto export_name = [&](const std::string& name, const Symbol& s) {
      scope[name] = s;
      out.exports.push_back(name);
    };
    if (out.binding) {
      export_name(out.bind_var, local.at(out.bind_var));
      // An ordered intersection's edges have no single witness, so they stay local (SPEC v2 §6.2;
      // a later use is E102).
      if (!out.order)
        for (const Operand& o : out.operands) {
          Symbol ws = symbol(Sym::edge, Ty::edge, local.at(o.edge).loc, EdgeRole::witness, st.id.s);
          ws.latest = out.latest_witness;
          export_name(o.edge, ws);
        }
    } else if (st.op == Op::exists) {
      // The witness exists only when the probe found one, i.e. only behind a gate.
      if (out.gate)
        export_name(out.operands[0].edge,
                    symbol(Sym::edge, Ty::edge, local.at(out.operands[0].edge).loc, EdgeRole::witness, st.id.s));
    } else {
      const bool is_count = st.op == Op::count;
      export_name(out.output_var, symbol(is_count ? Sym::int_var : Sym::set_var,
                                         is_count ? Ty::int_ : Ty::set, st.output->loc));
    }
    return out;
  }

  const dsl::Spec& s_;
  std::map<std::string, std::pair<Loc, std::string>> decls_;
};

}  // namespace

const Stage* Pattern::find_stage(std::string_view id) const {
  for (const Level& lv : levels) {
    if (lv.binding && lv.binding->id == id) return &*lv.binding;
    for (const Stage& s : lv.checks)
      if (s.id == id) return &s;
  }
  return nullptr;
}

Pattern lower(const dsl::Spec& spec) { return Lowerer(spec).run(); }

}  // namespace blz::ir

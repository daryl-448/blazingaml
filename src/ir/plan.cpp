// Planner (SPEC §7.3). Every decision is also recorded as a note, which the backends emit as a
// comment in the generated code.
#include <set>

#include "ir/ir.hpp"

namespace blz::ir {

namespace {

using dsl::Sym;
using dsl::Ty;

Expr int_lit(std::int64_t v, const Loc& loc) {
  Expr e;
  e.k = Expr::K::Int;
  e.ival = v;
  e.ty = Ty::int_;
  e.loc = loc;
  return e;
}

// x + d (d = +1 or -1), folded into a trailing integer literal when x has one:
// (h.t + 1) - 1 -> h.t, (t - 5) + 1 -> t - 4, t + 1 -> t + 1.
Expr shift(const Expr& x, int d) {
  if ((x.k == Expr::K::Add || x.k == Expr::K::Sub) && x.kids[1].k == Expr::K::Int) {
    const std::int64_t c = (x.k == Expr::K::Add ? x.kids[1].ival : -x.kids[1].ival) + d;
    if (c == 0) return x.kids[0];
    Expr e = x;
    e.k = c > 0 ? Expr::K::Add : Expr::K::Sub;
    e.kids[1] = int_lit(c > 0 ? c : -c, x.loc);
    return e;
  }
  Expr e;
  e.k = d > 0 ? Expr::K::Add : Expr::K::Sub;
  e.ty = x.ty;
  e.loc = x.loc;
  e.kids = {x, int_lit(1, x.loc)};
  return e;
}

std::vector<Expr> inclusive(const std::vector<TimeBound>& bounds, bool lower) {
  std::vector<Expr> out;
  std::set<std::string> seen;
  for (const TimeBound& b : bounds) {
    Expr e = b.strict ? shift(b.expr, lower ? +1 : -1) : b.expr;
    if (seen.insert(to_string(e)).second) out.push_back(std::move(e));
  }
  return out;
}

const char* dir_name(Dir d) { return d == Dir::in ? "in" : "out"; }

std::string list_of(const std::vector<Expr>& xs) {
  std::string s;
  for (const Expr& x : xs) s += (s.empty() ? "" : ", ") + to_string(x);
  return s;
}

// Visibility is a runtime parameter (SPEC v2 §6.1, D-22): the range end is chosen once per trigger.
std::string range_note(const Operand& o, const std::string& e0) {
  std::string s = o.edge + " in " + o.node + "." + dir_name(o.dir) + ": ts >= ";
  s += o.lo_incl.size() == 1 ? to_string(o.lo_incl[0]) : "max(" + list_of(o.lo_incl) + ")";
  if (!o.hi_incl.empty())
    s += ", ts <= " + (o.hi_incl.size() == 1 ? to_string(o.hi_incl[0]) : "min(" + list_of(o.hi_incl) + ")");
  s += ", visible: eid <= eid(" + e0 + ") (stream) or ts <= t (timestamp), set once per trigger";
  if (!o.hi_incl.empty()) s += "; the upper bounds fold into the range end (no per-element break)";
  return s;
}

void plan_stage(Stage& st, const Pattern& p) {
  for (Operand& o : st.operands) {
    o.lo_incl = inclusive(o.lo, true);
    o.hi_incl = inclusive(o.hi, false);
  }
  switch (st.op) {
    case Op::for_all: {
      Operand& o = st.operands[0];
      o.access = st.multiplicity == Mult::distinct ? Access::scan_distinct : Access::scan_edges;
      st.notes.push_back(st.multiplicity == Mult::distinct
                             ? "scan T-CSR " + o.node + "." + dir_name(o.dir) +
                                   " distinct-first: each " + st.bind_var + " once, " + o.edge +
                                   " = its earliest witness in range"
                             : "scan T-CSR " + o.node + "." + dir_name(o.dir) + ": every edge binds " +
                                   st.bind_var);
      break;
    }
    case Op::count: {
      Operand& o = st.operands[0];
      o.access = st.multiplicity == Mult::distinct ? Access::scan_distinct : Access::scan_edges;
      st.notes.push_back("scan T-CSR " + o.node + "." + dir_name(o.dir) +
                         (st.multiplicity == Mult::distinct ? " distinct-first" : ", every edge") +
                         "; " + st.output_var + " = number of elements that pass the filters (count only)");
      break;
    }
    case Op::exists: {
      Operand& o = st.operands[0];
      o.access = Access::probe;
      st.notes.push_back("probe P-CSR " + o.node + "." + dir_name(o.dir) + " for " + st.probe_node +
                         ": two binary searches; " + o.edge + " = the earliest witness in range");
      break;
    }
    case Op::intersection: {
      for (Operand& o : st.operands) o.access = Access::intersect;
      const Operand& a = st.operands[0];
      const Operand& b = st.operands[1];
      st.notes.push_back("intersection " + a.node + "." + dir_name(a.dir) + " ∩ " + b.node + "." +
                         dir_name(b.dir) +
                         " over distinct nodes: per trigger, scan the side with the smaller window "
                         "(T-CSR, distinct-first) and probe the other (P-CSR)");
      if (st.order) {
        // Which witness each side needs, and where it comes from (SPEC v2 §6.2).
        const Order& r = *st.order;
        auto ext = [](bool latest, const std::string& e) { return std::string(latest ? "max(" : "min(") + e + ")"; };
        auto from = [&](bool latest, bool scanned) {
          return std::string(latest ? "P-CSR pair_latest" : scanned ? "the scanned entry (distinct-first)" : "the probe (pair_exists)") +
                 (latest && scanned ? " on the scanned side" : latest ? " (the probe)" : "");
        };
        st.notes.push_back("order constraint: skip_if " + r.a + ".t " + dsl::cmp_text(r.skip) + " " + r.b +
                           ".t (existential over witness pairs), so an element is kept when " + ext(r.a_latest, r.a) +
                           " " + dsl::cmp_text(r.keep) + " " + ext(r.b_latest, r.b));
        st.notes.push_back("order witnesses: scanning " + a.node + "." + dir_name(a.dir) + ", " + ext(r.a_latest, r.a) +
                           " = " + from(r.a_latest, true) + " and " + ext(r.b_latest, r.b) + " = " +
                           from(r.b_latest, false) + "; scanning " + b.node + "." + dir_name(b.dir) + ", " +
                           ext(r.b_latest, r.b) + " = " + from(r.b_latest, true) + " and " + ext(r.a_latest, r.a) +
                           " = " + from(r.a_latest, false));
      }
      if (st.binding) {
        st.notes.push_back("binds " + st.bind_var + " once per common neighbour" +
                           (st.order ? std::string("; its edges are not exported (no single witness)") : ""));
      } else {
        st.materialize = false;  // the type system only lets a set appear inside len()
        st.notes.push_back("no materialization: " + st.output_var + " is used only through len(), so " +
                           "the common neighbours that pass the filters are counted");
      }
      break;
    }
  }
  for (const Operand& o : st.operands) st.notes.push_back(range_note(o, p.e0));
  if (st.gate) st.notes.push_back("gate: later stages run only when this check passes");
}

template <class F>
void for_each_stage(Pattern& p, F&& f) {
  for (Level& lv : p.levels) {
    if (lv.binding) f(*lv.binding);
    for (Stage& s : lv.checks) f(s);
  }
}

}  // namespace

void plan(Pattern& p) {
  for_each_stage(p, [&](Stage& s) { plan_stage(s, p); });

  Plan& pl = p.plan;
  pl = {};
  pl.depth = int(p.levels.size()) - 1;
  const Level& root = p.levels[0];
  std::string root_ids;
  for (const Stage& s : root.checks) root_ids += (root_ids.empty() ? "" : ", ") + s.id;

  if (pl.depth > 0) {
    const Stage& first = *p.levels[1].binding;
    pl.work_stage = first.id;
    pl.split_stage = first.id;
    pl.notes.push_back("work estimate (sorted schedule, CUDA bucketing): the window size of " +
                       first.id + "." + first.operands[0].edge + " (" + first.operands[0].node + "." +
                       dir_name(first.operands[0].dir) + "), two binary searches");
    pl.notes.push_back(
        "CUDA heavy kernel: " + (root_ids.empty() ? std::string("no root checks") : "root checks " + root_ids + " on lane 0") +
        "; the lanes stride over the scan of " + first.id +
        " and each runs every stage nested in its own bindings; lane bin counts are summed across the warp");
  } else {
    const Stage* work = nullptr;
    std::string counts;
    for (const Stage& s : root.checks)
      if (s.op == Op::count) {
        if (!work) work = &s;
        counts += (counts.empty() ? "" : ", ") + s.id;
      }
    if (!work && !root.checks.empty()) work = &root.checks[0];
    if (work) {
      pl.work_stage = work->id;
      pl.notes.push_back("work estimate (sorted schedule, CUDA bucketing): the window size of " +
                         work->id + "." + work->operands[0].edge + " (" + work->operands[0].node + "." +
                         dir_name(work->operands[0].dir) + "), two binary searches");
    }
    if (!counts.empty()) {
      pl.split_stage = counts.substr(0, counts.find(','));
      pl.split_raw_count = true;
      pl.notes.push_back("CUDA heavy kernel: no binding stage; the lanes split the scan of " + counts +
                         ", the raw count is summed across the warp and lane 0 emits once "
                         "(partial counts are never binned)");
    } else {
      pl.notes.push_back("CUDA heavy kernel: no binding stage and no root count; lane 0 runs every check");
    }
  }
}

}  // namespace blz::ir

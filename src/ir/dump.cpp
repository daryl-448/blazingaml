#include <sstream>

#include "ir/ir.hpp"

namespace blz::ir {

namespace {

const char* dir_name(Dir d) { return d == Dir::in ? "in" : "out"; }

const char* op_name(Op op) {
  switch (op) {
    case Op::for_all: return "for_all";
    case Op::intersection: return "intersection";
    case Op::exists: return "exists";
    case Op::count: return "count";
  }
  return "?";
}

const char* access_name(Access a) {
  switch (a) {
    case Access::scan_distinct: return "scan_distinct";
    case Access::scan_edges: return "scan_edges";
    case Access::probe: return "probe";
    case Access::intersect: return "intersect";
  }
  return "?";
}

std::string bounds(const std::vector<TimeBound>& bs, bool lower) {
  std::string s;
  for (const TimeBound& b : bs) {
    if (!s.empty()) s += ", ";
    s += lower ? (b.strict ? "> " : ">= ") : (b.strict ? "< " : "<= ");
    s += to_string(b.expr);
    if (b.from_window) s += " [window]";
  }
  return s.empty() ? "-" : s;
}

std::string exprs(const std::vector<Expr>& xs) {
  std::string s;
  for (const Expr& x : xs) s += (s.empty() ? "" : ", ") + to_string(x);
  return s.empty() ? "-" : s;
}

std::string extremum(const char* fn, const std::vector<Expr>& xs) {
  return xs.size() == 1 ? to_string(xs[0]) : std::string(fn) + "(" + exprs(xs) + ")";
}

void stage(std::ostringstream& o, const Stage& s, const std::string& ind) {
  o << ind << (s.binding ? "bind " : "check ") << op_name(s.op) << " " << s.id;
  if (!s.bind_var.empty()) o << " -> " << s.bind_var;
  if (!s.output_var.empty()) o << " -> " << s.output_var;
  if (s.op == Op::for_all || s.op == Op::count)
    o << " [" << (s.multiplicity == Mult::distinct ? "distinct" : "edges") << "]";
  if (s.gate) o << " [gate]";
  o << "\n";
  for (const Operand& op : s.operands) {
    o << ind << "  operand " << op.edge << ": " << op.node << "." << dir_name(op.dir) << " access "
      << access_name(op.access) << "\n";
    o << ind << "    lo " << bounds(op.lo, true) << "\n";
    o << ind << "    hi " << bounds(op.hi, false) << "\n";
    o << ind << "    range ts >= " << extremum("max", op.lo_incl);
    if (!op.hi_incl.empty()) o << ", ts <= " << extremum("min", op.hi_incl);
    o << "\n";
  }
  if (!s.probe_node.empty()) o << ind << "  probe for " << s.probe_node << "\n";
  if (!s.element_var.empty()) o << ind << "  element " << s.element_var << "\n";
  for (const Expr& f : s.filters) o << ind << "  skip if " << to_string(f) << "\n";
  if (s.order) {
    const Order& r = *s.order;
    o << ind << "  order skip if " << r.a << ".t " << dsl::cmp_text(r.skip) << " " << r.b << ".t: keep if "
      << (r.a_latest ? "max(" : "min(") << r.a << ") " << dsl::cmp_text(r.keep) << " " << (r.b_latest ? "max(" : "min(")
      << r.b << ")\n";
  }
  if (s.emit) o << ind << "  emit " << to_string(s.emit->size) << " (min " << s.emit->min_size << ")\n";
  if (!s.exports.empty()) {
    o << ind << "  exports";
    for (const auto& e : s.exports) o << " " << e;
    o << "\n";
  }
  for (const auto& n : s.notes) o << ind << "  // " << n << "\n";
}

}  // namespace

std::string dump(const Pattern& p) {
  std::ostringstream o;
  o << "pattern " << p.name << " block " << p.block << "\n";
  o << "window span " << p.span_param << " lower " << (p.lower_open ? "open" : "closed")
    << " visibility "
    << (p.visibility_loc.line == 0 ? "runtime"
                                   : p.visibility == Visibility::stream ? "pinned stream" : "pinned timestamp")
    << "\n";
  if (p.work_limit) o << "work limit: bindings per trigger\n";
  for (const auto& [k, v] : p.consts) o << "const " << k << " = " << v << "\n";
  o << "trigger " << p.e0 << ": " << p.src_var << " -> " << p.dst_var << "\n";
  for (const Expr& f : p.trigger_filters) o << "  skip if " << to_string(f) << "\n";
  for (std::size_t i = 0; i < p.levels.size(); ++i) {
    const Level& lv = p.levels[i];
    const std::string ind(2 * i, ' ');
    o << ind << "level " << i << (lv.binding ? "" : " (root)") << "\n";
    if (lv.binding) stage(o, *lv.binding, ind + "  ");
    for (const Stage& s : lv.checks) stage(o, s, ind + "  ");
  }
  o << "plan depth " << p.plan.depth << " work " << p.plan.work_stage << " split "
    << p.plan.split_stage << (p.plan.split_raw_count ? " (raw count)" : "")
    << (p.plan.sweep ? " cpu sweep " + p.plan.sweep_stage : "") << "\n";
  for (const auto& n : p.plan.notes) o << "  // " << n << "\n";
  return o.str();
}

}  // namespace blz::ir

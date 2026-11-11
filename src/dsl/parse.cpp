// Schema checking: expanded YAML tree -> Spec (SPEC §6.1, §6.3). E101 / E401 only; names are
// resolved by lowering.
#include <algorithm>
#include <charconv>
#include <initializer_list>
#include <string_view>
#include <vector>

#include "dsl/ast.hpp"
#include "dsl/names.hpp"

namespace blz::dsl {

namespace {

using Keys = std::initializer_list<std::string_view>;
using KeyList = std::vector<std::string_view>;

template <class R>
bool in(std::string_view s, const R& keys) {
  return std::find(keys.begin(), keys.end(), s) != keys.end();
}
bool in(std::string_view s, Keys keys) { return std::find(keys.begin(), keys.end(), s) != keys.end(); }

std::string join(Keys keys) {
  std::string s;
  for (auto k : keys) s += (s.empty() ? "" : ", ") + std::string(k);
  return s;
}

const YNode& map_of(const YNode& n, const std::string& what) {
  if (!n.is_map()) fail("E101", n.loc, what + " must be a map");
  return n;
}

const YNode& scalar_of(const YNode& n, const std::string& what) {
  if (!n.is_scalar()) fail("E101", n.loc, what + " must be a scalar");
  return n;
}

void check_keys(const YNode& m, Keys allowed, const std::string& where) {
  for (const auto& [k, v] : m.map)
    if (!in(k.scalar, allowed))
      fail("E101", k.loc, "unknown key '" + k.scalar + "' in " + where + " (expected one of: " + join(allowed) + ")");
}

// Never null: E101 when the key is missing. A pointer, not a reference, to keep
// -Wdangling-reference quiet about the temporary `where` strings.
const YNode* required(const YNode& m, std::string_view key, const std::string& where) {
  const YNode* v = m.get(key);
  if (!v) fail("E101", m.loc, where + " needs '" + std::string(key) + "'");
  return v;
}

Loc key_loc(const YNode& m, std::string_view key) {
  for (const auto& [k, v] : m.map)
    if (k.scalar == key) return k.loc;
  return m.loc;
}

Name ident(const YNode& n, const std::string& what) {
  scalar_of(n, what);
  if (!is_ident(n.scalar)) fail("E101", n.loc, what + " '" + n.scalar + "' is not an identifier");
  if (is_reserved(n.scalar)) fail("E102", n.loc, "'" + n.scalar + "' is a reserved word");
  return {n.scalar, n.loc};
}

std::int64_t integer(const YNode& n, const std::string& what) {
  scalar_of(n, what);
  std::int64_t v = 0;
  const std::string& s = n.scalar;
  const auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
  if (s.empty() || ec != std::errc() || p != s.data() + s.size())
    fail("E101", n.loc, what + " must be an integer, found '" + s + "'");
  return v;
}

bool boolean(const YNode& n, const std::string& what) {
  scalar_of(n, what);
  if (n.scalar == "true") return true;
  if (n.scalar == "false") return false;
  fail("E101", n.loc, what + " must be true or false, found '" + n.scalar + "'");
}

Port port(const YNode& n, const std::string& what) {
  scalar_of(n, what);
  const std::string& s = n.scalar;
  const std::size_t dot = s.rfind('.');
  if (dot == std::string::npos)
    fail("E101", n.loc, what + " must be <NodeVar>.in or <NodeVar>.out, found '" + s + "'");
  const std::string var = s.substr(0, dot), dir = s.substr(dot + 1);
  if (dir != "in" && dir != "out")
    fail("E101", n.loc, what + " must end in .in or .out, found '" + s + "'");
  if (!is_ident(var)) fail("E101", n.loc, what + ": '" + var + "' is not an identifier");
  if (is_reserved(var)) fail("E102", n.loc, "'" + var + "' is a reserved word");
  return {{var, n.loc}, dir == "in" ? Dir::in : Dir::out, n.loc};
}

Expr expr_of(const YNode& n, const std::string& what) {
  scalar_of(n, what);
  // Column offsets inside the string are exact only when the text stands verbatim in the source
  // (one line, no escapes or folding) and expansion left it alone; otherwise every diagnostic
  // points at the scalar itself (SPEC §6.7).
  const bool exact = n.verbatim && !n.rewritten;
  Loc base = n.loc;
  if (exact && n.quoted && base.col > 0) ++base.col;
  return parse_expr(n.scalar, base, exact);
}

std::vector<Constraint> constraints_of(const YNode& n, bool trigger_level) {
  std::vector<Constraint> out;
  if (n.kind == YNode::Kind::Null) return out;
  if (!n.is_seq()) fail("E101", n.loc, "'constraints' must be a list");
  for (const YNode& c : n.seq) {
    if (!c.is_map() || c.map.size() != 1)
      fail("E101", c.loc, "a constraint must be a map with exactly one key, skip_if or break_if");
    const auto& [k, v] = c.map[0];
    if (k.scalar != "skip_if" && k.scalar != "break_if")
      fail("E101", k.loc, "unknown constraint '" + k.scalar + "' (expected skip_if or break_if)");
    const bool brk = k.scalar == "break_if";
    if (brk && trigger_level)
      fail("E101", k.loc, "only skip_if is allowed in the trigger-level constraints");
    out.push_back({brk, expr_of(v, k.scalar), k.loc});
  }
  return out;
}

Emit emit_of(const YNode& n) {
  map_of(n, "'emit'");
  check_keys(n, {"size", "min_size"}, "emit");
  Emit e;
  e.loc = n.loc;
  e.size = expr_of(*required(n, "size", "emit"), "emit size");
  if (const YNode* m = n.get("min_size")) e.min_size = integer(*m, "emit min_size");
  return e;
}

constexpr Keys kStageKeys = {"id",      "op",     "src",     "dst",          "edge",
                             "edges",   "dst_var", "node",   "output",       "element",
                             "multiplicity", "gate", "constraints", "emit"};

Stage stage_of(const YNode& n) {
  if (!n.is_map()) fail("E101", n.loc, "a stage must be a map");
  Stage s;
  s.loc = n.loc;

  // The op first, so that a reserved op is reported as such (E401) whatever else is wrong.
  const YNode* opn = required(n, "op", "a stage");
  scalar_of(*opn, "'op'");
  s.op_loc = opn->loc;
  const std::string& op = opn->scalar;
  if (op == "union" || op == "difference" || op == "differentiate")
    fail("E401", opn->loc, "op '" + op + "' is reserved and not supported in v1");
  if (op == "for_all") s.op = Op::for_all;
  else if (op == "intersection") s.op = Op::intersection;
  else if (op == "exists") s.op = Op::exists;
  else if (op == "count") s.op = Op::count;
  else fail("E101", opn->loc, "unknown op '" + op + "' (expected for_all, intersection, exists or count)");

  check_keys(n, kStageKeys, "a stage");
  s.id = ident(*required(n, "id", "a stage"), "stage id");

  // Fields each op takes (SPEC §6.3); `id`, `op`, `constraints` and `emit` are common.
  KeyList allowed;
  KeyList needs;
  switch (s.op) {
    case Op::for_all:
      allowed = {"src", "edge", "dst_var", "multiplicity"};
      needs = {"src", "edge", "dst_var"};
      break;
    case Op::intersection:
      allowed = {"src", "dst", "edges", "dst_var", "output", "element", "gate"};
      needs = {"src", "dst", "edges"};
      break;
    case Op::exists:
      allowed = {"src", "node", "edge", "gate"};
      needs = {"src", "node", "edge"};
      break;
    case Op::count:
      allowed = {"src", "edge", "element", "multiplicity", "output", "gate"};
      needs = {"src", "edge", "output"};
      break;
  }
  for (const auto& [k, v] : n.map)
    if (!in(k.scalar, {"id", "op", "constraints", "emit"}) && !in(k.scalar, allowed))
      fail("E101", k.loc, "'" + k.scalar + "' is not a field of op '" + op + "'");
  for (auto key : needs) (void)required(n, key, "op '" + op + "' (stage '" + s.id.s + "')");

  if (const YNode* v = n.get("src")) s.src = port(*v, "'src'");
  if (const YNode* v = n.get("dst")) s.dst = port(*v, "'dst'");
  if (const YNode* v = n.get("edge")) s.edge = ident(*v, "'edge'");
  if (const YNode* v = n.get("dst_var")) s.dst_var = ident(*v, "'dst_var'");
  if (const YNode* v = n.get("node")) s.node = ident(*v, "'node'");
  if (const YNode* v = n.get("output")) s.output = ident(*v, "'output'");
  if (const YNode* v = n.get("element")) s.element = ident(*v, "'element'");
  if (const YNode* v = n.get("edges")) {
    if (!v->is_seq() || v->seq.size() != 2)
      fail("E101", v->loc, "'edges' must be a list of two edge variables [src side, dst side]");
    for (const YNode& e : v->seq) s.edges.push_back(ident(e, "edge variable"));
  }
  if (const YNode* v = n.get("multiplicity")) {
    scalar_of(*v, "'multiplicity'");
    s.multiplicity_loc = v->loc;
    if (v->scalar == "distinct") s.multiplicity = Mult::distinct;
    else if (v->scalar == "edges") s.multiplicity = Mult::edges;
    else fail("E101", v->loc, "multiplicity must be distinct or edges, found '" + v->scalar + "'");
  }
  if (const YNode* v = n.get("gate")) {
    s.gate = boolean(*v, "'gate'");
    s.gate_loc = v->loc;
  }
  if (s.op == Op::intersection) {
    if (s.dst_var.has_value() == s.output.has_value())
      fail("E101", s.loc,
           "intersection '" + s.id.s + "' needs exactly one of 'dst_var' (binding) or 'output' (check)");
    if (s.dst_var && s.element)
      fail("E101", key_loc(n, "element"), "'element' belongs to the check form of intersection (with 'output')");
    if (s.dst_var && s.gate)
      fail("E101", s.gate_loc, "'gate' applies to check stages; this intersection binds 'dst_var'");
  }
  if (const YNode* v = n.get("constraints")) s.constraints = constraints_of(*v, false);
  if (const YNode* v = n.get("emit")) s.emit = emit_of(*v);
  return s;
}

}  // namespace

Spec parse_spec(const YNode& root, const std::string& file) {
  map_of(root, "a spec");
  check_keys(root, {"version", "pattern", "block", "consts", "window", "visibility", "trigger",
                    "constraints", "stages"},
             "the spec");
  Spec s;
  s.file = file;

  const YNode* ver = required(root, "version", "the spec");
  if (integer(*ver, "'version'") != 1) fail("E101", ver->loc, "only version 1 is supported");
  s.pattern = ident(*required(root, "pattern", "the spec"), "'pattern'");
  s.block = ident(*required(root, "block", "the spec"), "'block'");
  s.consts = read_consts(root);
  if (const YNode* c = root.get("consts"); c && c->is_map())
    for (const auto& [k, v] : c->map) s.const_locs.push_back(k.loc);

  const YNode& w = *required(root, "window", "the spec");
  map_of(w, "'window'");
  check_keys(w, {"span", "lower"}, "window");
  s.span = ident(*required(w, "span", "window"), "window span");
  const YNode& lower = *required(w, "lower", "window");
  scalar_of(lower, "window lower");
  if (lower.scalar == "open") s.lower_open = true;
  else if (lower.scalar == "closed") s.lower_open = false;
  else fail("E101", lower.loc, "window lower must be open or closed, found '" + lower.scalar + "'");

  if (const YNode* v = root.get("visibility")) {
    scalar_of(*v, "'visibility'");
    s.visibility_loc = v->loc;
    if (v->scalar == "stream") s.visibility = Visibility::stream;
    else if (v->scalar == "timestamp") s.visibility = Visibility::timestamp;
    else fail("E101", v->loc, "visibility must be stream or timestamp, found '" + v->scalar + "'");
  }

  const YNode& trig = *required(root, "trigger", "the spec");
  map_of(trig, "'trigger'");
  check_keys(trig, {"edge", "src", "dst"}, "trigger");
  s.trig_edge = ident(*required(trig, "edge", "trigger"), "trigger edge");
  s.trig_src = ident(*required(trig, "src", "trigger"), "trigger src");
  s.trig_dst = ident(*required(trig, "dst", "trigger"), "trigger dst");

  if (const YNode* v = root.get("constraints")) s.constraints = constraints_of(*v, true);

  const YNode& stages = *required(root, "stages", "the spec");
  if (!stages.is_seq() || stages.seq.empty()) fail("E101", stages.loc, "'stages' must be a non-empty list");
  for (const YNode& st : stages.seq) s.stages.push_back(stage_of(st));
  return s;
}

}  // namespace blz::dsl

#include "dsl/ynode.hpp"

#include <yaml-cpp/yaml.h>

namespace blz::dsl {

std::string format_loc(const Loc& loc) {
  std::string s = loc.file.empty() ? "<input>" : loc.file;
  if (loc.line > 0) {
    s += ":" + std::to_string(loc.line);
    if (loc.col > 0) s += ":" + std::to_string(loc.col);
  }
  return s;
}

CompileError::CompileError(std::string code, Loc loc, std::string message)
    : std::runtime_error(format_loc(loc) + ": error " + code + ": " + message),
      code_(std::move(code)),
      loc_(std::move(loc)),
      message_(std::move(message)) {}

void fail(const char* code, const Loc& loc, const std::string& message) {
  throw CompileError(code, loc, message);
}

const YNode* YNode::get(std::string_view key) const {
  for (const auto& [k, v] : map)
    if (k.scalar == key) return &v;
  return nullptr;
}

YNode* YNode::get(std::string_view key) {
  for (auto& [k, v] : map)
    if (k.scalar == key) return &v;
  return nullptr;
}

namespace {

Loc to_loc(const YAML::Mark& m, const std::string& file) {
  if (m.is_null()) return {file, 0, 0};
  return {file, m.line + 1, m.column + 1};
}

YNode convert(const YAML::Node& n, std::string_view text, const std::string& file, const Loc& fallback) {
  YNode out;
  out.loc = to_loc(n.Mark(), file);
  switch (n.Type()) {
    case YAML::NodeType::Undefined:
    case YAML::NodeType::Null:
      out.kind = YNode::Kind::Null;
      out.loc = fallback;  // yaml-cpp marks an empty value at the next token
      break;
    case YAML::NodeType::Scalar:
      out.kind = YNode::Kind::Scalar;
      out.scalar = n.Scalar();
      out.quoted = n.Tag() == "!";
      if (!n.Mark().is_null() && n.Mark().pos >= 0) {
        const std::size_t at = std::size_t(n.Mark().pos) + (out.quoted ? 1 : 0);
        out.verbatim = out.scalar.find('\n') == std::string::npos && at <= text.size() &&
                       text.substr(at, out.scalar.size()) == out.scalar;
      }
      break;
    case YAML::NodeType::Sequence:
      out.kind = YNode::Kind::Seq;
      for (const auto& item : n) out.seq.push_back(convert(item, text, file, out.loc));
      break;
    case YAML::NodeType::Map:
      out.kind = YNode::Kind::Map;
      for (const auto& kv : n) {
        YNode key = convert(kv.first, text, file, out.loc);
        if (!key.is_scalar()) fail("E101", key.loc, "map keys must be plain scalars");
        for (const auto& [k, v] : out.map)
          if (k.scalar == key.scalar)
            fail("E101", key.loc,
                 "duplicate key '" + key.scalar + "' (first at " + format_loc(k.loc) + ")");
        YNode value = convert(kv.second, text, file, key.loc);
        out.map.emplace_back(std::move(key), std::move(value));
      }
      break;
  }
  return out;
}

void emit(YAML::Emitter& em, const YNode& n) {
  switch (n.kind) {
    case YNode::Kind::Null:
      em << YAML::Null;
      break;
    case YNode::Kind::Scalar:
      if (n.quoted) em << YAML::DoubleQuoted;
      em << n.scalar;
      break;
    case YNode::Kind::Seq:
      em << YAML::BeginSeq;
      for (const auto& item : n.seq) emit(em, item);
      em << YAML::EndSeq;
      break;
    case YNode::Kind::Map:
      em << YAML::BeginMap;
      for (const auto& [k, v] : n.map) {
        em << YAML::Key << k.scalar << YAML::Value;
        emit(em, v);
      }
      em << YAML::EndMap;
      break;
  }
}

}  // namespace

YNode load_yaml(std::string_view text, const std::string& file) {
  std::vector<YAML::Node> docs;
  try {
    docs = YAML::LoadAll(std::string(text));
  } catch (const YAML::Exception& e) {
    fail("E101", to_loc(e.mark, file), "YAML syntax error: " + e.msg);
  }
  if (docs.size() != 1) fail("E101", {file, 1, 1}, "expected exactly one YAML document");
  return convert(docs[0], text, file, {file, 1, 1});
}

std::string emit_yaml(const YNode& root) {
  YAML::Emitter em;
  em.SetIndent(2);
  emit(em, root);
  return std::string(em.c_str()) + "\n";
}

bool same_tree(const YNode& a, const YNode& b) {
  if (a.kind != b.kind) return false;
  switch (a.kind) {
    case YNode::Kind::Null:
      return true;
    case YNode::Kind::Scalar:
      return a.scalar == b.scalar;
    case YNode::Kind::Seq:
      if (a.seq.size() != b.seq.size()) return false;
      for (std::size_t i = 0; i < a.seq.size(); ++i)
        if (!same_tree(a.seq[i], b.seq[i])) return false;
      return true;
    case YNode::Kind::Map:
      if (a.map.size() != b.map.size()) return false;
      for (std::size_t i = 0; i < a.map.size(); ++i)
        if (a.map[i].first.scalar != b.map[i].first.scalar ||
            !same_tree(a.map[i].second, b.map[i].second))
          return false;
      return true;
  }
  return false;
}

}  // namespace blz::dsl

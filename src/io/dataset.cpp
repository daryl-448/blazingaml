#include "io/dataset.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

namespace blz {
namespace {

std::string at(const std::string& file, const YAML::Mark& m) {
  if (m.is_null() || m.line < 0) return file;
  return file + ":" + std::to_string(m.line + 1) + ":" + std::to_string(m.column + 1);
}

[[noreturn]] void fail(const std::string& file, const YAML::Node& n, const std::string& msg) {
  throw std::runtime_error(at(file, n.Mark()) + ": " + msg);
}

void check_keys(const YAML::Node& map, const std::string& file, std::span<const std::string_view> allowed,
                const std::string& where) {
  if (!map.IsMap()) fail(file, map, where + " must be a map");
  std::set<std::string> seen;
  for (auto kv : map) {
    if (!kv.first.IsScalar()) fail(file, kv.first, "a key of " + where + " must be a scalar");
    const std::string k = kv.first.Scalar();
    if (std::find(allowed.begin(), allowed.end(), k) == allowed.end())
      fail(file, kv.first, "unknown key '" + k + "' in " + where);
    if (!seen.insert(k).second) fail(file, kv.first, "duplicate key '" + k + "' in " + where);
  }
}

std::string text(const YAML::Node& map, const char* key, const std::string& file, const std::string& where,
                 bool required = true) {
  const YAML::Node n = map[key];
  if (!n) {
    if (required) fail(file, map, where + " needs '" + key + "'");
    return "";
  }
  if (!n.IsScalar() || n.Scalar().empty()) fail(file, n, "'" + std::string(key) + "' must be a non-empty string");
  return n.Scalar();
}

}  // namespace

DatasetDescriptor parse_dataset(std::string_view yaml_text, const std::string& file) {
  YAML::Node loaded;
  try {
    loaded = YAML::Load(std::string(yaml_text));
  } catch (const YAML::Exception& e) {
    throw std::runtime_error(file + ": " + e.what());
  }
  const YAML::Node& root = loaded;  // const: operator[] must not insert keys
  const std::string where = "the dataset descriptor";
  static constexpr std::string_view kKeys[] = {"id", "transactions", "schema", "split", "patterns", "node_tables",
                                               "invariants"};
  check_keys(root, file, kKeys, where);

  DatasetDescriptor d;
  d.file = file;
  d.sha256 = sha256(yaml_text.data(), yaml_text.size());
  d.id = text(root, "id", file, where);
  if (!std::all_of(d.id.begin(), d.id.end(), [](char c) { return std::isalnum((unsigned char)c) || c == '_'; }))
    fail(file, root["id"], "'id' must be letters, digits and '_', found '" + d.id + "'");
  d.transactions = text(root, "transactions", file, where);
  const std::string schema = text(root, "schema", file, where);
  d.schema = find_schema(schema);
  if (!d.schema) fail(file, root["schema"], "schema must be ibm or transxion, found '" + schema + "'");
  const std::string split = text(root, "split", file, where);
  const auto rule = parse_split_rule(split);
  if (!rule) fail(file, root["split"], "split must be legacy_hi_small or temporal_60_20_20, found '" + split + "'");
  d.split = *rule;
  d.patterns = text(root, "patterns", file, where, false);

  if (const YAML::Node nt = root["node_tables"]) {
    if (!nt.IsSequence() || nt.size() == 0) fail(file, nt, "node_tables must be a non-empty list");
    for (const YAML::Node& t : nt) {
      const std::string w = "a node table";
      static constexpr std::string_view kTableKeys[] = {"file", "bank", "account"};
      check_keys(t, file, kTableKeys, w);
      d.node_tables.push_back({text(t, "file", file, w), text(t, "bank", file, w), text(t, "account", file, w)});
    }
  }

  const YAML::Node inv = root["invariants"];
  if (!inv) fail(file, root, where + " needs 'invariants'");
  check_keys(inv, file, kInvariantNames, "invariants");
  for (auto kv : inv) {
    const std::string& v = kv.second.IsScalar() ? kv.second.Scalar() : std::string();
    if (v.empty() || !std::all_of(v.begin(), v.end(), [](char c) { return c >= '0' && c <= '9'; }) || v.size() > 19)
      fail(file, kv.second, "invariant " + kv.first.Scalar() + " must be a non-negative integer");
    d.invariants[kv.first.Scalar()] = std::stoull(v);
  }
  return d;
}

DatasetDescriptor load_dataset(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::stringstream ss;
  ss << in.rdbuf();
  return parse_dataset(ss.str(), path);
}

}  // namespace blz

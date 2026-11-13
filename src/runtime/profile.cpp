#include "runtime/profile.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

#include "common/sha256.hpp"
#include "runtime/cpu_backend.hpp"

namespace blz {

namespace {

std::string at(const std::string& file, const YAML::Mark& m) {
  if (m.is_null() || m.line < 0) return file;
  return file + ":" + std::to_string(m.line + 1) + ":" + std::to_string(m.column + 1);
}

[[noreturn]] void fail(const std::string& loc, const std::string& msg) { throw std::runtime_error(loc + ": " + msg); }
[[noreturn]] void fail_code(const std::string& loc, const char* code, const std::string& msg) {
  throw std::runtime_error(loc + ": error " + code + ": " + msg);
}

const ConstInfo* find_const(const PatternInfo& p, const char* name) {
  for (const ConstInfo& c : p.consts)
    if (std::strcmp(c.name, name) == 0) return &c;
  return nullptr;
}

bool is_ident(const std::string& s) {
  if (s.empty() || !(std::isalpha((unsigned char)s[0]) || s[0] == '_')) return false;
  return std::all_of(s.begin(), s.end(), [](char c) { return std::isalnum((unsigned char)c) || c == '_'; });
}

// The patterns whose definitions the oracle computes for each block (SPEC §4.3–§4.6, §9).
std::vector<std::string> oracle_patterns(Block b) {
  switch (b) {
    case Block::fan_in: return {"fan_in"};
    case Block::fan_out: return {"fan_out"};
    case Block::deg_in: return {"deg_in"};
    case Block::deg_out: return {"deg_out"};
    case Block::sg: return {"sg_gather", "sg_scatter"};
    case Block::tcyc: return {"tcyc"};
    case Block::sg_ord: return {"sg_ordered"};
  }
  return {};
}

std::string read_text(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// A scalar of `node` as T, or a located error naming `what`.
template <class T>
T scalar_as(const YAML::Node& node, const std::string& file, const std::string& what) {
  if (!node.IsScalar()) fail(at(file, node.Mark()), what + " must be a scalar");
  try {
    return node.as<T>();
  } catch (const YAML::Exception&) {
    fail(at(file, node.Mark()), what + " has the wrong type ('" + node.Scalar() + "')");
  }
}

void check_keys(const YAML::Node& map, const std::string& file, std::initializer_list<const char*> allowed,
                const std::string& where) {
  std::set<std::string> seen;
  for (auto kv : map) {
    const std::string k = scalar_as<std::string>(kv.first, file, "a key of " + where);
    if (std::none_of(allowed.begin(), allowed.end(), [&](const char* a) { return k == a; }))
      fail(at(file, kv.first.Mark()), "unknown key '" + k + "' in " + where);
    if (!seen.insert(k).second) fail(at(file, kv.first.Mark()), "duplicate key '" + k + "' in " + where);
  }
}

YAML::Node required(const YAML::Node& map, const char* key, const std::string& file, const std::string& where) {
  const YAML::Node n = map[key];
  if (!n) fail(at(file, map.Mark()), where + " needs '" + key + "'");
  return n;
}

BlockParams read_params(const YAML::Node& blk, const std::string& file, const std::string& bname) {
  BlockParams bp;
  const YAML::Node w = required(blk, "window", file, "block " + bname);
  bp.window = scalar_as<std::int32_t>(w, file, bname + " window");
  if (bp.window <= 0) fail(at(file, w.Mark()), bname + ": window must be > 0");
  const YAML::Node bins = required(blk, "bins", file, "block " + bname);
  if (!bins.IsSequence() || bins.size() == 0) fail(at(file, bins.Mark()), bname + ": bins must be a non-empty list");
  for (const YAML::Node& b : bins) {
    const auto v = scalar_as<std::int64_t>(b, file, bname + " bin");
    if (v < 0 || v > UINT32_MAX) fail(at(file, b.Mark()), bname + ": a bin must be in [0, 2^32)");
    if (!bp.bins.empty() && std::uint32_t(v) <= bp.bins.back())
      fail(at(file, b.Mark()), bname + ": bins must be strictly increasing");
    bp.bins.push_back(std::uint32_t(v));
  }
  return bp;
}

const PatternInfo* find_pattern(std::span<const PatternInfo> registry, const std::string& name) {
  for (const PatternInfo& p : registry)
    if (name == p.name) return &p;
  return nullptr;
}

Profile parse_v1(const YAML::Node& root, std::string_view text, const std::string& file,
                 std::span<const PatternInfo> registry) {
  const FeatureConfig cfg = parse_feature_config(text, file);  // validates, v1 messages
  Profile p;
  p.file = file;
  p.name = std::filesystem::path(file).stem().string();
  p.format = 1;
  p.k_max = cfg.k_max;
  p.k_max_loc = root["k_max"] ? at(file, root["k_max"].Mark()) : file;
  p.visibility_loc = p.credit_loc = file;
  for (auto kv : root["blocks"]) {
    const std::string name = kv.first.as<std::string>();
    ProfileBlock pb{*parse_block(name), {}, *cfg[*parse_block(name)], at(file, kv.first.Mark())};
    for (const PatternInfo& pi : registry)
      if (name == pi.block) pb.patterns.push_back(&pi);
    p.blocks.push_back(std::move(pb));
  }
  return p;
}

Profile parse_v2(const YAML::Node& root, const std::string& file, std::span<const PatternInfo> registry) {
  check_keys(root, file, {"version", "profile", "visibility", "credit", "k_max", "blocks"}, "the profile");
  Profile p;
  p.file = file;
  p.format = 2;
  const YAML::Node name = required(root, "profile", file, "the profile");
  p.name = scalar_as<std::string>(name, file, "'profile'");
  if (!is_ident(p.name)) fail(at(file, name.Mark()), "'profile' must be an identifier, found '" + p.name + "'");

  const YAML::Node vis = required(root, "visibility", file, "the profile");
  p.visibility_loc = at(file, vis.Mark());
  const auto v = scalar_as<std::string>(vis, file, "'visibility'");
  if (v == "stream") p.visibility = Visibility::stream;
  else if (v == "timestamp") p.visibility = Visibility::timestamp;
  else fail(p.visibility_loc, "visibility must be stream or timestamp, found '" + v + "'");

  const YAML::Node cr = required(root, "credit", file, "the profile");
  p.credit_loc = at(file, cr.Mark());
  const auto c = scalar_as<std::string>(cr, file, "'credit'");
  if (c == "trigger") p.credit = Credit::trigger;
  else if (c == "participants") p.credit = Credit::participants;
  else fail(p.credit_loc, "credit must be trigger or participants, found '" + c + "'");

  const YAML::Node km = required(root, "k_max", file, "the profile");
  p.k_max_loc = at(file, km.Mark());
  p.k_max = scalar_as<int>(km, file, "'k_max'");
  if (p.k_max < 2) fail(p.k_max_loc, "k_max must be >= 2");

  const YAML::Node blocks = required(root, "blocks", file, "the profile");
  if (!blocks.IsMap() || blocks.size() == 0) fail(at(file, blocks.Mark()), "'blocks' must be a non-empty map");
  std::set<std::string> seen;
  for (auto kv : blocks) {
    const std::string bname = scalar_as<std::string>(kv.first, file, "a block name");
    const std::string bloc = at(file, kv.first.Mark());
    const auto b = parse_block(bname);
    if (!b) fail(bloc, "unknown block '" + bname + "'");
    if (!seen.insert(bname).second) fail(bloc, "duplicate block '" + bname + "'");
    const YAML::Node blk = kv.second;
    if (!blk.IsMap()) fail(at(file, blk.Mark()), "block " + bname + " must be a map");
    check_keys(blk, file, {"patterns", "window", "bins"}, "block " + bname);
    ProfileBlock pb{*b, {}, read_params(blk, file, bname), bloc};
    const YAML::Node pats = required(blk, "patterns", file, "block " + bname);
    if (!pats.IsSequence() || pats.size() == 0)
      fail(at(file, pats.Mark()), bname + ": patterns must be a non-empty list");
    for (const YAML::Node& pn : pats) {
      const auto pname = scalar_as<std::string>(pn, file, bname + " pattern");
      const std::string ploc = at(file, pn.Mark());
      const PatternInfo* pi = find_pattern(registry, pname);
      if (!pi) fail_code(ploc, "E403", "unknown pattern '" + pname + "': no compiled pattern has this name");
      if (bname != pi->block)
        fail_code(ploc, "E403", "pattern '" + pname + "' (" + pi->file + ") emits into block '" + pi->block +
                                    "', not '" + bname + "'");
      if (std::find(pb.patterns.begin(), pb.patterns.end(), pi) != pb.patterns.end())
        fail(ploc, bname + ": pattern '" + pname + "' is listed twice");
      if (*pi->visibility && v != pi->visibility)
        fail_code(ploc, "E403", "pattern '" + pname + "' pins visibility: " + pi->visibility + " (" +
                                    pi->visibility_loc + "), but the profile has visibility: " + v + " (" +
                                    p.visibility_loc + ")");
      pb.patterns.push_back(pi);
    }
    p.blocks.push_back(std::move(pb));
  }
  return p;
}

}  // namespace

const char* visibility_name(Visibility v) { return v == Visibility::stream ? "stream" : "timestamp"; }
const char* credit_name(Credit c) { return c == Credit::trigger ? "trigger" : "participants"; }
const char* backend_name(Backend b) { return b == Backend::oracle ? "oracle" : b == Backend::cpu ? "cpu" : "cuda"; }
Backend parse_backend(const std::string& s) {
  if (s == "oracle") return Backend::oracle;
  if (s == "cpu") return Backend::cpu;
  if (s == "cuda") return Backend::cuda;
  throw std::runtime_error("unknown backend '" + s + "' (oracle, cpu, cuda)");
}

const ProfileBlock* Profile::find(Block b) const {
  for (const ProfileBlock& pb : blocks)
    if (pb.block == b) return &pb;
  return nullptr;
}

bool Profile::causal() const { return visibility == Visibility::stream && credit == Credit::trigger; }

std::string Profile::row_sees() const {
  std::string s = visibility == Visibility::stream
                      ? "the edges with eid <= the row's eid (stream visibility)"
                      : "the edges with ts <= the row's ts, including later edges of the same timestamp "
                        "(timestamp visibility)";
  if (credit == Credit::participants) s += "; participants credit also adds instances completed by later edges";
  return s;
}

FeatureConfig Profile::feature_config() const {
  FeatureConfig cfg;
  cfg.k_max = k_max;
  for (const ProfileBlock& pb : blocks) cfg[pb.block] = pb.params;
  return cfg;
}

nlohmann::json Profile::resolved() const {
  nlohmann::json order = nlohmann::json::array(), out = nlohmann::json::object();
  for (const ProfileBlock& pb : blocks) {
    const std::string bname(block_name(pb.block));
    order.push_back(bname);
    nlohmann::json pats = nlohmann::json::array();
    for (const PatternInfo* pi : pb.patterns)
      pats.push_back({{"name", pi->name}, {"spec", pi->file}, {"spec_sha256", pi->spec_sha256}});
    out[bname] = {{"patterns", pats}, {"window", pb.params.window}, {"bins", pb.params.bins}};
  }
  return {{"version", 2},          {"profile", name},   {"visibility", visibility_name(visibility)},
          {"credit", credit_name(credit)}, {"k_max", k_max}, {"block_order", order},
          {"blocks", out}};
}

nlohmann::json Profile::canonical() const {
  nlohmann::json order = nlohmann::json::array(), out = nlohmann::json::object();
  for (const ProfileBlock& pb : blocks) {
    const std::string bname(block_name(pb.block));
    order.push_back(bname);
    nlohmann::json pats = nlohmann::json::array();
    for (const PatternInfo* pi : pb.patterns) pats.push_back(pi->spec_sha256);
    out[bname] = {{"patterns", pats}, {"window", pb.params.window}, {"bins", pb.params.bins}};
  }
  return {{"visibility", visibility_name(visibility)},
          {"credit", credit_name(credit)},
          {"k_max", k_max},
          {"blocks", out},
          {"block_order", order}};
}

std::string Profile::id() const {
  const std::string text = canonical().dump();
  return to_hex(sha256(text.data(), text.size())).substr(0, 16);
}

Profile parse_profile(std::string_view yaml_text, const std::string& file, std::span<const PatternInfo> registry) {
  YAML::Node loaded;
  try {
    loaded = YAML::Load(std::string(yaml_text));
  } catch (const YAML::Exception& e) {
    throw std::runtime_error(file + ": " + e.what());
  }
  const YAML::Node& root = loaded;  // const: operator[] must not insert keys
  if (!root.IsMap()) fail(file, "top level must be a map");
  if (!root["version"]) return parse_v1(root, yaml_text, file, registry);
  const int version = scalar_as<int>(root["version"], file, "'version'");
  if (version != 2)
    fail(at(file, root["version"].Mark()),
         "a profile has version: 2 (a v1 feature config has no version key), found " + std::to_string(version));
  return parse_v2(root, file, registry);
}

Profile load_profile(const std::string& path, std::span<const PatternInfo> registry) {
  return parse_profile(read_text(path), path, registry);
}

Profile registry_profile(std::span<const PatternInfo> registry) {
  Profile p;
  p.file = "(registry)";
  p.name = "registry";
  p.format = 1;
  p.visibility_loc = p.credit_loc = p.k_max_loc = p.file;
  for (const PatternInfo& pi : registry)
    if (const ConstInfo* k = find_const(pi, "K_MAX")) p.k_max = int(k->value);
  for (Block b : kAllBlocks) {
    ProfileBlock pb{b, {}, {}, p.file};
    for (const PatternInfo& pi : registry)
      if (block_name(b) == pi.block) pb.patterns.push_back(&pi);
    if (!pb.patterns.empty()) p.blocks.push_back(std::move(pb));
  }
  return p;
}

void check_profile_backend(const Profile& p, Backend b) {
  const std::string be = backend_name(b);
  if (p.credit == Credit::participants)
    fail_code(p.credit_loc, "E404",
              "the " + be + " backend cannot run credit: participants (stretch item S1 of SPEC v2 implements it)");
  if (b == Backend::oracle) {
    for (const ProfileBlock& pb : p.blocks) {
      std::vector<std::string> got, want = oracle_patterns(pb.block);
      for (const PatternInfo* pi : pb.patterns) got.push_back(pi->name);
      std::sort(got.begin(), got.end());
      std::sort(want.begin(), want.end());
      if (got != want) {
        std::string w;
        for (const auto& s : want) w += (w.empty() ? "" : ", ") + s;
        fail_code(pb.loc, "E404",
                  "the oracle computes block " + std::string(block_name(pb.block)) + " as the patterns [" + w +
                      "] together; it cannot run another pattern list");
      }
    }
    return;
  }
  if (p.format == 1) {  // a v1 feature config: the v1 start-up checks (E402, K_MAX), unchanged
    check_backend_config(p.feature_config(), p.file);
    return;
  }
  for (const ProfileBlock& pb : p.blocks)
    for (const PatternInfo* pi : pb.patterns)
      if (const ConstInfo* k = find_const(*pi, "K_MAX"); k && k->value != p.k_max)
        fail_code(p.k_max_loc, "E404",
                  "the " + be + " backend cannot run k_max: " + std::to_string(p.k_max) + ": pattern '" + pi->name +
                      "' (" + pi->file + ") was compiled with K_MAX = " + std::to_string(k->value));
}

}  // namespace blz

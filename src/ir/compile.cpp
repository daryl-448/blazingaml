// The front end driver (SPEC §7.1 steps 1–7).
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "common/sha256.hpp"
#include "ir/ir.hpp"

namespace blz::ir {

std::string read_text_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

dsl::YNode expand_text(std::string_view text, const std::string& file, const IntEnv& overrides) {
  return dsl::expand_spec(dsl::load_yaml(text, file), overrides);
}

Pattern compile_text(std::string_view text, const std::string& file, const IntEnv& overrides) {
  const dsl::Spec spec = dsl::parse_spec(expand_text(text, file, overrides), file);
  Pattern p = lower(spec);
  plan(p);
  p.spec_sha256 = to_hex(sha256(text.data(), text.size()));
  return p;
}

Pattern compile_file(const std::string& path, const IntEnv& overrides) {
  return compile_text(read_text_file(path), path, overrides);
}

}  // namespace blz::ir

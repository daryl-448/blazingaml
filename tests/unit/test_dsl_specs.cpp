// The invalid-spec suite (SPEC §13.4, AC5) and the IR snapshots of the pattern library.
#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <filesystem>
#include <regex>
#include <set>
#include <string>

#include "ir/ir.hpp"

using namespace blz;
namespace fs = std::filesystem;

namespace {

std::string src(const std::string& rel) { return std::string(BLZ_SOURCE_DIR) + "/" + rel; }

std::vector<fs::path> yaml_files(const std::string& dir) {
  std::vector<fs::path> out;
  for (const auto& e : fs::directory_iterator(src(dir)))
    if (e.path().extension() == ".yaml") out.push_back(e.path());
  std::sort(out.begin(), out.end());
  return out;
}

}  // namespace

TEST_CASE("invalid-spec suite: every file fails with its declared code and position") {
  const auto files = yaml_files("tests/specs/invalid");
  REQUIRE(files.size() >= 40);
  const std::regex expect_re(R"(^# expect: (E[0-9]{3}) ([0-9]+):([0-9]+)\b)");
  std::set<std::string> covered;
  for (const fs::path& f : files) {
    INFO(f.filename().string());
    const std::string text = ir::read_text_file(f.string());
    std::smatch m;
    const std::string first = text.substr(0, text.find('\n'));
    REQUIRE(std::regex_search(first, m, expect_re));
    const std::string code = m[1];
    const int line = std::stoi(m[2]), col = std::stoi(m[3]);
    // File names start with their code, so a case cannot silently test something else.
    std::string prefix = f.filename().string().substr(0, 4);
    std::transform(prefix.begin(), prefix.end(), prefix.begin(), ::toupper);
    CHECK(prefix == code);
    try {
      ir::compile_text(text, f.string());
      FAIL_CHECK("accepted, expected " << code);
    } catch (const dsl::CompileError& e) {
      INFO(e.what());
      CHECK(e.code() == code);
      CHECK(e.loc().file == f.string());
      CHECK(e.loc().line == line);
      CHECK(e.loc().col == col);
      // The message carries file:line:col and the code (SPEC §6.7).
      CHECK(std::string(e.what()).rfind(f.string() + ":" + std::to_string(line) + ":" +
                                            std::to_string(col) + ": error " + code + ": ",
                                        0) == 0);
      covered.insert(code);
    }
  }
  // AC5: E101–E401 (E402 needs the feature config: test_registration.cpp), and E205 (SPEC v2 §6.2).
  for (const char* c : {"E101", "E102", "E103", "E201", "E202", "E203", "E204", "E205", "E301", "E401"})
    CHECK(covered.count(c) == 1);
}

TEST_CASE("library: every pattern compiles, and its IR equals the snapshot") {
  const auto files = yaml_files("patterns");
  REQUIRE(files.size() == 9);
  for (const fs::path& f : files) {
    INFO(f.filename().string() << " (regenerate with: blzc --dump-ir --out tests/specs/ir patterns/*.yaml)");
    const ir::Pattern p = ir::compile_file(f.string());
    CHECK(p.name == f.stem().string());
    const std::string snap = src("tests/specs/ir/" + p.name + ".ir.txt");
    REQUIRE(fs::exists(snap));
    CHECK(ir::dump(p) == ir::read_text_file(snap));
  }
}

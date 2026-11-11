#pragma once
#include <stdexcept>
#include <string>

namespace blz::dsl {

// A source position. line and col are 1-based; 0 means unknown.
struct Loc {
  std::string file;
  int line = 0;
  int col = 0;
};

// "file:line:col" (the parts that are known).
std::string format_loc(const Loc& loc);

// A spec diagnostic (SPEC §6.7). what() is "file:line:col: error E201: message".
class CompileError : public std::runtime_error {
 public:
  CompileError(std::string code, Loc loc, std::string message);
  const std::string& code() const { return code_; }
  const Loc& loc() const { return loc_; }
  const std::string& message() const { return message_; }

 private:
  std::string code_;
  Loc loc_;
  std::string message_;
};

[[noreturn]] void fail(const char* code, const Loc& loc, const std::string& message);

}  // namespace blz::dsl

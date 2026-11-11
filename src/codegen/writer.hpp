#pragma once
#include <string>

namespace blz::codegen {

// Indented line writer for generated sources (two spaces per level).
class Writer {
 public:
  void line(const std::string& s = "") {
    if (s.empty()) out_ += '\n';
    else out_ += std::string(2 * std::size_t(depth_), ' ') + s + '\n';
  }
  // A line that opens a brace; the following lines are indented until close().
  void open(const std::string& s) {
    line(s);
    ++depth_;
  }
  void close(const std::string& s = "}") {
    --depth_;
    line(s);
  }
  // Text that is already indented and ends with a newline.
  void raw(const std::string& s) { out_ += s; }
  const std::string& str() const { return out_; }

 private:
  std::string out_;
  int depth_ = 0;
};

}  // namespace blz::codegen

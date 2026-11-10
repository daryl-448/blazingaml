#pragma once
#include <cstddef>
#include <string>

namespace blz {

// Read-only memory map of a whole file. Throws std::runtime_error on failure.
class MappedFile {
 public:
  explicit MappedFile(const std::string& path);
  ~MappedFile();
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  const char* data() const { return data_; }
  std::size_t size() const { return size_; }

 private:
  const char* data_ = nullptr;
  std::size_t size_ = 0;
};

}  // namespace blz

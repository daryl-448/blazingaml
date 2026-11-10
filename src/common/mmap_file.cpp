#include "common/mmap_file.hpp"

#include <stdexcept>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace blz {

MappedFile::MappedFile(const std::string& path) {
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("cannot open " + path);
  struct stat st {};
  if (::fstat(fd, &st) != 0) {
    ::close(fd);
    throw std::runtime_error("cannot stat " + path);
  }
  size_ = std::size_t(st.st_size);
  if (size_ > 0) {
    void* p = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) {
      ::close(fd);
      throw std::runtime_error("cannot mmap " + path);
    }
    data_ = static_cast<const char*>(p);
  }
  ::close(fd);
}

MappedFile::~MappedFile() {
  if (data_) ::munmap(const_cast<char*>(data_), size_);
}

}  // namespace blz

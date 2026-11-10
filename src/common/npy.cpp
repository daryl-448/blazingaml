#include "common/npy.hpp"

#include <cstdio>
#include <memory>
#include <stdexcept>

namespace blz {

std::string npy_header(const std::string& descr, std::uint64_t rows, std::uint64_t cols) {
  std::string dict = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': (" +
                     std::to_string(rows) + ", " + std::to_string(cols) + "), }";
  std::size_t total = 10 + dict.size() + 1;  // magic(6) + version(2) + len(2) + dict + '\n'
  dict.append((64 - total % 64) % 64, ' ');
  dict += '\n';
  std::string h = "\x93NUMPY";
  h += char(1);
  h += char(0);
  h += char(dict.size() & 0xff);
  h += char(dict.size() >> 8);
  return h + dict;
}

void write_npy_i64(const std::string& path, std::span<const std::int64_t> data,
                   std::uint64_t rows, std::uint64_t cols) {
  if (data.size() != rows * cols) throw std::invalid_argument("npy: size != rows * cols");
  std::string tmp = path + ".tmp";
  std::unique_ptr<FILE, int (*)(FILE*)> f(std::fopen(tmp.c_str(), "wb"), &std::fclose);
  if (!f) throw std::runtime_error("cannot write " + tmp);
  std::string h = npy_header("<i8", rows, cols);
  bool ok = std::fwrite(h.data(), 1, h.size(), f.get()) == h.size() &&
            std::fwrite(data.data(), sizeof(std::int64_t), data.size(), f.get()) == data.size();
  if (std::fclose(f.release()) != 0 || !ok) throw std::runtime_error("write failed: " + tmp);
  if (std::rename(tmp.c_str(), path.c_str()) != 0) throw std::runtime_error("rename failed: " + path);
}

}  // namespace blz

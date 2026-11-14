#include "common/npy.hpp"

#include <cstdio>
#include <cstring>
#include <memory>
#include <regex>
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

namespace {

template <class T>
void write_npy(const std::string& path, const char* descr, std::span<const T> data,
               std::uint64_t rows, std::uint64_t cols) {
  if (data.size() != rows * cols) throw std::invalid_argument("npy: size != rows * cols");
  std::string tmp = path + ".tmp";
  std::unique_ptr<FILE, int (*)(FILE*)> f(std::fopen(tmp.c_str(), "wb"), &std::fclose);
  if (!f) throw std::runtime_error("cannot write " + tmp);
  std::string h = npy_header(descr, rows, cols);
  bool ok = std::fwrite(h.data(), 1, h.size(), f.get()) == h.size() &&
            std::fwrite(data.data(), sizeof(T), data.size(), f.get()) == data.size();
  if (std::fclose(f.release()) != 0 || !ok) throw std::runtime_error("write failed: " + tmp);
  if (std::rename(tmp.c_str(), path.c_str()) != 0) throw std::runtime_error("rename failed: " + path);
}

}  // namespace

void write_npy_i64(const std::string& path, std::span<const std::int64_t> data,
                   std::uint64_t rows, std::uint64_t cols) {
  write_npy(path, "<i8", data, rows, cols);
}

void write_npy_u32(const std::string& path, std::span<const std::uint32_t> data,
                   std::uint64_t rows, std::uint64_t cols) {
  write_npy(path, "<u4", data, rows, cols);
}

void write_npy_f64(const std::string& path, std::span<const double> data, std::uint64_t rows, std::uint64_t cols) {
  write_npy(path, "<f8", data, rows, cols);
}

namespace {

struct NpyLayout {
  std::uint64_t rows = 0, cols = 0;
  std::size_t offset = 0;
};

// The header of a C-order .npy file of dtype `descr` and element size `elem`, 2-D or 1-D (one
// column), checked against the file size.
NpyLayout npy_layout(const MappedFile& file, const std::string& path, const char* descr, std::size_t elem) {
  const char* p = file.data();
  const std::size_t size = file.size();
  auto bad = [&](const std::string& why) { return std::runtime_error(path + ": " + why); };
  if (size < 10 || std::memcmp(p, "\x93NUMPY", 6) != 0) throw bad("not a .npy file");
  const int major = std::uint8_t(p[6]);
  std::size_t hlen = 0, hstart = 0;
  if (major == 1) {
    hlen = std::uint8_t(p[8]) | (std::size_t(std::uint8_t(p[9])) << 8);
    hstart = 10;
  } else if (major == 2 && size >= 12) {
    for (int i = 0; i < 4; ++i) hlen |= std::size_t(std::uint8_t(p[8 + i])) << (8 * i);
    hstart = 12;
  } else {
    throw bad("unsupported .npy version");
  }
  if (hstart + hlen > size) throw bad("truncated header");
  const std::string dict(p + hstart, hlen);
  if (dict.find(std::string("'descr': '") + descr + "'") == std::string::npos)
    throw bad(std::string("dtype is not '") + descr + "'");
  if (dict.find("'fortran_order': False") == std::string::npos) throw bad("not C order");
  NpyLayout l;
  std::smatch m;
  if (std::regex_search(dict, m, std::regex(R"('shape': \((\d+), (\d+)\))"))) {
    l.rows = std::stoull(m[1]);
    l.cols = std::stoull(m[2]);
  } else if (std::regex_search(dict, m, std::regex(R"('shape': \((\d+),\))"))) {
    l.rows = std::stoull(m[1]);
    l.cols = 1;
  } else {
    throw bad("shape is not 1-D or 2-D");
  }
  l.offset = hstart + hlen;
  if (l.offset % elem != 0) throw bad("misaligned data");
  if (size - l.offset != l.rows * l.cols * elem) throw bad("data size does not match shape");
  return l;
}

}  // namespace

NpyU32View::NpyU32View(const std::string& path) : file_(path) {
  const NpyLayout l = npy_layout(file_, path, "<u4", sizeof(std::uint32_t));
  rows_ = l.rows;
  cols_ = l.cols;
  data_ = {reinterpret_cast<const std::uint32_t*>(file_.data() + l.offset), rows_ * cols_};
}

NpyU8View::NpyU8View(const std::string& path) : file_(path) {
  const NpyLayout l = npy_layout(file_, path, "|u1", 1);
  rows_ = l.rows;
  cols_ = l.cols;
  data_ = {reinterpret_cast<const std::uint8_t*>(file_.data() + l.offset), rows_ * cols_};
}

}  // namespace blz

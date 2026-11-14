#pragma once
#include <cstdint>
#include <span>
#include <string>

#include "common/mmap_file.hpp"

namespace blz {

// Writes `rows x cols` little-endian int64 values (C order) as a .npy v1.0 file.
void write_npy_i64(const std::string& path, std::span<const std::int64_t> data,
                   std::uint64_t rows, std::uint64_t cols);
// Same for uint32 ('<u4'), the feature block format (SPEC §8).
void write_npy_u32(const std::string& path, std::span<const std::uint32_t> data,
                   std::uint64_t rows, std::uint64_t cols);
// Same for float64 ('<f8'): vertex statistics (SPEC v2 §8 S2).
void write_npy_f64(const std::string& path, std::span<const double> data, std::uint64_t rows, std::uint64_t cols);
// The .npy v1.0 header (magic through the newline) for a C-order array.
std::string npy_header(const std::string& descr, std::uint64_t rows, std::uint64_t cols);

// Read-only, mmap-backed view of a C-order '<u4' .npy file (v1.0 or v2.0 header), 2-D, or 1-D
// as one column. Throws std::runtime_error on any other layout.
class NpyU32View {
 public:
  explicit NpyU32View(const std::string& path);
  std::uint64_t rows() const { return rows_; }
  std::uint64_t cols() const { return cols_; }
  std::span<const std::uint32_t> row(std::uint64_t r) const { return data_.subspan(r * cols_, cols_); }
  std::span<const std::uint32_t> data() const { return data_; }

 private:
  MappedFile file_;
  std::uint64_t rows_ = 0, cols_ = 0;
  std::span<const std::uint32_t> data_;
};

// The same for '|u1' (blz coverage --pred, SPEC v2 §5).
class NpyU8View {
 public:
  explicit NpyU8View(const std::string& path);
  std::uint64_t rows() const { return rows_; }
  std::uint64_t cols() const { return cols_; }
  std::span<const std::uint8_t> data() const { return data_; }

 private:
  MappedFile file_;
  std::uint64_t rows_ = 0, cols_ = 0;
  std::span<const std::uint8_t> data_;
};

}  // namespace blz

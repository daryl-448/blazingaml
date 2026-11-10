#pragma once
#include <cstdint>
#include <span>
#include <string>

namespace blz {

// Writes `rows x cols` little-endian int64 values (C order) as a .npy v1.0 file.
void write_npy_i64(const std::string& path, std::span<const std::int64_t> data,
                   std::uint64_t rows, std::uint64_t cols);
// The .npy v1.0 header (magic through the newline) for a C-order array.
std::string npy_header(const std::string& descr, std::uint64_t rows, std::uint64_t cols);

}  // namespace blz

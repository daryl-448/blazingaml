#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace blz {

using Sha256Digest = std::array<std::uint8_t, 32>;

// SHA-256 of a byte buffer (FIPS 180-4).
Sha256Digest sha256(const void* data, std::size_t len);
std::string to_hex(const Sha256Digest& d);

// The same, fed in pieces (blz vstats hashes its output as it writes it).
class Sha256 {
 public:
  void update(const void* data, std::size_t len);
  Sha256Digest finish();  // once

 private:
  std::uint32_t h_[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                         0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  std::uint8_t buf_[64] = {};
  std::size_t used_ = 0;
  std::uint64_t len_ = 0;
};

}  // namespace blz

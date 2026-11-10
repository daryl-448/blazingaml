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

}  // namespace blz

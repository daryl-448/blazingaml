#pragma once
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

#include "blz/edges.hpp"
#include "common/mmap_file.hpp"
#include "common/sha256.hpp"

namespace blz {

// .blzg binary cache (SPEC §3.6), little-endian, version 2 (SPEC v2 §3.5, D-31):
//   header (64 B): "BLZG", u32 version, u64 m, u64 n, u8 sha256[32] of the source CSV,
//                  u64 file size;
//   then, each section starting at a 64-byte aligned offset:
//     src u32[m], dst u32[m], ts i32[m], label u8[m], amount_paid f64[m],
//     amount_received f64[m], pay_ccy u8[m], recv_ccy u8[m], format u8[m],
//     file_row u32[m] (new in version 2: each edge's 0-based data row in the CSV),
//     string tables ccy_dict, format_dict, node_bank[n], node_account[n].
//   A string table is u64 count, u64 offsets[count + 1] (into its chars), chars.
// Version 1 (v1, no file_row) is rejected with a message to re-run `blz prepare`.
inline constexpr std::uint32_t kBlzgVersion = 2;

void write_blzg(const std::string& path, const EdgeTable& t);

struct StringTable {
  std::span<const std::uint64_t> offsets;  // size() + 1 entries
  const char* chars = nullptr;
  std::size_t size() const { return offsets.empty() ? 0 : offsets.size() - 1; }
  std::string_view operator[](std::size_t i) const {
    return {chars + offsets[i], std::size_t(offsets[i + 1] - offsets[i])};
  }
};

// Read-only, mmap-backed view of a .blzg file. Throws std::runtime_error if the file is
// malformed.
class BlzgView {
 public:
  explicit BlzgView(const std::string& path);

  std::uint64_t m() const { return m_; }
  std::uint64_t n() const { return n_; }
  const Sha256Digest& csv_sha256() const { return sha_; }

  std::span<const std::uint32_t> src, dst;
  std::span<const std::int32_t> ts;
  std::span<const std::uint8_t> label;
  std::span<const double> amount_paid, amount_received;
  std::span<const std::uint8_t> pay_ccy, recv_ccy, format;
  std::span<const std::uint32_t> file_row;
  StringTable ccy_dict, format_dict, node_bank, node_account;

 private:
  MappedFile file_;
  std::uint64_t m_ = 0, n_ = 0;
  Sha256Digest sha_{};
};

}  // namespace blz

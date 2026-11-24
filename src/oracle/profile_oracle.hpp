#pragma once
#include <cstdint>
#include <span>
#include <variant>

#include "oracle/clean_oracle.hpp"
#include "oracle/oracle.hpp"
#include "runtime/profile.hpp"

namespace blz {

// The reference of a profile, wherever one is used (blz mine --backend oracle, fuzz-diff,
// metamorphic, coverage): the clean oracle for a clean profile (SPEC v3 §3.2), with E16's quirks
// `q`, and the legacy oracle at the profile's visibility for a legacy one. Quirks are for clean
// profiles only (std::invalid_argument otherwise). Each block is mined at the profile's k_max, as
// the oracle of its family does it; check_profile_backend(prof, Backend::oracle) says which
// pattern lists that covers.
class ProfileOracle {
 public:
  ProfileOracle(const EdgeStream& s, const Profile& prof, Quirks q = {});

  std::uint64_t mine(Block b, const BlockParams& p, std::uint64_t first, std::uint64_t last,
                     std::uint32_t* out) const;
  std::uint64_t mine(Block b, const BlockParams& p, std::span<const std::uint32_t> eids, std::uint32_t* out) const;
  // Block sg's gather role alone (the order-relaxation bound of sg_ord).
  std::uint64_t mine_sg_gather(const BlockParams& p, std::uint64_t first, std::uint64_t last,
                               std::uint32_t* out) const;

 private:
  std::variant<Oracle, CleanOracle> o_;
  int k_max_;
};

}  // namespace blz

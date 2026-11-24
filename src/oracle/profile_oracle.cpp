#include "oracle/profile_oracle.hpp"

#include <stdexcept>

namespace blz {
namespace {

std::variant<Oracle, CleanOracle> make(const EdgeStream& s, const Profile& prof, Quirks q, std::span<const double> amount) {
  if (prof.family == Family::clean) return std::variant<Oracle, CleanOracle>(std::in_place_type<CleanOracle>, s, amount, q);
  if (!quirk_ids(q).empty()) throw std::invalid_argument("quirks are for clean profiles only (SPEC v3 §3.6)");
  return std::variant<Oracle, CleanOracle>(std::in_place_type<Oracle>, s, prof.visibility);
}

}  // namespace

ProfileOracle::ProfileOracle(const EdgeStream& s, const Profile& prof, Quirks q, std::span<const double> amount)
    : o_(make(s, prof, q, amount)), k_max_(prof.k_max) {}

std::uint64_t ProfileOracle::mine(Block b, const BlockParams& p, std::uint64_t first, std::uint64_t last,
                                  std::uint32_t* out) const {
  return std::visit([&](const auto& o) { return o.mine(b, p, k_max_, first, last, out); }, o_);
}

std::uint64_t ProfileOracle::mine(Block b, const BlockParams& p, std::span<const std::uint32_t> eids,
                                  std::uint32_t* out) const {
  return std::visit([&](const auto& o) { return o.mine(b, p, k_max_, eids, out); }, o_);
}

std::uint64_t ProfileOracle::mine_sg_gather(const BlockParams& p, std::uint64_t first, std::uint64_t last,
                                            std::uint32_t* out) const {
  return std::visit([&](const auto& o) { return o.mine_sg_gather(p, first, last, out); }, o_);
}

void ProfileOracle::vstats(std::int32_t window, const TriggerSel& sel, double* out) const {
  const auto* o = std::get_if<CleanOracle>(&o_);
  if (!o) throw std::logic_error("vstats: a legacy profile has no vertex statistics (SPEC v3 R8)");
  if (sel.sampled) o->vstats(window, sel.eids, out);
  else o->vstats(window, sel.first, sel.last, out);
}

}  // namespace blz

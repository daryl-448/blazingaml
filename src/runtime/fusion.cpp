#include "runtime/fusion.hpp"

#include <algorithm>

namespace blz {

std::string CudaUnit::name() const {
  return fusion ? std::string(fusion->name) : std::string(block_name(blocks.front()->block));
}

std::vector<CudaUnit> cuda_units(std::span<const ProfileBlock* const> blocks, bool fuse,
                                 std::span<const FusionInfo> fusions) {
  std::vector<bool> used(blocks.size(), false);
  std::vector<std::pair<std::size_t, CudaUnit>> units;  // (first block's position, unit)
  const auto sorted = [](std::vector<const PatternInfo*> v) {
    std::sort(v.begin(), v.end());
    return v;
  };
  for (const FusionInfo& f : fuse ? fusions : std::span<const FusionInfo>{}) {
    std::uint32_t nslots = 0;
    for (std::uint32_t s : f.slots) nslots = std::max(nslots, s + 1);
    std::vector<std::size_t> at(nslots, blocks.size());  // slot -> position in blocks
    bool all = true;
    for (std::uint32_t s = 0; s < nslots && all; ++s) {
      std::vector<const PatternInfo*> want;
      for (std::size_t k = 0; k < f.members.size(); ++k)
        if (f.slots[k] == s) want.push_back(f.members[k]);
      want = sorted(want);
      for (std::size_t i = 0; i < blocks.size() && at[s] == blocks.size(); ++i)
        if (!used[i] && sorted(blocks[i]->patterns) == want) at[s] = i;
      all = at[s] != blocks.size();
    }
    if (!all) continue;
    CudaUnit u{&f, {}};
    for (std::size_t i : at) {
      used[i] = true;
      u.blocks.push_back(blocks[i]);
    }
    units.emplace_back(*std::min_element(at.begin(), at.end()), std::move(u));
  }
  for (std::size_t i = 0; i < blocks.size(); ++i)
    if (!used[i]) units.emplace_back(i, CudaUnit{nullptr, {blocks[i]}});
  std::stable_sort(units.begin(), units.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
  std::vector<CudaUnit> out;
  for (auto& [pos, u] : units) out.push_back(std::move(u));
  return out;
}

}  // namespace blz

#pragma once
// Host-only helpers of the ablation variants (`blzc --disable`, SPEC v2 §7.2, E11). Generated code
// includes this header only when distinct_first or count_only is disabled; such code is CPU-only.
#include <cstddef>
#include <cstdint>
#include <vector>

namespace blz {

// distinct_first disabled: the neighbours a scan has seen, a set of node ids with linear probing.
// It grows to twice the largest scan's distinct count and never shrinks; clear() costs O(size),
// not O(capacity), so a large earlier scan does not slow down the small ones after it.
class DistinctSet {
 public:
  // Adds y; false if y was already in the set.
  bool insert(std::uint32_t y) {
    if (2 * (used_.size() + 1) > slots_.size()) grow();
    std::size_t i = slot(y);
    for (; slots_[i] != kEmpty; i = (i + 1) & mask_)
      if (slots_[i] == y) return false;
    slots_[i] = y;
    used_.push_back(std::uint32_t(i));
    return true;
  }
  void clear() {
    for (std::uint32_t i : used_) slots_[i] = kEmpty;
    used_.clear();
  }
  std::size_t size() const { return used_.size(); }

 private:
  static constexpr std::uint32_t kEmpty = UINT32_MAX;  // node ids are < n <= 2^32 - 1
  std::size_t slot(std::uint32_t y) const { return std::size_t((std::uint64_t(y) * 0x9E3779B97F4A7C15ull) >> shift_); }
  void grow() {
    std::vector<std::uint32_t> keys;
    keys.reserve(used_.size());
    for (std::uint32_t i : used_) keys.push_back(slots_[i]);
    const std::size_t cap = slots_.empty() ? 16 : 2 * slots_.size();
    slots_.assign(cap, kEmpty);
    mask_ = cap - 1;
    shift_ = 64;
    for (std::size_t c = cap; c > 1; c >>= 1) --shift_;
    used_.clear();
    for (std::uint32_t y : keys) insert(y);
  }
  std::vector<std::uint32_t> slots_;
  std::vector<std::uint32_t> used_;  // the occupied slots
  std::size_t mask_ = 0;
  int shift_ = 64;
};

}  // namespace blz

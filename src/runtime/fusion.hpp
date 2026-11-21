#pragma once
// Which of a profile's blocks the CUDA backend mines together (SPEC v2 §8 S7): the fusions blzc
// compiled (blzc --fuse) against the blocks a run selects.
#include <span>
#include <string>
#include <vector>

#include "blz/registry.hpp"
#include "runtime/profile.hpp"

namespace blz {

// One CUDA run: a fusion's blocks, or one block whose patterns run one after another (unfused).
struct CudaUnit {
  const FusionInfo* fusion = nullptr;       // null: unfused
  std::vector<const ProfileBlock*> blocks;  // fused: slot s's block at s; unfused: the one block
  std::string name() const;                 // the fusion's name, or the block's
};

// The units of `blocks` (a run's selection, in profile order). With `fuse`, a fusion applies when
// `blocks` holds, for each of its slots, a block whose patterns are exactly that slot's members (in
// any order); fusions are tried in `fusions` order, and a block joins at most one. Every other
// block is a unit of its own. Units are ordered by their first block's position in `blocks`.
// Without `fuse`, every block is a unit of its own, in order.
std::vector<CudaUnit> cuda_units(std::span<const ProfileBlock* const> blocks, bool fuse,
                                 std::span<const FusionInfo> fusions = registered_fusions());

}  // namespace blz

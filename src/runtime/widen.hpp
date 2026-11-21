#pragma once
// Packed D2H (SPEC v2 §8 S7): the host side widens a block's rows from the narrow type they crossed
// PCIe in back to uint32. Plain C++ (OpenMP), so nvcc-compiled code can call it.
#include <cstdint>

namespace blz {

// dst[i] = src[i] for i < n, where src holds n cells of `width` bytes (1: uint8, 2: uint16).
// Parallel over the OpenMP team, with streaming stores where the target supports them (the rows
// are not read again soon, and a streaming store does not first read the line it overwrites).
void widen_cells(const void* src, std::uint32_t width, std::uint32_t* dst, std::uint64_t n);

}  // namespace blz

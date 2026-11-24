// The CUDA side of clean_vstats.cpp (SPEC v3 R8): one thread per distinct side, the same BLZ_HD
// clean_vs_side as the CPU driver. Built with --fmad=false (CMakeLists.txt); division and sqrt are
// IEEE-rounded (nvcc's defaults), so the sides equal the CPU's and the clean oracle's bit for bit.
#include <algorithm>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include "runtime/clean_vstats.hpp"
#include "runtime/cuda_backend.hpp"

namespace blz {

namespace {

void check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) throw std::runtime_error(std::string("CUDA: ") + what + ": " + cudaGetErrorString(err));
}
#define BLZ_CUDA(call) check((call), #call)

// A device allocation that grows on demand and is freed on scope exit.
struct DeviceBuffer {
  void* ptr = nullptr;
  std::size_t bytes = 0;
  void reserve(std::size_t want) {
    if (want <= bytes) return;
    if (ptr) BLZ_CUDA(cudaFree(ptr));
    ptr = nullptr;
    BLZ_CUDA(cudaMalloc(&ptr, want));
    bytes = want;
  }
  ~DeviceBuffer() {
    if (ptr) cudaFree(ptr);
  }
};

__global__ void clean_vs_kernel(TCsrView out_l, TCsrView in_l, VsOrderView out_o, VsOrderView in_o, std::int32_t window,
                                const VsSide* sides, std::uint32_t count, double* res) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const VsSide s = sides[i];
  clean_vs_side(s.in ? in_l : out_l, s.in ? in_o : out_o, window, s, res + std::size_t(i) * kVstatsPerSide);
}

}  // namespace

struct CleanVstats::Device {
  TCsrView lists[2];
  DeviceBuffer pos[2], amount[2];
  VsOrderView order[2];
  DeviceBuffer sides, res;
};

void CleanVstats::DeviceDelete::operator()(Device* d) const { delete d; }

void CleanVstats::init_device(const DeviceGraph& dg) {
  dev_.reset(new Device);
  Device& d = *dev_;
  d.lists[0] = dg.view().out;
  d.lists[1] = dg.view().in;
  for (int k = 0; k < 2; ++k) {
    d.pos[k].reserve(std::max<std::size_t>(pos_[k].size() * sizeof(std::uint32_t), 1));
    d.amount[k].reserve(std::max<std::size_t>(amount_[k].size() * sizeof(double), 1));
    BLZ_CUDA(cudaMemcpy(d.pos[k].ptr, pos_[k].data(), pos_[k].size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice));
    BLZ_CUDA(cudaMemcpy(d.amount[k].ptr, amount_[k].data(), amount_[k].size() * sizeof(double), cudaMemcpyHostToDevice));
    d.order[k] = {static_cast<const std::uint32_t*>(d.pos[k].ptr), static_cast<const double*>(d.amount[k].ptr)};
  }
}

void CleanVstats::sides_gpu(std::span<const VsSide> sides, double* out) {
  if (sides.empty()) return;
  Device& d = *dev_;
  d.sides.reserve(sides.size_bytes());
  d.res.reserve(sides.size() * kVstatsPerSide * sizeof(double));
  BLZ_CUDA(cudaMemcpy(d.sides.ptr, sides.data(), sides.size_bytes(), cudaMemcpyHostToDevice));
  const auto count = std::uint32_t(sides.size());
  clean_vs_kernel<<<(count + 127) / 128, 128>>>(d.lists[0], d.lists[1], d.order[0], d.order[1], window_,
                                                static_cast<const VsSide*>(d.sides.ptr), count,
                                                static_cast<double*>(d.res.ptr));
  BLZ_CUDA(cudaGetLastError());
  BLZ_CUDA(cudaMemcpy(out, d.res.ptr, sides.size() * kVstatsPerSide * sizeof(double), cudaMemcpyDeviceToHost));
}

}  // namespace blz

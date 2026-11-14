// The CUDA twin of vstats.cpp (SPEC v2 §8 S2): one thread per row, the same BLZ_HD code as the
// CPU driver. Built with --fmad=false (CMakeLists.txt); division and sqrt are IEEE-rounded
// (nvcc's defaults), so the rows equal the CPU's bit for bit.
#include <algorithm>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include "runtime/cuda_backend.hpp"
#include "runtime/vstats.hpp"

namespace blz {

namespace {

void check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) throw std::runtime_error(std::string("CUDA: ") + what + ": " + cudaGetErrorString(err));
}
#define BLZ_CUDA(call) check((call), #call)

__global__ void vstats_kernel(GraphView g, const double* amount, VstatsParams p, std::uint32_t first,
                              std::uint32_t count, double* out) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) vstats_row(g, amount, p, first + i, out + std::size_t(i) * kVstatsCols);
}

// A device allocation freed on scope exit.
struct DeviceBuffer {
  void* ptr = nullptr;
  explicit DeviceBuffer(std::size_t bytes) { BLZ_CUDA(cudaMalloc(&ptr, std::max<std::size_t>(bytes, 1))); }
  ~DeviceBuffer() { cudaFree(ptr); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

}  // namespace

void vstats_cuda(const DeviceGraph& dg, std::span<const double> amount, const VstatsParams& p, std::uint64_t first,
                 std::uint64_t last, double* out) {
  const GraphView& g = dg.view();
  if (amount.size() != g.m) throw std::invalid_argument("vstats: one amount per edge is needed");
  if (p.window <= 0) throw std::invalid_argument("vstats: window must be > 0");
  if (first > last || last > g.m) throw std::invalid_argument("vstats: bad trigger range");
  DeviceBuffer d_amount(amount.size() * sizeof(double));
  BLZ_CUDA(cudaMemcpy(d_amount.ptr, amount.data(), amount.size() * sizeof(double), cudaMemcpyHostToDevice));
  // Rows in chunks, so the device output stays at 1 << 19 rows (218 MB).
  const std::uint64_t chunk = std::min<std::uint64_t>(last - first, 1u << 19);
  DeviceBuffer d_out(chunk * kVstatsCols * sizeof(double));
  for (std::uint64_t a = first; a < last; a += chunk) {
    const std::uint32_t count = std::uint32_t(std::min(chunk, last - a));
    const unsigned blocks = (count + 255) / 256;
    vstats_kernel<<<blocks, 256>>>(g, static_cast<const double*>(d_amount.ptr), p, std::uint32_t(a), count,
                                   static_cast<double*>(d_out.ptr));
    BLZ_CUDA(cudaGetLastError());
    BLZ_CUDA(cudaMemcpy(out + (a - first) * kVstatsCols, d_out.ptr, std::size_t(count) * kVstatsCols * sizeof(double),
                        cudaMemcpyDeviceToHost));
  }
}

}  // namespace blz

// The CUDA backend in a build without BLZ_ENABLE_CUDA: every entry point throws.
#include <stdexcept>

#include "runtime/cuda_backend.hpp"

namespace blz {

namespace {

[[noreturn]] void no_cuda() {
  throw std::runtime_error("this build has no CUDA backend; configure with the rel-cuda preset (BLZ_ENABLE_CUDA=ON)");
}

}  // namespace

bool cuda_built() { return false; }
CudaDevice cuda_device() { no_cuda(); }

PinnedBuffer::PinnedBuffer(std::uint64_t) { no_cuda(); }
PinnedBuffer::~PinnedBuffer() = default;

struct DeviceGraph::Impl {};
DeviceGraph::DeviceGraph(const GraphView&) { no_cuda(); }
DeviceGraph::~DeviceGraph() = default;
const GraphView& DeviceGraph::view() const { no_cuda(); }
std::uint64_t DeviceGraph::bytes() const { no_cuda(); }
double DeviceGraph::upload_seconds() const { no_cuda(); }

struct CudaMiner::Impl {};
CudaMiner::CudaMiner() { no_cuda(); }
CudaMiner::~CudaMiner() = default;
CudaBlockTiming CudaMiner::mine(const DeviceGraph&, Block, const BlockParams&, std::uint64_t, std::uint64_t,
                                std::uint32_t, std::uint32_t*) {
  no_cuda();
}

}  // namespace blz

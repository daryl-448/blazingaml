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
CudaMiner::CudaMiner(bool) { no_cuda(); }
CudaMiner::~CudaMiner() = default;
CudaBlockTiming CudaMiner::mine(const DeviceGraph&, Block, const BlockParams&, std::uint64_t, std::uint64_t,
                                std::uint32_t, std::uint32_t*) {
  no_cuda();
}
CudaBlockTiming CudaMiner::mine(const DeviceGraph&, std::span<const PatternInfo* const>, const BlockParams&,
                                Visibility, std::uint64_t, std::uint64_t, std::uint32_t, std::uint32_t*, bool) {
  no_cuda();
}
CudaMiner::FusedTiming CudaMiner::mine_fused(const DeviceGraph&, const FusionInfo&, std::span<const BlockParams* const>,
                                             Visibility, std::uint64_t, std::uint64_t, std::uint32_t,
                                             std::span<std::uint32_t* const>, bool) {
  no_cuda();
}
CudaBlockTiming CudaMiner::fetch_for_test(std::span<const std::uint32_t>, std::uint32_t*, bool) { no_cuda(); }
CudaBlockTiming CudaMiner::mine_subset(const DeviceGraph&, std::span<const PatternInfo* const>, const BlockParams&,
                                       Visibility, std::uint64_t, std::uint64_t, std::span<const std::uint32_t>,
                                       std::uint32_t, std::uint32_t*) {
  no_cuda();
}
CudaBlockTiming CudaMiner::mine_split(const DeviceGraph&, std::span<const PatternInfo* const>, const BlockParams&,
                                      Visibility, std::uint64_t, std::uint64_t, std::uint32_t, std::uint32_t,
                                      std::uint32_t*, const CpuListFn&) {
  no_cuda();
}

}  // namespace blz

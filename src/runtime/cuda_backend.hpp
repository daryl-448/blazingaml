#pragma once
// The CUDA backend's host side (SPEC §7.5): the device mirror of the graph index, light/heavy
// bucketing, and the per-block driver with its timings. Plain C++ interface; a build without
// BLZ_ENABLE_CUDA links cuda_backend_stub.cpp, whose functions throw.
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "blz/primitives.hpp"
#include "blz/registry.hpp"
#include "runtime/features.hpp"

namespace blz {

// The light/heavy threshold H (SPEC §7.5): a trigger whose work estimate is >= H goes to the
// heavy kernel (warp per trigger), the others to the light kernel (thread per trigger).
// H = 0 sends every trigger to the heavy kernel, kAllLight every trigger to the light kernel.
inline constexpr std::uint32_t kDefaultHeavyThreshold = 32;
inline constexpr std::uint32_t kAllLight = UINT32_MAX;

// Whether this build has the CUDA backend.
bool cuda_built();

struct CudaDevice {
  std::string name;
  int cc_major = 0, cc_minor = 0, sm_count = 0;
  std::uint64_t total_bytes = 0, free_bytes = 0;
  int driver_version = 0, runtime_version = 0;  // e.g. 13040
};
// Device 0, which every CUDA call here uses. Throws if there is none.
CudaDevice cuda_device();

// Host memory pinned for fast device-to-host copies (cudaMallocHost).
class PinnedBuffer {
 public:
  explicit PinnedBuffer(std::uint64_t count);  // uint32 elements, uninitialized
  ~PinnedBuffer();
  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;
  std::uint32_t* data() const { return data_; }
  std::uint64_t size() const { return size_; }

 private:
  std::uint32_t* data_ = nullptr;
  std::uint64_t size_ = 0;
};

// The device mirror of an indexed graph (SPEC §7.5 "Memory"): the edge arrays and both
// directions of T-CSR and P-CSR, copied once from a host GraphView into one device allocation
// and shared by every block.
class DeviceGraph {
 public:
  explicit DeviceGraph(const GraphView& host);
  ~DeviceGraph();
  DeviceGraph(const DeviceGraph&) = delete;
  DeviceGraph& operator=(const DeviceGraph&) = delete;

  const GraphView& view() const;  // device pointers
  std::uint64_t bytes() const;
  double upload_seconds() const;  // the host-to-device copies, from pageable memory (cudaEvents)

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

struct CudaBlockTiming {
  double kernel_seconds = 0;  // zeroing + bucketing + light + heavy kernels, every pattern (cudaEvents)
  double bucket_seconds = 0;  // work estimates, partition, and the read-back of the heavy count
  double light_seconds = 0, heavy_seconds = 0;
  // The block's rows to the host. Unpacked: the copy (cudaEvents). Packed (SPEC v2 §8 S7): from the
  // first narrow chunk's copy to the last row widened into the host buffer (host clock).
  double d2h_seconds = 0;
  double pack_seconds = 0;    // packed: the pack kernels and the read-back of the block's maximum (cudaEvents)
  double widen_seconds = 0;   // packed: the host's widening, summed over the chunks (inside d2h_seconds)
  std::uint32_t d2h_width = 4;  // bytes per cell over PCIe: 1, 2 or 4
  std::uint64_t d2h_bytes = 0;  // bytes over PCIe
  double h2d_seconds = 0;     // mine_subset: the trigger list to the device (cudaEvents)
  double split_seconds = 0;   // mine_split: the CPU's triggers found and copied to the host (cudaEvents)
  std::uint64_t cpu_triggers = 0;  // mine_split: how many
  double wall_seconds = 0;    // host wall clock of the whole call
  std::uint64_t light = 0, heavy = 0;  // triggers per kernel, summed over the block's patterns
  std::uint64_t saturated = 0;
};

// The per-block driver. Owns a stream, events and scratch buffers, which grow as needed and are
// reused by later calls. Not thread-safe, but it may be used from any one thread at a time.
// `blocking_sync` makes every host wait sleep instead of spin (cudaEventBlockingSync), so a
// thread driving the GPU next to an OpenMP team (the hybrid, SPEC v2 §8 S6) does not take a core.
class CudaMiner {
 public:
  explicit CudaMiner(bool blocking_sync = false);
  ~CudaMiner();
  CudaMiner(const CudaMiner&) = delete;
  CudaMiner& operator=(const CudaMiner&) = delete;

  // Mines block b for the triggers [first, last) with every pattern of the block, under stream
  // visibility, bucketed at `heavy_threshold`, and copies the rows to host_out: (last - first)
  // rows of p.bins.size() columns, overwritten. Pinned host memory makes the copy faster. Throws
  // if no pattern emits into b or the build has no CUDA launchers.
  CudaBlockTiming mine(const DeviceGraph& g, Block b, const BlockParams& p, std::uint64_t first,
                       std::uint64_t last, std::uint32_t heavy_threshold, std::uint32_t* host_out);
  // The same with the patterns `pats` (a profile's list for the block, SPEC v2 §3.1) under
  // visibility `vis` (the profile's, SPEC v2 §6.1).
  //
  // `pack` (SPEC v2 §8 S7) copies the rows as uint8 when every cell is <= 255, else as uint16 when
  // every cell is <= 65535, else as uint32, and widens them on the host: the same rows, fewer bytes
  // over PCIe. A pack kernel writes the uint8 copy and the block's maximum (the saturation flag); the
  // host reads the maximum before any row is copied, so a block that overflows uint8 is packed again
  // as uint16 (or copied as uint32) and never crosses PCIe twice.
  CudaBlockTiming mine(const DeviceGraph& g, std::span<const PatternInfo* const> pats, const BlockParams& p,
                       Visibility vis, std::uint64_t first, std::uint64_t last, std::uint32_t heavy_threshold,
                       std::uint32_t* host_out, bool pack = false);
  // A fusion's blocks at once (SPEC v2 §8 S7): one kernel runs every member of `f` for each trigger
  // in [first, last). params[s] and host_out[s] are slot s's block parameters and rows ((last - first)
  // rows of params[s]->bins.size() columns, overwritten); member k runs with params[f.slots[k]]'s
  // window. The work estimate of a trigger is the largest of its members'. `group` holds the kernel
  // times and the trigger counts (each trigger once), `slots[s]` slot s's pack, D2H and saturation;
  // group's d2h, pack, widen, bytes and saturated are the slots' sums.
  struct FusedTiming {
    CudaBlockTiming group;
    std::vector<CudaBlockTiming> slots;
  };
  FusedTiming mine_fused(const DeviceGraph& g, const FusionInfo& f, std::span<const BlockParams* const> params,
                         Visibility vis, std::uint64_t first, std::uint64_t last, std::uint32_t heavy_threshold,
                         std::span<std::uint32_t* const> host_out, bool pack = false);
  // For tests: uploads `values` as one block's device rows and copies them to host_out as mine()
  // does with `pack`.
  CudaBlockTiming fetch_for_test(std::span<const std::uint32_t> values, std::uint32_t* host_out, bool pack);
  // The hybrid's GPU side (SPEC v2 §8 S6): the same, but only for the triggers `subset` (ascending
  // eids in [first, last)), which are uploaded first (h2d_seconds). All (last - first) rows are
  // still written to host_out; the rows of the other triggers are zero.
  CudaBlockTiming mine_subset(const DeviceGraph& g, std::span<const PatternInfo* const> pats, const BlockParams& p,
                              Visibility vis, std::uint64_t first, std::uint64_t last,
                              std::span<const std::uint32_t> subset, std::uint32_t heavy_threshold,
                              std::uint32_t* host_out);
  // The hybrid's GPU side with the split on the device (SPEC v2 §8 S6): the triggers in [first,
  // last) whose work estimate (pats.front()'s, the CPU's estimate compiled for the device) is
  // >= cpu_threshold are the CPU's. Their eids, ascending, go to on_cpu_list as soon as they are
  // known (split_seconds), on this call's thread; then the GPU mines every other trigger as
  // mine_subset does. The CPU's rows are written as zero.
  using CpuListFn = std::function<void(std::vector<std::uint32_t>)>;
  CudaBlockTiming mine_split(const DeviceGraph& g, std::span<const PatternInfo* const> pats, const BlockParams& p,
                             Visibility vis, std::uint64_t first, std::uint64_t last, std::uint32_t cpu_threshold,
                             std::uint32_t heavy_threshold, std::uint32_t* host_out, const CpuListFn& on_cpu_list);

 private:
  struct Split {
    std::uint32_t cpu_threshold;
    const CpuListFn* on_cpu_list;
  };
  CudaBlockTiming run(const DeviceGraph& g, std::span<const PatternInfo* const> pats, const BlockParams& p,
                      Visibility vis, std::uint64_t first, std::uint64_t last,
                      const std::span<const std::uint32_t>* subset, const Split* split,
                      std::uint32_t heavy_threshold, std::uint32_t* host_out, bool pack);
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace blz

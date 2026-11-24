// The CUDA backend's host side (SPEC §7.5). See cuda_backend.hpp.
#include "runtime/cuda_backend.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <cub/device/device_partition.cuh>
#include <cuda_runtime.h>

#include "blz/cuda_kernels.cuh"
#include "blz/registry.hpp"
#include "runtime/cpu_backend.hpp"
#include "runtime/widen.hpp"

namespace blz {

namespace {

using Clock = std::chrono::steady_clock;

void check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) throw std::runtime_error(std::string("CUDA: ") + what + ": " + cudaGetErrorString(err));
}
#define BLZ_CUDA(call) check((call), #call)

double elapsed(cudaEvent_t a, cudaEvent_t b) {
  float ms = 0;
  BLZ_CUDA(cudaEventElapsedTime(&ms, a, b));
  return double(ms) / 1e3;
}

// A device buffer that only grows.
struct DeviceBuffer {
  void* ptr = nullptr;
  std::size_t cap = 0;
  void ensure(std::size_t bytes) {
    if (bytes <= cap) return;
    if (ptr) BLZ_CUDA(cudaFree(ptr));
    ptr = nullptr;
    cap = 0;
    BLZ_CUDA(cudaMalloc(&ptr, bytes));
    cap = bytes;
  }
  ~DeviceBuffer() {
    if (ptr) cudaFree(ptr);
  }
  template <class T>
  T* as() const {
    return static_cast<T*>(ptr);
  }
};

__global__ void iota_kernel(std::uint32_t* ids, std::uint64_t first, std::uint32_t count) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) ids[i] = std::uint32_t(first + i);
}

// mine_subset: the heavy flag of each listed trigger, from the flags of the whole range.
__global__ void gather_flags_kernel(const std::uint32_t* ids, std::uint32_t count, std::uint64_t first,
                                    const std::uint8_t* all, std::uint8_t* flags) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) flags[i] = all[ids[i] - first];
}

// The largest bin count the heavy kernel's shared histograms allow (48 KB per block).
// heavy_shared_bytes(nbins) <= 48 KiB: one histogram and one work counter (SPEC v2 §8 S3) per warp.
constexpr std::uint32_t kMaxBins = 48 * 1024 / (cuda::kHeavyWarps * sizeof(unsigned long long)) - 1;

// The device counters (u64 each): the emits lost to saturation per slot (an unfused block uses slot
// 0), the heavy count of a partition (int), the smallest aborted eid (SPEC v2 §8 S3), and the
// block's maximum cell of a pack (unsigned int, SPEC v2 §8 S7).
enum Counter { kSaturated = 0, kHeavyCount = kMaxFusedSlots, kAborted, kMaxCell, kCounters };

// Packed D2H (SPEC v2 §8 S7): out[i] = min(in[i], max of T) and *max_cell = the largest in[i]
// (when max_cell is set). `in` is 16-byte aligned; four cells per load.
template <class T>
__global__ void pack_kernel(const std::uint32_t* __restrict__ in, std::uint64_t n, T* __restrict__ out,
                            unsigned int* max_cell) {
  constexpr std::uint32_t kTop = T(~T(0));
  const std::uint64_t stride = std::uint64_t(gridDim.x) * blockDim.x;
  const std::uint64_t tid = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t n4 = n / 4;
  unsigned int m = 0;
  for (std::uint64_t i = tid; i < n4; i += stride) {
    const uint4 v = reinterpret_cast<const uint4*>(in)[i];
    m = max(m, max(max(v.x, v.y), max(v.z, v.w)));
    if constexpr (sizeof(T) == 1)
      reinterpret_cast<uchar4*>(out)[i] = make_uchar4(std::uint8_t(min(v.x, kTop)), std::uint8_t(min(v.y, kTop)),
                                                      std::uint8_t(min(v.z, kTop)), std::uint8_t(min(v.w, kTop)));
    else
      reinterpret_cast<ushort4*>(out)[i] = make_ushort4(std::uint16_t(min(v.x, kTop)), std::uint16_t(min(v.y, kTop)),
                                                        std::uint16_t(min(v.z, kTop)), std::uint16_t(min(v.w, kTop)));
  }
  for (std::uint64_t i = n4 * 4 + tid; i < n; i += stride) {
    m = max(m, in[i]);
    out[i] = T(min(in[i], kTop));
  }
  if (!max_cell) return;  // the same for every thread
  for (int o = 16; o > 0; o >>= 1) m = max(m, __shfl_xor_sync(cuda::kFullMask, m, o));
  if (threadIdx.x % cuda::kWarpSize == 0 && m) atomicMax(max_cell, m);
}

// Pinned staging for packed D2H: the narrow rows cross PCIe in chunks of this many bytes, into two
// buffers in turn, and the host widens one chunk while the next one is copied.
constexpr std::size_t kStageBytes = 4 << 20;

// Device row regions start on 256-byte boundaries (the pack kernel reads 16-byte vectors).
std::uint64_t round_cells(std::uint64_t cells) { return (cells + 63) / 64 * 64; }

}  // namespace

bool cuda_built() { return true; }

CudaDevice cuda_device() {
  int count = 0;
  BLZ_CUDA(cudaGetDeviceCount(&count));
  if (count == 0) throw std::runtime_error("CUDA: no device");
  cudaDeviceProp prop{};
  BLZ_CUDA(cudaGetDeviceProperties(&prop, 0));
  CudaDevice d;
  d.name = prop.name;
  d.cc_major = prop.major;
  d.cc_minor = prop.minor;
  d.sm_count = prop.multiProcessorCount;
  std::size_t free_b = 0, total_b = 0;
  BLZ_CUDA(cudaMemGetInfo(&free_b, &total_b));
  d.free_bytes = free_b;
  d.total_bytes = total_b;
  BLZ_CUDA(cudaDriverGetVersion(&d.driver_version));
  BLZ_CUDA(cudaRuntimeGetVersion(&d.runtime_version));
  return d;
}

PinnedBuffer::PinnedBuffer(std::uint64_t count) : size_(count) {
  void* p = nullptr;
  if (count > 0) BLZ_CUDA(cudaMallocHost(&p, count * sizeof(std::uint32_t)));
  data_ = static_cast<std::uint32_t*>(p);
}

PinnedBuffer::~PinnedBuffer() {
  if (data_) cudaFreeHost(data_);
}

// --- DeviceGraph ------------------------------------------------------------------------------

struct DeviceGraph::Impl {
  void* base = nullptr;
  std::uint64_t bytes = 0;
  GraphView view;
  double upload_s = 0;
  ~Impl() {
    if (base) cudaFree(base);
  }
};

DeviceGraph::DeviceGraph(const GraphView& h) : impl_(std::make_unique<Impl>()) {
  const std::uint64_t m = h.m, n = h.n;
  const std::uint64_t pairs_out = h.pout.off[n], pairs_in = h.pin.off[n];
  struct Copy {
    std::uint64_t off;
    const void* host;
    std::uint64_t bytes;
  };
  std::vector<Copy> copies;
  GraphView& d = impl_->view;
  std::uintptr_t base = 0;
  std::uint64_t used = 0;
  // Lays every array out in one allocation (256-byte aligned) and points d's fields into it.
  // Run once with base 0 to size the allocation, then with the real base.
  auto layout = [&] {
    used = 0;
    copies.clear();
    auto put = [&](auto*& field, auto* host, std::uint64_t count) {
      using T = std::remove_reference_t<decltype(*host)>;
      const std::uint64_t bytes = count * sizeof(T);
      field = reinterpret_cast<T*>(base + used);
      copies.push_back({used, host, bytes});
      used += (bytes + 255) / 256 * 256;
    };
    d.m = h.m;
    d.n = h.n;
    put(d.src, h.src, m);
    put(d.dst, h.dst, m);
    put(d.ts, h.ts, m);
    // The lists hold off[n] entries: m, less the self-loops of an index that drops them (SPEC v3 R1).
    for (auto [dv, hv] : {std::pair{&d.out, &h.out}, std::pair{&d.in, &h.in}}) {
      const std::uint64_t len = hv->off[n];
      put(dv->off, hv->off, n + 1);
      put(dv->nbr, hv->nbr, len);
      put(dv->ts, hv->ts, len);
      put(dv->eid, hv->eid, len);
      put(dv->prev_same, hv->prev_same, len);
    }
    for (auto [dv, hv, pairs] :
         {std::tuple{&d.pout, &h.pout, pairs_out}, std::tuple{&d.pin, &h.pin, pairs_in}}) {
      const std::uint64_t len = hv->run[pairs];
      put(dv->off, hv->off, n + 1);
      put(dv->nbr, hv->nbr, pairs);
      put(dv->run, hv->run, pairs + 1);
      put(dv->run_ts, hv->run_ts, len);
      put(dv->run_eid, hv->run_eid, len);
    }
  };
  layout();
  impl_->bytes = used;
  BLZ_CUDA(cudaMalloc(&impl_->base, used));
  base = reinterpret_cast<std::uintptr_t>(impl_->base);
  layout();

  cudaEvent_t a, b;
  BLZ_CUDA(cudaEventCreate(&a));
  BLZ_CUDA(cudaEventCreate(&b));
  BLZ_CUDA(cudaEventRecord(a));
  for (const Copy& c : copies)
    if (c.bytes > 0)
      BLZ_CUDA(cudaMemcpy(static_cast<char*>(impl_->base) + c.off, c.host, c.bytes, cudaMemcpyHostToDevice));
  BLZ_CUDA(cudaEventRecord(b));
  BLZ_CUDA(cudaEventSynchronize(b));
  impl_->upload_s = elapsed(a, b);
  cudaEventDestroy(a);
  cudaEventDestroy(b);
}

DeviceGraph::~DeviceGraph() = default;
const GraphView& DeviceGraph::view() const { return impl_->view; }
std::uint64_t DeviceGraph::bytes() const { return impl_->bytes; }
double DeviceGraph::upload_seconds() const { return impl_->upload_s; }

// --- CudaMiner --------------------------------------------------------------------------------

struct CudaMiner::Impl {
  cudaStream_t stream = nullptr;
  enum {
    kStart, kBucket, kPart, kLight, kHeavy, kDone, kD2h0, kD2h1, kH2d0, kH2d1, kSplit0, kSplit1,
    kPack0, kPack1, kStage0, kStage1, kEvents
  };
  cudaEvent_t ev[kEvents] = {};
  DeviceBuffer out, ids, flags, flags_all, part, temp, bins, counters, packed;
  std::uint64_t* host_counters = nullptr;  // pinned: kCounters
  void* stage[2] = {};                     // pinned: kStageBytes each, allocated on the first packed copy

  explicit Impl(bool blocking_sync) {
    BLZ_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (cudaEvent_t& e : ev)
      BLZ_CUDA(cudaEventCreateWithFlags(&e, blocking_sync ? cudaEventBlockingSync : cudaEventDefault));
    BLZ_CUDA(cudaMallocHost(reinterpret_cast<void**>(&host_counters), kCounters * sizeof(std::uint64_t)));
    counters.ensure(kCounters * sizeof(std::uint64_t));
  }
  ~Impl() {
    for (cudaEvent_t e : ev)
      if (e) cudaEventDestroy(e);
    if (stream) cudaStreamDestroy(stream);
    if (host_counters) cudaFreeHost(host_counters);
    for (void* p : stage)
      if (p) cudaFreeHost(p);
  }

  std::uint64_t* counter(Counter c) const { return counters.as<std::uint64_t>() + c; }

  // The partition of ids[0, count) by flags into part: the flagged ids first, in order, then the
  // others in reverse order. Returns how many were flagged (a host read-back).
  int partition(std::uint32_t count, std::size_t temp_bytes) {
    int* d_heavy = reinterpret_cast<int*>(counter(kHeavyCount));
    BLZ_CUDA(cub::DevicePartition::Flagged(temp.ptr, temp_bytes, ids.as<std::uint32_t>(), flags.as<std::uint8_t>(),
                                           part.as<std::uint32_t>(), d_heavy, int(count), stream));
    BLZ_CUDA(cudaMemcpyAsync(host_counters + kHeavyCount, d_heavy, sizeof(int), cudaMemcpyDeviceToHost, stream));
    BLZ_CUDA(cudaEventRecord(ev[kPart], stream));
    BLZ_CUDA(cudaEventSynchronize(ev[kPart]));
    int n = 0;
    std::memcpy(&n, host_counters + kHeavyCount, sizeof(int));
    if (n < 0 || std::uint32_t(n) > count) throw std::runtime_error("CUDA: bad partition count");
    return n;
  }

  // The scratch that partition() needs for `count` ids.
  std::size_t partition_temp(std::uint32_t count) {
    std::size_t bytes = 0;
    BLZ_CUDA(cub::DevicePartition::Flagged(nullptr, bytes, ids.as<std::uint32_t>(), flags.as<std::uint8_t>(),
                                           part.as<std::uint32_t>(), reinterpret_cast<int*>(counter(kHeavyCount)),
                                           int(count), stream));
    temp.ensure(std::max<std::size_t>(bytes, 1));
    return bytes;
  }

  // `cells` uint32 rows at device address d (256-byte aligned) to host_out, after the stream's work
  // so far: one copy, or packed (SPEC v2 §8 S7, see CudaMiner::mine). Sets tm's D2H fields.
  void fetch(const std::uint32_t* d, std::uint64_t cells, std::uint32_t* host_out, bool pack, CudaBlockTiming& tm) {
    const cudaStream_t st = stream;
    std::uint32_t width = 4;
    if (pack && cells > 0) {
      packed.ensure(cells * sizeof(std::uint16_t));
      unsigned int* d_max = reinterpret_cast<unsigned int*>(counter(kMaxCell));
      const unsigned blocks = unsigned(std::clamp<std::uint64_t>((cells / 4 + 255) / 256, 1, 2048));
      BLZ_CUDA(cudaEventRecord(ev[kPack0], st));
      BLZ_CUDA(cudaMemsetAsync(d_max, 0, sizeof(unsigned int), st));
      pack_kernel<std::uint8_t><<<blocks, 256, 0, st>>>(d, cells, packed.as<std::uint8_t>(), d_max);
      BLZ_CUDA(cudaGetLastError());
      BLZ_CUDA(cudaMemcpyAsync(host_counters + kMaxCell, d_max, sizeof(unsigned int), cudaMemcpyDeviceToHost, st));
      BLZ_CUDA(cudaEventRecord(ev[kPack1], st));
      BLZ_CUDA(cudaEventSynchronize(ev[kPack1]));
      unsigned int max_cell = 0;
      std::memcpy(&max_cell, host_counters + kMaxCell, sizeof(unsigned int));
      width = max_cell <= 0xffu ? 1 : max_cell <= 0xffffu ? 2 : 4;
      if (width == 2) {  // the uint8 copy saturated: pack again as uint16
        pack_kernel<std::uint16_t><<<blocks, 256, 0, st>>>(d, cells, packed.as<std::uint16_t>(), nullptr);
        BLZ_CUDA(cudaGetLastError());
        BLZ_CUDA(cudaEventRecord(ev[kPack1], st));
        BLZ_CUDA(cudaEventSynchronize(ev[kPack1]));
      }
      tm.pack_seconds = elapsed(ev[kPack0], ev[kPack1]);
    }
    tm.d2h_width = width;
    tm.d2h_bytes = cells * width;
    if (width == 4) {
      BLZ_CUDA(cudaEventRecord(ev[kD2h0], st));
      if (cells > 0)
        BLZ_CUDA(cudaMemcpyAsync(host_out, d, cells * sizeof(std::uint32_t), cudaMemcpyDeviceToHost, st));
      BLZ_CUDA(cudaEventRecord(ev[kD2h1], st));
      BLZ_CUDA(cudaEventSynchronize(ev[kD2h1]));
      tm.d2h_seconds = elapsed(ev[kD2h0], ev[kD2h1]);
      return;
    }
    for (void*& p : stage)
      if (!p) BLZ_CUDA(cudaMallocHost(&p, kStageBytes));
    const char* src = packed.as<char>();
    const std::uint64_t chunk = kStageBytes / width, chunks = (cells + chunk - 1) / chunk;
    const auto issue = [&](std::uint64_t k) {
      const std::uint64_t a = k * chunk, len = std::min(chunk, cells - a);
      BLZ_CUDA(cudaMemcpyAsync(stage[k % 2], src + a * width, len * width, cudaMemcpyDeviceToHost, st));
      BLZ_CUDA(cudaEventRecord(ev[kStage0 + k % 2], st));
    };
    const auto t0 = Clock::now();
    for (std::uint64_t k = 0; k < std::min<std::uint64_t>(chunks, 2); ++k) issue(k);
    double widen_s = 0;
    for (std::uint64_t k = 0; k < chunks; ++k) {
      BLZ_CUDA(cudaEventSynchronize(ev[kStage0 + k % 2]));
      const auto w0 = Clock::now();
      const std::uint64_t a = k * chunk, len = std::min(chunk, cells - a);
      widen_cells(stage[k % 2], width, host_out + a, len);
      widen_s += std::chrono::duration<double>(Clock::now() - w0).count();
      if (k + 2 < chunks) issue(k + 2);  // into the buffer just widened
    }
    tm.d2h_seconds = std::chrono::duration<double>(Clock::now() - t0).count();
    tm.widen_seconds = widen_s;
  }
};

CudaMiner::CudaMiner(bool blocking_sync) : impl_(std::make_unique<Impl>(blocking_sync)) {}
CudaMiner::~CudaMiner() = default;

CudaBlockTiming CudaMiner::mine(const DeviceGraph& g, Block b, const BlockParams& p, std::uint64_t first,
                                std::uint64_t last, std::uint32_t heavy_threshold, std::uint32_t* host_out) {
  const auto pats = patterns_for(b);
  if (pats.empty()) throw std::runtime_error("no registered pattern emits into block " + std::string(block_name(b)));
  return mine(g, pats, p, Visibility::stream, first, last, heavy_threshold, host_out);
}

CudaBlockTiming CudaMiner::mine(const DeviceGraph& g, std::span<const PatternInfo* const> pats, const BlockParams& p,
                                Visibility vis, std::uint64_t first, std::uint64_t last, std::uint32_t heavy_threshold,
                                std::uint32_t* host_out, bool pack) {
  return run(g, pats, p, vis, first, last, nullptr, nullptr, heavy_threshold, host_out, pack);
}

CudaBlockTiming CudaMiner::mine_subset(const DeviceGraph& g, std::span<const PatternInfo* const> pats,
                                       const BlockParams& p, Visibility vis, std::uint64_t first, std::uint64_t last,
                                       std::span<const std::uint32_t> subset, std::uint32_t heavy_threshold,
                                       std::uint32_t* host_out) {
  for (std::size_t i = 0; i < subset.size(); ++i)
    if (subset[i] < first || subset[i] >= last || (i > 0 && subset[i] <= subset[i - 1]))
      throw std::invalid_argument("CudaMiner::mine_subset: the subset must be ascending eids in [first, last)");
  return run(g, pats, p, vis, first, last, &subset, nullptr, heavy_threshold, host_out, false);
}

CudaBlockTiming CudaMiner::mine_split(const DeviceGraph& g, std::span<const PatternInfo* const> pats,
                                      const BlockParams& p, Visibility vis, std::uint64_t first, std::uint64_t last,
                                      std::uint32_t cpu_threshold, std::uint32_t heavy_threshold,
                                      std::uint32_t* host_out, const CpuListFn& on_cpu_list) {
  const Split split{cpu_threshold, &on_cpu_list};
  return run(g, pats, p, vis, first, last, nullptr, &split, heavy_threshold, host_out, false);
}

CudaBlockTiming CudaMiner::run(const DeviceGraph& g, std::span<const PatternInfo* const> pats, const BlockParams& p,
                               Visibility vis, std::uint64_t first, std::uint64_t last,
                               const std::span<const std::uint32_t>* subset, const Split* split,
                               std::uint32_t heavy_threshold, std::uint32_t* host_out, bool pack) {
  const auto t0 = Clock::now();
  Impl& s = *impl_;
  const GraphView& dg = g.view();
  if (pats.empty()) throw std::runtime_error("CudaMiner::mine: no pattern to run");
  for (const PatternInfo* pi : pats)
    if (!pi->cuda || !pi->cuda_bucket) throw std::runtime_error(std::string("pattern '") + pi->name + "' has no CUDA launcher");
  if (first > last || last > dg.m) throw std::invalid_argument("CudaMiner::mine: bad trigger range");
  const std::uint32_t nbins = block_cols(p);  // 1 for a count block (SPEC v3 R7)
  if (nbins > kMaxBins)
    throw std::invalid_argument("CudaMiner::mine: the CUDA backend supports 1.." + std::to_string(kMaxBins) + " bins");
  CudaBlockTiming tm;
  const std::uint32_t rows = std::uint32_t(last - first);
  // The triggers mined: every row's, the subset's, or (split) the ones the CPU does not take.
  std::uint32_t count = subset ? std::uint32_t(subset->size()) : rows;
  const bool listed = subset || split;  // the mined triggers are a list, not the whole range
  const std::size_t out_bytes = std::size_t(rows) * nbins * sizeof(std::uint32_t);
  if (rows == 0) {
    if (split) (*split->on_cpu_list)({});
    return tm;
  }

  s.out.ensure(out_bytes);
  s.ids.ensure(std::max<std::size_t>(count, 1) * sizeof(std::uint32_t));
  s.flags.ensure(std::max<std::uint32_t>(count, 1));
  if (listed) s.flags_all.ensure(rows);
  s.part.ensure(std::max<std::size_t>(count, 1) * sizeof(std::uint32_t));
  s.bins.ensure(nbins * sizeof(std::uint32_t));
  const std::size_t temp_bytes = s.partition_temp(count);

  const cudaStream_t st = s.stream;
  const Params params{p.window, vis == Visibility::timestamp, p.work_limit};
  if (!p.bins.empty())
    BLZ_CUDA(cudaMemcpyAsync(s.bins.ptr, p.bins.data(), nbins * sizeof(std::uint32_t), cudaMemcpyHostToDevice, st));
  if (subset && count > 0) {
    BLZ_CUDA(cudaEventRecord(s.ev[Impl::kH2d0], st));
    BLZ_CUDA(cudaMemcpyAsync(s.ids.ptr, subset->data(), count * sizeof(std::uint32_t), cudaMemcpyHostToDevice, st));
    BLZ_CUDA(cudaEventRecord(s.ev[Impl::kH2d1], st));
  }
  if (split) {
    // The CPU's triggers first, in eid order (the partition's selected side), to the host; the
    // others, in reverse eid order, become this run's list.
    BLZ_CUDA(cudaEventRecord(s.ev[Impl::kSplit0], st));
    iota_kernel<<<(rows + 255) / 256, 256, 0, st>>>(s.ids.as<std::uint32_t>(), first, rows);
    BLZ_CUDA(cudaGetLastError());
    pats.front()->cuda_bucket(dg, Params{p.window, vis == Visibility::timestamp, p.work_limit}, first, rows,
                              split->cpu_threshold, s.flags.as<std::uint8_t>(), st);
    BLZ_CUDA(cudaGetLastError());
    const int ncpu = s.partition(rows, temp_bytes);
    std::vector<std::uint32_t> cpu_list(static_cast<std::size_t>(ncpu));
    if (ncpu > 0)
      BLZ_CUDA(cudaMemcpyAsync(cpu_list.data(), s.part.ptr, std::size_t(ncpu) * sizeof(std::uint32_t),
                               cudaMemcpyDeviceToHost, st));
    count = rows - std::uint32_t(ncpu);
    if (count > 0)
      BLZ_CUDA(cudaMemcpyAsync(s.ids.ptr, s.part.as<std::uint32_t>() + ncpu, std::size_t(count) * sizeof(std::uint32_t),
                               cudaMemcpyDeviceToDevice, st));
    BLZ_CUDA(cudaEventRecord(s.ev[Impl::kSplit1], st));
    BLZ_CUDA(cudaEventSynchronize(s.ev[Impl::kSplit1]));
    tm.split_seconds = elapsed(s.ev[Impl::kSplit0], s.ev[Impl::kSplit1]);
    tm.cpu_triggers = std::uint64_t(ncpu);
    (*split->on_cpu_list)(std::move(cpu_list));
  }
  BLZ_CUDA(cudaEventRecord(s.ev[Impl::kStart], st));
  BLZ_CUDA(cudaMemsetAsync(s.out.ptr, 0, out_bytes, st));
  BLZ_CUDA(cudaMemsetAsync(s.counter(kSaturated), 0, sizeof(std::uint64_t), st));
  BLZ_CUDA(cudaMemsetAsync(s.counter(kAborted), 0xff, sizeof(std::uint64_t), st));  // no abort
  if (!listed) {
    iota_kernel<<<(count + 255) / 256, 256, 0, st>>>(s.ids.as<std::uint32_t>(), first, count);
    BLZ_CUDA(cudaGetLastError());
  }

  CudaRun run;
  run.out = s.out.as<std::uint32_t>();
  run.first = first;
  run.bins = s.bins.as<std::uint32_t>();
  run.nbins = nbins;
  run.count_block = p.bins.empty();
  run.saturated = reinterpret_cast<unsigned long long*>(s.counter(kSaturated));
  run.aborted = reinterpret_cast<unsigned long long*>(s.counter(kAborted));
  run.stream = st;
  // H = 0 and H = kAllLight need no work estimates (a window size is < 2^31): every trigger goes
  // to one kernel, in eid order.
  const bool one_kernel = heavy_threshold == 0 || heavy_threshold == kAllLight;
  // Kernel time = the zeroing (start -> the first pattern's bucket event) + each pattern's
  // bucket..done interval; the host bookkeeping between patterns is not in it.
  double setup_seconds = 0;
  for (const PatternInfo* pi : pats) {
    if (count == 0) break;  // an empty subset: the rows stay zero
    // Bucketing: work estimates -> flags -> heavy triggers first (in eid order), light after
    // them (in reverse eid order).
    BLZ_CUDA(cudaEventRecord(s.ev[Impl::kBucket], st));
    int heavy = heavy_threshold == 0 ? int(count) : 0;
    const std::uint32_t* list = s.ids.as<std::uint32_t>();
    if (!one_kernel) {
      if (listed) {  // the whole range's flags, then the list's
        pi->cuda_bucket(dg, params, first, rows, heavy_threshold, s.flags_all.as<std::uint8_t>(), st);
        BLZ_CUDA(cudaGetLastError());
        gather_flags_kernel<<<(count + 255) / 256, 256, 0, st>>>(s.ids.as<std::uint32_t>(), count, first,
                                                                 s.flags_all.as<std::uint8_t>(), s.flags.as<std::uint8_t>());
      } else {
        pi->cuda_bucket(dg, params, first, count, heavy_threshold, s.flags.as<std::uint8_t>(), st);
      }
      BLZ_CUDA(cudaGetLastError());
      heavy = s.partition(count, temp_bytes);
      list = s.part.as<std::uint32_t>();
    }
    BLZ_CUDA(cudaEventRecord(s.ev[Impl::kLight], st));
    run.heavy = false;
    run.triggers = list + heavy;
    run.count = count - std::uint32_t(heavy);
    pi->cuda(dg, params, run);
    BLZ_CUDA(cudaGetLastError());
    BLZ_CUDA(cudaEventRecord(s.ev[Impl::kHeavy], st));
    run.heavy = true;
    run.triggers = list;
    run.count = std::uint32_t(heavy);
    pi->cuda(dg, params, run);
    BLZ_CUDA(cudaGetLastError());
    BLZ_CUDA(cudaEventRecord(s.ev[Impl::kDone], st));
    BLZ_CUDA(cudaEventSynchronize(s.ev[Impl::kDone]));
    if (pi == pats.front()) setup_seconds = elapsed(s.ev[Impl::kStart], s.ev[Impl::kBucket]);
    tm.bucket_seconds += elapsed(s.ev[Impl::kBucket], s.ev[Impl::kLight]);
    tm.light_seconds += elapsed(s.ev[Impl::kLight], s.ev[Impl::kHeavy]);
    tm.heavy_seconds += elapsed(s.ev[Impl::kHeavy], s.ev[Impl::kDone]);
    tm.heavy += std::uint64_t(heavy);
    tm.light += count - std::uint64_t(heavy);
  }
  if (count == 0) {  // no pattern ran: the zeroing alone
    BLZ_CUDA(cudaEventRecord(s.ev[Impl::kDone], st));
    BLZ_CUDA(cudaEventSynchronize(s.ev[Impl::kDone]));
    setup_seconds = elapsed(s.ev[Impl::kStart], s.ev[Impl::kDone]);
  }
  tm.kernel_seconds = setup_seconds + tm.bucket_seconds + tm.light_seconds + tm.heavy_seconds;
  // An abort (SPEC v2 §8 S3) is reported before any row is copied.
  BLZ_CUDA(cudaMemcpyAsync(s.host_counters, s.counters.ptr, kCounters * sizeof(std::uint64_t), cudaMemcpyDeviceToHost, st));
  BLZ_CUDA(cudaStreamSynchronize(st));
  tm.saturated = s.host_counters[kSaturated];
  if (s.host_counters[kAborted] != UINT64_MAX)
    throw WorkLimitExceeded(std::string("cuda, block ") + pats.front()->block, s.host_counters[kAborted], p.work_limit);
  s.fetch(s.out.as<std::uint32_t>(), std::uint64_t(rows) * nbins, host_out, pack, tm);
  if (subset && count > 0) tm.h2d_seconds = elapsed(s.ev[Impl::kH2d0], s.ev[Impl::kH2d1]);
  tm.wall_seconds = std::chrono::duration<double>(Clock::now() - t0).count();
  return tm;
}

CudaMiner::FusedTiming CudaMiner::mine_fused(const DeviceGraph& g, const FusionInfo& f,
                                             std::span<const BlockParams* const> params, Visibility vis,
                                             std::uint64_t first, std::uint64_t last, std::uint32_t heavy_threshold,
                                             std::span<std::uint32_t* const> host_out, bool pack) {
  const auto t0 = Clock::now();
  Impl& s = *impl_;
  const GraphView& dg = g.view();
  if (!f.cuda || !f.cuda_bucket) throw std::runtime_error(std::string("fusion '") + f.name + "' has no CUDA launcher");
  std::uint32_t nslots = 0;
  for (std::uint32_t sl : f.slots) nslots = std::max(nslots, sl + 1);
  if (f.members.size() != f.slots.size() || f.members.size() > kMaxFusedMembers || nslots > kMaxFusedSlots ||
      params.size() != nslots || host_out.size() != nslots)
    throw std::invalid_argument(std::string("CudaMiner::mine_fused: fusion '") + f.name + "' needs " +
                                std::to_string(nslots) + " blocks");
  if (first > last || last > dg.m) throw std::invalid_argument("CudaMiner::mine_fused: bad trigger range");
  FusedTiming tm;
  tm.slots.resize(nslots);
  const std::uint32_t rows = std::uint32_t(last - first);
  if (rows == 0) return tm;

  // Slot sl's rows at s.out + off[sl], its bins at s.bins + bin_off[sl].
  std::vector<std::uint64_t> off(nslots), cells(nslots);
  std::vector<std::uint32_t> all_bins, bin_off(nslots);
  std::uint64_t total = 0;
  for (std::uint32_t sl = 0; sl < nslots; ++sl) {
    const std::vector<std::uint32_t>& b = params[sl]->bins;
    if (b.size() > kMaxBins)  // as mine(), so a block runs fused exactly when it runs alone
      throw std::invalid_argument("CudaMiner::mine_fused: the CUDA backend supports 1.." + std::to_string(kMaxBins) +
                                  " bins");
    bin_off[sl] = std::uint32_t(all_bins.size());
    all_bins.insert(all_bins.end(), b.begin(), b.end());
    off[sl] = total;
    cells[sl] = std::uint64_t(rows) * block_cols(*params[sl]);
    total += round_cells(cells[sl]);
  }
  s.out.ensure(total * sizeof(std::uint32_t));
  s.ids.ensure(std::size_t(rows) * sizeof(std::uint32_t));
  s.flags.ensure(rows);
  s.part.ensure(std::size_t(rows) * sizeof(std::uint32_t));
  s.bins.ensure(std::max<std::size_t>(all_bins.size(), 1) * sizeof(std::uint32_t));
  const std::size_t temp_bytes = s.partition_temp(rows);

  const cudaStream_t st = s.stream;
  FusedParams fp;
  for (std::size_t k = 0; k < f.members.size(); ++k) {
    const BlockParams& bp = *params[f.slots[k]];
    fp.member[k] = Params{bp.window, vis == Visibility::timestamp, bp.work_limit};
  }
  if (!all_bins.empty())
    BLZ_CUDA(cudaMemcpyAsync(s.bins.ptr, all_bins.data(), all_bins.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice,
                             st));
  BLZ_CUDA(cudaEventRecord(s.ev[Impl::kStart], st));
  BLZ_CUDA(cudaMemsetAsync(s.out.ptr, 0, total * sizeof(std::uint32_t), st));
  BLZ_CUDA(cudaMemsetAsync(s.counter(kSaturated), 0, kMaxFusedSlots * sizeof(std::uint64_t), st));
  iota_kernel<<<(rows + 255) / 256, 256, 0, st>>>(s.ids.as<std::uint32_t>(), first, rows);
  BLZ_CUDA(cudaGetLastError());

  FusedRun run;
  for (std::uint32_t sl = 0; sl < nslots; ++sl) {
    run.out[sl] = s.out.as<std::uint32_t>() + off[sl];
    run.bins[sl] = s.bins.as<std::uint32_t>() + bin_off[sl];
    run.nbins[sl] = block_cols(*params[sl]);
    run.count_block[sl] = params[sl]->bins.empty();
  }
  run.first = first;
  run.saturated = reinterpret_cast<unsigned long long*>(s.counter(kSaturated));
  run.stream = st;
  // As in run(): bucketing (one estimate per trigger, the largest of its members'), then the light
  // and the heavy kernel.
  BLZ_CUDA(cudaEventRecord(s.ev[Impl::kBucket], st));
  int heavy = heavy_threshold == 0 ? int(rows) : 0;
  const std::uint32_t* list = s.ids.as<std::uint32_t>();
  if (heavy_threshold != 0 && heavy_threshold != kAllLight) {
    f.cuda_bucket(dg, fp, first, rows, heavy_threshold, s.flags.as<std::uint8_t>(), st);
    BLZ_CUDA(cudaGetLastError());
    heavy = s.partition(rows, temp_bytes);
    list = s.part.as<std::uint32_t>();
  }
  BLZ_CUDA(cudaEventRecord(s.ev[Impl::kLight], st));
  run.heavy = false;
  run.triggers = list + heavy;
  run.count = rows - std::uint32_t(heavy);
  f.cuda(dg, fp, run);
  BLZ_CUDA(cudaGetLastError());
  BLZ_CUDA(cudaEventRecord(s.ev[Impl::kHeavy], st));
  run.heavy = true;
  run.triggers = list;
  run.count = std::uint32_t(heavy);
  f.cuda(dg, fp, run);
  BLZ_CUDA(cudaGetLastError());
  BLZ_CUDA(cudaEventRecord(s.ev[Impl::kDone], st));
  BLZ_CUDA(cudaMemcpyAsync(s.host_counters, s.counters.ptr, kCounters * sizeof(std::uint64_t), cudaMemcpyDeviceToHost, st));
  BLZ_CUDA(cudaStreamSynchronize(st));
  CudaBlockTiming& gt = tm.group;
  gt.bucket_seconds = elapsed(s.ev[Impl::kBucket], s.ev[Impl::kLight]);
  gt.light_seconds = elapsed(s.ev[Impl::kLight], s.ev[Impl::kHeavy]);
  gt.heavy_seconds = elapsed(s.ev[Impl::kHeavy], s.ev[Impl::kDone]);
  gt.kernel_seconds = elapsed(s.ev[Impl::kStart], s.ev[Impl::kBucket]) + gt.bucket_seconds + gt.light_seconds +
                      gt.heavy_seconds;
  gt.heavy = std::uint64_t(heavy);
  gt.light = rows - std::uint64_t(heavy);
  std::vector<std::uint64_t> saturated(s.host_counters + kSaturated, s.host_counters + kSaturated + nslots);
  for (std::uint32_t sl = 0; sl < nslots; ++sl) {
    CudaBlockTiming& t = tm.slots[sl];
    s.fetch(s.out.as<std::uint32_t>() + off[sl], cells[sl], host_out[sl], pack, t);
    t.saturated = saturated[sl];
    gt.d2h_seconds += t.d2h_seconds;
    gt.pack_seconds += t.pack_seconds;
    gt.widen_seconds += t.widen_seconds;
    gt.d2h_bytes += t.d2h_bytes;
    gt.saturated += t.saturated;
  }
  gt.wall_seconds = std::chrono::duration<double>(Clock::now() - t0).count();
  return tm;
}

CudaBlockTiming CudaMiner::fetch_for_test(std::span<const std::uint32_t> values, std::uint32_t* host_out, bool pack) {
  Impl& s = *impl_;
  s.out.ensure(std::max<std::size_t>(values.size(), 1) * sizeof(std::uint32_t));
  if (!values.empty())
    BLZ_CUDA(cudaMemcpyAsync(s.out.ptr, values.data(), values.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice,
                             s.stream));
  CudaBlockTiming tm;
  s.fetch(s.out.as<std::uint32_t>(), values.size(), host_out, pack, tm);
  return tm;
}

}  // namespace blz

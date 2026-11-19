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
    for (auto [dv, hv] : {std::pair{&d.out, &h.out}, std::pair{&d.in, &h.in}}) {
      put(dv->off, hv->off, n + 1);
      put(dv->nbr, hv->nbr, m);
      put(dv->ts, hv->ts, m);
      put(dv->eid, hv->eid, m);
      put(dv->prev_same, hv->prev_same, m);
    }
    for (auto [dv, hv, pairs] :
         {std::tuple{&d.pout, &h.pout, pairs_out}, std::tuple{&d.pin, &h.pin, pairs_in}}) {
      put(dv->off, hv->off, n + 1);
      put(dv->nbr, hv->nbr, pairs);
      put(dv->run, hv->run, pairs + 1);
      put(dv->run_ts, hv->run_ts, m);
      put(dv->run_eid, hv->run_eid, m);
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
  enum { kStart, kBucket, kPart, kLight, kHeavy, kDone, kD2h0, kD2h1, kH2d0, kH2d1, kSplit0, kSplit1, kEvents };
  cudaEvent_t ev[kEvents] = {};
  // counters: saturated (u64), heavy count (int), aborted eid (u64, SPEC v2 §8 S3)
  DeviceBuffer out, ids, flags, flags_all, part, temp, bins, counters;
  std::uint64_t* host_counters = nullptr;  // pinned: the same three

  explicit Impl(bool blocking_sync) {
    BLZ_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (cudaEvent_t& e : ev)
      BLZ_CUDA(cudaEventCreateWithFlags(&e, blocking_sync ? cudaEventBlockingSync : cudaEventDefault));
    BLZ_CUDA(cudaMallocHost(reinterpret_cast<void**>(&host_counters), 3 * sizeof(std::uint64_t)));
    counters.ensure(3 * sizeof(std::uint64_t));
  }
  ~Impl() {
    for (cudaEvent_t e : ev)
      if (e) cudaEventDestroy(e);
    if (stream) cudaStreamDestroy(stream);
    if (host_counters) cudaFreeHost(host_counters);
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
                                std::uint32_t* host_out) {
  return run(g, pats, p, vis, first, last, nullptr, nullptr, heavy_threshold, host_out);
}

CudaBlockTiming CudaMiner::mine_subset(const DeviceGraph& g, std::span<const PatternInfo* const> pats,
                                       const BlockParams& p, Visibility vis, std::uint64_t first, std::uint64_t last,
                                       std::span<const std::uint32_t> subset, std::uint32_t heavy_threshold,
                                       std::uint32_t* host_out) {
  for (std::size_t i = 0; i < subset.size(); ++i)
    if (subset[i] < first || subset[i] >= last || (i > 0 && subset[i] <= subset[i - 1]))
      throw std::invalid_argument("CudaMiner::mine_subset: the subset must be ascending eids in [first, last)");
  return run(g, pats, p, vis, first, last, &subset, nullptr, heavy_threshold, host_out);
}

CudaBlockTiming CudaMiner::mine_split(const DeviceGraph& g, std::span<const PatternInfo* const> pats,
                                      const BlockParams& p, Visibility vis, std::uint64_t first, std::uint64_t last,
                                      std::uint32_t cpu_threshold, std::uint32_t heavy_threshold,
                                      std::uint32_t* host_out, const CpuListFn& on_cpu_list) {
  const Split split{cpu_threshold, &on_cpu_list};
  return run(g, pats, p, vis, first, last, nullptr, &split, heavy_threshold, host_out);
}

CudaBlockTiming CudaMiner::run(const DeviceGraph& g, std::span<const PatternInfo* const> pats, const BlockParams& p,
                               Visibility vis, std::uint64_t first, std::uint64_t last,
                               const std::span<const std::uint32_t>* subset, const Split* split,
                               std::uint32_t heavy_threshold, std::uint32_t* host_out) {
  const auto t0 = Clock::now();
  Impl& s = *impl_;
  const GraphView& dg = g.view();
  if (pats.empty()) throw std::runtime_error("CudaMiner::mine: no pattern to run");
  for (const PatternInfo* pi : pats)
    if (!pi->cuda || !pi->cuda_bucket) throw std::runtime_error(std::string("pattern '") + pi->name + "' has no CUDA launcher");
  if (first > last || last > dg.m) throw std::invalid_argument("CudaMiner::mine: bad trigger range");
  const std::uint32_t nbins = std::uint32_t(p.bins.size());
  if (nbins == 0 || nbins > kMaxBins)
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
  std::size_t temp_bytes = 0;
  int* d_heavy = reinterpret_cast<int*>(s.counters.as<std::uint64_t>() + 1);
  BLZ_CUDA(cub::DevicePartition::Flagged(nullptr, temp_bytes, s.ids.as<std::uint32_t>(), s.flags.as<std::uint8_t>(),
                                         s.part.as<std::uint32_t>(), d_heavy, int(count), s.stream));
  s.temp.ensure(std::max<std::size_t>(temp_bytes, 1));

  const cudaStream_t st = s.stream;
  const Params params{p.window, vis == Visibility::timestamp, p.work_limit};
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
    BLZ_CUDA(cub::DevicePartition::Flagged(s.temp.ptr, temp_bytes, s.ids.as<std::uint32_t>(), s.flags.as<std::uint8_t>(),
                                           s.part.as<std::uint32_t>(), d_heavy, int(rows), st));
    BLZ_CUDA(cudaMemcpyAsync(s.host_counters + 1, d_heavy, sizeof(int), cudaMemcpyDeviceToHost, st));
    BLZ_CUDA(cudaEventRecord(s.ev[Impl::kPart], st));
    BLZ_CUDA(cudaEventSynchronize(s.ev[Impl::kPart]));
    int ncpu = 0;
    std::memcpy(&ncpu, s.host_counters + 1, sizeof(int));
    if (ncpu < 0 || std::uint32_t(ncpu) > rows) throw std::runtime_error("CUDA: bad partition count");
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
  BLZ_CUDA(cudaMemsetAsync(s.counters.ptr, 0, sizeof(std::uint64_t), st));
  BLZ_CUDA(cudaMemsetAsync(s.counters.as<std::uint64_t>() + 2, 0xff, sizeof(std::uint64_t), st));  // no abort
  if (!listed) {
    iota_kernel<<<(count + 255) / 256, 256, 0, st>>>(s.ids.as<std::uint32_t>(), first, count);
    BLZ_CUDA(cudaGetLastError());
  }

  CudaRun run;
  run.out = s.out.as<std::uint32_t>();
  run.first = first;
  run.bins = s.bins.as<std::uint32_t>();
  run.nbins = nbins;
  run.saturated = reinterpret_cast<unsigned long long*>(s.counters.as<std::uint64_t>());
  run.aborted = reinterpret_cast<unsigned long long*>(s.counters.as<std::uint64_t>() + 2);
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
      BLZ_CUDA(cub::DevicePartition::Flagged(s.temp.ptr, temp_bytes, s.ids.as<std::uint32_t>(),
                                             s.flags.as<std::uint8_t>(), s.part.as<std::uint32_t>(), d_heavy,
                                             int(count), st));
      BLZ_CUDA(cudaMemcpyAsync(s.host_counters + 1, d_heavy, sizeof(int), cudaMemcpyDeviceToHost, st));
      BLZ_CUDA(cudaEventRecord(s.ev[Impl::kPart], st));
      BLZ_CUDA(cudaEventSynchronize(s.ev[Impl::kPart]));
      std::memcpy(&heavy, s.host_counters + 1, sizeof(int));
      if (heavy < 0 || std::uint32_t(heavy) > count) throw std::runtime_error("CUDA: bad partition count");
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
  BLZ_CUDA(cudaEventRecord(s.ev[Impl::kD2h0], st));
  BLZ_CUDA(cudaMemcpyAsync(host_out, s.out.ptr, out_bytes, cudaMemcpyDeviceToHost, st));
  BLZ_CUDA(cudaMemcpyAsync(s.host_counters, s.counters.ptr, 3 * sizeof(std::uint64_t), cudaMemcpyDeviceToHost, st));
  BLZ_CUDA(cudaEventRecord(s.ev[Impl::kD2h1], st));
  BLZ_CUDA(cudaEventSynchronize(s.ev[Impl::kD2h1]));
  if (count == 0) setup_seconds = elapsed(s.ev[Impl::kStart], s.ev[Impl::kD2h0]);
  tm.kernel_seconds = setup_seconds + tm.bucket_seconds + tm.light_seconds + tm.heavy_seconds;
  tm.d2h_seconds = elapsed(s.ev[Impl::kD2h0], s.ev[Impl::kD2h1]);
  if (subset && count > 0) tm.h2d_seconds = elapsed(s.ev[Impl::kH2d0], s.ev[Impl::kH2d1]);
  tm.saturated = s.host_counters[0];
  if (s.host_counters[2] != UINT64_MAX)
    throw WorkLimitExceeded(std::string("cuda, block ") + pats.front()->block, s.host_counters[2], p.work_limit);
  tm.wall_seconds = std::chrono::duration<double>(Clock::now() - t0).count();
  return tm;
}

}  // namespace blz

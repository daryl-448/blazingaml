// The CUDA backend's host side (SPEC §7.5). See cuda_backend.hpp.
#include "runtime/cuda_backend.hpp"

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

// The largest bin count the heavy kernel's shared histograms allow (48 KB per block).
constexpr std::uint32_t kMaxBins = 48 * 1024 / (cuda::kHeavyWarps * sizeof(unsigned long long));

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
  enum { kStart, kBucket, kLight, kHeavy, kDone, kD2h0, kD2h1, kEvents };
  cudaEvent_t ev[kEvents] = {};
  DeviceBuffer out, ids, flags, part, temp, bins, counters;  // counters: saturated (u64), heavy count (int)
  std::uint64_t* host_counters = nullptr;                   // pinned: saturated, heavy count

  Impl() {
    BLZ_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (cudaEvent_t& e : ev) BLZ_CUDA(cudaEventCreate(&e));
    BLZ_CUDA(cudaMallocHost(reinterpret_cast<void**>(&host_counters), 2 * sizeof(std::uint64_t)));
    counters.ensure(2 * sizeof(std::uint64_t));
  }
  ~Impl() {
    for (cudaEvent_t e : ev)
      if (e) cudaEventDestroy(e);
    if (stream) cudaStreamDestroy(stream);
    if (host_counters) cudaFreeHost(host_counters);
  }
};

CudaMiner::CudaMiner() : impl_(std::make_unique<Impl>()) {}
CudaMiner::~CudaMiner() = default;

CudaBlockTiming CudaMiner::mine(const DeviceGraph& g, Block b, const BlockParams& p, std::uint64_t first,
                                std::uint64_t last, std::uint32_t heavy_threshold, std::uint32_t* host_out) {
  const auto t0 = Clock::now();
  Impl& s = *impl_;
  const GraphView& dg = g.view();
  const auto pats = patterns_for(b);
  if (pats.empty()) throw std::runtime_error("no registered pattern emits into block " + std::string(block_name(b)));
  for (const PatternInfo* pi : pats)
    if (!pi->cuda || !pi->cuda_bucket) throw std::runtime_error(std::string("pattern '") + pi->name + "' has no CUDA launcher");
  if (first > last || last > dg.m) throw std::invalid_argument("CudaMiner::mine: bad trigger range");
  const std::uint32_t nbins = std::uint32_t(p.bins.size());
  if (nbins == 0 || nbins > kMaxBins)
    throw std::invalid_argument("CudaMiner::mine: the CUDA backend supports 1.." + std::to_string(kMaxBins) + " bins");
  CudaBlockTiming tm;
  const std::uint32_t count = std::uint32_t(last - first);
  const std::size_t out_bytes = std::size_t(count) * nbins * sizeof(std::uint32_t);
  if (count == 0) return tm;

  s.out.ensure(out_bytes);
  s.ids.ensure(count * sizeof(std::uint32_t));
  s.flags.ensure(count);
  s.part.ensure(count * sizeof(std::uint32_t));
  s.bins.ensure(nbins * sizeof(std::uint32_t));
  std::size_t temp_bytes = 0;
  int* d_heavy = reinterpret_cast<int*>(s.counters.as<std::uint64_t>() + 1);
  BLZ_CUDA(cub::DevicePartition::Flagged(nullptr, temp_bytes, s.ids.as<std::uint32_t>(), s.flags.as<std::uint8_t>(),
                                         s.part.as<std::uint32_t>(), d_heavy, int(count), s.stream));
  s.temp.ensure(temp_bytes);

  const cudaStream_t st = s.stream;
  const Params params{p.window};
  BLZ_CUDA(cudaMemcpyAsync(s.bins.ptr, p.bins.data(), nbins * sizeof(std::uint32_t), cudaMemcpyHostToDevice, st));
  BLZ_CUDA(cudaEventRecord(s.ev[Impl::kStart], st));
  BLZ_CUDA(cudaMemsetAsync(s.out.ptr, 0, out_bytes, st));
  BLZ_CUDA(cudaMemsetAsync(s.counters.ptr, 0, sizeof(std::uint64_t), st));
  iota_kernel<<<(count + 255) / 256, 256, 0, st>>>(s.ids.as<std::uint32_t>(), first, count);
  BLZ_CUDA(cudaGetLastError());

  CudaRun run;
  run.out = s.out.as<std::uint32_t>();
  run.first = first;
  run.bins = s.bins.as<std::uint32_t>();
  run.nbins = nbins;
  run.saturated = reinterpret_cast<unsigned long long*>(s.counters.as<std::uint64_t>());
  run.stream = st;
  // H = 0 and H = kAllLight need no work estimates (a window size is < 2^31): every trigger goes
  // to one kernel, in eid order.
  const bool one_kernel = heavy_threshold == 0 || heavy_threshold == kAllLight;
  // Kernel time = the zeroing (start -> the first pattern's bucket event) + each pattern's
  // bucket..done interval; the host bookkeeping between patterns is not in it.
  double setup_seconds = 0;
  for (const PatternInfo* pi : pats) {
    // Bucketing: work estimates -> flags -> heavy triggers first (in eid order), light after
    // them (in reverse eid order).
    BLZ_CUDA(cudaEventRecord(s.ev[Impl::kBucket], st));
    int heavy = heavy_threshold == 0 ? int(count) : 0;
    const std::uint32_t* list = s.ids.as<std::uint32_t>();
    if (!one_kernel) {
      pi->cuda_bucket(dg, params, first, count, heavy_threshold, s.flags.as<std::uint8_t>(), st);
      BLZ_CUDA(cudaGetLastError());
      BLZ_CUDA(cub::DevicePartition::Flagged(s.temp.ptr, temp_bytes, s.ids.as<std::uint32_t>(),
                                             s.flags.as<std::uint8_t>(), s.part.as<std::uint32_t>(), d_heavy,
                                             int(count), st));
      BLZ_CUDA(cudaMemcpyAsync(s.host_counters + 1, d_heavy, sizeof(int), cudaMemcpyDeviceToHost, st));
      BLZ_CUDA(cudaStreamSynchronize(st));
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
  BLZ_CUDA(cudaMemcpyAsync(s.host_counters, s.counters.ptr, sizeof(std::uint64_t), cudaMemcpyDeviceToHost, st));
  BLZ_CUDA(cudaEventRecord(s.ev[Impl::kD2h1], st));
  BLZ_CUDA(cudaEventSynchronize(s.ev[Impl::kD2h1]));
  tm.kernel_seconds = setup_seconds + tm.bucket_seconds + tm.light_seconds + tm.heavy_seconds;
  tm.d2h_seconds = elapsed(s.ev[Impl::kD2h0], s.ev[Impl::kD2h1]);
  tm.saturated = s.host_counters[0];
  tm.wall_seconds = std::chrono::duration<double>(Clock::now() - t0).count();
  return tm;
}

}  // namespace blz

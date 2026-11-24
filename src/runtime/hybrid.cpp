#include "runtime/hybrid.hpp"

#include <chrono>
#include <cstring>
#include <exception>
#include <future>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "runtime/triggers.hpp"

namespace blz {

namespace {

using Clock = std::chrono::steady_clock;
double since(Clock::time_point a) { return std::chrono::duration<double>(Clock::now() - a).count(); }

}  // namespace

HybridTiming mine_hybrid(const GraphView& host, const DeviceGraph& dev, CudaMiner& miner,
                         std::span<const PatternInfo* const> pats, const BlockParams& p, Visibility vis,
                         std::uint64_t first, std::uint64_t last, std::uint32_t cpu_threshold,
                         std::uint32_t gpu_threshold, Schedule schedule, std::uint32_t* host_out) {
  if (pats.empty()) throw std::runtime_error("mine_hybrid: no pattern to run");
  if (first > last || last > host.m) throw std::invalid_argument("mine_hybrid: bad trigger range");
  const auto t0 = Clock::now();
  HybridTiming tm;
  const std::size_t nb = block_cols(p);
  const std::uint64_t rows = last - first;

  // The CPU's triggers. The GPU finds them (CudaMiner::mine_split: the first pattern's work
  // estimate, the same code as the CPU's sorted schedule runs) and hands them over before it mines
  // the rest. Every pattern of the block runs on the side its trigger went to, so the rows stay
  // disjoint. Threshold 0 keeps every trigger on the CPU, kAllGpu none.
  TriggerSel heavy;
  heavy.spec = "hybrid:cpu";
  heavy.first = first;
  heavy.last = last;
  heavy.sampled = true;
  std::promise<std::vector<std::uint32_t>> cpu_list;
  std::future<std::vector<std::uint32_t>> cpu_list_ready = cpu_list.get_future();
  bool list_set = false;  // the GPU thread's own flag: whether it fulfilled the promise
  std::exception_ptr gpu_error;
  std::thread gpu_side;
  const bool gpu_runs = cpu_threshold != 0 && rows > 0;
  if (!gpu_runs) {
    heavy.eids.resize(rows);
    std::iota(heavy.eids.begin(), heavy.eids.end(), std::uint32_t(first));
  } else {
    gpu_side = std::thread([&] {
      try {
        const auto g0 = Clock::now();
        if (cpu_threshold == kAllGpu) {
          list_set = true;
          cpu_list.set_value({});
          tm.gpu = miner.mine(dev, pats, p, vis, first, last, gpu_threshold, host_out);
        } else {
          tm.gpu = miner.mine_split(dev, pats, p, vis, first, last, cpu_threshold, gpu_threshold, host_out,
                                    [&](std::vector<std::uint32_t> eids) {
                                      list_set = true;
                                      cpu_list.set_value(std::move(eids));
                                    });
        }
        tm.gpu_seconds = since(g0);
      } catch (...) {
        gpu_error = std::current_exception();
        if (!list_set) cpu_list.set_exception(gpu_error);  // the CPU side is still waiting for its list
      }
    });
    try {
      heavy.eids = cpu_list_ready.get();
    } catch (...) {
      gpu_side.join();
      throw;
    }
  }
  tm.split_seconds = since(t0);
  tm.cpu_triggers = heavy.eids.size();
  tm.gpu_triggers = rows - heavy.eids.size();

  // The CPU side on this thread's OpenMP team, while the GPU mines the rest.
  const auto t1 = Clock::now();
  std::vector<std::uint32_t> cpu_rows;
  std::exception_ptr cpu_error;
  if (!heavy.eids.empty()) {
    try {  // the GPU thread is joined below whatever happens here, the allocation included
      cpu_rows.assign(heavy.eids.size() * nb, 0);
      const auto c0 = Clock::now();
      tm.cpu = mine_cpu(host, pats, p, vis, heavy, schedule, cpu_rows.data());
      tm.cpu_seconds = since(c0);
    } catch (...) {
      cpu_error = std::current_exception();
    }
  }
  if (gpu_runs) gpu_side.join();
  tm.overlap_seconds = since(t1);

  // A work-limit abort (SPEC v2 §8 S3) on both sides: the smaller eid, as one backend would report.
  if (cpu_error || gpu_error) {
    std::exception_ptr report = cpu_error ? cpu_error : gpu_error;
    std::uint64_t best = UINT64_MAX;
    for (const std::exception_ptr& e : {cpu_error, gpu_error}) {
      if (!e) continue;
      try {
        std::rethrow_exception(e);
      } catch (const WorkLimitExceeded& x) {
        if (x.eid < best) {
          best = x.eid;
          report = e;
        }
      } catch (...) {
        std::rethrow_exception(e);  // anything else is a real error
      }
    }
    std::rethrow_exception(report);
  }

  // Merge: the CPU's rows over the GPU's (zero) rows of the same triggers.
  const auto t2 = Clock::now();
  if (!gpu_runs) std::memset(host_out, 0, rows * nb * sizeof(std::uint32_t));
  for (std::size_t r = 0; r < heavy.eids.size(); ++r)
    std::memcpy(host_out + (heavy.eids[r] - first) * nb, cpu_rows.data() + r * nb, nb * sizeof(std::uint32_t));
  tm.merge_seconds = since(t2);
  tm.saturated = tm.cpu.saturated + tm.gpu.saturated;
  tm.total_seconds = since(t0);
  return tm;
}

}  // namespace blz

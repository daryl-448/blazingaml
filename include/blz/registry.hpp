#pragma once
// What generated pattern code and the runtime share (SPEC §7.4, §7.6): the CPU output sink, the
// driver and work-estimate signatures, and the registry blzc writes (registry.cpp).
#include <cstdint>
#include <span>

#include "blz/primitives.hpp"

namespace blz {

// One trigger's output row. emit(size) bins one instance (SPEC §4.2): sizes below 2 or below
// b_0 are dropped, the last bin is open-ended, counts saturate at UINT32_MAX.
struct RowSink {
  std::uint32_t* row = nullptr;
  const std::uint32_t* bins = nullptr;
  std::uint32_t nbins = 0;
  std::uint64_t saturated = 0;

  BLZ_HD void emit(std::int64_t size) {
    if (size < 2 || size < std::int64_t(bins[0])) return;
    std::uint32_t i = nbins - 1;
    while (size < std::int64_t(bins[i])) --i;
    if (row[i] == UINT32_MAX) ++saturated;
    else ++row[i];
  }
};

// A block's output rows: row r belongs to trigger eid first + r.
struct BlockOut {
  std::uint32_t* data = nullptr;
  std::uint64_t first = 0;
  const std::uint32_t* bins = nullptr;
  std::uint32_t nbins = 0;
  std::uint64_t saturated = 0;  // summed by the drivers

  RowSink sink(std::uint32_t e) const { return {data + (e - first) * nbins, bins, nbins, 0}; }
};

// <pattern>_cpu: runs <pattern>_trigger for every eid in `triggers`, in that order, with
// `#pragma omp parallel for schedule(dynamic, chunk)`, adding into `out`.
using CpuFn = void (*)(const GraphView&, const Params&, std::span<const std::uint32_t> triggers,
                       BlockOut& out, int chunk);
// <pattern>_work: the trigger's work estimate, the window size of the pattern's first scan
// (SPEC §7.4 `sorted`, §7.5 bucketing).
using WorkFn = std::uint32_t (*)(const GraphView&, const Params&, std::uint32_t e);

struct ConstInfo {
  const char* name;
  std::int64_t value;
};

struct PatternInfo {
  const char* name;
  const char* block;
  const char* block_loc;   // "<spec>:line:col" of its `block:` value, for E402
  const char* file;        // the spec it was generated from
  const char* span_param;  // the window span parameter; its value is the block's window
  std::span<const ConstInfo> consts;  // compile-time consts, after any --const override
  CpuFn cpu;
  WorkFn work;
};

// Every pattern blzc compiled into this build, in the order of its spec arguments.
std::span<const PatternInfo> registered_patterns();

}  // namespace blz

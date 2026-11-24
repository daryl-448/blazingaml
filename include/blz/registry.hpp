#pragma once
// What generated pattern code and the runtime share (SPEC §7.4–§7.6): the output sink, the CPU
// driver, work-estimate and CUDA launcher signatures, and the registry blzc writes (registry.cpp).
#include <cstdint>
#include <span>
#include <string_view>

#include "blz/primitives.hpp"

namespace blz {

// One trigger's output row. emit(size, min_size) records one instance (SPEC §4.2, §6.6). A
// histogram row bins it: sizes below min_size (>= 2) or below b_0 are dropped, the last bin is
// open-ended, counts saturate at UINT32_MAX. A count row (`count`, SPEC v3 R7) has one column, to
// which the emit adds the size, min_size not applying; a sum past UINT32_MAX clamps there. Each
// emit that saturation loses or clamps adds one to `saturated`. A count block's patterns emit at
// most once per trigger (PatternInfo::one_instance, E403), so its row is the instance's size.
//
// over(limit) counts one binding of a pattern compiled with `work_limit: true` (SPEC v2 §8 S3).
// Past `limit` bindings it marks the trigger aborted and returns true; the generated code then
// stops the trigger, and the driver reports it: a row is never truncated silently.
struct RowSink {
  std::uint32_t* row = nullptr;
  const std::uint32_t* bins = nullptr;
  std::uint32_t nbins = 0;  // the row's columns: 1 for a count row
  bool count = false;
  std::uint64_t saturated = 0;
  std::uint64_t work = 0;
  bool aborted = false;

  BLZ_HD void emit(std::int64_t size, std::int64_t min_size) {
    if (count) {
      const std::uint64_t sum = std::uint64_t(row[0]) + std::uint64_t(size);
      if (sum > UINT32_MAX) {
        row[0] = UINT32_MAX;
        ++saturated;
      } else {
        row[0] = std::uint32_t(sum);
      }
      return;
    }
    if (size < min_size || size < std::int64_t(bins[0])) return;
    std::uint32_t i = nbins - 1;
    while (size < std::int64_t(bins[i])) --i;
    if (row[i] == UINT32_MAX) ++saturated;
    else ++row[i];
  }
  BLZ_HD bool over(std::uint64_t limit) {
    if (++work <= limit) return false;
    aborted = true;
    return true;
  }
};

// A block's output rows: row r belongs to trigger eid first + r, for the eids [first, last), or,
// when `rows` is set (a trigger sample, SPEC v2 §11), to rows[r], the r-th eid of an ascending list
// of `nrows` eids.
struct BlockOut {
  std::uint32_t* data = nullptr;
  std::uint64_t first = 0;
  const std::uint32_t* bins = nullptr;
  std::uint32_t nbins = 0;  // the row's columns: 1 for a count block
  bool count = false;       // a count block (SPEC v3 R7): see RowSink
  std::uint64_t saturated = 0;  // summed by the drivers
  const std::uint32_t* rows = nullptr;
  std::uint64_t nrows = 0;
  std::uint64_t last = 0;  // without `rows`: the end of the eid range
  // The smallest eid whose trigger exceeded the work limit (SPEC v2 §8 S3), or UINT64_MAX. Set
  // by the drivers of `work_limit` patterns, which then skip the triggers not yet started.
  std::uint64_t aborted = UINT64_MAX;

  RowSink sink(std::uint32_t e) const { return {data + row_of(e) * nbins, bins, nbins, count}; }
  // Whether trigger e has a row (the sweep driver walks every list entry, SPEC v2 §7.1).
  bool selects(std::uint32_t e) const {
    if (!rows) return e >= first && e < last;
    const std::uint64_t r = row_of(e);
    return r < nrows && rows[r] == e;
  }
  std::uint64_t row_of(std::uint32_t e) const {
    if (!rows) return e - first;
    std::uint64_t a = 0, b = nrows;  // the first position with rows[pos] >= e
    while (a < b) {
      const std::uint64_t mid = a + (b - a) / 2;
      if (rows[mid] < e) a = mid + 1;
      else b = mid;
    }
    return a;
  }
};

// <pattern>_cpu: runs <pattern>_trigger for every eid in `triggers`, in that order, with
// `#pragma omp parallel for schedule(dynamic, chunk)`, adding into `out`. Under the sweep plan
// (SPEC v2 §7.1) it walks every node's list instead and emits for the triggers `out` selects;
// `triggers` and `chunk` are then unused.
using CpuFn = void (*)(const GraphView&, const Params&, std::span<const std::uint32_t> triggers,
                       BlockOut& out, int chunk);
// <pattern>_work: the trigger's work estimate, the window size of the pattern's first scan
// (SPEC §7.4 `sorted`, §7.5 bucketing).
using WorkFn = std::uint32_t (*)(const GraphView&, const Params&, std::uint32_t e);

// The CUDA launchers (SPEC §7.5), defined in the generated <stem>_cuda.cu. Every pointer is a
// device pointer, the GraphView included, and `stream` is a cudaStream_t. They only enqueue work.
//
// <pattern>_cuda_bucket: flags[i] = (<pattern>_work(first + i) >= threshold), for i < count.
using CudaBucketFn = void (*)(const GraphView& g, const Params& p, std::uint64_t first, std::uint32_t count,
                              std::uint32_t threshold, std::uint8_t* flags, void* stream);

// One launch of a pattern's light or heavy kernel over a trigger list, adding into the block's
// rows (row r belongs to trigger eid first + r).
struct CudaRun {
  std::uint32_t* out = nullptr;
  std::uint64_t first = 0;
  const std::uint32_t* bins = nullptr;
  std::uint32_t nbins = 0;    // the row's columns: 1 for a count block
  bool count_block = false;  // a count block (SPEC v3 R7): see RowSink
  unsigned long long* saturated = nullptr;  // emits lost to saturation are added here
  unsigned long long* aborted = nullptr;    // atomicMin of the eids over the work limit (SPEC v2 §8 S3)
  const std::uint32_t* triggers = nullptr;
  std::uint32_t count = 0;
  bool heavy = false;  // heavy: warp per trigger; light: thread per trigger
  void* stream = nullptr;
};
// <pattern>_cuda: the light or heavy kernel over run.triggers.
using CudaFn = void (*)(const GraphView& g, const Params& p, const CudaRun& run);

struct ConstInfo {
  const char* name;
  std::int64_t value;
};

struct PatternInfo {
  const char* name;
  const char* block;
  const char* block_loc;   // "<spec>:line:col" of its `block:` value, for E402
  const char* file;        // the spec it was generated from
  const char* spec_sha256; // hex SHA-256 of that spec's text; profile ids use it (SPEC v2 §3.1)
  const char* visibility;  // "stream" or "timestamp" if the spec pins it (§6.1 of v2), else ""
  const char* visibility_loc;  // "<spec>:line:col" of its `visibility:` value, or ""
  const char* span_param;  // the window span parameter; its value is the block's window
  std::span<const ConstInfo> consts;  // compile-time consts, after any --const override
  CpuFn cpu;
  WorkFn work;
  CudaBucketFn cuda_bucket;  // null in a build without CUDA
  CudaFn cuda;               // null in a build without CUDA
  bool cpu_sweep;            // the CPU driver is the sweep plan (SPEC v2 §7.1)
  bool cpu_reach;            // the CPU driver prunes by reachability when Params::reach_prune (SPEC v2 §8 S6)
  const char* family;        // "legacy" or "clean" (SPEC v3 §3.2); a profile mixes neither (E403)
  bool one_instance;         // at most one emit per trigger, at the root: may fill a count block (SPEC v3 R7)
};

inline bool is_clean(const PatternInfo& p) { return std::string_view(p.family) == "clean"; }

// Every pattern blzc compiled into this build, in the order of its spec arguments.
std::span<const PatternInfo> registered_patterns();

// Fusions (blzc --fuse, SPEC v2 §8 S7, D-46): one CUDA kernel body counts several sweep-shaped
// patterns with one scan of the trigger list they share. Each member writes its block's rows; the
// members' distinct blocks are the fusion's slots.
inline constexpr std::uint32_t kMaxFusedMembers = 4;
inline constexpr std::uint32_t kMaxFusedSlots = 4;

// Per member: its parameters (its block's window, the profile's visibility).
struct FusedParams {
  Params member[kMaxFusedMembers];
};

// One launch of a fusion's light or heavy kernel over a trigger list, adding into every slot's rows
// (row r of slot s belongs to trigger eid first + r). Every pointer is a device pointer.
struct FusedRun {
  std::uint32_t* out[kMaxFusedSlots] = {};
  const std::uint32_t* bins[kMaxFusedSlots] = {};
  std::uint32_t nbins[kMaxFusedSlots] = {};  // per slot, its row's columns: 1 for a count block
  bool count_block[kMaxFusedSlots] = {};      // per slot, a count block (SPEC v3 R7): see RowSink
  std::uint64_t first = 0;
  unsigned long long* saturated = nullptr;  // kMaxFusedSlots counters: emits lost to saturation, per slot
  const std::uint32_t* triggers = nullptr;
  std::uint32_t count = 0;
  bool heavy = false;
  void* stream = nullptr;
};
// fused_<id>_cuda_bucket: flags[i] = (the largest member work estimate of trigger first + i >= threshold).
using FusedBucketFn = void (*)(const GraphView& g, const FusedParams& p, std::uint64_t first, std::uint32_t count,
                               std::uint32_t threshold, std::uint8_t* flags, void* stream);
// fused_<id>_cuda: the light or heavy kernel over run.triggers.
using FusedFn = void (*)(const GraphView& g, const FusedParams& p, const FusedRun& run);

struct FusionInfo {
  const char* name;                             // the members' names joined by '+'
  std::span<const PatternInfo* const> members;  // in fusion order, entries of registered_patterns()
  std::span<const std::uint32_t> slots;         // per member: its block's slot
  FusedBucketFn cuda_bucket;                    // null in a build without CUDA
  FusedFn cuda;                                 // null in a build without CUDA
};

// Every fusion blzc compiled into this build (its --fuse groups), in that order.
std::span<const FusionInfo> registered_fusions();

// The optimizations this build's patterns were compiled without (`blzc --disable`, SPEC v2 §7.2);
// empty for the fully optimized build.
std::span<const char* const> disabled_optimizations();

}  // namespace blz

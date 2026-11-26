#pragma once
// Mining in trigger-range chunks (SPEC v3 H7, §5.1): a profile's blocks on one backend, each block's
// rows delivered to its own stream of a BlockSink, a chunk of at most chunk_rows triggers at a time,
// in eid order. A row depends only on its trigger and the whole graph, so the chunks' rows are the
// whole selection's rows. `blz mine` (NpySink) and `blz measure` (HashSink) run through it.
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "graph/index.hpp"
#include "oracle/profile_oracle.hpp"
#include "runtime/cpu_backend.hpp"
#include "runtime/cuda_backend.hpp"
#include "runtime/fusion.hpp"
#include "runtime/hybrid.hpp"
#include "runtime/profile.hpp"
#include "runtime/block_sink.hpp"
#include "runtime/triggers.hpp"

namespace blz {

// The chunk of the harness (SPEC v3 §5.1), frozen at P15 (D-81).
inline constexpr std::uint64_t kDefaultChunkRows = std::uint64_t(1) << 20;

// A run's backend and its options (blz mine's flags).
struct MineOptions {
  Backend backend = Backend::cpu;
  Schedule schedule = Schedule::dynamic;                   // cpu, and the hybrid's CPU side
  std::uint32_t heavy_threshold = kDefaultHeavyThreshold;  // cuda, hybrid: the light/heavy split
  std::uint32_t cpu_threshold = kDefaultCpuThreshold;      // hybrid: from it a trigger runs on the CPU
  bool reach_prune = false;                                // cpu, hybrid (SPEC v2 §8 S6)
  bool fuse = false, pack = false;                         // cuda (SPEC v2 §8 S7)
  std::uint64_t work_limit = kDefaultWorkLimit;            // every block's (R11)
  Quirks quirks;                                           // oracle, clean profile (E16)
  std::uint64_t chunk_rows = kDefaultChunkRows;            // triggers per chunk, > 0
};

// One block of a run, or R8's vstats.
struct BlockRun {
  std::string name;                     // the block's name, or "vstats"
  const ProfileBlock* block = nullptr;  // null for vstats
  BlockParams params;                   // as mined: the profile's, with the run's work limit and pruning
  StreamTally out;
  // The backend's calls, summed over the chunks: a fused unit's time, shared by its blocks; vstats'
  // includes its amount order. sink_seconds: the stream's (hash, tally, the sink's bytes).
  double mine_seconds = 0, sink_seconds = 0;
  // A block with a CPU sweep plan, on cpu or hybrid, is mined once over the selection (a sweep walks
  // every list entry whatever the range, D-79) and delivered in chunks.
  bool whole = false;
  const FusionInfo* fusion = nullptr;  // cuda --fuse: the unit it ran in
  // Summed over the chunks, for the backend that ran: cpu; cuda (a fused unit: the group's kernels and
  // trigger counts with this block's D2H and saturation; d2h_width the widest of its chunks, from 0);
  // hybrid.
  CpuBlockTiming cpu;
  CudaBlockTiming cuda{.d2h_width = 0};
  HybridTiming hybrid;
  std::uint64_t vstats_sides = 0;  // the distinct sides CleanVstats computed (path hit)
};

class ChunkDriver {
 public:
  // Builds what o.backend mines from: the profile's oracle, or its index (Profile::self_loops), and
  // for cuda and hybrid the index's device mirror and a miner. `s`, `amount` (Amount Received by
  // eid, read by vstats only) and `prof` must outlive the driver. Throws when an option does not
  // apply to the backend, or the profile cannot run on it (E404).
  ChunkDriver(const EdgeStream& s, std::span<const double> amount, const Profile& prof, const MineOptions& o);
  ~ChunkDriver();
  ChunkDriver(const ChunkDriver&) = delete;
  ChunkDriver& operator=(const ChunkDriver&) = delete;

  // Mines `blocks` (blocks of the profile; cuda --fuse joins a compiled fusion's blocks into one
  // unit), then R8's vstats if `vstats`, for `sel`, each into its own stream of `sink`. A clean
  // profile's lcyc runs first, so a work-limit abort (R11) comes before any other stream opens; the
  // exception leaves the open streams unfinished, which discards their rows. on_block runs after each
  // block's stream finished. cuda and hybrid take an eid range, not a sample.
  std::vector<BlockRun> run(std::span<const ProfileBlock* const> blocks, bool vstats, const TriggerSel& sel,
                            BlockSink& sink, const std::function<void(const BlockRun&)>& on_block = {});

  const MineOptions& options() const { return o_; }
  double build_seconds() const { return build_s_; }  // the index, or the oracle's lists
  const GraphIndex* index() const { return index_ ? &*index_ : nullptr; }
  const DeviceGraph* device() const { return dev_ ? &*dev_ : nullptr; }  // cuda and hybrid
  // The host row buffers held for chunks, vstats' included (pinned for cuda and hybrid), in bytes.
  std::uint64_t row_buffer_bytes() const;

 private:
  BlockRun run_block(const ProfileBlock& pb, const TriggerSel& sel, BlockSink& sink);
  std::vector<BlockRun> run_fused(const CudaUnit& unit, const TriggerSel& sel, BlockSink& sink);
  BlockRun run_vstats(const TriggerSel& sel, BlockSink& sink);
  // Block pb's rows of `part` into out (part.rows() rows); returns the lost emits.
  std::uint64_t mine_part(const ProfileBlock& pb, BlockRun& r, const TriggerSel& part, std::uint32_t* out);
  // Row buffer `slot`, at least `cells` cells: pinned for cuda and hybrid, zeroed host memory otherwise
  // (the cpu backend and the oracle add into their rows). Kept for later chunks and runs.
  std::uint32_t* rows_buffer(std::size_t slot, std::uint64_t cells);

  const Profile& prof_;
  MineOptions o_;
  std::span<const double> amount_;
  bool gpu_ = false;  // cuda or hybrid
  double build_s_ = 0;
  std::optional<ProfileOracle> oracle_;
  std::optional<GraphIndex> index_;
  std::optional<DeviceGraph> dev_;
  std::optional<CudaMiner> miner_;
  std::vector<std::unique_ptr<PinnedBuffer>> pinned_;
  std::vector<std::uint32_t> host_;
  std::vector<double> vstats_;
};

}  // namespace blz

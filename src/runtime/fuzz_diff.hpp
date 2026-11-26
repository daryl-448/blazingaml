#pragma once
#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "graph/exhaustive.hpp"
#include "graph/fuzz_gen.hpp"
#include "runtime/cpu_backend.hpp"
#include "runtime/profile.hpp"

namespace blz {

// One graph that fuzz_diff checks: an edge stream in stream order, the windows at which every block
// (and the vertex statistics) is checked, and one amount per edge for the vertex statistics.
struct DiffCase {
  std::uint64_t id = 0;  // a fuzz graph's seed; an exhaustive batch's index; a member's canonical index
  EdgeList g;
  std::vector<std::int32_t> windows;
  std::vector<double> amount;
  FuzzHits hits;  // a fuzz graph's (fuzz_hits); none for an exhaustive or a given case
  // An exhaustive case (GraphUnion): per eid, its canonical graph and its eid in that graph.
  std::vector<std::uint64_t> graph;
  std::vector<std::uint32_t> local;
};

// Where fuzz_diff's graphs come from.
//   - fuzz(s0, s1, setting): fuzz_graph(s) for the seeds [s0, s1), each at its own window, with
//     fuzz_amounts(s).
//   - exhaustive(b, batch_edges): L2's domain (SPEC v3 §4.3), every canonical graph of bound b
//     (ExhaustiveGraphs), as disjoint unions of up to batch_edges edges (at least one graph; 1 gives
//     each graph alone), each checked at every window 1..t. A batch never mixes edge counts, so
//     batches run m ascending too. Canonical graph i's amounts are fuzz_amounts(i, m), by its own
//     eids, alone or in a batch.
//   - given(cases): the cases as they are, each alone (L3's generated graphs, SPEC v3 §4.3). Each
//     has one amount per edge, at least one window, and no `graph` or `local`; std::invalid_argument
//     otherwise.
class DiffSource {
 public:
  enum class Kind { fuzz, exhaustive, given };
  static DiffSource fuzz(std::uint64_t s0, std::uint64_t s1, FuzzSetting setting = FuzzSetting::dense);
  static DiffSource exhaustive(ExhaustiveBound b, std::uint64_t batch_edges);
  static DiffSource given(std::vector<DiffCase> cases);

  Kind kind() const { return kind_; }
  std::uint64_t size() const { return size_; }  // cases
  DiffCase get(std::uint64_t i) const;
  // An exhaustive batch's canonical graphs, each alone, in canonical order; none for a fuzz graph.
  std::vector<DiffCase> members(const DiffCase& c) const;
  const ExhaustiveGraphs* graphs() const { return graphs_ ? &*graphs_ : nullptr; }
  std::uint64_t batch_edges() const { return batch_edges_; }
  FuzzSetting setting() const { return setting_; }

 private:
  DiffCase member(std::uint64_t graph, const std::vector<std::int32_t>& windows) const;

  Kind kind_ = Kind::fuzz;
  std::uint64_t s0_ = 0, size_ = 0, batch_edges_ = 0;
  FuzzSetting setting_ = FuzzSetting::dense;
  std::vector<DiffCase> given_;
  std::optional<ExhaustiveGraphs> graphs_;
  std::vector<std::uint64_t> batch_first_;  // per m - 1: the first batch with m edges; m_max + 1 entries
  std::vector<std::uint64_t> per_batch_;    // per m - 1: graphs per batch
};

// AC1 on fuzz graphs (SPEC §13.3), and L2's bounded code check on an exhaustive source (SPEC v3
// §4.3): the generated CPU backend against the profile's oracle (ProfileOracle: the clean oracle
// for a clean profile, SPEC v3 §3.2), every row of every block, on every case of `src`, at each of
// the case's windows, with `bins`, and the compiled K_MAX for the oracle's tcyc cap. Both schedules
// are checked, and for a block with a pattern planned for reachability pruning (tcyc, SPEC v2 §8
// S6) both again with it. The blocks, their patterns and k_max come from `prof`; the windows come
// from the cases, and the bins from fuzz_bins(): `bins`, or each profile block's own when `bins` is
// empty, and none for a count block (SPEC v3 R7). The graphs are indexed as the profile says
// (Profile::self_loops). registry_profile() gives every registered legacy block (v1's six, and
// sg_ord from P10).
//
// Work limit (SPEC v2 §8 S3, SPEC v3 R11): every run has `work_limit`. Where the reference stops a
// block (lcyc) at a window, the case is limited for that block, i.e. inconclusive: its rows are not
// compared and it counts in no gate, but the backend must stop at the same eid, the smallest over
// the limit; a backend that stops elsewhere, or not at all, or stops where the reference does not,
// counts as one mismatched row.
// A profile with vstats (SPEC v3 R8) also has its vertex statistics compared, bit for bit, at each
// window with the case's amounts: CleanVstats on the backend's device (the GPU for cuda and the
// hybrid) against the reference's.
// What the rows are compared against: the oracle, or the CPU backend with the dynamic schedule
// (backend against backend, cheaper than the oracle). With Reference::cpu, fuzz_diff_cpu checks the
// sorted schedule (and reachability pruning) against it, and no vstats: the CPU's are the reference.
enum class Reference { oracle, cpu };

// The first row a check found different, in check order (window, block, path, trigger).
struct Mismatch {
  std::string block;  // a block's name, or "vstats"
  std::string path;   // the run that computed `got`: "cpu sorted reach", "cuda H=4 fused packed", "hybrid H=0 C=4", ...
  std::int32_t window = 0;
  std::uint64_t eid = 0;                  // the trigger (a work-limit mismatch: the smaller stop eid)
  std::vector<std::uint32_t> got, want;   // a block's rows
  std::vector<double> got_vstats, want_vstats;
  // A work-limit mismatch (SPEC v3 R11): where each side stopped, UINT64_MAX if it completed; no rows.
  std::uint64_t got_stop = UINT64_MAX, want_stop = UINT64_MAX;
  // A chunked run's mismatch that is not a row (fuzz_diff_chunks): a stream's tally against the
  // reference's rows, or the chunk driver's error. `eid` is then the selection's first trigger.
  std::string note{};
};

// What a failing run reports (SPEC v3 §4.3): the mismatching case with the smallest id and, for an
// exhaustive source, its first canonical graph that fails alone, re-run through the same paths.
struct Counterexample {
  std::uint64_t case_id = 0;  // the seed, or the batch
  std::uint64_t graph = 0;    // the seed, or the canonical index of `g`
  // false: no graph of the batch fails alone (the bug needs a graph's position in the batch); `first`
  // is then the batch's own first mismatch, its eid that of its graph `g`.
  bool alone = true;
  EdgeList g;
  Mismatch first;  // its eids are g's
};

// Edge e of g as u->v@t.
std::string edge_text(const EdgeList& g, std::uint64_t e);
// A counterexample in the code's terms (SPEC v3 §4.3), as `blz fuzz-diff` prints it: the graph (by
// seed, canonical index or given case, as `src` names them), the block, window and path, the edges
// in eid order (up to 64 of them), and the trigger's got and want rows, the vstats columns that
// differ, or both sides' stop eids. Lines end in '\n'.
std::string counterexample_text(const Counterexample& cx, const DiffSource& src);

struct FuzzDiff {
  std::uint64_t cases = 0;           // fuzz graphs, or exhaustive batches
  std::uint64_t graphs = 0;          // fuzz graphs, or the canonical graphs in the batches
  std::uint64_t edges = 0;           // of the cases
  std::uint64_t trigger_checks = 0;  // (trigger, window) pairs
  std::uint64_t row_checks = 0;      // rows compared: per trigger check, block (and vstats) and path
  std::array<std::uint64_t, kAllBlocks.size()> mismatched_rows{};  // per block, every path
  std::array<std::uint64_t, kAllBlocks.size()> limited_graphs{};   // per block: cases where the reference hit the limit
  // Per block, the comparisons that ran, every path: rows compared, and runs whose stop eids were
  // compared instead (either side hit the work limit).
  std::array<std::uint64_t, kAllBlocks.size()> compared_rows{}, compared_stops{};
  // Path hit (SPEC v3 §4.4 item 6): what the cases exercise (fuzz graphs only), and per block the
  // reference's nonzero rows, summed over windows, where it completed.
  FuzzHits hits;
  std::array<std::uint64_t, kAllBlocks.size()> nonzero_rows{};
  // vstats: the cases compared and completed, the rows that differ in any bit, the reference's rows
  // with a nonzero column, and the distinct sides the backend computed (path hit), summed over
  // windows; and the cases where a chunked run stopped at the work limit before vstats (R11).
  std::uint64_t vstats_graphs = 0, vstats_mismatched_rows = 0, vstats_nonzero_rows = 0, vstats_sides = 0;
  std::uint64_t vstats_limited_graphs = 0;
  // Path hits of the optimised paths: the compared rows that each one computed (SPEC v3 §4.1 T2).
  std::uint64_t sweep_rows = 0;  // the CPU sweep drivers (SPEC v2 §7.1)
  std::uint64_t reach_rows = 0;  // reachability pruning (SPEC v2 §8 S6), on the CPU or the hybrid's CPU side
  std::uint64_t fused_rows = 0;  // fused CUDA kernels (SPEC v2 §8 S7)
  std::array<std::uint64_t, 3> packed_rows{};  // --pack: rows copied at 1, 2 and 4 bytes per cell
  std::uint64_t light_triggers = 0, heavy_triggers = 0;            // cuda: triggers per kernel
  std::uint64_t hybrid_cpu_triggers = 0, hybrid_gpu_triggers = 0;  // hybrid: triggers per side
  // The chunk driver (fuzz_diff_chunks, SPEC v3 H7): per block, and for vstats, the streams compared,
  // the puts they took, and the streams of more than one put; the streams of sweep blocks mined once
  // and put in chunks (D-79) and of fused units; the runs over a range and over a sample.
  struct ChunkHits {
    std::uint64_t streams = 0, puts = 0, multi = 0;
  };
  std::array<ChunkHits, kAllBlocks.size()> chunk_hits{};
  ChunkHits vstats_chunk_hits;
  std::uint64_t whole_streams = 0, fused_streams = 0, range_runs = 0, sampled_runs = 0;
  std::vector<std::uint64_t> bad_seeds;      // the mismatching cases' ids, ascending
  std::vector<std::uint64_t> limited_seeds;  // ascending: limited in some block
  std::optional<Counterexample> counterexample;
};

FuzzDiff fuzz_diff_cpu(const Profile& prof, const DiffSource& src, std::span<const std::uint32_t> bins,
                       std::uint64_t work_limit = kDefaultWorkLimit, Reference ref = Reference::oracle);

// AC1 for the generated CUDA backend, the same way: every row of every block against the oracle,
// once per light/heavy threshold in `thresholds` (0 sends every trigger to the heavy kernel,
// kAllLight every trigger to the light kernel). mismatched_rows sums over the thresholds.
// SPEC v2 §8 S7: `opts.fuse` mines each compiled fusion's blocks in one kernel (cuda_units), and
// `opts.pack` copies the rows packed; the rows must not change. A fused kernel runs with every
// member at the window, and again (a case with more than one window) with member k at the k-th
// window after it, cyclically, since a profile may give a fusion's members different windows.
struct CudaOptions {
  bool fuse = false;
  bool pack = false;
};
FuzzDiff fuzz_diff_cuda(const Profile& prof, const DiffSource& src, std::span<const std::uint32_t> bins,
                        std::span<const std::uint32_t> thresholds, std::uint64_t work_limit = kDefaultWorkLimit,
                        CudaOptions opts = {}, Reference ref = Reference::oracle);

// AC1 for the hybrid (SPEC v2 §8 S6): mine_hybrid with reachability pruning on its CPU side and
// the sorted schedule, once per pair of a GPU light/heavy threshold in `thresholds` and a CPU
// threshold in `cpu_thresholds` (0: every trigger on the CPU, kAllGpu: every trigger on the GPU).
// With no CPU threshold it is fuzz_diff_cuda.
// `opts` is for the plain CUDA runs only (no CPU threshold).
FuzzDiff fuzz_diff_hybrid(const Profile& prof, const DiffSource& src, std::span<const std::uint32_t> bins,
                          std::span<const std::uint32_t> thresholds, std::span<const std::uint32_t> cpu_thresholds,
                          std::uint64_t work_limit = kDefaultWorkLimit, CudaOptions opts = {},
                          Reference ref = Reference::oracle);

// L2 and V-diff through the chunk driver (SPEC v3 H7, §4.4 V-code): `backend` (cpu, cuda or hybrid)
// runs every block of `prof` and its vstats through ChunkDriver into a MemorySink, as `blz mine` and
// `blz measure` do, once per chunk size in `chunk_rows` (each > 0), per light/heavy threshold in
// `thresholds` (cuda, hybrid) and per CPU threshold in `cpu_thresholds` (hybrid), with `opts` (cuda),
// over each case's selections: every trigger; the range [m/4, m - m/5) when it is neither empty nor
// every trigger; and on the cpu a sample of about 40 % of the triggers. The cpu runs the dynamic
// schedule, the hybrid's CPU side the sorted one with reachability pruning, as fuzz_diff_hybrid.
// Every stream is compared with the reference's rows at its triggers, bit for bit, and its tally
// with them: rows, ⌈rows / chunk_rows⌉ puts, the cells' sum and FNV-1a (`fnv1a_bytes`). A chunked
// run that throws std::logic_error (the streams' and slice()'s guards) is a mismatch of the block
// whose stream opened last. Under the work limit (R11) the run over every trigger must stop at the
// reference's lcyc stop eid and keep no stream; a sub-selection of a case the reference stopped is
// inconclusive. The comparison is fuzz_diff's own (drive), so counterexamples narrow the same way.
// The cases run in parallel on every backend, each driver with its own device graph, miner and
// pinned buffers: a chunk of one trigger costs a backend call per row and block, and the GPU's calls
// from several host threads overlap (the hybrid's CPU side then runs on one thread per case).
FuzzDiff fuzz_diff_chunks(const Profile& prof, const DiffSource& src, std::span<const std::uint32_t> bins,
                          Backend backend, std::span<const std::uint64_t> chunk_rows,
                          std::span<const std::uint32_t> thresholds = {}, std::span<const std::uint32_t> cpu_thresholds = {},
                          std::uint64_t work_limit = kDefaultWorkLimit, CudaOptions opts = {},
                          Reference ref = Reference::oracle);

}  // namespace blz

#include "runtime/chunk_driver.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>

#include "runtime/clean_vstats.hpp"
#include "runtime/fusion.hpp"

namespace blz {

namespace {

using Clock = std::chrono::steady_clock;
double since(Clock::time_point a) { return std::chrono::duration<double>(Clock::now() - a).count(); }

void add(CpuBlockTiming& a, const CpuBlockTiming& b) {
  a.schedule_seconds += b.schedule_seconds;
  a.loop_seconds += b.loop_seconds;
  a.sweep_seconds += b.sweep_seconds;
  a.saturated += b.saturated;
}

void add(CudaBlockTiming& a, const CudaBlockTiming& b) {
  a.kernel_seconds += b.kernel_seconds;
  a.bucket_seconds += b.bucket_seconds;
  a.light_seconds += b.light_seconds;
  a.heavy_seconds += b.heavy_seconds;
  a.d2h_seconds += b.d2h_seconds;
  a.pack_seconds += b.pack_seconds;
  a.widen_seconds += b.widen_seconds;
  a.d2h_width = std::max(a.d2h_width, b.d2h_width);
  a.d2h_bytes += b.d2h_bytes;
  a.h2d_seconds += b.h2d_seconds;
  a.split_seconds += b.split_seconds;
  a.cpu_triggers += b.cpu_triggers;
  a.wall_seconds += b.wall_seconds;
  a.light += b.light;
  a.heavy += b.heavy;
  a.saturated += b.saturated;
}

void add(HybridTiming& a, const HybridTiming& b) {
  a.split_seconds += b.split_seconds;
  a.cpu_seconds += b.cpu_seconds;
  a.gpu_seconds += b.gpu_seconds;
  a.overlap_seconds += b.overlap_seconds;
  a.merge_seconds += b.merge_seconds;
  a.total_seconds += b.total_seconds;
  a.cpu_triggers += b.cpu_triggers;
  a.gpu_triggers += b.gpu_triggers;
  a.saturated += b.saturated;
  add(a.cpu, b.cpu);
  add(a.gpu, b.gpu);
}

}  // namespace

ChunkDriver::ChunkDriver(const EdgeStream& s, std::span<const double> amount, const Profile& prof,
                         const MineOptions& o)
    : prof_(prof), o_(o), amount_(amount) {
  const Backend be = o.backend;
  gpu_ = be == Backend::cuda || be == Backend::hybrid;
  if (o.chunk_rows == 0) throw std::invalid_argument("--chunk-rows must be > 0");
  if ((o.fuse || o.pack) && be != Backend::cuda) throw std::runtime_error("--fuse and --pack are for the cuda backend");
  if (o.reach_prune && be != Backend::cpu && be != Backend::hybrid)
    throw std::runtime_error("--reach-prune is for the cpu and hybrid backends");
  if (!quirk_ids(o.quirks).empty() && (be != Backend::oracle || prof.family != Family::clean))
    throw std::runtime_error("--quirks is for the oracle backend and a clean profile (SPEC v3 §3.6)");
  if (gpu_ && !cuda_built()) cuda_device();  // throws: no CUDA backend in this build
  check_profile_backend(prof, be);           // E404 (v1 configs: E402, K_MAX)
  const auto t0 = Clock::now();
  if (be == Backend::oracle) oracle_.emplace(s, prof, o.quirks, amount);
  else index_.emplace(s.src, s.dst, s.ts, s.n, prof.self_loops());
  build_s_ = since(t0);
  if (gpu_) {
    dev_.emplace(index_->view());
    miner_.emplace(be == Backend::hybrid);  // hybrid: the GPU's host thread sleeps while the CPU team works
  }
}

ChunkDriver::~ChunkDriver() = default;

std::uint32_t* ChunkDriver::rows_buffer(std::size_t slot, std::uint64_t cells) {
  if (!gpu_) {
    host_.assign(cells, 0);
    return host_.data();
  }
  if (pinned_.size() <= slot) pinned_.resize(slot + 1);
  if (!pinned_[slot] || pinned_[slot]->size() < cells)
    pinned_[slot] = std::make_unique<PinnedBuffer>(std::max<std::uint64_t>(cells, 1));
  return pinned_[slot]->data();
}

std::uint64_t ChunkDriver::mine_part(const ProfileBlock& pb, BlockRun& r, const TriggerSel& part, std::uint32_t* out) {
  const BlockParams& p = r.params;
  switch (o_.backend) {
    case Backend::oracle:
      return part.sampled ? oracle_->mine(pb.block, p, part.eids, out)
                          : oracle_->mine(pb.block, p, part.first, part.last, out);
    case Backend::cpu: {
      const CpuBlockTiming tm = mine_cpu(index_->view(), pb.patterns, p, prof_.visibility, part, o_.schedule, out);
      add(r.cpu, tm);
      return tm.saturated;
    }
    case Backend::cuda: {
      const CudaBlockTiming tm = miner_->mine(*dev_, pb.patterns, p, prof_.visibility, part.first, part.last,
                                              o_.heavy_threshold, out, o_.pack);
      add(r.cuda, tm);
      return tm.saturated;
    }
    case Backend::hybrid: {
      const HybridTiming tm = mine_hybrid(index_->view(), *dev_, *miner_, pb.patterns, p, prof_.visibility, part.first,
                                          part.last, o_.cpu_threshold, o_.heavy_threshold, o_.schedule, out);
      add(r.hybrid, tm);
      return tm.saturated;
    }
  }
  throw std::logic_error("mine_part: unknown backend");
}

BlockRun ChunkDriver::run_block(const ProfileBlock& pb, const TriggerSel& sel, BlockSink& sink) {
  BlockRun r;
  r.name = std::string(block_name(pb.block));
  r.block = &pb;
  r.params = pb.params;
  r.params.work_limit = o_.work_limit;
  r.params.reach_prune = o_.reach_prune;
  const std::uint32_t cols = block_cols(r.params);
  const std::uint64_t rows = sel.rows();
  std::unique_ptr<BlockStream> stream = sink.open({r.name, Cell::u32, rows, cols});
  r.whole = (o_.backend == Backend::cpu || o_.backend == Backend::hybrid) &&
            std::any_of(pb.patterns.begin(), pb.patterns.end(), [](const PatternInfo* pi) { return pi->cpu_sweep; });
  const std::uint64_t part_rows = r.whole ? std::max<std::uint64_t>(rows, 1) : o_.chunk_rows;
  for (std::uint64_t a = 0; a < rows; a += part_rows) {
    const std::uint64_t b = std::min(rows, a + part_rows);
    std::uint32_t* buf = rows_buffer(0, (b - a) * cols);
    auto t0 = Clock::now();
    std::uint64_t sat = mine_part(pb, r, slice(sel, a, b), buf);
    r.mine_seconds += since(t0);
    t0 = Clock::now();
    for (std::uint64_t c = a; c < b; c += o_.chunk_rows) {
      const std::uint64_t e = std::min(b, c + o_.chunk_rows);
      stream->put(std::span<const std::uint32_t>(buf + (c - a) * cols, (e - c) * cols), sat);
      sat = 0;  // a whole part's lost emits go with its first chunk
    }
    r.sink_seconds += since(t0);
  }
  const auto t1 = Clock::now();
  r.out = stream->finish();
  r.sink_seconds += since(t1);
  return r;
}

std::vector<BlockRun> ChunkDriver::run_fused(const CudaUnit& unit, const TriggerSel& sel, BlockSink& sink) {
  const std::size_t nslots = unit.blocks.size();
  const std::uint64_t rows = sel.rows();
  std::vector<BlockRun> runs(nslots);
  std::vector<const BlockParams*> params;
  std::vector<std::unique_ptr<BlockStream>> streams;
  for (std::size_t sl = 0; sl < nslots; ++sl) {
    BlockRun& r = runs[sl];
    r.name = std::string(block_name(unit.blocks[sl]->block));
    r.block = unit.blocks[sl];
    r.params = unit.blocks[sl]->params;
    r.params.work_limit = o_.work_limit;
    r.fusion = unit.fusion;
    params.push_back(&r.params);
    streams.push_back(sink.open({r.name, Cell::u32, rows, block_cols(r.params)}));
  }
  std::vector<std::uint32_t*> outs(nslots);
  for (std::uint64_t a = 0; a < rows; a += o_.chunk_rows) {
    const std::uint64_t b = std::min(rows, a + o_.chunk_rows);
    for (std::size_t sl = 0; sl < nslots; ++sl) outs[sl] = rows_buffer(sl, (b - a) * block_cols(runs[sl].params));
    const auto t0 = Clock::now();
    const CudaMiner::FusedTiming ftm = miner_->mine_fused(*dev_, *unit.fusion, params, prof_.visibility, sel.first + a,
                                                          sel.first + b, o_.heavy_threshold, outs, o_.pack);
    const double mine_s = since(t0);
    for (std::size_t sl = 0; sl < nslots; ++sl) {
      BlockRun& r = runs[sl];
      r.mine_seconds += mine_s;
      CudaBlockTiming t = ftm.group;  // the kernels and trigger counts are the fusion's
      const CudaBlockTiming& own = ftm.slots[sl];
      t.d2h_seconds = own.d2h_seconds;
      t.pack_seconds = own.pack_seconds;
      t.widen_seconds = own.widen_seconds;
      t.d2h_width = own.d2h_width;
      t.d2h_bytes = own.d2h_bytes;
      t.saturated = own.saturated;
      add(r.cuda, t);
      const auto t1 = Clock::now();
      streams[sl]->put(std::span<const std::uint32_t>(outs[sl], (b - a) * block_cols(r.params)), own.saturated);
      r.sink_seconds += since(t1);
    }
  }
  for (std::size_t sl = 0; sl < nslots; ++sl) {
    const auto t1 = Clock::now();
    runs[sl].out = streams[sl]->finish();
    runs[sl].sink_seconds += since(t1);
  }
  return runs;
}

BlockRun ChunkDriver::run_vstats(const TriggerSel& sel, BlockSink& sink) {
  BlockRun r;
  r.name = "vstats";
  const std::uint64_t rows = sel.rows();
  std::unique_ptr<BlockStream> stream = sink.open({r.name, Cell::f64, rows, std::uint32_t(kVstatsCols)});
  const std::int32_t window = prof_.vstats->window;
  auto t0 = Clock::now();
  std::optional<CleanVstats> vs;  // the CPU's or (cuda, hybrid) the GPU's; the oracle has its own
  if (!oracle_) vs.emplace(index_->view(), amount_, window, gpu_ ? &*dev_ : nullptr);
  r.mine_seconds += since(t0);
  std::vector<double> buf;
  for (std::uint64_t a = 0; a < rows; a += o_.chunk_rows) {
    const std::uint64_t b = std::min(rows, a + o_.chunk_rows);
    const TriggerSel part = slice(sel, a, b);
    buf.resize((b - a) * kVstatsCols);
    t0 = Clock::now();
    if (vs) vs->rows(part, buf.data());
    else oracle_->vstats(window, part, buf.data());
    r.mine_seconds += since(t0);
    t0 = Clock::now();
    stream->put(std::span<const double>(buf));
    r.sink_seconds += since(t0);
  }
  t0 = Clock::now();
  r.out = stream->finish();
  r.sink_seconds += since(t0);
  if (vs) r.vstats_sides = vs->sides();
  return r;
}

std::vector<BlockRun> ChunkDriver::run(std::span<const ProfileBlock* const> blocks, bool vstats,
                                       const TriggerSel& sel, BlockSink& sink,
                                       const std::function<void(const BlockRun&)>& on_block) {
  if (gpu_ && sel.sampled)
    throw std::runtime_error("--triggers sample: runs on the oracle and cpu backends (the preflight of SPEC v2 §2.5 "
                             "uses cpu); the CUDA kernels take an eid range");
  if (vstats && !prof_.vstats) throw std::runtime_error("vstats: " + prof_.file + " has no vstats (SPEC v3 R8)");
  std::vector<CudaUnit> units = cuda_units(blocks, o_.fuse && o_.backend == Backend::cuda);
  // R11: a clean run that reaches the work limit aborts with no rows. lcyc is the one block that
  // counts work, so it runs first: an abort comes before any other block is mined.
  if (prof_.family == Family::clean)
    std::stable_partition(units.begin(), units.end(),
                          [](const CudaUnit& u) { return u.blocks.front()->block == Block::lcyc; });
  std::vector<BlockRun> out;
  const auto done = [&](BlockRun r) {
    if (on_block) on_block(r);
    out.push_back(std::move(r));
  };
  for (const CudaUnit& unit : units) {
    if (unit.fusion) {
      for (BlockRun& r : run_fused(unit, sel, sink)) done(std::move(r));
    } else {
      done(run_block(*unit.blocks.front(), sel, sink));
    }
  }
  if (vstats) done(run_vstats(sel, sink));
  return out;
}

}  // namespace blz

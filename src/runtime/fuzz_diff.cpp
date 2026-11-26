#include "runtime/fuzz_diff.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <type_traits>

#include <omp.h>

#include "graph/fuzz_gen.hpp"
#include "graph/index.hpp"
#include "oracle/profile_oracle.hpp"
#include "runtime/block_sink.hpp"
#include "runtime/chunk_driver.hpp"
#include "runtime/clean_vstats.hpp"
#include "runtime/cuda_backend.hpp"
#include "runtime/fusion.hpp"
#include "runtime/hybrid.hpp"

namespace blz {

DiffSource DiffSource::fuzz(std::uint64_t s0, std::uint64_t s1, FuzzSetting setting) {
  if (s1 < s0) throw std::invalid_argument("fuzz-diff: seeds a:b need a <= b");
  DiffSource s;
  s.s0_ = s0;
  s.size_ = s1 - s0;
  s.setting_ = setting;
  return s;
}

DiffSource DiffSource::given(std::vector<DiffCase> cases) {
  for (const DiffCase& c : cases)
    if (c.amount.size() != c.g.src.size() || c.windows.empty() || !c.graph.empty() || !c.local.empty())
      throw std::invalid_argument("fuzz-diff: a given case needs one amount per edge, a window, and no batch");
  DiffSource s;
  s.kind_ = Kind::given;
  s.size_ = cases.size();
  s.given_ = std::move(cases);
  return s;
}

DiffSource DiffSource::exhaustive(ExhaustiveBound b, std::uint64_t batch_edges) {
  DiffSource s;
  s.kind_ = Kind::exhaustive;
  s.graphs_.emplace(b);
  s.batch_edges_ = std::max<std::uint64_t>(batch_edges, 1);
  s.batch_first_.assign(b.m_max + 1, 0);
  s.per_batch_.assign(b.m_max, 0);
  for (std::uint32_t m = 1; m <= b.m_max; ++m) {
    const std::uint64_t graphs = s.graphs_->first(m + 1) - s.graphs_->first(m);
    s.per_batch_[m - 1] = std::max<std::uint64_t>(1, s.batch_edges_ / m);
    s.batch_first_[m] = s.batch_first_[m - 1] + graphs / s.per_batch_[m - 1] + (graphs % s.per_batch_[m - 1] != 0);
  }
  s.size_ = s.batch_first_.back();
  return s;
}

DiffCase DiffSource::member(std::uint64_t graph, const std::vector<std::int32_t>& windows) const {
  DiffCase c;
  c.id = graph;
  c.g = graphs_->graph(graph);
  const std::uint64_t m = c.g.src.size();
  c.windows = windows;
  c.amount = fuzz_amounts(graph, m);
  c.graph.assign(m, graph);
  c.local.resize(m);
  std::iota(c.local.begin(), c.local.end(), 0u);
  return c;
}

DiffCase DiffSource::get(std::uint64_t i) const {
  if (i >= size_) throw std::out_of_range("fuzz-diff: case index past the source");
  if (kind_ == Kind::given) return given_[i];
  DiffCase c;
  if (kind_ == Kind::fuzz) {
    FuzzGraph fg = fuzz_graph(s0_ + i, setting_);
    c.id = fg.seed;
    c.hits = fuzz_hits(fg);
    c.windows = {fg.window};
    c.amount = fuzz_amounts(fg.seed, fg.src.size());
    c.g = {fg.n, std::move(fg.src), std::move(fg.dst), std::move(fg.ts)};
    return c;
  }
  const std::uint32_t m = std::uint32_t(std::upper_bound(batch_first_.begin(), batch_first_.end(), i) - batch_first_.begin());
  const std::uint64_t first = graphs_->first(m) + (i - batch_first_[m - 1]) * per_batch_[m - 1];
  const std::uint64_t last = std::min(first + per_batch_[m - 1], graphs_->first(m + 1));
  GraphUnion u = disjoint_union(*graphs_, first, last);
  c.id = i;
  for (std::int32_t w = 1; w <= graphs_->bound().t; ++w) c.windows.push_back(w);
  std::vector<std::vector<double>> amounts(last - first);  // each member's own, by its local eid
  for (std::uint64_t k = first; k < last; ++k) amounts[k - first] = fuzz_amounts(k, m);
  c.amount.resize(u.g.src.size());
  for (std::uint64_t e = 0; e < u.g.src.size(); ++e) c.amount[e] = amounts[u.graph[e] - first][u.local[e]];
  c.g = std::move(u.g);
  c.graph = std::move(u.graph);
  c.local = std::move(u.local);
  return c;
}

std::vector<DiffCase> DiffSource::members(const DiffCase& c) const {
  if (kind_ != Kind::exhaustive || c.graph.empty()) return {};
  const auto [lo, hi] = std::minmax_element(c.graph.begin(), c.graph.end());
  std::vector<DiffCase> out;
  for (std::uint64_t k = *lo; k <= *hi; ++k) out.push_back(member(k, c.windows));
  return out;
}

namespace {

std::array<std::vector<std::uint32_t>, kAllBlocks.size()> all_fuzz_bins(const Profile& prof,
                                                                        std::span<const std::uint32_t> bins) {
  std::array<std::vector<std::uint32_t>, kAllBlocks.size()> out;
  for (const ProfileBlock& pb : prof.blocks) try {
      out[int(pb.block)] = fuzz_bins(prof, pb, bins);
    } catch (const std::invalid_argument& e) {
      throw std::invalid_argument(std::string("fuzz-diff: ") + e.what());
    }
  return out;
}

// What every case of a run is checked with.
struct Run {
  const Profile& prof;
  std::array<std::vector<std::uint32_t>, kAllBlocks.size()> bins;
  std::uint64_t work_limit;
  Reference ref;

  BlockParams params(Block b, std::int32_t window) const {
    BlockParams p{window, bins[int(b)]};
    p.work_limit = work_limit;
    return p;
  }
};

bool has(const ProfileBlock& pb, bool PatternInfo::*flag) {
  return std::any_of(pb.patterns.begin(), pb.patterns.end(), [&](const PatternInfo* pi) { return pi->*flag; });
}

// An sg block's gather pattern, when the block sums both roles (§4.5), else null. Its rows are the
// sum, so the gather role is also checked alone against the oracle's (T1 per pattern).
const PatternInfo* sg_gather_of(const ProfileBlock& pb) {
  if (pb.block != Block::sg || pb.patterns.size() != 2) return nullptr;
  for (const PatternInfo* pi : pb.patterns)
    if (std::string_view(pi->name).ends_with("sg_gather")) return pi;
  return nullptr;
}

std::string threshold_text(std::uint32_t h) { return h == kAllLight ? "all" : std::to_string(h); }

// The reference rows of one block: the profile's oracle's, or the CPU backend's (dynamic schedule,
// no pruning) when `oracle` is null.
void reference_rows(const ProfileOracle* oracle, const GraphIndex& index, const ProfileBlock& pb, const BlockParams& p,
                    const Profile& prof, std::uint64_t m, std::uint32_t* out) {
  if (oracle) {
    oracle->mine(pb.block, p, 0, m, out);
    return;
  }
  TriggerSel all;
  all.last = m;
  BlockParams q = p;
  q.reach_prune = false;
  mine_cpu(index.view(), pb.patterns, q, prof.visibility, all, Schedule::dynamic, out);
}

// The reference's vstats rows of a graph (SPEC v3 R8): the profile's oracle's, or the CPU's.
std::vector<double> reference_vstats(const ProfileOracle* oracle, const GraphIndex& index,
                                     std::span<const double> amount, std::int32_t window, std::uint64_t m) {
  std::vector<double> out(m * kVstatsCols);
  TriggerSel all;
  all.last = m;
  if (oracle) oracle->vstats(window, all, out.data());
  else CleanVstats(index.view(), amount, window).rows(all, out.data());
  return out;
}

// Runs f; returns UINT64_MAX, or the eid at which the work limit stopped it (SPEC v2 §8 S3).
template <class F>
std::uint64_t stop_eid(F&& f) {
  try {
    f();
  } catch (const WorkLimitExceeded& x) {
    return x.eid;
  }
  return UINT64_MAX;
}

// The reference's outcome at one window of a case: every block's rows and stop eid, vstats, and
// with `sg_gather` (and the oracle) the oracle's sg gather role alone (sg_gather_of).
struct Want {
  std::array<std::vector<std::uint32_t>, kAllBlocks.size()> rows;
  std::array<std::uint64_t, kAllBlocks.size()> stop;  // the reference's stop eid, or UINT64_MAX
  std::vector<double> vstats;
  std::vector<std::uint32_t> sg_gather;
};

Want reference_want(const Run& run, const ProfileOracle* oracle, const GraphIndex& index, const DiffCase& c,
                    std::int32_t w, bool sg_gather) {
  const Profile& prof = run.prof;
  const std::uint64_t m = c.g.src.size();
  Want x;
  if (prof.vstats) x.vstats = reference_vstats(oracle, index, c.amount, w, m);
  x.stop.fill(UINT64_MAX);
  for (const ProfileBlock& pb : prof.blocks) {
    const BlockParams p = run.params(pb.block, w);
    x.rows[int(pb.block)].assign(m * block_cols(p), 0);
    x.stop[int(pb.block)] =
        stop_eid([&] { reference_rows(oracle, index, pb, p, prof, m, x.rows[int(pb.block)].data()); });
    if (sg_gather && sg_gather_of(pb) && oracle) {
      x.sg_gather.assign(m * block_cols(p), 0);
      oracle->mine_sg_gather(p, 0, m, x.sg_gather.data());
    }
  }
  return x;
}

// One case's tally (a FuzzDiff of that case alone) and its first mismatch.
struct CaseResult {
  FuzzDiff d;
  std::optional<Mismatch> first;
  std::array<bool, kAllBlocks.size()> limited{};
  bool vstats_compared = false;  // the case's vstats rows were compared (counted once per case)
  bool vstats_limited = false;   // a chunked run of the case stopped at the work limit before vstats

  explicit CaseResult(const DiffCase& c) {
    const std::uint64_t m = c.g.src.size();
    d.cases = 1;
    d.graphs = 1;
    if (!c.graph.empty()) {  // an exhaustive batch: its canonical graphs are a contiguous run
      const auto [lo, hi] = std::minmax_element(c.graph.begin(), c.graph.end());
      d.graphs = *hi - *lo + 1;
    }
    d.edges = m;
    d.trigger_checks = m * c.windows.size();
    d.hits = c.hits;
  }

  bool bad() const {
    return d.vstats_mismatched_rows != 0 ||
           std::any_of(d.mismatched_rows.begin(), d.mismatched_rows.end(), [](std::uint64_t x) { return x != 0; });
  }

  // A block's rows, `got` against `want`, row by row; row r is trigger r, or sel->eid(r).
  void rows(Block b, const std::string& path, std::int32_t w, std::span<const std::uint32_t> got,
            std::span<const std::uint32_t> want, std::size_t cols, const TriggerSel* sel = nullptr) {
    const std::uint64_t m = want.size() / cols;
    d.row_checks += m;
    d.compared_rows[int(b)] += m;
    for (std::uint64_t e = 0; e < m; ++e) {
      const auto g = got.subspan(e * cols, cols), x = want.subspan(e * cols, cols);
      if (std::equal(g.begin(), g.end(), x.begin())) continue;
      ++d.mismatched_rows[int(b)];
      if (!first)
        first = Mismatch{std::string(block_name(b)), path, w, sel ? sel->eid(e) : e, {g.begin(), g.end()},
                         {x.begin(), x.end()}, {}, {}};
    }
  }

  // A chunked run's mismatch that is not a row (a stream's tally, the driver's error): one
  // mismatched row of block b, or of vstats when b is null.
  void note(const Block* b, const std::string& path, std::int32_t w, std::uint64_t eid, std::string text) {
    if (b) ++d.mismatched_rows[int(*b)];
    else ++d.vstats_mismatched_rows;
    if (!first) {
      first = Mismatch{b ? std::string(block_name(*b)) : "vstats", path, w, eid, {}, {}, {}, {}};
      first->note = std::move(text);
    }
  }

  // Where either side stopped at the work limit (SPEC v3 R11), the stop eids are compared instead of
  // the rows; returns whether that was so.
  bool stops(Block b, const std::string& path, std::int32_t w, std::uint64_t got, std::uint64_t want) {
    if (got == UINT64_MAX && want == UINT64_MAX) return false;
    ++d.compared_stops[int(b)];
    if (got != want) {
      ++d.mismatched_rows[int(b)];
      if (!first) {
        first = Mismatch{std::string(block_name(b)), path, w, std::min(got, want), {}, {}, {}, {}};
        first->got_stop = got;
        first->want_stop = want;
      }
    }
    return true;
  }

  // vstats rows, bit for bit, and the reference's nonzero rows; row r is trigger r, or sel->eid(r).
  void vstats(const std::string& path, std::int32_t w, const std::vector<double>& got, const std::vector<double>& want,
              const TriggerSel* sel = nullptr) {
    const std::uint64_t m = want.size() / kVstatsCols;
    vstats_compared = true;
    d.row_checks += m;
    for (std::uint64_t r = 0; r < m; ++r) {
      const double* g = got.data() + r * kVstatsCols;
      const double* x = want.data() + r * kVstatsCols;
      d.vstats_nonzero_rows += std::any_of(x, x + kVstatsCols, [](double v) { return v != 0; });
      if (std::memcmp(g, x, kVstatsCols * sizeof(double)) == 0) continue;
      ++d.vstats_mismatched_rows;
      if (!first) {
        first = Mismatch{"vstats", path, w, sel ? sel->eid(r) : r, {}, {}, {g, g + kVstatsCols}, {x, x + kVstatsCols}};
      }
    }
  }

  // The reference's outcome of block b at one window: limited, or its nonzero rows (path hit).
  void reference(Block b, std::uint64_t stop, std::span<const std::uint32_t> want, std::size_t cols) {
    if (stop != UINT64_MAX) limited[int(b)] = true;
    else d.nonzero_rows[int(b)] += nonzero_rows(want, cols);
  }

  // Closes the tally: per block, whether the case was limited; the case's id in the id lists.
  void close(std::uint64_t id) {
    bool any_limited = false;
    for (std::size_t i = 0; i < limited.size(); ++i) {
      d.limited_graphs[i] += limited[i];
      any_limited = any_limited || limited[i];
    }
    d.vstats_graphs += vstats_compared && !vstats_limited;
    d.vstats_limited_graphs += vstats_limited;
    if (bad()) d.bad_seeds.push_back(id);
    if (any_limited) d.limited_seeds.push_back(id);
  }
};

void add(FuzzDiff& a, const FuzzDiff& b) {
  a.cases += b.cases;
  a.graphs += b.graphs;
  a.edges += b.edges;
  a.trigger_checks += b.trigger_checks;
  a.row_checks += b.row_checks;
  for (std::size_t i = 0; i < kAllBlocks.size(); ++i) {
    a.mismatched_rows[i] += b.mismatched_rows[i];
    a.limited_graphs[i] += b.limited_graphs[i];
    a.nonzero_rows[i] += b.nonzero_rows[i];
    a.compared_rows[i] += b.compared_rows[i];
    a.compared_stops[i] += b.compared_stops[i];
  }
  a.hits += b.hits;
  a.vstats_graphs += b.vstats_graphs;
  a.vstats_limited_graphs += b.vstats_limited_graphs;
  a.vstats_mismatched_rows += b.vstats_mismatched_rows;
  a.vstats_nonzero_rows += b.vstats_nonzero_rows;
  a.vstats_sides += b.vstats_sides;
  a.sweep_rows += b.sweep_rows;
  a.reach_rows += b.reach_rows;
  a.fused_rows += b.fused_rows;
  for (std::size_t i = 0; i < a.packed_rows.size(); ++i) a.packed_rows[i] += b.packed_rows[i];
  a.light_triggers += b.light_triggers;
  a.heavy_triggers += b.heavy_triggers;
  a.hybrid_cpu_triggers += b.hybrid_cpu_triggers;
  a.hybrid_gpu_triggers += b.hybrid_gpu_triggers;
  const auto chunks = [](FuzzDiff::ChunkHits& x, const FuzzDiff::ChunkHits& y) {
    x.streams += y.streams;
    x.puts += y.puts;
    x.multi += y.multi;
  };
  for (std::size_t i = 0; i < kAllBlocks.size(); ++i) chunks(a.chunk_hits[i], b.chunk_hits[i]);
  chunks(a.vstats_chunk_hits, b.vstats_chunk_hits);
  a.whole_streams += b.whole_streams;
  a.fused_streams += b.fused_streams;
  a.range_runs += b.range_runs;
  a.sampled_runs += b.sampled_runs;
  a.bad_seeds.insert(a.bad_seeds.end(), b.bad_seeds.begin(), b.bad_seeds.end());
  a.limited_seeds.insert(a.limited_seeds.end(), b.limited_seeds.begin(), b.limited_seeds.end());
}

// The graph loop of every backend, and so the one owner of the comparison's bookkeeping. The cases
// run in groups: `prepare` runs on each case of a group in parallel (the CPU's whole check, or the
// GPU's reference), then `finish` on each in case order on this thread (the GPU's runs), which
// returns the case's CaseResult. The oracle's and the drivers' own parallel loops inside `prepare`
// run nested, i.e. on one thread each, unless the group is one case (L3 checks one given case at a
// time): its loops then get the whole team, so its drivers cross their chunks on several threads. Then the mismatching case with the smallest id is narrowed to its counterexample: an
// exhaustive batch's first canonical graph that fails alone, through the same prepare and finish.
template <class Prepare, class Finish>
FuzzDiff drive(const DiffSource& src, std::uint64_t group, Prepare prepare, Finish finish) {
  using Prepared = std::invoke_result_t<Prepare, const DiffCase&>;
  FuzzDiff res;
  std::optional<std::uint64_t> bad_case;  // its index in src
  std::optional<Mismatch> bad_first;
  for (std::uint64_t g0 = 0; g0 < src.size(); g0 += group) {
    const std::uint64_t g1 = std::min(g0 + group, src.size());
    std::vector<std::optional<Prepared>> items(g1 - g0);
    std::exception_ptr error;
#pragma omp parallel for schedule(dynamic, 1) if (items.size() > 1)
    for (std::int64_t i = 0; i < std::int64_t(items.size()); ++i) try {
        items[std::size_t(i)].emplace(prepare(src.get(g0 + std::uint64_t(i))));
      } catch (...) {
#pragma omp critical
        if (!error) error = std::current_exception();
      }
    if (error) std::rethrow_exception(error);
    for (std::size_t i = 0; i < items.size(); ++i) {
      CaseResult r = finish(*items[i]);
      items[i].reset();
      add(res, r.d);
      if (r.first && !bad_case) {
        bad_case = g0 + i;
        bad_first = r.first;
      }
    }
  }
  if (bad_case) {
    const DiffCase c = src.get(*bad_case);
    Counterexample cx;
    cx.case_id = c.id;
    bool found = false;
    for (const DiffCase& mc : src.members(c)) {
      Prepared p = prepare(mc);
      CaseResult r = finish(p);
      if (!r.first) continue;
      cx.graph = mc.id;
      cx.g = mc.g;
      cx.first = *r.first;
      found = true;
      break;
    }
    if (!found) {
      cx.first = *bad_first;
      if (c.graph.empty()) {  // a fuzz graph or a given case: the case itself
        cx.graph = c.id;
        cx.g = c.g;
      } else {  // no graph fails alone: the batch's mismatch, at its graph's eid
        cx.alone = false;
        cx.graph = c.graph[cx.first.eid];
        cx.g = src.graphs()->graph(cx.graph);
        cx.first.eid = c.local[cx.first.eid];
      }
    }
    res.counterexample = std::move(cx);
  }
  std::sort(res.bad_seeds.begin(), res.bad_seeds.end());
  std::sort(res.limited_seeds.begin(), res.limited_seeds.end());
  return res;
}

// Cases per group: enough to keep every thread busy; the GPU's groups hold their references.
std::uint64_t group_size(int per_thread) { return std::uint64_t(per_thread) * std::uint64_t(omp_get_max_threads()); }

// The CPU's whole check of one case: at each window, vstats, then each block's reference and runs.
CaseResult check_cpu(const Run& run, const DiffCase& c) {
  const Profile& prof = run.prof;
  const std::uint64_t m = c.g.src.size();
  CaseResult r(c);
  std::optional<ProfileOracle> oracle;
  if (run.ref == Reference::oracle) oracle.emplace(EdgeStream{c.g.src, c.g.dst, c.g.ts, c.g.n}, prof, Quirks{}, c.amount);
  const GraphIndex index(c.g.src, c.g.dst, c.g.ts, c.g.n, prof.self_loops());
  TriggerSel all;
  all.last = m;
  for (const std::int32_t w : c.windows) {
    if (prof.vstats && oracle) {
      const std::vector<double> want = reference_vstats(&*oracle, index, c.amount, w, m);
      std::vector<double> got(m * kVstatsCols);
      CleanVstats cv(index.view(), c.amount, w);
      cv.rows(all, got.data());
      r.d.vstats_sides += cv.sides();
      r.vstats("cpu", w, got, want);
    }
    for (const ProfileBlock& pb : prof.blocks) {
      const Block b = pb.block;
      const BlockParams p = run.params(b, w);
      const std::size_t nb = block_cols(p);
      std::vector<std::uint32_t> want(m * nb, 0);
      const std::uint64_t stop = stop_eid([&] { reference_rows(oracle ? &*oracle : nullptr, index, pb, p, prof, m, want.data()); });
      r.reference(b, stop, want, nb);
      // A block with a pattern planned for reachability pruning (SPEC v2 §8 S6) runs both ways.
      const bool reach = has(pb, &PatternInfo::cpu_reach), sweep = has(pb, &PatternInfo::cpu_sweep);
      for (int k = 0; k < (reach ? 4 : 2); ++k) {
        if (k == 0 && run.ref == Reference::cpu) continue;  // the reference itself
        const Schedule s = k % 2 == 0 ? Schedule::dynamic : Schedule::sorted;
        BlockParams q = p;
        q.reach_prune = k >= 2;
        const std::string path = std::string("cpu ") + schedule_name(s) + (q.reach_prune ? " reach" : "");
        std::vector<std::uint32_t> got(m * nb, 0);
        const std::uint64_t got_stop =
            stop_eid([&] { mine_cpu(index.view(), pb.patterns, q, prof.visibility, all, s, got.data()); });
        if (r.stops(b, path, w, got_stop, stop)) continue;
        r.rows(b, path, w, got, want, nb);
        if (sweep) r.d.sweep_rows += m;
        if (q.reach_prune) r.d.reach_rows += m;
      }
      if (const PatternInfo* g = sg_gather_of(pb); g && oracle) {
        std::vector<std::uint32_t> want_g(m * nb, 0), got_g(m * nb, 0);
        oracle->mine_sg_gather(p, 0, m, want_g.data());
        mine_cpu(index.view(), std::span(&g, 1), p, prof.visibility, all, Schedule::dynamic, got_g.data());
        r.rows(b, "cpu dynamic sg_gather alone", w, got_g, want_g, nb);
      }
    }
  }
  r.close(c.id);
  return r;
}

}  // namespace

FuzzDiff fuzz_diff_cpu(const Profile& prof, const DiffSource& src, std::span<const std::uint32_t> bins,
                       std::uint64_t work_limit, Reference ref) {
  const Run run{prof, all_fuzz_bins(prof, bins), work_limit, ref};
  return drive(
      src, group_size(16), [&](const DiffCase& c) { return check_cpu(run, c); }, [](CaseResult& r) { return std::move(r); });
}

FuzzDiff fuzz_diff_cuda(const Profile& prof, const DiffSource& src, std::span<const std::uint32_t> bins,
                        std::span<const std::uint32_t> thresholds, std::uint64_t work_limit, CudaOptions opts,
                        Reference ref) {
  return fuzz_diff_hybrid(prof, src, bins, thresholds, {}, work_limit, opts, ref);
}

FuzzDiff fuzz_diff_hybrid(const Profile& prof, const DiffSource& src, std::span<const std::uint32_t> bins,
                          std::span<const std::uint32_t> thresholds, std::span<const std::uint32_t> cpu_thresholds,
                          std::uint64_t work_limit, CudaOptions opts, Reference ref) {
  const Run run{prof, all_fuzz_bins(prof, bins), work_limit, ref};
  const bool hybrid = !cpu_thresholds.empty();
  if (hybrid && (opts.fuse || opts.pack)) throw std::invalid_argument("fuzz-diff: the hybrid neither fuses nor packs");
  std::vector<const ProfileBlock*> blocks;
  for (const ProfileBlock& pb : prof.blocks) blocks.push_back(&pb);
  const std::vector<CudaUnit> units = cuda_units(blocks, opts.fuse);
  CudaMiner miner(hybrid);
  // The plain CUDA runs have one pass per GPU threshold; the hybrid one per (GPU, CPU) pair.
  const std::uint32_t plain[] = {0};
  const std::span<const std::uint32_t> splits = hybrid ? cpu_thresholds : std::span<const std::uint32_t>(plain);

  // A case's host side, built in parallel: its index and the reference at each window.
  struct Item {
    DiffCase c;
    std::unique_ptr<GraphIndex> index;
    std::vector<Want> want;  // per window
  };
  const auto prepare = [&](DiffCase case_) {
    // The index points into the case's edge arrays, which keep their buffers when the item moves.
    Item it{std::move(case_), nullptr, {}};
    const DiffCase& c = it.c;
    std::optional<ProfileOracle> oracle;
    if (ref == Reference::oracle) oracle.emplace(EdgeStream{c.g.src, c.g.dst, c.g.ts, c.g.n}, prof, Quirks{}, c.amount);
    it.index = std::make_unique<GraphIndex>(c.g.src, c.g.dst, c.g.ts, c.g.n, prof.self_loops());
    for (const std::int32_t w : c.windows)
      it.want.push_back(reference_want(run, oracle ? &*oracle : nullptr, *it.index, c, w, !hybrid));
    return it;
  };
  // The GPU's runs of one case, in case order on this thread.
  const auto finish = [&](Item& it) {
    const DiffCase& c = it.c;
    const std::uint64_t m = c.g.src.size();
    CaseResult r(c);
    const DeviceGraph dg(it.index->view());
    TriggerSel all;
    all.last = m;
    for (std::size_t wi = 0; wi < c.windows.size(); ++wi) {
      const std::int32_t w = c.windows[wi];
      const Want& x = it.want[wi];
      for (const ProfileBlock& pb : prof.blocks)
        r.reference(pb.block, x.stop[int(pb.block)], x.rows[int(pb.block)], block_cols(run.bins[int(pb.block)]));
      // One block's rows (or stop eid) at window c.windows[wk] against the reference's.
      const auto check = [&](const ProfileBlock& pb, const std::string& path, std::size_t wk,
                             const std::vector<std::uint32_t>& got, std::uint64_t got_stop) {
        const Block b = pb.block;
        if (r.stops(b, path, c.windows[wk], got_stop, it.want[wk].stop[int(b)])) return false;
        r.rows(b, path, c.windows[wk], got, it.want[wk].rows[int(b)], block_cols(run.bins[int(b)]));
        return true;
      };
      const auto packed = [&](const CudaBlockTiming& t) {
        if (opts.pack) r.d.packed_rows[t.d2h_width == 1 ? 0 : t.d2h_width == 2 ? 1 : 2] += m;
      };
      for (const CudaUnit& unit : units)
        for (std::uint32_t h : thresholds)
          for (std::uint32_t hc : splits) {
            const std::string run_text = hybrid ? "hybrid H=" + threshold_text(h) + " C=" + threshold_text(hc)
                                                : "cuda H=" + threshold_text(h);
            const std::string pack_text = opts.pack ? " packed" : "";
            const std::string path = run_text + pack_text;  // an unfused run's
            std::vector<BlockParams> params;
            for (const ProfileBlock* pb : unit.blocks) {
              BlockParams& p = params.emplace_back(run.params(pb->block, w));
              p.reach_prune = hybrid;  // the hybrid's CPU side prunes where it can (SPEC v2 §8 S6)
            }
            std::vector<std::vector<std::uint32_t>> got(unit.blocks.size());
            for (std::size_t sl = 0; sl < got.size(); ++sl) got[sl].assign(m * block_cols(params[sl]), 0xdeadbeefu);
            // A fused run (no fused pattern counts work, so it never stops) has every member at w, and
            // then, with more than one window, member k at the k-th window after w (cyclically): a
            // profile may give the members different windows, so their starts in the shared scan differ.
            for (std::size_t shift = 0; unit.fusion && shift < std::min<std::size_t>(2, c.windows.size()); ++shift) {
              std::vector<std::size_t> at(got.size());
              std::string windows_text;
              std::vector<const BlockParams*> pp;
              std::vector<std::uint32_t*> outs;
              for (std::size_t sl = 0; sl < got.size(); ++sl) {
                at[sl] = (wi + shift * sl) % c.windows.size();
                params[sl] = run.params(unit.blocks[sl]->block, c.windows[at[sl]]);
                got[sl].assign(m * block_cols(params[sl]), 0xdeadbeefu);
                if (shift) windows_text += (sl ? "," : " at windows ") + std::to_string(c.windows[at[sl]]);
                pp.push_back(&params[sl]);
                outs.push_back(got[sl].data());
              }
              const std::string fused_path = run_text + " fused" + windows_text + pack_text;
              const CudaMiner::FusedTiming t =
                  miner.mine_fused(dg, *unit.fusion, pp, prof.visibility, 0, m, h, outs, opts.pack);
              r.d.light_triggers += t.group.light;
              r.d.heavy_triggers += t.group.heavy;
              for (std::size_t sl = 0; sl < got.size(); ++sl)
                if (check(*unit.blocks[sl], fused_path, at[sl], got[sl], UINT64_MAX)) {
                  r.d.fused_rows += m;
                  packed(t.slots[sl]);
                }
            }
            if (unit.fusion) continue;
            const ProfileBlock& pb = *unit.blocks.front();
            CudaBlockTiming t;
            HybridTiming ht;
            const std::uint64_t got_stop = stop_eid([&] {
              if (hybrid)
                ht = mine_hybrid(it.index->view(), dg, miner, pb.patterns, params[0], prof.visibility, 0, m, hc, h,
                                 Schedule::sorted, got[0].data());
              else
                t = miner.mine(dg, pb.patterns, params[0], prof.visibility, 0, m, h, got[0].data(), opts.pack);
            });
            if (!check(pb, path, wi, got[0], got_stop)) continue;
            if (hybrid) {
              r.d.hybrid_cpu_triggers += ht.cpu_triggers;
              r.d.hybrid_gpu_triggers += ht.gpu_triggers;
              t = ht.gpu;
              if (has(pb, &PatternInfo::cpu_reach)) r.d.reach_rows += ht.cpu_triggers;
            } else {
              packed(t);
            }
            r.d.light_triggers += t.light;
            r.d.heavy_triggers += t.heavy;
          }
      for (const ProfileBlock& pb : prof.blocks)
        if (const PatternInfo* g = sg_gather_of(pb); g && !x.sg_gather.empty())
          for (std::uint32_t h : thresholds) {
            const BlockParams p = run.params(pb.block, w);
            std::vector<std::uint32_t> got(x.sg_gather.size(), 0xdeadbeefu);
            miner.mine(dg, std::span(&g, 1), p, prof.visibility, 0, m, h, got.data(), opts.pack);
            r.rows(pb.block, "cuda H=" + threshold_text(h) + " sg_gather alone", w, got, x.sg_gather, block_cols(p));
          }
      if (prof.vstats) {  // on the GPU, for cuda and the hybrid alike
        std::vector<double> got(m * kVstatsCols);
        CleanVstats cv(it.index->view(), c.amount, w, &dg);
        cv.rows(all, got.data());
        r.d.vstats_sides += cv.sides();
        r.vstats(hybrid ? "hybrid" : "cuda", w, got, x.vstats);
      }
    }
    r.close(c.id);
    return r;
  };
  return drive(src, group_size(2), prepare, finish);
}

namespace {

// The profile as fuzz_diff checks it at window w: every block at the run's parameters (its bins, or
// --bins, and the run's work limit), and vstats at w.
Profile at_window(const Run& run, std::int32_t w) {
  Profile p = run.prof;
  for (ProfileBlock& pb : p.blocks) pb.params = run.params(pb.block, w);
  if (p.vstats) p.vstats->window = w;
  return p;
}

// A chunked run's selections of a case of m triggers: every trigger; the range [m/4, m - m/5) when
// it is neither empty nor every trigger; with `sample`, about 40 % of the triggers, drawn from `seed`.
std::vector<TriggerSel> chunk_selections(std::uint64_t m, std::uint64_t seed, bool sample) {
  const Splits none;
  std::vector<TriggerSel> out = {select_triggers("all", m, none)};
  const std::uint64_t a = m / 4, b = m - m / 5;
  if (a < b && (a > 0 || b < m))
    out.push_back(select_triggers("range:" + std::to_string(a) + ":" + std::to_string(b), m, none));
  if (sample) out.push_back(select_triggers("sample:40:" + std::to_string(seed), m, none));
  return out;
}

// The cells of `rows` (one row of `cols` cells per trigger) at sel's triggers, in row order.
template <class T>
std::vector<T> at_triggers(const std::vector<T>& rows, std::size_t cols, const TriggerSel& sel) {
  std::vector<T> out;
  out.reserve(sel.rows() * cols);
  for (std::uint64_t r = 0; r < sel.rows(); ++r) {
    const auto at = rows.begin() + std::ptrdiff_t(sel.eid(r) * cols);
    out.insert(out.end(), at, at + std::ptrdiff_t(cols));
  }
  return out;
}

// A MemorySink that names the stream it opened last: the block a driver error belongs to.
class LastOpened final : public BlockSink {
 public:
  std::unique_ptr<BlockStream> open(const StreamSpec& spec) override {
    last = spec.name;
    return mem.open(spec);
  }
  MemorySink mem;
  std::string last;
};

// What a stream's tally should be for the reference's cells `want` of `rows` rows put in chunks of
// k: the rows, ⌈rows / k⌉ puts, the cells' sum (uint32 only) and FNV-1a over their bytes. Empty
// when it is.
template <class T>
std::string tally_diff(const StreamTally& t, const std::vector<T>& want, std::uint64_t rows, std::uint64_t k) {
  std::uint64_t sum = 0;
  if constexpr (std::is_same_v<T, std::uint32_t>)
    for (const std::uint32_t v : want) sum += v;
  const std::uint64_t puts = rows == 0 ? 0 : (rows - 1) / k + 1;
  const std::uint64_t fnv = fnv1a_bytes(want.data(), want.size() * sizeof(T));
  std::string out;
  const auto diff = [&](const char* what, std::uint64_t got, std::uint64_t exp) {
    if (got != exp) out += (out.empty() ? "" : ", ") + std::string(what) + " " + std::to_string(got) + ", want " +
                           std::to_string(exp);
  };
  diff("rows", t.rows, rows);
  diff("puts", t.chunks, puts);
  diff("fnv1a", t.fnv1a, fnv);
  if constexpr (std::is_same_v<T, std::uint32_t>) diff("instances", t.instances, sum);
  return out.empty() ? out : "stream tally: " + out;
}

// One chunk-driver run over `sel` against the reference at window w (`x`; m triggers), as
// fuzz_diff_chunks says.
void chunked_run(CaseResult& r, ChunkDriver& d, const Profile& pw, const Want& x, std::uint64_t m, std::int32_t w,
                 const TriggerSel& sel, std::uint64_t k, const std::string& path) {
  std::vector<const ProfileBlock*> blocks;
  for (const ProfileBlock& pb : pw.blocks) blocks.push_back(&pb);
  const std::uint64_t first = sel.rows() ? sel.eid(0) : 0;
  const auto block_of = [&](const std::string& name) -> const Block* {
    for (const ProfileBlock& pb : pw.blocks)
      if (block_name(pb.block) == name) return &pb.block;
    return nullptr;  // vstats
  };
  LastOpened sink;
  std::vector<BlockRun> runs;
  std::uint64_t got_stop = UINT64_MAX;
  try {
    runs = d.run(blocks, pw.vstats.has_value(), sel, sink);
  } catch (const WorkLimitExceeded& e) {
    got_stop = e.eid;
  } catch (const std::logic_error& e) {
    const Block* b = sink.last.empty() ? &pw.blocks.front().block : block_of(sink.last);
    r.note(b, path, w, first, std::string("the chunk driver threw: ") + e.what());
    return;
  }
  const bool all = !sel.sampled && sel.first == 0 && sel.last == m;
  const std::uint64_t want_stop = x.stop[int(Block::lcyc)];
  if (got_stop != UINT64_MAX || want_stop != UINT64_MAX) {
    // R11: over every trigger the run stops where the reference does, before any stream finished. A
    // sub-selection of a case the reference stopped may or may not reach the limit: inconclusive.
    if (all || want_stop == UINT64_MAX) r.stops(Block::lcyc, path, w, got_stop, want_stop);
    // A run that stops compares no rows of any block, so the case is not completed for them (R11).
    // A legacy profile keeps D-39's stop, after the blocks mined before lcyc, so it may keep their
    // rows; a clean one may not.
    for (const ProfileBlock& pb : pw.blocks) r.limited[int(pb.block)] = true;
    r.vstats_limited = r.vstats_limited || pw.vstats.has_value();
    if (pw.family == Family::clean && got_stop != UINT64_MAX && (!sink.mem.u32.empty() || !sink.mem.f64.empty())) {
      const Block lcyc = Block::lcyc;
      r.note(&lcyc, path, w, got_stop,
             "an aborted run kept the rows of " + (sink.mem.u32.empty() ? "vstats" : sink.mem.u32.begin()->first));
    }
    return;
  }
  if (runs.size() != blocks.size() + pw.vstats.has_value()) {
    r.note(&pw.blocks.front().block, path, w, first,
           std::to_string(runs.size()) + " streams for " + std::to_string(blocks.size()) + " blocks");
    return;
  }
  const std::uint64_t n = sel.rows();
  std::vector<const FusionInfo*> fusions;  // the fused units counted, for their kernels' triggers
  for (const BlockRun& br : runs) {
    FuzzDiff::ChunkHits& hit = br.block ? r.d.chunk_hits[int(br.block->block)] : r.d.vstats_chunk_hits;
    ++hit.streams;
    hit.puts += br.out.chunks;
    hit.multi += br.out.chunks > 1;
    if (!br.block) {  // vstats
      const std::vector<double> want = at_triggers(x.vstats, kVstatsCols, sel);
      const auto it = sink.mem.f64.find("vstats");
      if (it == sink.mem.f64.end() || it->second.size() != want.size()) {
        r.note(nullptr, path, w, first, "no vstats rows, or the wrong number of cells");
        continue;
      }
      r.vstats(path, w, it->second, want, &sel);
      r.d.vstats_sides += br.vstats_sides;
      if (std::string t = tally_diff(br.out, want, n, k); !t.empty()) r.note(nullptr, path, w, first, std::move(t));
      continue;
    }
    const ProfileBlock& pb = *br.block;
    const Block b = pb.block;
    const std::size_t cols = block_cols(br.params);
    const std::vector<std::uint32_t> want = at_triggers(x.rows[int(b)], cols, sel);
    const auto it = sink.mem.u32.find(br.name);
    if (it == sink.mem.u32.end() || it->second.size() != want.size()) {
      r.note(&b, path, w, first, "no rows, or the wrong number of cells");
      continue;
    }
    r.rows(b, path, w, it->second, want, cols, &sel);
    if (std::string t = tally_diff(br.out, want, n, k); !t.empty()) r.note(&b, path, w, first, std::move(t));
    // The optimised paths the run took (SPEC v3 §4.4 item 6).
    if (br.whole) {
      ++r.d.whole_streams;
      r.d.sweep_rows += n;
    }
    if (br.fusion) {
      ++r.d.fused_streams;
      r.d.fused_rows += n;
    }
    if (d.options().backend == Backend::cuda) {
      if (!br.fusion || std::find(fusions.begin(), fusions.end(), br.fusion) == fusions.end()) {
        r.d.light_triggers += br.cuda.light;
        r.d.heavy_triggers += br.cuda.heavy;
        if (br.fusion) fusions.push_back(br.fusion);
      }
      if (d.options().pack) r.d.packed_rows[br.cuda.d2h_width == 1 ? 0 : br.cuda.d2h_width == 2 ? 1 : 2] += n;
    }
    if (d.options().backend == Backend::hybrid) {
      r.d.hybrid_cpu_triggers += br.hybrid.cpu_triggers;
      r.d.hybrid_gpu_triggers += br.hybrid.gpu_triggers;
      r.d.light_triggers += br.hybrid.gpu.light;
      r.d.heavy_triggers += br.hybrid.gpu.heavy;
      if (has(pb, &PatternInfo::cpu_reach)) r.d.reach_rows += br.hybrid.cpu_triggers;
    }
  }
  if (sel.sampled) ++r.d.sampled_runs;
  else if (!all) ++r.d.range_runs;
}

std::string chunk_path(const MineOptions& o, std::uint64_t k, const TriggerSel& sel) {
  std::string p = backend_name(o.backend);
  if (o.backend == Backend::cpu) p += std::string(" ") + schedule_name(o.schedule);
  if (o.backend != Backend::cpu) p += " H=" + threshold_text(o.heavy_threshold);
  if (o.backend == Backend::hybrid) p += " C=" + threshold_text(o.cpu_threshold);
  p += " chunks of " + std::to_string(k);
  if (o.fuse) p += " fused";
  if (o.pack) p += " packed";
  return p + ", " + sel.spec;
}

}  // namespace

FuzzDiff fuzz_diff_chunks(const Profile& prof, const DiffSource& src, std::span<const std::uint32_t> bins,
                          Backend backend, std::span<const std::uint64_t> chunk_rows,
                          std::span<const std::uint32_t> thresholds, std::span<const std::uint32_t> cpu_thresholds,
                          std::uint64_t work_limit, CudaOptions opts, Reference ref) {
  const bool gpu = backend == Backend::cuda || backend == Backend::hybrid;
  if (!gpu && backend != Backend::cpu) throw std::invalid_argument("fuzz-diff --chunk-rows: the backend is cpu, cuda or hybrid");
  if (chunk_rows.empty() || std::find(chunk_rows.begin(), chunk_rows.end(), 0) != chunk_rows.end())
    throw std::invalid_argument("fuzz-diff --chunk-rows: one or more chunk sizes, each > 0");
  if (gpu && thresholds.empty()) throw std::invalid_argument("fuzz-diff --chunk-rows: no light/heavy threshold");
  if (backend == Backend::hybrid && cpu_thresholds.empty())
    throw std::invalid_argument("fuzz-diff --chunk-rows: no CPU threshold for the hybrid");
  if ((opts.fuse || opts.pack) && backend != Backend::cuda)
    throw std::invalid_argument("fuzz-diff: --fuse and --pack are for the cuda backend");
  const Run run{prof, all_fuzz_bins(prof, bins), work_limit, ref};
  const std::uint32_t none[] = {0};
  const std::span<const std::uint32_t> hs = gpu ? thresholds : std::span<const std::uint32_t>(none);
  const std::span<const std::uint32_t> cs = backend == Backend::hybrid ? cpu_thresholds : std::span<const std::uint32_t>(none);
  // Every chunked run of one case: at each window, chunk size and threshold, over its selections.
  const auto check = [&](const DiffCase& c, const std::vector<Want>& want) {
    const std::uint64_t m = c.g.src.size();
    CaseResult r(c);
    const std::vector<TriggerSel> sels = chunk_selections(m, c.id, backend == Backend::cpu);
    for (std::size_t wi = 0; wi < c.windows.size(); ++wi) {
      const std::int32_t w = c.windows[wi];
      const Profile pw = at_window(run, w);
      for (const ProfileBlock& pb : prof.blocks)
        r.reference(pb.block, want[wi].stop[int(pb.block)], want[wi].rows[int(pb.block)],
                    block_cols(run.bins[int(pb.block)]));
      for (const std::uint64_t k : chunk_rows)
        for (const std::uint32_t h : hs)
          for (const std::uint32_t hc : cs) {
            MineOptions o;
            o.backend = backend;
            o.schedule = backend == Backend::hybrid ? Schedule::sorted : Schedule::dynamic;
            o.heavy_threshold = h;
            o.cpu_threshold = hc;
            o.reach_prune = backend == Backend::hybrid;  // as fuzz_diff_hybrid's CPU side
            o.fuse = opts.fuse;
            o.pack = opts.pack;
            o.work_limit = work_limit;
            o.chunk_rows = k;
            ChunkDriver d(EdgeStream{c.g.src, c.g.dst, c.g.ts, c.g.n}, c.amount, pw, o);
            for (const TriggerSel& sel : sels) chunked_run(r, d, pw, want[wi], m, w, sel, k, chunk_path(o, k, sel));
          }
    }
    r.close(c.id);
    return r;
  };
  // A case's whole check, the cases in parallel, on every backend: each case's drivers own their
  // device graph, miner and pinned buffers, so the GPU's calls from several host threads overlap.
  const auto prepare = [&](const DiffCase& c) {
    std::optional<ProfileOracle> oracle;
    if (ref == Reference::oracle) oracle.emplace(EdgeStream{c.g.src, c.g.dst, c.g.ts, c.g.n}, prof, Quirks{}, c.amount);
    const GraphIndex index(c.g.src, c.g.dst, c.g.ts, c.g.n, prof.self_loops());
    std::vector<Want> want;
    for (const std::int32_t w : c.windows) want.push_back(reference_want(run, oracle ? &*oracle : nullptr, index, c, w, false));
    return check(c, want);
  };
  return drive(src, group_size(16), prepare, [](CaseResult& r) { return std::move(r); });
}

std::string edge_text(const EdgeList& g, std::uint64_t e) {
  return std::to_string(g.src[e]) + "->" + std::to_string(g.dst[e]) + "@" + std::to_string(g.ts[e]);
}

std::string counterexample_text(const Counterexample& cx, const DiffSource& src) {
  const Mismatch& f = cx.first;
  const EdgeList& g = cx.g;
  const std::uint64_t m = g.src.size();
  const std::string what = src.kind() == DiffSource::Kind::exhaustive
                               ? "graph " + std::to_string(cx.graph) + " of the bound (batch " + std::to_string(cx.case_id) + ")"
                           : src.kind() == DiffSource::Kind::given ? "case " + std::to_string(cx.graph)
                                                                   : "seed " + std::to_string(cx.graph);
  std::string out = "  counterexample: " + what + ", m = " + std::to_string(m) + ", n = " + std::to_string(g.n) +
                    (cx.alone ? "" : " (fails only inside its batch: no graph of the batch fails alone)") + "\n";
  out += "    block " + f.block + ", window " + std::to_string(f.window) + ", path " + f.path + "\n";
  if (m <= 64) {
    out += "    edges (eid: u->v@t):";
    for (std::uint64_t e = 0; e < m; ++e) out += (e ? "  " : " ") + std::to_string(e) + ": " + edge_text(g, e);
    out += "\n";
  }
  const auto stop = [](std::uint64_t x) { return x == UINT64_MAX ? std::string("completed") : std::to_string(x); };
  if (f.got_stop != UINT64_MAX || f.want_stop != UINT64_MAX)
    return out + "    work limit: got " + stop(f.got_stop) + ", want " + stop(f.want_stop) + " (stop eids)\n";
  if (!f.note.empty()) return out + "    " + f.note + " (the selection's first trigger: eid " + std::to_string(f.eid) + ")\n";
  const std::string trigger = "    trigger eid " + std::to_string(f.eid) + (f.eid < m ? " (" + edge_text(g, f.eid) + ")" : "");
  if (f.block == "vstats") {
    const std::vector<std::string> names = clean_vstats_column_names();
    out += trigger + ", the columns that differ:\n";
    for (std::size_t k = 0; k < f.got_vstats.size(); ++k)
      if (std::memcmp(&f.got_vstats[k], &f.want_vstats[k], sizeof(double)) != 0) {
        char buf[256];
        std::snprintf(buf, sizeof buf, "      %s: got %.17g, want %.17g\n", names[k].c_str(), f.got_vstats[k],
                      f.want_vstats[k]);
        out += buf;
      }
    return out;
  }
  const auto row = [](const std::vector<std::uint32_t>& r) {
    std::string t = "[";
    for (std::size_t k = 0; k < r.size(); ++k) t += (k ? " " : "") + std::to_string(r[k]);
    return t + "]";
  };
  return out + trigger + ": got " + row(f.got) + ", want " + row(f.want) + "\n";
}

}  // namespace blz

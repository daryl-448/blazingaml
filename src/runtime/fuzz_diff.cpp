#include "runtime/fuzz_diff.hpp"

#include <algorithm>
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

DiffSource DiffSource::exhaustive(ExhaustiveBound b, std::uint64_t batch_edges) {
  DiffSource s;
  s.graphs_.emplace(b);
  s.batch_edges_ = std::max<std::uint64_t>(batch_edges, 1);
  s.batch_first_.assign(b.m_max + 1, 0);
  s.per_batch_.assign(b.m_max, 0);
  for (std::uint32_t m = 1; m <= b.m_max; ++m) {
    const std::uint64_t graphs = s.graphs_->first(m + 1) - s.graphs_->first(m);
    s.per_batch_[m - 1] = std::max<std::uint64_t>(1, s.batch_edges_ / m);
    s.batch_first_[m] = s.batch_first_[m - 1] + (graphs + s.per_batch_[m - 1] - 1) / s.per_batch_[m - 1];
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
  DiffCase c;
  if (!graphs_) {
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
  if (!graphs_ || c.graph.empty()) return {};
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

// One case's tally (a FuzzDiff of that case alone) and its first mismatch.
struct CaseResult {
  FuzzDiff d;
  std::optional<Mismatch> first;
  std::array<bool, kAllBlocks.size()> limited{};

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

  // A block's rows, `got` against `want`, row by row.
  void rows(Block b, const std::string& path, std::int32_t w, std::span<const std::uint32_t> got,
            std::span<const std::uint32_t> want, std::size_t cols) {
    const std::uint64_t m = want.size() / cols;
    d.row_checks += m;
    for (std::uint64_t e = 0; e < m; ++e) {
      const auto g = got.subspan(e * cols, cols), x = want.subspan(e * cols, cols);
      if (std::equal(g.begin(), g.end(), x.begin())) continue;
      ++d.mismatched_rows[int(b)];
      if (!first)
        first = Mismatch{std::string(block_name(b)), path, w, e, {g.begin(), g.end()}, {x.begin(), x.end()}, {}, {}};
    }
  }

  // Where either side stopped at the work limit (SPEC v3 R11), the stop eids are compared instead of
  // the rows; returns whether that was so.
  bool stops(Block b, const std::string& path, std::int32_t w, std::uint64_t got, std::uint64_t want) {
    if (got == UINT64_MAX && want == UINT64_MAX) return false;
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

  // vstats rows, bit for bit, and the reference's nonzero rows.
  void vstats(const std::string& path, std::int32_t w, const std::vector<double>& got, const std::vector<double>& want) {
    const std::uint64_t m = want.size() / kVstatsCols;
    ++d.vstats_graphs;
    d.row_checks += m;
    for (std::uint64_t r = 0; r < m; ++r) {
      const double* g = got.data() + r * kVstatsCols;
      const double* x = want.data() + r * kVstatsCols;
      d.vstats_nonzero_rows += std::any_of(x, x + kVstatsCols, [](double v) { return v != 0; });
      if (std::memcmp(g, x, kVstatsCols * sizeof(double)) == 0) continue;
      ++d.vstats_mismatched_rows;
      if (!first) {
        first = Mismatch{"vstats", path, w, r, {}, {}, {g, g + kVstatsCols}, {x, x + kVstatsCols}};
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
  }
  a.hits += b.hits;
  a.vstats_graphs += b.vstats_graphs;
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
  a.bad_seeds.insert(a.bad_seeds.end(), b.bad_seeds.begin(), b.bad_seeds.end());
  a.limited_seeds.insert(a.limited_seeds.end(), b.limited_seeds.begin(), b.limited_seeds.end());
}

// The graph loop of every backend, and so the one owner of the comparison's bookkeeping. The cases
// run in groups: `prepare` runs on each case of a group in parallel (the CPU's whole check, or the
// GPU's reference), then `finish` on each in case order on this thread (the GPU's runs), which
// returns the case's CaseResult. The oracle's and the drivers' own parallel loops inside `prepare`
// run nested, i.e. on one thread each. Then the mismatching case with the smallest id is narrowed to
// its counterexample: an exhaustive batch's first canonical graph that fails alone, through the same
// prepare and finish.
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
#pragma omp parallel for schedule(dynamic, 1)
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
      if (c.graph.empty()) {  // a fuzz graph: the case itself
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
  struct Want {
    std::array<std::vector<std::uint32_t>, kAllBlocks.size()> rows;
    std::array<std::uint64_t, kAllBlocks.size()> stop;  // the reference's stop eid, or UINT64_MAX
    std::vector<double> vstats;
  };
  struct Item {
    DiffCase c;
    std::unique_ptr<GraphIndex> index;
    std::vector<Want> want;  // per window
  };
  const auto prepare = [&](DiffCase case_) {
    // The index points into the case's edge arrays, which keep their buffers when the item moves.
    Item it{std::move(case_), nullptr, {}};
    const DiffCase& c = it.c;
    const std::uint64_t m = c.g.src.size();
    std::optional<ProfileOracle> oracle;
    if (ref == Reference::oracle) oracle.emplace(EdgeStream{c.g.src, c.g.dst, c.g.ts, c.g.n}, prof, Quirks{}, c.amount);
    it.index = std::make_unique<GraphIndex>(c.g.src, c.g.dst, c.g.ts, c.g.n, prof.self_loops());
    for (const std::int32_t w : c.windows) {
      Want& x = it.want.emplace_back();
      if (prof.vstats) x.vstats = reference_vstats(oracle ? &*oracle : nullptr, *it.index, c.amount, w, m);
      x.stop.fill(UINT64_MAX);
      for (const ProfileBlock& pb : prof.blocks) {
        const BlockParams p = run.params(pb.block, w);
        x.rows[int(pb.block)].assign(m * block_cols(p), 0);
        x.stop[int(pb.block)] = stop_eid(
            [&] { reference_rows(oracle ? &*oracle : nullptr, *it.index, pb, p, prof, m, x.rows[int(pb.block)].data()); });
      }
    }
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
            std::string path = hybrid ? "hybrid H=" + threshold_text(h) + " C=" + threshold_text(hc)
                                      : "cuda H=" + threshold_text(h);
            if (unit.fusion) path += " fused";
            if (opts.pack) path += " packed";
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
              std::string fused_path = path;
              std::vector<const BlockParams*> pp;
              std::vector<std::uint32_t*> outs;
              for (std::size_t sl = 0; sl < got.size(); ++sl) {
                at[sl] = (wi + shift * sl) % c.windows.size();
                params[sl] = run.params(unit.blocks[sl]->block, c.windows[at[sl]]);
                got[sl].assign(m * block_cols(params[sl]), 0xdeadbeefu);
                if (shift) fused_path += (sl ? "," : " at windows ") + std::to_string(c.windows[at[sl]]);
                pp.push_back(&params[sl]);
                outs.push_back(got[sl].data());
              }
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

}  // namespace blz

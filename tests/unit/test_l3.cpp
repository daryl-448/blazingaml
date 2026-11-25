// L3 (SPEC v3 §4.3): RapidCheck properties for what L2's exhaustive bound (m <= 5, n <= 5, t = 4;
// m = 6 on 4 nodes) cannot reach: cycles and chains at k_max, saturation (R10), packing widths and
// chunk boundaries, timestamps at the int32 limits, and graphs alone, past the bound, through every
// path. Rows are compared by fuzz_diff alone (DiffSource::given) against the profile's oracle; a
// failure shrinks to a small graph and prints as `blz fuzz-diff` prints a counterexample.
//
// Seeds: a property's seed is a hash of its test name, so ctest's runs (label fast) repeat. The
// environment changes that: BLZ_L3_SEED=<n> or BLZ_L3_SEED=random, and BLZ_L3_SCALE=<k> for k times
// the cases (ctest's l3_random, label verify, runs both). A failure prints the line that reruns it:
//   BLZ_L3_SEED=<seed> BLZ_L3_SCALE=<k> build/<preset>/tests/blz_unit_tests "<test name>"
#include <omp.h>
#include <rapidcheck.h>
#include <rapidcheck/detail/TestListenerAdapter.h>

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/interfaces/catch_interfaces_capture.hpp>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "graph/index.hpp"
#include "oracle/profile_oracle.hpp"
#include "runtime/cpu_backend.hpp"
#include "runtime/fuzz_diff.hpp"
#if defined(BLZ_WITH_CUDA)
#include "runtime/cuda_backend.hpp"
#include "runtime/hybrid.hpp"
#endif

using namespace blz;

namespace {

constexpr std::uint32_t kMax = UINT32_MAX;
constexpr std::int64_t kTsMax = INT32_MAX, kTsMin = INT32_MIN;

// ---- The runner ---------------------------------------------------------------------------------

std::uint64_t name_seed(const std::string& s) {  // FNV-1a
  std::uint64_t h = 0xcbf29ce484222325ULL;
  for (const unsigned char c : s) h = (h ^ c) * 0x100000001b3ULL;
  return h;
}

struct ShrinkTries : rc::detail::TestListenerAdapter {
  std::uint64_t tries = 0;
  void onShrinkTried(const rc::detail::CaseDescription&, bool) override { ++tries; }
};

// Shrinking stops kShrinkSeconds after it starts, at the smallest failing case so far: a case of
// 65,536 cycles costs the oracle a tenth of a second a try, and its shrinking thousands of tries.
constexpr int kShrinkSeconds = 60;
struct ShrinkClock {
  std::optional<std::chrono::steady_clock::time_point> start;
  bool stopped = false;
};
using Described = rc::Shrinkable<rc::detail::CaseDescription>;
Described timed(Described s, std::shared_ptr<ShrinkClock> clock) {
  return rc::shrinkable::mapShrinks(std::move(s), [clock](rc::Seq<Described>&& shrinks) {
    if (!clock->start) clock->start = std::chrono::steady_clock::now();
    const auto more = [clock](const Described&) {
      clock->stopped =
          clock->stopped || std::chrono::steady_clock::now() - *clock->start > std::chrono::seconds(kShrinkSeconds);
      return !clock->stopped;
    };
    return rc::seq::map(rc::seq::takeWhile(std::move(shrinks), more),
                        [clock](Described&& d) { return timed(std::move(d), clock); });
  });
}

// Runs `property` on `cases` cases (times BLZ_L3_SCALE) from the test's seed. A failure shrinks for up
// to kShrinkSeconds, then FAILs with RapidCheck's report (the shrunk case and the property's message)
// and the command that reruns it.
template <class Property>
void check(int cases, Property&& property) {
  const std::string name = Catch::getResultCapture().getCurrentTestName();
  std::uint64_t seed = name_seed(name);
  if (const char* s = std::getenv("BLZ_L3_SEED"))
    seed = std::string(s) == "random" ? (std::uint64_t(std::random_device{}()) << 32) ^ std::random_device{}()
                                      : std::strtoull(s, nullptr, 0);
  int scale = 1;
  if (const char* s = std::getenv("BLZ_L3_SCALE")) scale = std::max(1, std::atoi(s));
  rc::detail::TestParams params;
  params.seed = seed;
  params.maxSuccess = cases * scale;
  params.maxSize = 100;
  ShrinkTries listener;
  // A case runs alone, so fuzz_diff gives its drivers the whole OpenMP team: two threads cross the
  // drivers' chunks without twelve threads' fork cost on every small loop.
  const int threads = omp_get_max_threads();
  omp_set_num_threads(std::min(threads, 2));
  const rc::detail::Property prop = rc::detail::toProperty(std::forward<Property>(property));
  const auto clock = std::make_shared<ShrinkClock>();
  const rc::detail::Property timed_prop(
      [prop, clock](const rc::Random& r, int size) { return timed(prop(r, size), clock); });
  const rc::detail::TestResult result = rc::detail::checkProperty(timed_prop, {name, name}, params, listener, {});
  omp_set_num_threads(threads);
  std::ostringstream report;
  rc::detail::printResultMessage(result, report);
  if (result.is<rc::detail::SuccessResult>()) {
    std::printf("%s: seed %llu, %s\n", name.c_str(), (unsigned long long)seed, report.str().c_str());
    return;
  }
  FAIL(report.str() << "\n\n"
                    << listener.tries << " shrinks tried"
                    << (clock->stopped ? ", then shrinking stopped at " + std::to_string(kShrinkSeconds) + " s" : "")
                    << "\nrerun: BLZ_L3_SEED=" << seed << " BLZ_L3_SCALE=" << scale << " blz_unit_tests \"" << name
                    << "\"");
}

// ---- Cases --------------------------------------------------------------------------------------

struct Rng {  // SplitMix64, seeded from RapidCheck's Random
  std::uint64_t s;
  std::uint64_t next() {
    std::uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }
  std::uint64_t below(std::uint64_t k) { return next() % k; }
  std::int64_t in(std::int64_t a, std::int64_t b) { return a + std::int64_t(below(std::uint64_t(b - a) + 1)); }
  bool chance(int pct) { return std::int64_t(below(100)) < pct; }
};

// One graph alone, as fuzz_diff checks it, with the settings its paths take.
struct Case {
  EdgeList g;                         // stream order
  std::vector<std::int32_t> windows;  // every block, and vstats, at each
  std::vector<double> amount;         // per edge, for vstats
  std::vector<std::uint32_t> bins;    // every histogram block's; empty: each block's own
  std::uint32_t heavy = 1;            // cuda: a light/heavy threshold, besides 0 and all
  std::uint32_t cpu = 1;              // the hybrid's CPU threshold
  std::uint32_t gpu = 0;              // which CUDA runs, on which profile, at which thresholds: see diff()
  std::uint64_t work_limit = kDefaultWorkLimit;
  // What the generator built (path hits); a shrunk case has none.
  std::vector<std::string> tags;
  std::uint32_t plant_eid = UINT32_MAX;  // the trigger that closes or ends a complete planted instance
  Block plant_block = Block::tcyc;
  std::uint32_t plant_k = 0;  // its edges
};

void showValue(const Case& c, std::ostream& os) {
  os << "n " << c.g.n << ", m " << c.g.src.size() << ", windows";
  for (const std::int32_t w : c.windows) os << ' ' << w;
  os << ", bins";
  if (c.bins.empty()) os << " (each block's own)";
  for (const std::uint32_t b : c.bins) os << ' ' << b;
  os << ", cuda H " << c.heavy << " (runs " << c.gpu << "), hybrid C " << c.cpu << ", work limit " << c.work_limit;
  for (const std::string& t : c.tags) os << "; " << t;
  os << "\nedges (eid: u->v@t):";
  for (std::uint64_t e = 0; e < c.g.src.size(); ++e) os << (e ? "  " : " ") << e << ": " << edge_text(c.g, e);
  os << "\namounts:";
  for (const double a : c.amount) os << ' ' << a;
}

// A graph under construction: any node ids, int64 times.
struct Draft {
  std::uint32_t n = 0;
  std::vector<std::uint32_t> src, dst;
  std::vector<std::int64_t> ts;
  std::uint32_t mark = UINT32_MAX;  // one edge to follow through the sort (a planted trigger)

  std::uint32_t add(std::uint32_t u, std::uint32_t v, std::int64_t t) {
    src.push_back(u);
    dst.push_back(v);
    ts.push_back(t);
    n = std::max(n, std::max(u, v) + 1);
    return std::uint32_t(src.size() - 1);
  }
  std::uint64_t m() const { return src.size(); }
  std::int64_t span() const { return m() ? ts.back() - ts.front() : 0; }  // in stream order

  void sort() {  // stream order: a stable sort by ts (SPEC §3.3)
    std::vector<std::uint32_t> order(m());
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(), [&](std::uint32_t a, std::uint32_t b) { return ts[a] < ts[b]; });
    Draft s;
    s.n = n;
    for (const std::uint32_t e : order) {
      if (e == mark) s.mark = std::uint32_t(s.m());
      s.src.push_back(src[e]);
      s.dst.push_back(dst[e]);
      s.ts.push_back(ts[e]);
    }
    *this = std::move(s);
  }
};

// Node ids in first-appearance order over src0, dst0, src1, ... (L2's canonical labels), n = the
// nodes used.
template <class G>
void canonical_labels(G& g) {
  std::map<std::uint32_t, std::uint32_t> id;
  for (std::uint64_t e = 0; e < g.src.size(); ++e)
    for (std::uint32_t* x : {&g.src[e], &g.dst[e]}) *x = id.try_emplace(*x, std::uint32_t(id.size())).first->second;
  g.n = std::uint32_t(id.size());
}

// The labels L2 never has (SPEC v3 §4.3, a graph's position): node ids as drawn, canonical, reversed
// or shuffled, then with isolated nodes before them (node 0 unused) or after them.
void relabel(Draft& d, Rng& r) {
  switch (r.below(4)) {
    case 0:
      break;
    case 1:
      canonical_labels(d);
      break;
    case 2:
      for (std::uint64_t e = 0; e < d.m(); ++e) d.src[e] = d.n - 1 - d.src[e], d.dst[e] = d.n - 1 - d.dst[e];
      break;
    default: {
      std::vector<std::uint32_t> p(d.n);
      std::iota(p.begin(), p.end(), 0u);
      for (std::uint32_t i = d.n; i > 1; --i) std::swap(p[i - 1], p[r.below(i)]);
      for (std::uint64_t e = 0; e < d.m(); ++e) d.src[e] = p[d.src[e]], d.dst[e] = p[d.dst[e]];
    }
  }
  if (r.chance(20)) {
    const std::uint32_t k = std::uint32_t(r.in(1, 3));
    for (std::uint64_t e = 0; e < d.m(); ++e) d.src[e] += k, d.dst[e] += k;
    d.n += k;
  }
  if (r.chance(20)) d.n += std::uint32_t(r.in(1, 3));
}

// Every window that matters to a graph of this span (R2: the window (t - delta, t], the legacy
// [t - delta, t]; SPEC v3 §4.3: delta near the span): 1, the span and its neighbours, a gap between
// two edges, the profiles' windows, and the largest.
std::vector<std::int64_t> windows_near(const Draft& d, Rng& r) {
  const std::int64_t s = d.span();
  std::vector<std::int64_t> w = {1, s - 1, s, s + 1, 21600, 86400, kTsMax - 1, kTsMax};
  if (d.m() > 1) {
    std::uint64_t a = r.below(d.m()), b = r.below(d.m());
    w.push_back(d.ts[std::max(a, b)] - d.ts[std::min(a, b)]);
  }
  return w;
}

// The case of a sorted draft: one or two of `windows` (clamped to [1, INT32_MAX]), the amounts,
// sometimes a subset of the profiles' bins for every block, and the CUDA thresholds.
Case finish(const Draft& d, Rng& r, const std::vector<std::int64_t>& windows) {
  Case c;
  c.g.n = d.n;
  c.g.src = d.src;
  c.g.dst = d.dst;
  for (const std::int64_t t : d.ts) {
    if (t < kTsMin || t > kTsMax) throw std::logic_error("L3: a generated time past int32");
    c.g.ts.push_back(std::int32_t(t));
  }
  for (int k = int(r.in(1, 2)); k > 0; --k) {
    const std::int32_t w = std::int32_t(std::clamp<std::int64_t>(windows[r.below(windows.size())], 1, kTsMax));
    if (std::find(c.windows.begin(), c.windows.end(), w) == c.windows.end()) c.windows.push_back(w);
  }
  c.amount = fuzz_amounts(r.next(), d.m());
  if (r.chance(25)) {  // the profiles' bins: 2..10 and clean sg's 13, 16, 20
    for (const std::uint32_t b : {2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 13u, 16u, 20u})
      if (r.chance(40)) c.bins.push_back(b);
  }
  c.heavy = std::uint32_t(r.in(1, 64));
  c.cpu = std::uint32_t(r.in(1, 64));
  c.gpu = std::uint32_t(r.below(36));
  c.plant_eid = d.mark;
  return c;
}

// Shrinks: drop windows, remove runs of edges (the longest first), canonical labels, times shifted
// to 0 and then compressed to their ranks, each block's own bins, smaller windows and thresholds.
// Each keeps stream order; a shrunk case plants nothing.
rc::Seq<Case> shrinks(const Case& c0) {
  const auto c = std::make_shared<Case>(c0);
  c->tags.clear();
  c->plant_eid = UINT32_MAX;
  const auto edit = [c](auto f) {
    Case x = *c;
    f(x);
    return x;
  };
  const std::uint64_t m = c->g.src.size();
  rc::Seq<Case> fewer_windows;
  if (c->windows.size() > 1)
    fewer_windows = rc::seq::map(rc::seq::range<std::size_t>(0, c->windows.size()),
                                 [=](std::size_t i) { return edit([&](Case& x) { x.windows = {c->windows[i]}; }); });
  std::vector<std::pair<std::uint64_t, std::uint64_t>> runs;  // (first, length)
  for (std::uint64_t len = m / 2; len > 0; len /= 2)
    for (std::uint64_t a = 0; a < m; a += len) runs.push_back({a, std::min(len, m - a)});
  if (m == 1) runs.push_back({0, 1});
  rc::Seq<Case> fewer_edges =
      rc::seq::map(rc::seq::fromContainer(runs), [=](std::pair<std::uint64_t, std::uint64_t> run) {
        return edit([&](Case& x) {
          const auto cut = [&](auto& v) { v.erase(v.begin() + run.first, v.begin() + run.first + run.second); };
          cut(x.g.src), cut(x.g.dst), cut(x.g.ts), cut(x.amount);
        });
      });
  std::vector<Case> simpler;
  Case x = *c;
  canonical_labels(x.g);
  if (x.g.src != c->g.src || x.g.dst != c->g.dst || x.g.n != c->g.n) simpler.push_back(x);
  if (m > 0 && c->g.ts.front() != 0 && std::int64_t(c->g.ts.back()) - c->g.ts.front() <= kTsMax) {
    x = *c;
    for (std::int32_t& t : x.g.ts) t = std::int32_t(std::int64_t(t) - c->g.ts.front());
    simpler.push_back(x);
  }
  x = *c;
  for (std::uint64_t e = 0, rank = 0; e < m; ++e)
    x.g.ts[e] = std::int32_t(rank += (e > 0 && c->g.ts[e] != c->g.ts[e - 1]));
  if (x.g.ts != c->g.ts) simpler.push_back(x);
  if (!c->bins.empty()) simpler.push_back(edit([](Case& y) { y.bins.clear(); }));
  rc::Seq<Case> smaller_windows;
  for (std::size_t i = 0; i < c->windows.size(); ++i)
    smaller_windows =
        rc::seq::concat(std::move(smaller_windows),
                        rc::seq::map(rc::shrink::towards<std::int32_t>(c->windows[i], 1),
                                     [=](std::int32_t w) { return edit([&](Case& y) { y.windows[i] = w; }); }));
  rc::Seq<Case> thresholds =
      rc::seq::concat(rc::seq::map(rc::shrink::towards<std::uint32_t>(c->heavy, 1),
                                   [=](std::uint32_t h) { return edit([&](Case& y) { y.heavy = h; }); }),
                      rc::seq::map(rc::shrink::towards<std::uint32_t>(c->cpu, 1),
                                   [=](std::uint32_t h) { return edit([&](Case& y) { y.cpu = h; }); }));
  return rc::seq::concat(std::move(fewer_windows), std::move(fewer_edges), rc::seq::fromContainer(std::move(simpler)),
                         std::move(smaller_windows), std::move(thresholds));
}

// A generator of cases from build(rng, size): RapidCheck draws the seed, and shrinks() shrinks.
template <class Build>
rc::Gen<Case> cases(Build build) {
  return rc::Gen<Case>([build](const rc::Random& random, int size) {
    rc::Random r = random;
    Rng rng{r.next()};
    return rc::shrinkable::shrinkRecur(build(rng, size), &shrinks);
  });
}

// ---- fuzz_diff over every profile and path ------------------------------------------------------

// The L2 profiles (SPEC v3 §4.3): clean_all (every clean pattern and vstats), and every legacy
// pattern under each visibility.
const std::vector<Profile>& profiles() {
  static const std::vector<Profile> p = {
      load_profile(std::string(BLZ_SOURCE_DIR) + "/configs/profiles/clean_all.yaml"),
      load_profile(std::string(BLZ_SOURCE_DIR) + "/tests/profiles/l2_legacy_stream.yaml"),
      load_profile(std::string(BLZ_SOURCE_DIR) + "/tests/profiles/l2_legacy_tsvis.yaml")};
  return p;
}

// What the checked cases exercised (SPEC v3 §4.4 item 6), summed over fuzz_diff runs, and the
// generators' tags.
struct Hits {
  std::uint64_t row_checks = 0, limited = 0, reach = 0, sweep = 0, fused = 0, light = 0, heavy = 0;
  std::uint64_t hybrid_cpu = 0, hybrid_gpu = 0, vstats_sides = 0;
  std::array<std::uint64_t, 3> packed{};
  std::array<std::uint64_t, kAllBlocks.size()> nonzero{};
  std::map<std::string, std::uint64_t> tags;

  void add(const FuzzDiff& d) {
    row_checks += d.row_checks;
    reach += d.reach_rows, sweep += d.sweep_rows, fused += d.fused_rows;
    light += d.light_triggers, heavy += d.heavy_triggers;
    hybrid_cpu += d.hybrid_cpu_triggers, hybrid_gpu += d.hybrid_gpu_triggers;
    vstats_sides += d.vstats_sides;
    for (int k = 0; k < 3; ++k) packed[k] += d.packed_rows[k];
    for (std::size_t b = 0; b < kAllBlocks.size(); ++b) nonzero[b] += d.nonzero_rows[b], limited += d.limited_graphs[b];
  }
  void tag(const Case& c) {
    for (const std::string& t : c.tags) ++tags[t];
  }
  std::uint64_t at(const std::string& t) const {
    const auto it = tags.find(t);
    return it == tags.end() ? 0 : it->second;
  }
  void print() const {
    std::printf(
        "  hits: %llu rows, %llu limited, reach %llu, sweep %llu, fused %llu, packed %llu/%llu/%llu, "
        "light %llu, heavy %llu, hybrid cpu %llu gpu %llu, vstats sides %llu\n",
        (unsigned long long)row_checks, (unsigned long long)limited, (unsigned long long)reach,
        (unsigned long long)sweep, (unsigned long long)fused, (unsigned long long)packed[0],
        (unsigned long long)packed[1], (unsigned long long)packed[2], (unsigned long long)light,
        (unsigned long long)heavy, (unsigned long long)hybrid_cpu, (unsigned long long)hybrid_gpu,
        (unsigned long long)vstats_sides);
    for (const auto& [t, k] : tags) std::printf("  %s: %llu\n", t.c_str(), (unsigned long long)k);
  }
  // Every block of the profiles had nonzero rows.
  bool every_block() const {
    return std::all_of(nonzero.begin(), nonzero.end(), [](std::uint64_t k) { return k > 0; });
  }
};

// Which CUDA runs a case gets, on one profile, as a GPU run costs milliseconds even on a small
// graph: `rotating`, two of cuda plain, packed, fused and packed, and the hybrid at C = c.cpu, each at
// one of H = 0, c.heavy and all; `packed_heavy`, cuda packed at H = 0 alone. c.gpu picks the profile
// (c.gpu % 3), the thresholds' rotation (c.gpu / 3 % 3) and the pair (c.gpu / 9).
enum class Gpu { rotating, packed_heavy };
enum class Run { plain, packed, fused, hybrid };
constexpr std::array<std::array<Run, 2>, 4> kRunPairs = {
    {{Run::plain, Run::packed}, {Run::fused, Run::hybrid}, {Run::plain, Run::fused}, {Run::packed, Run::hybrid}}};

// fuzz_diff on c alone: the CPU on every profile (both schedules, reachability pruning both ways)
// against the oracle; in a CUDA build, the `gpu` runs against the CPU's rows and vstats, which equal
// the oracle's once the CPU's check passed. Returns the first counterexample, as fuzz-diff prints
// it.
std::optional<std::string> diff(const Case& c, Hits& hits, [[maybe_unused]] Gpu gpu = Gpu::rotating) {
  DiffCase dc;
  dc.g = c.g;
  dc.windows = c.windows;
  dc.amount = c.amount;
  const DiffSource src = DiffSource::given({dc});
  const auto bad = [&](const Profile& p, const FuzzDiff& d) -> std::optional<std::string> {
    hits.add(d);
    if (d.bad_seeds.empty()) return std::nullopt;
    return "profile " + p.name + "\n" + counterexample_text(*d.counterexample, src);
  };
  for (const Profile& p : profiles())
    if (auto b = bad(p, fuzz_diff_cpu(p, src, c.bins, c.work_limit))) return b;
#if defined(BLZ_WITH_CUDA)
  const Profile& p = profiles()[c.gpu % 3];
  const std::uint32_t h[] = {0, c.heavy, kAllLight}, cpu[] = {c.cpu};
  if (gpu == Gpu::packed_heavy)
    return bad(p, fuzz_diff_cuda(p, src, c.bins, {h, 1}, c.work_limit, {false, true}, Reference::cpu));
  for (std::uint32_t k = 0; k < 2; ++k) {
    const std::span<const std::uint32_t> at(&h[(c.gpu / 3 + k) % 3], 1);
    const Run run = kRunPairs[c.gpu / 9 % 4][k];
    const FuzzDiff d = run == Run::hybrid ? fuzz_diff_hybrid(p, src, c.bins, at, cpu, c.work_limit, {}, Reference::cpu)
                                          : fuzz_diff_cuda(p, src, c.bins, at, c.work_limit,
                                                           {run == Run::fused, run != Run::plain}, Reference::cpu);
    if (auto b = bad(p, d)) return b;
  }
#endif
  return std::nullopt;
}

// The property of every graph property here: c passes fuzz_diff (tagging its hits).
void diff_property(const Case& c, Hits& hits, Gpu gpu = Gpu::rotating) {
  hits.tag(c);
  if (const std::optional<std::string> bad = diff(c, hits, gpu)) RC_FAIL(*bad);
}

// ---- Generators ---------------------------------------------------------------------------------

// Random edges past L2's bound: up to 12 nodes and 40 edges over up to 30 times, with parallel
// edges, duplicates, self-loops, reversed edges and a hub at either end of the ids; sorted.
Draft random_draft(Rng& r, int size) {
  Draft d;
  const std::uint32_t n = std::uint32_t(r.in(2, 2 + 10 * size / 100));
  const std::int64_t times = r.in(1, 30), m = r.in(1, 4 + 36 * size / 100);
  const auto node = [&] { return std::uint32_t(r.below(n)); };
  for (std::int64_t i = 0; i < m; ++i) {
    const std::int64_t t = r.in(0, times - 1);
    const std::uint64_t k = r.below(100);
    const std::uint32_t e = d.m() ? std::uint32_t(r.below(d.m())) : 0, u = node();
    if (k < 55 || d.m() == 0)
      d.add(u, node(), t);
    else if (k < 70)
      d.add(d.src[e], d.dst[e], t);  // parallel
    else if (k < 78)
      d.add(d.src[e], d.dst[e], d.ts[e]);  // duplicate
    else if (k < 86)
      d.add(d.dst[e], d.src[e], t);  // reversed
    else if (k < 92)
      d.add(u, u, t);  // self-loop
    else if (r.chance(50))
      d.add(r.chance(50) ? 0 : n - 1, u, t);  // a hub at either end
    else
      d.add(u, r.chance(50) ? 0 : n - 1, t);
  }
  d.n = std::max(d.n, n);
  d.sort();
  return d;
}

// lcyc's unordered search on a dense graph is bounded; a limited case still compares stop eids (R11).
constexpr std::uint64_t kRandomWorkLimit = 200'000;

// Random graphs, labelled as relabel() says, at windows near their span.
Case random_graph(Rng& r, int size) {
  Draft d = random_draft(r, size);
  relabel(d, r);
  Case c = finish(d, r, windows_near(d, r));
  c.work_limit = kRandomWorkLimit;
  return c;
}

// Arithmetic extremes: random graphs moved to the int32 limits: the latest time at INT32_MAX (or 1 or 2 below), the
// earliest at INT32_MIN (or above), or split between the two, a span past INT32_MAX; windows 1,
// near the span, and INT32_MAX.
Case extremes(Rng& r, int size) {
  Draft d = random_draft(r, size);
  relabel(d, r);
  const std::uint64_t mode = d.m() > 1 ? r.below(3) : r.below(2);
  const std::int64_t lo = d.ts.front(), hi = d.ts.back();
  const std::int64_t to_max = kTsMax - r.in(0, 2) - hi, to_min = kTsMin + r.in(0, 2) - lo;
  const std::uint64_t cut = mode == 2 ? 1 + r.below(d.m() - 1) : 0;
  for (std::uint64_t e = 0; e < d.m(); ++e) d.ts[e] += mode == 0 || (mode == 2 && e >= cut) ? to_max : to_min;
  Case c = finish(d, r, windows_near(d, r));
  c.work_limit = kRandomWorkLimit;
  c.tags.push_back(mode == 0 ? "ts at INT32_MAX" : mode == 1 ? "ts at INT32_MIN" : "ts at both limits");
  const std::int64_t span = d.span();
  for (const std::int32_t w : c.windows) {
    if (w == 1) c.tags.push_back("window 1");
    if (w == kTsMax) c.tags.push_back("window INT32_MAX");
    if (std::abs(std::int64_t(w) - span) <= 1) c.tags.push_back("window span or 1 off it");
  }
  return c;
}

// k_max: a cycle (tcyc, lcyc) or a chain (tchain) of k edges, k mostly K_MAX - 1, K_MAX or K_MAX + 1
// (past the cap), the trigger last of its hops. Hop gaps of 0 to 3 (ties), sometimes two hop times
// swapped, parallel copies of hops, and noise edges; windows at the instance's span and around it.
Case planted(Rng& r, int) {
  const int K = compiled_k_max();  // the patterns' K_MAX const (PatternInfo::consts)
  const bool cycle = r.chance(50);
  const int k = r.chance(70) ? int(r.in(K - 1, K + 1)) : int(r.in(2, K + 2));
  const std::uint32_t n = std::uint32_t(cycle ? k : k + 1) + std::uint32_t(r.in(0, 4));
  std::vector<std::uint32_t> id(n);
  std::iota(id.begin(), id.end(), 0u);
  for (std::uint32_t i = n; i > 1; --i) std::swap(id[i - 1], id[r.below(i)]);
  // The hops id[0] -> id[1] -> ... -> id[k - 1], then the trigger id[k - 1] -> id[0] (a cycle) or
  // -> id[k] (a chain).
  // Most instances are complete: strictly increasing hop times before the trigger (legacy and
  // clean both count it) and every window past the span (R2's open lower end). The others may have
  // ties, two hop times swapped, or a window at the span or below it.
  const bool complete = r.chance(60);
  std::vector<std::int64_t> hop_t;
  std::int64_t t = 0;
  for (int i = 0; i + 1 < k; ++i) {
    if (i > 0) t += !complete && r.chance(15) ? 0 : r.in(1, 3);
    hop_t.push_back(t);
  }
  t += !complete && r.chance(20) ? 0 : r.in(1, 3);
  if (!complete && hop_t.size() > 1 && r.chance(20))
    std::swap(hop_t[r.below(hop_t.size())], hop_t[r.below(hop_t.size())]);
  Draft d;
  d.n = n;
  for (int i = 0; i + 1 < k; ++i) d.add(id[i], id[i + 1], hop_t[i]);
  d.mark = d.add(id[k - 1], id[cycle ? 0 : k], t);
  for (std::int64_t c = r.chance(25) ? r.in(1, 3) : 0; c > 0 && k > 1; --c) {  // parallel hops
    const std::uint64_t h = r.below(std::uint64_t(k - 1));
    d.add(d.src[h], d.dst[h], std::clamp<std::int64_t>(d.ts[h] + r.in(-1, 1), 0, t));
  }
  for (std::int64_t c = r.in(0, 6); c > 0; --c)
    d.add(std::uint32_t(r.below(n)), std::uint32_t(r.below(n)), r.in(0, t + 2));
  const std::int64_t span = hop_t.empty() ? 0 : t - *std::min_element(hop_t.begin(), hop_t.end());
  d.sort();
  Case c = complete ? finish(d, r, {span + 1, span + 2, 86400, kTsMax})
                    : finish(d, r, {span - 1, span, span + 1, span + 2, 86400, kTsMax});
  const std::string what = cycle ? "cycle" : "chain";
  c.tags.push_back(what + " of " + std::to_string(k) + " edges");
  if (complete && (k == K || k == K + 1))
    c.tags.push_back("complete " + what + " of K_MAX" + (k == K ? "" : " + 1") + " edges");
  if (complete && k == K) {
    c.plant_k = std::uint32_t(k);
    c.plant_block = cycle ? Block::tcyc : Block::tchain;
  } else {
    c.plant_eid = UINT32_MAX;
  }
  return c;
}

// A star of s leaves around a hub at node 0 or s: 17 to 64 (a heavy warp's lanes past 16), or 250
// to 300 (fan and degree counts past 255 with distinct leaves, degree counts with parallel edges).
// Its times are all tied, increasing (the largest count last), or random. `cell` is its largest
// count.
Draft star(Rng& r, bool increasing, std::uint64_t& cell) {
  const std::array<std::pair<std::int64_t, std::int64_t>, 2> bands = {{{17, 64}, {250, 300}}};
  const auto [lo, hi] = bands[r.below(bands.size())];
  const std::uint32_t s = std::uint32_t(r.in(lo, hi)), hub = r.chance(50) ? 0 : s;
  const std::uint64_t dir = r.below(3), times = increasing ? 1 : r.below(3);
  const bool one_leaf = r.chance(15);
  Draft d;
  for (std::uint32_t i = 0; i < s; ++i) {
    const std::uint32_t leaf = one_leaf ? (hub == 0 ? 1 : 0) : i + (hub == 0);
    const std::int64_t t = times == 0 ? 0 : times == 1 ? i : r.in(0, s / 4);
    for (std::int64_t c = r.chance(30) ? r.in(1, 3) : 1; c > 0; --c) {
      if (dir == 0 || (dir == 2 && r.chance(50)))
        d.add(hub, leaf, t);
      else
        d.add(leaf, hub, t);
    }
  }
  cell = d.m();
  return d;
}

// V -> L1 -> ... -> Lj -> U with layers of w_i nodes, and the trigger U -> V last: prod w_i cycles
// of j + 2 edges at it, and chains of j + 1. The product lands in `band` 0 (up to 255), 1
// (256..65,535) or 2 (65,536..131,071: j = K_MAX - 2, 4^8 = 65,536 cycles of K_MAX edges and
// more). Ties (every edge at one time, so every edge closes cycles) below band 2 only. `at_end`:
// j = K_MAX - 2 and no ties, so the cycle blocks' one large cell is their last (the last trigger's
// last bin). `cell` is the product.
Draft layered(Rng& r, int band, bool at_end, std::uint64_t& cell) {
  const int K = compiled_k_max();
  const bool ties = band < 2 && !at_end && r.chance(20);
  const int j = band == 2 || at_end ? K - 2 : band == 1 ? int(r.in(4, K - 2)) : int(r.in(1, K - 1));
  std::vector<std::uint64_t> w(std::size_t(j), band == 2 ? 4 : 1);
  const auto prod = [&] { return std::accumulate(w.begin(), w.end(), std::uint64_t(1), std::multiplies<>()); };
  const std::uint64_t lo = band == 0 ? 1 : band == 1 ? 256 : 65536, hi = band == 0 ? 255 : band == 1 ? 65535 : 131071;
  for (int tries = 0; tries < 200 && (prod() < lo || r.chance(50)); ++tries) {
    std::uint64_t& x = w[r.below(w.size())];
    if (x < 6 && prod() / x * (x + 1) <= hi) ++x;
  }
  cell = prod();
  Draft d;
  std::uint32_t next = 1;
  std::vector<std::uint32_t> prev = {0};  // V
  std::int64_t t = 0;
  for (int i = 0; i <= j; ++i, t += ties ? 0 : 1) {
    std::vector<std::uint32_t> layer;
    for (std::uint64_t q = 0; q < (i < j ? w[std::size_t(i)] : 1); ++q) layer.push_back(next++);
    for (const std::uint32_t a : prev)
      for (const std::uint32_t b : layer)
        for (std::int64_t c = r.chance(10) ? 2 : 1; c > 0; --c) d.add(a, b, t);
    prev = layer;
  }
  d.mark = d.add(prev.front(), 0, t);  // U -> V
  return d;
}

void tag_cell(Case& c, std::uint64_t cell) {
  c.tags.push_back(cell > 65535 ? "a cell past 65,535" : cell > 255 ? "a cell past 255" : "cells up to 255");
}

// Packing: a star or a layered graph (cells up to 65,535), noise edges anywhere, sometimes relabelled;
// or, with `band` 2, a layered graph of cells past 65,535.
Case wide_at(Rng& r, int band) {
  std::uint64_t cell = 0;
  Draft d = band < 2 && r.chance(55) ? star(r, false, cell) : layered(r, band, false, cell);
  for (std::int64_t c = r.in(0, 5); c > 0; --c)
    d.add(std::uint32_t(r.below(d.n)), std::uint32_t(r.below(d.n)), r.in(0, d.ts.back()));
  d.sort();
  if (r.chance(50)) relabel(d, r);
  Case c = finish(d, r, windows_near(d, r));
  c.plant_eid = UINT32_MAX;
  tag_cell(c, cell);
  return c;
}
Case wide(Rng& r, int) { return wide_at(r, int(r.below(2))); }
Case four_bytes(Rng& r, int) { return wide_at(r, 2); }

// Position: a graph alone with its largest rows at the arrays' ends, as no member of an L2 batch but the
// last has them (SPEC v3 §4.3, GraphUnion): a star with increasing times, or a layered graph whose
// cycle blocks' one large cell (up to 65,535) is their last, its edges after a few earlier noise
// edges, its hub or V at node 0, or relabelled.
Case position(Rng& r, int) {
  std::uint64_t cell = 0;
  Draft s = r.chance(50) ? star(r, true, cell) : layered(r, int(r.below(2)), true, cell);
  Draft d;
  d.n = s.n;
  for (std::int64_t c = r.in(0, 4); c > 0; --c)
    d.add(std::uint32_t(r.below(s.n)), std::uint32_t(r.below(s.n)), r.in(0, 2));
  for (std::uint64_t e = 0; e < s.m(); ++e) d.add(s.src[e], s.dst[e], s.ts[e] + 3);
  d.sort();
  if (r.chance(60)) relabel(d, r);
  Case c = finish(d, r, windows_near(d, r));
  c.plant_eid = UINT32_MAX;
  tag_cell(c, cell);
  return c;
}

// ---- R10's model --------------------------------------------------------------------------------

std::string row_text(std::span<const std::uint32_t> r) {
  std::string t = "[";
  for (std::size_t k = 0; k < r.size(); ++k) t += (k ? " " : "") + std::to_string(r[k]);
  return t + "]";
}

// A subset of the profiles' bins, nonempty.
rc::Gen<std::vector<std::uint32_t>> some_bins() {
  return rc::gen::map(
      rc::gen::suchThat(rc::gen::container<std::vector<bool>>(12, rc::gen::arbitrary<bool>()),
                        [](const std::vector<bool>& v) { return std::find(v.begin(), v.end(), true) != v.end(); }),
      [](const std::vector<bool>& v) {
        static constexpr std::uint32_t all[] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 16, 20};
        std::vector<std::uint32_t> b;
        for (std::size_t i = 0; i < v.size(); ++i)
          if (v[i]) b.push_back(all[i]);
        return b;
      });
}

// A cell before the emits: 0, at or just below UINT32_MAX, or anything.
rc::Gen<std::uint32_t> near_max() {
  return rc::gen::resize(
      100, rc::gen::oneOf(rc::gen::just<std::uint32_t>(0),
                          rc::gen::map(rc::gen::inRange<std::uint32_t>(0, 5), [](std::uint32_t k) { return kMax - k; }),
                          rc::gen::arbitrary<std::uint32_t>()));
}

}  // namespace

// ---- Properties ---------------------------------------------------------------------------------

TEST_CASE("L3 k_max: cycles and chains of K_MAX - 1 to K_MAX + 1 edges on every profile and path", "[l3]") {
  const int K = compiled_k_max();
  REQUIRE(K == profiles().front().k_max);
  Hits hits;
  check(100, [&] {
    const Case c = *cases(planted);
    // The generator's claim: a complete instance of K_MAX edges is counted at its size by every
    // profile's oracle, in tcyc and lcyc for a cycle, in tchain for a chain.
    if (c.plant_eid != UINT32_MAX) {
      for (const Profile& p : profiles()) {
        const ProfileOracle o(EdgeStream{c.g.src, c.g.dst, c.g.ts, c.g.n}, p, Quirks{}, c.amount);
        for (const Block b : {c.plant_block, c.plant_block == Block::tcyc ? Block::lcyc : Block::tchain}) {
          BlockParams bp{c.windows.front(), fuzz_bins(p, *p.find(b), c.bins)};
          std::vector<std::uint32_t> row(bp.bins.size(), 0);
          const std::uint32_t eid[] = {c.plant_eid};
          o.mine(b, bp, eid, row.data());
          const std::size_t bin =
              std::size_t(std::upper_bound(bp.bins.begin(), bp.bins.end(), c.plant_k) - bp.bins.begin());
          RC_ASSERT(bin > std::size_t(0));
          RC_ASSERT(row[bin - 1] > 0u);
        }
      }
    }
    diff_property(c, hits);
  });
  hits.print();
  CHECK(hits.at("complete cycle of K_MAX edges") > 0);
  CHECK(hits.at("complete cycle of K_MAX + 1 edges") > 0);
  CHECK(hits.at("complete chain of K_MAX edges") > 0);
  CHECK(hits.at("complete chain of K_MAX + 1 edges") > 0);
  CHECK(hits.reach > 0);
}

TEST_CASE("L3 R10: a histogram row saturates at UINT32_MAX and counts each lost instance", "[l3]") {
  check(400, [] {
    const std::vector<std::uint32_t> bins = *some_bins();
    const std::vector<std::uint32_t> pre = *rc::gen::container<std::vector<std::uint32_t>>(bins.size(), near_max());
    const std::int64_t top = bins.back() + 4;
    const auto emits = *rc::gen::container<std::vector<std::pair<std::int64_t, std::int64_t>>>(
        rc::gen::pair(rc::gen::resize(100, rc::gen::inRange<std::int64_t>(0, top + 1)),
                      rc::gen::resize(100, rc::gen::inRange<std::int64_t>(2, top + 1))));
    // The model: each instance of size >= min_size and >= b_0 goes to the last bin b_i <= size;
    // past UINT32_MAX a bin keeps UINT32_MAX and each instance more is lost.
    std::vector<std::uint64_t> sum(pre.begin(), pre.end());
    for (const auto& [size, min_size] : emits) {
      std::size_t i = bins.size();
      while (i > 0 && std::int64_t(bins[i - 1]) > size) --i;
      if (size >= min_size && i > 0) ++sum[i - 1];
    }
    std::vector<std::uint32_t> want(bins.size());
    std::uint64_t lost = 0;
    for (std::size_t i = 0; i < bins.size(); ++i) {
      want[i] = std::uint32_t(std::min<std::uint64_t>(sum[i], kMax));
      lost += sum[i] - want[i];
    }
    std::vector<std::uint32_t> row = pre;
    RowSink sink{row.data(), bins.data(), std::uint32_t(bins.size()), false};
    for (const auto& [size, min_size] : emits) sink.emit(size, min_size);
    if (row != want || sink.saturated != lost)
      RC_FAIL("RowSink, bins " + row_text(bins) + ", from " + row_text(pre) + ": got " + row_text(row) + " (" +
              std::to_string(sink.saturated) + " lost), want " + row_text(want) + " (" + std::to_string(lost) +
              " lost)");
  });
}

TEST_CASE("L3 R10 and R7: a count row clamps at UINT32_MAX and loses one emit per clamped sum", "[l3]") {
  check(400, [] {
    const std::uint32_t pre = *near_max();
    const auto sizes = *rc::gen::container<std::vector<std::uint32_t>>(
        rc::gen::resize(100, rc::gen::oneOf(rc::gen::inRange<std::uint32_t>(0, 9), rc::gen::arbitrary<std::uint32_t>(),
                                            rc::gen::just(kMax))));
    // The model: the row is min(pre + the sizes, UINT32_MAX); an emit is lost when it adds to a sum
    // that is past UINT32_MAX.
    std::uint64_t sum = pre, lost = 0;
    for (const std::uint32_t s : sizes) {
      sum += s;
      lost += s > 0 && sum > kMax;
    }
    const std::uint32_t want = std::uint32_t(std::min<std::uint64_t>(sum, kMax));
    std::uint32_t row = pre;
    RowSink sink{&row, nullptr, 1, true};
    for (const std::uint32_t s : sizes) sink.emit(s, 2);
    if (row != want || sink.saturated != lost)
      RC_FAIL("RowSink, a count row from " + std::to_string(pre) + ": got " + std::to_string(row) + " (" +
              std::to_string(sink.saturated) + " lost), want " + std::to_string(want) + " (" + std::to_string(lost) +
              " lost)");
  });
}

TEST_CASE("L3 R10: the oracles and the CPU saturate prefilled rows as R10 says with count blocks apart", "[l3]") {
  Hits hits;
  check(30, [&] {
    const Case c = *cases([](Rng& r, int size) { return random_graph(r, std::min(size, 30)); });
    const std::uint64_t pick = *rc::gen::arbitrary<std::uint64_t>();
    const std::uint64_t m = c.g.src.size();
    const std::int32_t w = c.windows.front();
    for (const Profile& p : profiles()) {
      const ProfileOracle o(EdgeStream{c.g.src, c.g.dst, c.g.ts, c.g.n}, p, Quirks{}, c.amount);
      const GraphIndex index(c.g.src, c.g.dst, c.g.ts, c.g.n, p.self_loops());
      TriggerSel all;
      all.last = m;
      for (const ProfileBlock& pb : p.blocks) {
        BlockParams bp{w, fuzz_bins(p, pb, c.bins)};
        bp.work_limit = c.work_limit;
        const bool count = bp.bins.empty();
        RC_ASSERT(!count || pb.patterns.size() == 1);  // R7: one emit per trigger at most
        const std::size_t nb = block_cols(bp);
        std::vector<std::uint32_t> base(m * nb, 0);
        try {
          o.mine(pb.block, bp, 0, m, base.data());
        } catch (const WorkLimitExceeded&) {
          continue;
        }
        // Before the emits: per cell, 0, UINT32_MAX, or such that the sum lands 2 below to 1 past it.
        std::vector<std::uint32_t> pre(base.size()), want(base.size());
        std::uint64_t lost = 0;
        Rng r{pick ^ name_seed(p.name + "/" + std::string(block_name(pb.block)))};
        for (std::size_t i = 0; i < base.size(); ++i) {
          const std::int64_t k = r.in(0, 5);
          pre[i] = k == 0   ? 0
                   : k == 1 ? kMax
                            : std::uint32_t(std::clamp<std::int64_t>(std::int64_t(kMax) + k - 4 - base[i], 0, kMax));
          const std::uint64_t sum = std::uint64_t(pre[i]) + base[i];
          want[i] = std::uint32_t(std::min<std::uint64_t>(sum, kMax));
          lost += count ? (sum > kMax) : sum - want[i];
          hits.tags[sum > kMax ? (count ? "a count row clamped" : "a bin saturated") : "below UINT32_MAX"]++;
        }
        const auto compare = [&](const std::string& path, const std::vector<std::uint32_t>& got,
                                 std::uint64_t got_lost) {
          if (got == want && got_lost == lost) return;
          std::size_t r0 = 0;
          while (r0 + 1 < m && std::equal(got.begin() + r0 * nb, got.begin() + (r0 + 1) * nb, want.begin() + r0 * nb))
            ++r0;
          const auto at = [&](const std::vector<std::uint32_t>& v) { return row_text({v.data() + r0 * nb, nb}); };
          RC_FAIL("profile " + p.name + ", block " + std::string(block_name(pb.block)) + ", window " +
                  std::to_string(w) + ", path " + path + "\ntrigger eid " + std::to_string(r0) + " (" +
                  edge_text(c.g, r0) + "): from " + at(pre) + " got " + at(got) + ", want " + at(want) +
                  "; lost: got " + std::to_string(got_lost) + ", want " + std::to_string(lost));
        };
        std::vector<std::uint32_t> got = pre;
        compare("oracle", got, o.mine(pb.block, bp, 0, m, got.data()));
        const bool reach =
            std::any_of(pb.patterns.begin(), pb.patterns.end(), [](const PatternInfo* q) { return q->cpu_reach; });
        for (int k = 0; k < (reach ? 4 : 2); ++k) {
          const Schedule s = k % 2 == 0 ? Schedule::dynamic : Schedule::sorted;
          BlockParams q = bp;
          q.reach_prune = k >= 2;
          got = pre;
          const CpuBlockTiming tm = mine_cpu(index.view(), pb.patterns, q, p.visibility, all, s, got.data());
          compare(std::string("cpu ") + schedule_name(s) + (q.reach_prune ? " reach" : ""), got, tm.saturated);
        }
      }
    }
  });
  hits.print();
  CHECK(hits.at("a count row clamped") > 0);
  CHECK(hits.at("a bin saturated") > 0);
}

TEST_CASE("L3 packing and chunks: stars and layered cycles with cells past 255", "[l3]") {
  Hits hits;
  check(24, [&] { diff_property(*cases(wide), hits); });
  hits.print();
  CHECK(hits.at("a cell past 255") > 0);
#if defined(BLZ_WITH_CUDA)
  CHECK(hits.packed[0] > 0);  // 1 byte a cell
  CHECK(hits.packed[1] > 0);  // 2
  CHECK(hits.heavy > 0);
  CHECK(hits.light > 0);
#endif
}

TEST_CASE("L3 packing at 4 bytes: 4^8 and more cycles of K_MAX edges at one trigger", "[l3]") {
  Hits hits;
  check(3, [&] { diff_property(*cases(four_bytes), hits, Gpu::packed_heavy); });
  hits.print();
  CHECK(hits.at("a cell past 65,535") > 0);
#if defined(BLZ_WITH_CUDA)
  CHECK(hits.packed[2] > 0);  // 4 bytes a cell
#endif
}

TEST_CASE("L3 arithmetic: timestamps at the int32 limits and windows from 1 to INT32_MAX", "[l3]") {
  Hits hits;
  check(60, [&] { diff_property(*cases(extremes), hits); });
  hits.print();
  for (const char* t : {"ts at INT32_MAX", "ts at INT32_MIN", "ts at both limits", "window 1", "window INT32_MAX",
                        "window span or 1 off it"})
    CHECK(hits.at(t) > 0);
}

TEST_CASE("L3 position: a graph alone with its largest rows at the arrays' ends", "[l3]") {
  Hits hits;
  check(16, [&] { diff_property(*cases(position), hits); });
  hits.print();
  CHECK(hits.at("a cell past 255") > 0);
}

TEST_CASE("L3 random graphs past L2's bound alone through every pattern and path", "[l3]") {
  Hits hits;
  check(100, [&] { diff_property(*cases(random_graph), hits); });
  hits.print();
  CHECK(hits.every_block());
  CHECK(hits.reach > 0);
  CHECK(hits.sweep > 0);
  CHECK(hits.vstats_sides > 0);
#if defined(BLZ_WITH_CUDA)
  CHECK(hits.fused > 0);
  CHECK(hits.packed[0] > 0);
  CHECK(hits.light > 0);
  CHECK(hits.heavy > 0);
  CHECK(hits.hybrid_cpu > 0);
  CHECK(hits.hybrid_gpu > 0);
#endif
}

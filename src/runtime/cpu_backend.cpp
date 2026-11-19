#include "runtime/cpu_backend.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <numeric>
#include <stdexcept>

#include "runtime/registration.hpp"

namespace blz {

namespace {

using Clock = std::chrono::steady_clock;
double since(Clock::time_point a) { return std::chrono::duration<double>(Clock::now() - a).count(); }

const ConstInfo* find_const(const PatternInfo& p, const char* name) {
  for (const ConstInfo& c : p.consts)
    if (std::strcmp(c.name, name) == 0) return &c;
  return nullptr;
}

}  // namespace

const char* schedule_name(Schedule s) { return s == Schedule::sorted ? "sorted" : "dynamic"; }

Schedule parse_schedule(const std::string& s) {
  if (s == "dynamic") return Schedule::dynamic;
  if (s == "sorted") return Schedule::sorted;
  throw std::runtime_error("--schedule must be dynamic or sorted, got '" + s + "'");
}

std::vector<const PatternInfo*> patterns_for(Block b) {
  std::vector<const PatternInfo*> out;
  for (const PatternInfo& p : registered_patterns())
    if (block_name(b) == p.block) out.push_back(&p);
  return out;
}

void check_backend_config(const FeatureConfig& cfg, const std::string& config_name) {
  // A v1 config covers the v1 blocks. A pattern of a v2 block (sg_ordered, SPEC v2 §6.3) runs
  // only from a profile that lists it, so a v1 config that leaves its block out is not E402.
  std::vector<PatternRef> refs;
  for (const PatternInfo& p : registered_patterns())
    if (const auto b = parse_block(p.block); !b || is_v1_block(*b)) refs.push_back({p.name, p.block, p.block_loc});
  check_registration(refs, cfg, config_name);
  for (const PatternInfo& p : registered_patterns())
    if (const auto b = parse_block(p.block); b && !is_v1_block(*b)) continue;  // not run by a v1 config
    else if (const ConstInfo* k = find_const(p, "K_MAX"); k && k->value != cfg.k_max)
      throw std::runtime_error(std::string(p.file) + ": pattern '" + p.name + "' was compiled with K_MAX = " +
                               std::to_string(k->value) + ", but " + config_name + " has k_max: " +
                               std::to_string(cfg.k_max));
}

int compiled_k_max() {
  int k = -1;
  for (const PatternInfo& p : registered_patterns())
    if (const ConstInfo* c = find_const(p, "K_MAX")) {
      if (k >= 0 && k != c->value) throw std::runtime_error("patterns were compiled with different K_MAX values");
      k = int(c->value);
    }
  if (k < 0) throw std::runtime_error("no registered pattern declares K_MAX");
  return k;
}

CpuBlockTiming mine_cpu(const GraphView& g, std::span<const PatternInfo* const> pats, const BlockParams& p,
                        Visibility vis, const TriggerSel& sel, Schedule schedule, std::uint32_t* out) {
  if (pats.empty()) throw std::runtime_error("mine_cpu: no pattern to run");
  if (sel.first > sel.last || sel.last > g.m) throw std::invalid_argument("mine_cpu: bad trigger range");
  for (std::uint32_t e : sel.eids)
    if (e < sel.first || e >= sel.last) throw std::invalid_argument("mine_cpu: sample eid outside its range");
  const Params params{p.window, vis == Visibility::timestamp, p.work_limit, p.reach_prune};
  BlockOut bo{out, sel.first, p.bins.data(), std::uint32_t(p.bins.size()), 0,
              sel.sampled ? sel.eids.data() : nullptr, sel.sampled ? sel.eids.size() : 0, sel.last};
  CpuBlockTiming timing;
  const std::size_t count = sel.rows();
  std::vector<std::uint32_t> triggers(count);
  std::vector<std::uint32_t> work(schedule == Schedule::sorted ? count : 0);
  bool listed = false;  // `triggers` holds the eid-order list
  for (const PatternInfo* pat : pats) {
    auto t0 = Clock::now();
    if (pat->cpu_sweep) {
      // The sweep plan (SPEC v2 §7.1) walks every node's list in its own order: no trigger order.
    } else if (schedule == Schedule::dynamic) {
      if (!listed) {
        if (sel.sampled) std::copy(sel.eids.begin(), sel.eids.end(), triggers.begin());
        else std::iota(triggers.begin(), triggers.end(), std::uint32_t(sel.first));
        listed = true;
      }
    } else {
      // Descending work, ties in eid order: a stable counting sort on the work estimate (a
      // window size, so at most m).
      std::uint32_t max_work = 0;
#pragma omp parallel for schedule(static) reduction(max : max_work)
      for (std::int64_t i = 0; i < std::int64_t(count); ++i) {
        work[std::size_t(i)] = pat->work(g, params, sel.eid(std::uint64_t(i)));
        max_work = std::max(max_work, work[std::size_t(i)]);
      }
      std::vector<std::uint64_t> start(std::size_t(max_work) + 2, 0);  // by max_work - work
      for (std::uint32_t w : work) ++start[max_work - w + 1];
      std::partial_sum(start.begin(), start.end(), start.begin());
      for (std::size_t i = 0; i < count; ++i) triggers[start[max_work - work[i]]++] = sel.eid(i);
    }
    timing.schedule_seconds += since(t0);
    auto t1 = Clock::now();
    pat->cpu(g, params, triggers, bo, schedule == Schedule::sorted ? kSortedChunk : kDynamicChunk);
    timing.loop_seconds += since(t1);
    if (bo.aborted != UINT64_MAX)
      throw WorkLimitExceeded(std::string("cpu, pattern ") + pat->name, bo.aborted, p.work_limit);
    if (pat->cpu_sweep) timing.sweep_seconds += since(t1);
  }
  timing.saturated = bo.saturated;
  return timing;
}

CpuBlockTiming mine_cpu(const GraphView& g, Block b, const BlockParams& p, std::uint64_t first,
                        std::uint64_t last, Schedule schedule, std::uint32_t* out) {
  const auto pats = patterns_for(b);
  if (pats.empty()) throw std::runtime_error("no registered pattern emits into block " + std::string(block_name(b)));
  if (first > last || last > g.m) throw std::invalid_argument("mine_cpu: bad trigger range");
  TriggerSel sel;
  sel.spec = "range";
  sel.first = first;
  sel.last = last;
  return mine_cpu(g, pats, p, Visibility::stream, sel, schedule, out);
}

}  // namespace blz

#include "graph/index.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace blz {

GraphIndex::GraphIndex(std::span<const std::uint32_t> src, std::span<const std::uint32_t> dst,
                       std::span<const std::int32_t> ts, std::uint64_t n) {
  const std::uint64_t m = src.size();
  if (dst.size() != m || ts.size() != m) throw std::invalid_argument("index: src/dst/ts sizes differ");
  // Positions are u32 and prev_same is i32 (SPEC §5).
  if (m >= std::uint64_t(std::numeric_limits<std::int32_t>::max()))
    throw std::invalid_argument("index: too many edges for i32 positions");
  if (n >= std::numeric_limits<std::uint32_t>::max()) throw std::invalid_argument("index: too many nodes");
  for (std::uint64_t e = 0; e < m; ++e) {
    if (src[e] >= n || dst[e] >= n) throw std::invalid_argument("index: node id >= n");
    if (e > 0 && ts[e] < ts[e - 1]) throw std::invalid_argument("index: ts not sorted (SPEC §3.3)");
  }
  build_tcsr(out_, src, dst, ts, n);
  build_tcsr(in_, dst, src, ts, n);
  build_pcsr(pout_, out_, n);
  build_pcsr(pin_, in_, n);

  view_.m = std::uint32_t(m);
  view_.n = std::uint32_t(n);
  view_.src = src.data();
  view_.dst = dst.data();
  view_.ts = ts.data();
  auto tv = [](const TCsr& t) {
    return TCsrView{t.off.data(), t.nbr.data(), t.ts.data(), t.eid.data(), t.prev_same.data()};
  };
  auto pv = [](const PCsr& p) {
    return PCsrView{p.off.data(), p.nbr.data(), p.run.data(), p.run_ts.data(), p.run_eid.data()};
  };
  view_.out = tv(out_);
  view_.in = tv(in_);
  view_.pout = pv(pout_);
  view_.pin = pv(pin_);
}

// Counting sort of the edges by `key` (src for out-lists, dst for in-lists). Edges are visited in
// eid order, so every list is sorted by eid, hence by ts.
void GraphIndex::build_tcsr(TCsr& t, std::span<const std::uint32_t> key, std::span<const std::uint32_t> nbr,
                            std::span<const std::int32_t> ts, std::uint64_t n) {
  const std::uint64_t m = key.size();
  t.off.assign(n + 1, 0);
  for (std::uint64_t e = 0; e < m; ++e) ++t.off[key[e] + 1];
  std::partial_sum(t.off.begin(), t.off.end(), t.off.begin());
  t.nbr.resize(m);
  t.eid.resize(m);
  t.ts.resize(m);
  t.prev_same.resize(m);
  std::vector<std::uint32_t> cursor(t.off.begin(), t.off.end() - 1);
  for (std::uint64_t e = 0; e < m; ++e) {
    const std::uint32_t j = cursor[key[e]]++;
    t.nbr[j] = nbr[e];
    t.eid[j] = std::uint32_t(e);
    t.ts[j] = ts[e];
  }
  // prev_same: the previous position in the same list with the same neighbour. `owner[y]` is the
  // list in which `last[y]` was recorded, so the n-sized arrays need no reset between lists.
  std::vector<std::uint32_t> owner(n, std::numeric_limits<std::uint32_t>::max());
  std::vector<std::int32_t> last(n, -1);
  for (std::uint32_t x = 0; x < n; ++x)
    for (std::uint32_t j = t.off[x]; j < t.off[x + 1]; ++j) {
      const std::uint32_t y = t.nbr[j];
      t.prev_same[j] = owner[y] == x ? last[y] : -1;
      owner[y] = x;
      last[y] = std::int32_t(j);
    }
}

// Per node, its T-CSR list regrouped by neighbour: the distinct neighbours sorted by id, each with
// its run of (ts, eid) in eid order. A node's runs occupy the same positions [off[x], off[x+1])
// as its T-CSR list.
void GraphIndex::build_pcsr(PCsr& p, const TCsr& t, std::uint64_t n) {
  const std::uint64_t m = t.nbr.size();
  p.off.assign(n + 1, 0);
  for (std::uint32_t x = 0; x < n; ++x) {
    std::uint32_t distinct = 0;
    for (std::uint32_t j = t.off[x]; j < t.off[x + 1]; ++j) distinct += t.prev_same[j] < 0;
    p.off[x + 1] = p.off[x] + distinct;
  }
  const std::uint32_t pairs = p.off[n];
  p.nbr.resize(pairs);
  p.run.resize(std::uint64_t(pairs) + 1);
  p.run_ts.resize(m);
  p.run_eid.resize(m);
  p.run[pairs] = std::uint32_t(m);
#pragma omp parallel
  {
    std::vector<std::uint32_t> pos;
#pragma omp for schedule(dynamic, 1024)
    for (std::int64_t xi = 0; xi < std::int64_t(n); ++xi) {
      const std::uint32_t x = std::uint32_t(xi);
      const std::uint32_t a = t.off[x], b = t.off[x + 1];
      pos.resize(b - a);
      std::iota(pos.begin(), pos.end(), a);
      // Stable: within a neighbour, positions (eids) stay increasing.
      std::stable_sort(pos.begin(), pos.end(), [&](std::uint32_t i, std::uint32_t j) { return t.nbr[i] < t.nbr[j]; });
      std::uint32_t pair = p.off[x];
      for (std::uint32_t k = 0; k < pos.size(); ++k) {
        const std::uint32_t j = pos[k];
        if (k == 0 || t.nbr[j] != t.nbr[pos[k - 1]]) {
          p.nbr[pair] = t.nbr[j];
          p.run[pair] = a + k;
          ++pair;
        }
        p.run_ts[a + k] = t.ts[j];
        p.run_eid[a + k] = t.eid[j];
      }
    }
  }
}

std::uint64_t GraphIndex::bytes() const {
  auto sz = [](const auto& v) { return std::uint64_t(v.size() * sizeof(v[0])); };
  std::uint64_t b = 0;
  for (const TCsr* t : {&out_, &in_}) b += sz(t->off) + sz(t->nbr) + sz(t->eid) + sz(t->ts) + sz(t->prev_same);
  for (const PCsr* p : {&pout_, &pin_}) b += sz(p->off) + sz(p->nbr) + sz(p->run) + sz(p->run_eid) + sz(p->run_ts);
  return b;
}

}  // namespace blz

#include "runtime/coverage.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>

#include <omp.h>

namespace blz {

namespace {

enum Split { kFit, kVal, kTest };
constexpr std::array<const char*, 3> kSplitNames = {"fit", "val", "test"};

Split split_of(std::uint64_t e, std::uint64_t b_fv, std::uint64_t b_test) {
  return e < b_fv ? kFit : e < b_test ? kVal : kTest;
}

bool row_nonzero(std::span<const std::uint32_t> block, std::uint64_t cols, std::uint64_t e) {
  const std::uint32_t* r = block.data() + e * cols;
  return std::any_of(r, r + cols, [](std::uint32_t c) { return c != 0; });
}

nlohmann::ordered_json null_or(double v) { return std::isnan(v) ? nlohmann::ordered_json() : nlohmann::ordered_json(v); }

}  // namespace

double percentile_linear(std::vector<double> x, double q) {
  if (x.empty()) return std::numeric_limits<double>::quiet_NaN();
  std::sort(x.begin(), x.end());
  const double pos = (double(x.size()) - 1.0) * q / 100.0;
  const std::size_t lo = std::size_t(std::floor(pos)), hi = std::min(lo + 1, x.size() - 1);
  return x[lo] + (x[hi] - x[lo]) * (pos - double(lo));
}

std::vector<CoverageAttempt> coverage_attempts(const nlohmann::json& gt, std::uint64_t m) {
  if (!gt.contains("attempts") || !gt["attempts"].is_array()) throw std::runtime_error("gt-join: no attempts array");
  std::vector<CoverageAttempt> out;
  for (const auto& a : gt["attempts"]) {
    CoverageAttempt c;
    c.id = a.at("id").get<std::int64_t>();
    c.typology = a.at("typology").get<std::string>();
    c.header = a.at("header").get<std::string>();
    c.status = a.at("status").get<std::string>();
    c.split = a.at("split").is_null() ? "none" : a.at("split").get<std::string>();
    c.span = a.at("span").is_null() ? 0 : a.at("span").get<std::int64_t>();
    for (const auto& r : a.at("rows")) {
      const std::string st = r.at("status").get<std::string>();
      if (st != "unique" && st != "resolved-duplicate") continue;
      const std::uint64_t e = r.at("eid").get<std::uint64_t>();
      if (e >= m) throw std::runtime_error("gt-join: attempt " + std::to_string(c.id) + " maps eid " + std::to_string(e) + " >= m");
      c.eids.push_back(std::uint32_t(e));
    }
    std::sort(c.eids.begin(), c.eids.end());
    if (std::adjacent_find(c.eids.begin(), c.eids.end()) != c.eids.end())
      throw std::runtime_error("gt-join: attempt " + std::to_string(c.id) + " maps one eid twice");
    if (c.status == "complete" && c.eids.empty())
      throw std::runtime_error("gt-join: complete attempt " + std::to_string(c.id) + " has no eid");
    out.push_back(std::move(c));
  }
  return out;
}

IsolatedGraph isolated_graph(const CoverageGraph& g, std::span<const std::uint32_t> eids) {
  IsolatedGraph ig;
  ig.eids.assign(eids.begin(), eids.end());
  if (!std::is_sorted(ig.eids.begin(), ig.eids.end()) ||
      std::adjacent_find(ig.eids.begin(), ig.eids.end()) != ig.eids.end())
    throw std::invalid_argument("isolated_graph: eids must be strictly ascending");
  std::unordered_map<std::uint32_t, std::uint32_t> id;
  auto local = [&](std::uint32_t v) {
    auto [it, fresh] = id.emplace(v, std::uint32_t(id.size()));
    return it->second;
  };
  for (std::uint32_t e : ig.eids) {
    if (e >= g.src.size()) throw std::invalid_argument("isolated_graph: eid >= m");
    ig.src.push_back(local(g.src[e]));
    ig.dst.push_back(local(g.dst[e]));
    ig.ts.push_back(g.ts[e]);
  }
  ig.n = id.size();
  return ig;
}

std::vector<std::uint64_t> oracle_eligible_rows(const IsolatedGraph& ig, const Profile& prof) {
  const Oracle oracle(ig.stream(), prof.visibility);
  const std::uint64_t k = ig.eids.size();
  std::vector<std::uint64_t> out;
  for (const ProfileBlock& pb : prof.blocks) {
    const std::uint64_t cols = pb.params.bins.size();
    std::vector<std::uint32_t> rows(k * cols, 0);
    oracle.mine(pb.block, pb.params, prof.k_max, 0, k, rows.data());
    std::uint64_t nz = 0;
    for (std::uint64_t i = 0; i < k; ++i) nz += row_nonzero(rows, cols, i);
    out.push_back(nz);
  }
  return out;
}

nlohmann::ordered_json coverage(const CoverageInputs& in) {
  using J = nlohmann::ordered_json;
  const Profile& prof = *in.profile;
  const CoverageGraph& g = in.graph;
  const std::uint64_t m = g.src.size();
  const std::size_t nb = prof.blocks.size();
  if (g.dst.size() != m || g.ts.size() != m || g.label.size() != m) throw std::runtime_error("coverage: graph arrays differ in size");
  if (!(in.b_fv <= in.b_test && in.b_test <= m)) throw std::runtime_error("coverage: bad split boundaries");
  if (in.blocks.size() != nb) throw std::runtime_error("coverage: one full-mining block per profile block is needed");
  std::vector<std::uint64_t> cols(nb);
  for (std::size_t b = 0; b < nb; ++b) {
    cols[b] = prof.blocks[b].params.bins.size();
    if (in.blocks[b].size() != m * cols[b])
      throw std::runtime_error("coverage: block " + std::string(block_name(prof.blocks[b].block)) + " is not m x bins");
  }
  const bool external = !in.external_eligible.empty();

  // Which splits the predictions score (fit never: it is training data).
  std::array<bool, 3> scored = {false, false, false};
  const bool have_pred = !in.pred.empty();
  if (have_pred) {
    if (in.pred.size() != m) throw std::runtime_error("coverage: --pred has " + std::to_string(in.pred.size()) + " rows, not m");
    const std::array<std::pair<std::uint64_t, std::uint64_t>, 3> rng = {
        std::pair{std::uint64_t(0), in.b_fv}, std::pair{in.b_fv, in.b_test}, std::pair{in.b_test, m}};
    for (int s = 0; s < 3; ++s) {
      std::uint64_t n_scored = 0;
      for (std::uint64_t e = rng[s].first; e < rng[s].second; ++e) {
        const std::uint8_t p = in.pred[e];
        if (p == kPredNegative || p == kPredPositive) ++n_scored;
        else if (p != kPredNotScored)
          throw std::runtime_error("coverage: --pred code " + std::to_string(int(p)) + " at eid " + std::to_string(e));
      }
      const std::uint64_t len = rng[s].second - rng[s].first;
      if (n_scored != 0 && n_scored != len)
        throw std::runtime_error(std::string("coverage: --pred scores only part of the ") + kSplitNames[s] + " split");
      scored[s] = len > 0 && n_scored == len;
    }
    if (scored[kFit]) throw std::runtime_error("coverage: --pred scores fit rows, which are training data (SPEC v2 §5)");
  }

  // Per attempt.
  struct Row {
    std::array<std::uint64_t, 3> edges{}, predicted{};
    std::vector<std::uint64_t> eligible, eligible_oracle, activated;
  };
  const std::size_t na = in.attempts.size();
  std::vector<Row> rows(na);
  std::vector<std::string> errors(na);
#pragma omp parallel for schedule(dynamic)
  for (std::size_t i = 0; i < na; ++i) {
    try {
      const CoverageAttempt& a = in.attempts[i];
      Row& r = rows[i];
      for (std::uint32_t e : a.eids) {
        const Split s = split_of(e, in.b_fv, in.b_test);
        ++r.edges[s];
        if (have_pred && in.pred[e] == kPredPositive) ++r.predicted[s];
      }
      if (a.status != "complete") continue;
      r.eligible_oracle = oracle_eligible_rows(isolated_graph(g, a.eids), prof);
      if (external) {
        auto it = in.external_eligible.find(a.id);
        if (it == in.external_eligible.end() || it->second.size() != nb)
          throw std::runtime_error("coverage: the external eligibility has no row of " + std::to_string(nb) +
                                   " blocks for attempt " + std::to_string(a.id));
        r.eligible = it->second;
        for (std::size_t b = 0; b < nb; ++b)
          if (r.eligible[b] > a.eids.size())
            throw std::runtime_error("coverage: external eligibility of attempt " + std::to_string(a.id) +
                                     " exceeds its edge count");
      } else {
        r.eligible = r.eligible_oracle;
      }
      r.activated.assign(nb, 0);
      for (std::size_t b = 0; b < nb; ++b)
        for (std::uint32_t e : a.eids) r.activated[b] += row_nonzero(in.blocks[b], cols[b], e);
    } catch (const std::exception& ex) {
      errors[i] = ex.what();
    }
  }
  for (const std::string& err : errors)
    if (!err.empty()) throw std::runtime_error(err);

  J out;
  J jblocks = J::array();
  for (const ProfileBlock& pb : prof.blocks) jblocks.push_back(std::string(block_name(pb.block)));
  out["blocks"] = jblocks;
  out["eligibility"] = external ? "external" : "oracle";
  out["splits"] = {{"fit", {0, in.b_fv}}, {"val", {in.b_fv, in.b_test}}, {"test", {in.b_test, m}}};
  J sc;
  for (int s = 1; s < 3; ++s) sc[kSplitNames[s]] = scored[s];
  out["scored"] = have_pred ? sc : J();

  // By typology, then "ALL".
  std::map<std::string, std::vector<std::size_t>> by_typ;
  for (std::size_t i = 0; i < na; ++i) {
    by_typ[in.attempts[i].typology].push_back(i);
    by_typ["\x7f" "ALL"].push_back(i);  // sorts last
  }
  J jtyp;
  for (const auto& [key, idx] : by_typ) {
    const std::string name = key == "\x7f" "ALL" ? "ALL" : key;
    J t;
    t["attempts"] = idx.size();
    std::map<std::string, std::uint64_t> status = {{"complete", 0}, {"partial", 0}, {"unmatched", 0}};
    std::map<std::string, std::uint64_t> split = {{"fit", 0}, {"val", 0}, {"test", 0}, {"straddles", 0}, {"none", 0}};
    std::vector<double> spans;
    std::uint64_t complete = 0, edges = 0;
    std::vector<std::uint64_t> elig(nb + 1, 0), act(nb + 1, 0), elig_not_act(nb + 1, 0), elig_or(nb + 1, 0);
    std::array<std::uint64_t, 3> d_att{}, d_edges{}, d_pred{};
    std::array<double, 3> d_share{};
    for (std::size_t i : idx) {
      const CoverageAttempt& a = in.attempts[i];
      const Row& r = rows[i];
      ++status[a.status];
      ++split[a.split];
      edges += a.eids.size();
      if (a.status != "complete") continue;
      ++complete;
      spans.push_back(double(a.span) / 3600.0);
      bool any_e = false, any_a = false, any_o = false;
      for (std::size_t b = 0; b < nb; ++b) {
        const bool e = r.eligible[b] > 0, ac = r.activated[b] > 0, o = r.eligible_oracle[b] > 0;
        elig[b] += e;
        act[b] += ac;
        elig_not_act[b] += e && !ac;
        elig_or[b] += o;
        any_e |= e;
        any_a |= ac;
        any_o |= o;
      }
      elig[nb] += any_e;
      act[nb] += any_a;
      elig_not_act[nb] += any_e && !any_a;
      elig_or[nb] += any_o;
      for (int s = 1; s < 3; ++s)
        if (r.edges[s] > 0) {
          ++d_att[s];
          d_edges[s] += r.edges[s];
          d_pred[s] += r.predicted[s];
          d_share[s] += double(r.predicted[s]) / double(r.edges[s]);
        }
    }
    t["status"] = status;
    t["split"] = split;
    t["edges"] = edges;
    t["complete"] = complete;
    J sp;
    for (double q : {10.0, 25.0, 50.0, 75.0, 90.0}) sp["p" + std::to_string(int(q))] = null_or(percentile_linear(spans, q));
    sp["max"] = null_or(spans.empty() ? NAN : *std::max_element(spans.begin(), spans.end()));
    t["span_hours"] = sp;
    auto per_block = [&](const std::vector<std::uint64_t>& v) {
      J j;
      for (std::size_t b = 0; b < nb; ++b) j[std::string(block_name(prof.blocks[b].block))] = v[b];
      j["any"] = v[nb];
      return j;
    };
    t["eligible"] = per_block(elig);
    t["activated"] = per_block(act);
    t["eligible_not_activated"] = per_block(elig_not_act);
    if (external) t["eligible_oracle"] = per_block(elig_or);
    for (int s = 1; s < 3; ++s) {
      J d;
      d["attempts"] = d_att[s];
      d["edges"] = d_edges[s];
      if (have_pred && scored[s]) {
        d["predicted"] = d_pred[s];
        d["mean"] = d_att[s] ? J(d_share[s] / double(d_att[s])) : J();
        d["pooled"] = d_edges[s] ? J(double(d_pred[s]) / double(d_edges[s])) : J();
      } else {
        d["predicted"] = J();
        d["mean"] = J();
        d["pooled"] = J();
      }
      t[std::string("det_") + kSplitNames[s]] = d;
    }
    jtyp[name] = t;
  }
  out["by_typology"] = jtyp;

  // Positives, and base rates over every edge.
  std::vector<std::uint8_t> listed(m, 0);
  for (const CoverageAttempt& a : in.attempts)
    for (std::uint32_t e : a.eids) listed[e] = 1;
  // counts[b][split][kind]: kind 0 = all edges, 1 = positives; b = nb is "any".
  using Counts = std::vector<std::array<std::array<std::uint64_t, 2>, 3>>;
  Counts nz(nb + 1);
  std::array<std::array<std::uint64_t, 2>, 3> tot{};
  // pos[listed][split]: positives, those no block activates, predicted positives.
  std::array<std::array<std::uint64_t, 3>, 2> pos{}, pos_none{}, pos_pred{};
  const int nthreads = omp_get_max_threads();
  std::vector<Counts> nz_t(nthreads, Counts(nb + 1));
  std::vector<std::array<std::array<std::uint64_t, 2>, 3>> tot_t(nthreads);
  std::vector<std::array<std::array<std::uint64_t, 3>, 2>> pos_t(nthreads), none_t(nthreads), pred_t(nthreads);
#pragma omp parallel
  {
    const int t = omp_get_thread_num();
#pragma omp for schedule(static)
    for (std::int64_t ie = 0; ie < std::int64_t(m); ++ie) {
      const std::uint64_t e = std::uint64_t(ie);
      const Split s = split_of(e, in.b_fv, in.b_test);
      const int kind = g.label[e] != 0;
      ++tot_t[t][s][0];
      if (kind) ++tot_t[t][s][1];
      bool any = false;
      for (std::size_t b = 0; b < nb; ++b)
        if (row_nonzero(in.blocks[b], cols[b], e)) {
          any = true;
          ++nz_t[t][b][s][0];
          if (kind) ++nz_t[t][b][s][1];
        }
      if (any) {
        ++nz_t[t][nb][s][0];
        if (kind) ++nz_t[t][nb][s][1];
      }
      if (kind) {
        ++pos_t[t][listed[e]][s];
        if (!any) ++none_t[t][listed[e]][s];
        if (have_pred && in.pred[e] == kPredPositive) ++pred_t[t][listed[e]][s];
      }
    }
  }
  for (int t = 0; t < nthreads; ++t) {
    for (std::size_t b = 0; b <= nb; ++b)
      for (int s = 0; s < 3; ++s)
        for (int k = 0; k < 2; ++k) nz[b][s][k] += nz_t[t][b][s][k];
    for (int s = 0; s < 3; ++s) {
      for (int k = 0; k < 2; ++k) tot[s][k] += tot_t[t][s][k];
      for (int l = 0; l < 2; ++l) {
        pos[l][s] += pos_t[t][l][s];
        pos_none[l][s] += none_t[t][l][s];
        pos_pred[l][s] += pred_t[t][l][s];
      }
    }
  }
  auto per_split = [&](auto get) {
    J j;
    std::uint64_t sum = 0;
    for (int s = 0; s < 3; ++s) {
      j[kSplitNames[s]] = get(s);
      sum += get(s);
    }
    j["all"] = sum;
    return j;
  };
  J jp;
  for (int l = 1; l >= 0; --l) {
    const char* k = l ? "listed" : "unlisted";
    jp[k] = per_split([&](int s) { return pos[l][s]; });
  }
  J none;
  for (int l = 1; l >= 0; --l) none[l ? "listed" : "unlisted"] = per_split([&](int s) { return pos_none[l][s]; });
  jp["no_block_activates"] = none;
  if (have_pred) {
    J pr;
    for (int s = 1; s < 3; ++s) {
      if (!scored[s]) {
        pr[kSplitNames[s]] = J();
        continue;
      }
      J js;
      for (int l = 1; l >= 0; --l) js[l ? "listed" : "unlisted"] = {{"positives", pos[l][s]}, {"predicted", pos_pred[l][s]}};
      pr[kSplitNames[s]] = js;
    }
    jp["predicted"] = pr;
  }
  out["positives"] = jp;
  J br;
  br["edges"] = per_split([&](int s) { return tot[s][0]; });
  br["positives"] = per_split([&](int s) { return tot[s][1]; });
  J brb;
  for (std::size_t b = 0; b <= nb; ++b) {
    const std::string name = b < nb ? std::string(block_name(prof.blocks[b].block)) : "any";
    brb[name] = {{"edges", per_split([&](int s) { return nz[b][s][0]; })},
                 {"positives", per_split([&](int s) { return nz[b][s][1]; })}};
  }
  br["nonzero"] = brb;
  out["base_rates"] = br;

  J ja = J::array();
  for (std::size_t i = 0; i < na; ++i) {
    const CoverageAttempt& a = in.attempts[i];
    const Row& r = rows[i];
    J x;
    x["id"] = a.id;
    x["typology"] = a.typology;
    x["status"] = a.status;
    x["split"] = a.split;
    x["span"] = a.span;
    x["edges"] = {{"fit", r.edges[kFit]}, {"val", r.edges[kVal]}, {"test", r.edges[kTest]}};
    if (a.status == "complete") {
      x["eligible"] = r.eligible;
      if (external) x["eligible_oracle"] = r.eligible_oracle;
      x["activated"] = r.activated;
    }
    if (have_pred) {
      J p;
      for (int s = 1; s < 3; ++s) p[kSplitNames[s]] = scored[s] ? J(r.predicted[s]) : J();
      x["predicted"] = p;
    }
    ja.push_back(x);
  }
  out["attempts"] = ja;
  return out;
}

std::string coverage_dump(const nlohmann::ordered_json& j) {
  std::string s = "{\n";
  bool first = true;
  for (auto it = j.begin(); it != j.end(); ++it) {
    if (!first) s += ",\n";
    first = false;
    s += " " + nlohmann::ordered_json(it.key()).dump() + ": ";
    if (it.key() == "attempts" && it->is_array()) {
      s += "[\n";
      for (std::size_t a = 0; a < it->size(); ++a) s += "  " + (*it)[a].dump() + (a + 1 < it->size() ? ",\n" : "\n");
      s += " ]";
    } else {
      std::string v = it->dump(1);
      for (std::size_t p = 0; (p = v.find('\n', p)) != std::string::npos; p += 2) v.insert(p + 1, " ");
      s += v;
    }
  }
  return s + "\n}\n";
}

}  // namespace blz

#include "runtime/triggers.hpp"

#include <charconv>
#include <cmath>
#include <stdexcept>

namespace blz {

namespace {

std::uint64_t mix64(std::uint64_t z) {  // the splitmix64 finalizer
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

std::uint64_t parse_u64(const std::string& s, const std::string& spec) {
  std::uint64_t v = 0;
  const auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
  if (s.empty() || ec != std::errc() || p != s.data() + s.size())
    throw std::runtime_error("--triggers " + spec + ": '" + s + "' is not a non-negative integer");
  return v;
}

}  // namespace

bool sample_selected(double percent, std::uint64_t seed, std::uint64_t e) {
  const std::uint64_t h = mix64(mix64(seed + 0x9e3779b97f4a7c15ULL) ^ (e * 0xd1342543de82ef95ULL + 1));
  return double(h >> 11) * 0x1.0p-53 * 100.0 < percent;  // uniform in [0, 100)
}

nlohmann::json TriggerSel::json() const {
  nlohmann::json j = {{"spec", spec}, {"first", first}, {"last", last}, {"rows", rows()}, {"sampled", sampled}};
  if (sampled) {
    j["percent"] = percent;
    j["seed"] = seed;
  }
  return j;
}

TriggerSel slice(const TriggerSel& sel, std::uint64_t a, std::uint64_t b) {
  if (a >= b || b > sel.rows()) throw std::invalid_argument("slice: rows [a, b) must be a non-empty part of the selection");
  TriggerSel part;
  part.spec = sel.spec;
  part.sampled = sel.sampled;
  part.percent = sel.percent;
  part.seed = sel.seed;
  if (sel.sampled) {
    part.eids.assign(sel.eids.begin() + std::ptrdiff_t(a), sel.eids.begin() + std::ptrdiff_t(b));
    part.first = part.eids.front();
    part.last = std::uint64_t(part.eids.back()) + 1;
  } else {
    part.first = sel.first + a;
    part.last = sel.first + b;
  }
  return part;
}

TriggerSel select_triggers(const std::string& spec, std::uint64_t m, const Splits& splits) {
  TriggerSel s;
  s.spec = spec;
  s.last = m;
  const auto named = [&](std::uint64_t first, std::uint64_t last) {
    if (first > last || last > m)
      throw std::runtime_error("--triggers " + spec + ": the split [" + std::to_string(first) + ", " +
                               std::to_string(last) + ") does not lie in the graph's m = " + std::to_string(m));
    s.first = first;
    s.last = last;
  };
  if (spec == "all") {
  } else if (spec == "train") {
    named(0, splits.b_test);
  } else if (spec == "test") {
    named(splits.b_test, m);
  } else if (spec == "fit") {
    named(0, splits.b_fv);
  } else if (spec == "val") {
    named(splits.b_fv, splits.b_test);
  } else if (spec.rfind("range:", 0) == 0) {
    const std::string r = spec.substr(6);
    const auto c = r.find(':');
    if (c == std::string::npos) throw std::runtime_error("--triggers " + spec + ": expected range:a:b");
    s.first = parse_u64(r.substr(0, c), spec);
    s.last = parse_u64(r.substr(c + 1), spec);
    if (s.first > s.last) throw std::runtime_error("--triggers " + spec + ": empty range");
    if (s.last > m) throw std::runtime_error("--triggers " + spec + ": range past m = " + std::to_string(m));
  } else if (spec.rfind("sample:", 0) == 0) {
    const std::string r = spec.substr(7);
    const auto c = r.find(':');
    if (c == std::string::npos) throw std::runtime_error("--triggers " + spec + ": expected sample:<pct>:<seed>");
    const std::string pct = r.substr(0, c);
    std::size_t used = 0;
    try {
      s.percent = std::stod(pct, &used);
    } catch (const std::exception&) {
      used = 0;
    }
    if (used == 0 || used != pct.size() || !std::isfinite(s.percent) || s.percent <= 0 || s.percent > 100)
      throw std::runtime_error("--triggers " + spec + ": the percentage must be a number in (0, 100]");
    s.seed = parse_u64(r.substr(c + 1), spec);
    s.sampled = true;
    for (std::uint64_t e = 0; e < m; ++e)
      if (sample_selected(s.percent, s.seed, e)) s.eids.push_back(std::uint32_t(e));
    s.first = s.eids.empty() ? 0 : s.eids.front();
    s.last = s.eids.empty() ? 0 : std::uint64_t(s.eids.back()) + 1;
  } else {
    throw std::runtime_error("--triggers must be all, train, test, fit, val, range:a:b or sample:<pct>:<seed>, got '" +
                             spec + "'");
  }
  return s;
}

}  // namespace blz

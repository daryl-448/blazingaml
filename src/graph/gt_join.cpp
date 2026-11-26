#include "graph/gt_join.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "blz/edges.hpp"
#include "io/csv.hpp"

namespace blz {

namespace {

using Fields = std::array<std::string_view, 11>;

// Calls f(line_number, line) for every line, without its '\n' and a trailing '\r'.
template <class F>
void for_each_line(std::string_view text, F&& f) {
  std::size_t pos = 0, line = 0;
  while (pos < text.size()) {
    std::size_t end = text.find('\n', pos);
    if (end == std::string_view::npos) end = text.size();
    std::string_view l = text.substr(pos, end - pos);
    if (!l.empty() && l.back() == '\r') l.remove_suffix(1);
    f(++line, l);
    pos = end + 1;
  }
}

bool split_fields(std::string_view l, Fields& f) {
  std::size_t nf = 0, start = 0;
  for (std::size_t i = 0; i <= l.size(); ++i)
    if (i == l.size() || l[i] == ',') {
      if (nf == f.size()) return false;
      f[nf++] = l.substr(start, i - start);
      start = i + 1;
    }
  return nf == f.size();
}

bool parse_amount(std::string_view s, double& v) {
  const auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
  return !s.empty() && ec == std::errc() && p == s.data() + s.size();
}

// Field-wise identity (SPEC v2 §3.5): the strings of timestamp, banks and accounts, the amounts
// as decimals, the currencies and the format. The label is not compared.
struct FieldKey {
  std::string_view ts, from_bank, from_acct, to_bank, to_acct, recv_ccy, pay_ccy, format;
  double recv = 0, paid = 0;
  bool operator==(const FieldKey&) const = default;
};
bool field_key(const Fields& f, FieldKey& k) {
  k = {f[0], f[1], f[2], f[3], f[4], f[6], f[8], f[9]};
  return parse_amount(f[5], k.recv) && parse_amount(f[7], k.paid);
}
std::string primary_key(const Fields& f) {  // the string part used to find candidates
  std::string s;
  for (int i : {0, 1, 2, 3, 4}) (s += f[i]) += '\x1f';
  return s;
}

enum class Status { unique, resolved, ambiguous, unmatched };
enum class Rule { text, fields, none };
const char* status_name(Status s) {
  switch (s) {
    case Status::unique: return "unique";
    case Status::resolved: return "resolved-duplicate";
    case Status::ambiguous: return "ambiguous";
    case Status::unmatched: return "unmatched";
  }
  return "?";
}
const char* rule_name(Rule r) { return r == Rule::text ? "text" : r == Rule::fields ? "fields" : "none"; }

struct Row {
  std::size_t line = 0, attempt = 0;
  std::string_view text;
  Fields f;
  Status status = Status::unmatched;
  Rule rule = Rule::none;
  std::int64_t eid = -1;
  std::vector<std::uint32_t> candidates;  // ambiguous only
};

struct Attempt {
  std::size_t line = 0;
  std::string typology, header;
  std::vector<std::size_t> rows;
};

std::string_view trim(std::string_view s) {
  while (!s.empty() && s.front() == ' ') s.remove_prefix(1);
  while (!s.empty() && s.back() == ' ') s.remove_suffix(1);
  return s;
}

// A duplicate group (SPEC v2 §3.5): its listed rows (Patterns.txt order) and matching CSV rows.
struct Group {
  std::vector<std::size_t> listed;
  std::vector<std::uint32_t> eids;
};

void resolve(Group& g, Rule rule, std::vector<Row>& rows) {
  std::sort(g.eids.begin(), g.eids.end());  // stream order
  for (std::size_t k = 0; k < g.listed.size(); ++k) {
    Row& r = rows[g.listed[k]];
    r.rule = rule;
    if (g.eids.size() == g.listed.size()) {
      r.status = g.listed.size() == 1 ? Status::unique : Status::resolved;
      r.eid = g.eids[k];
    } else {
      r.status = Status::ambiguous;
      r.candidates = g.eids;
    }
  }
}

}  // namespace

nlohmann::ordered_json gt_join(std::string_view csv_text, std::string_view patterns_text, const GtGraph& g) {
  using json = nlohmann::ordered_json;
  const std::uint64_t m = g.file_row.size();
  if (g.ts.size() != m || g.label.size() != m) throw std::runtime_error("gt-join: graph arrays differ in length");

  // 1. Patterns.txt: attempts and their transaction rows.
  std::vector<Attempt> attempts;
  std::vector<Row> rows;
  bool open = false;
  for_each_line(patterns_text, [&](std::size_t ln, std::string_view l) {
    auto bad = [&](const std::string& msg) {
      throw std::runtime_error("patterns line " + std::to_string(ln) + ": " + msg);
    };
    static constexpr std::string_view kBegin = "BEGIN LAUNDERING ATTEMPT - ", kEnd = "END LAUNDERING ATTEMPT - ";
    if (l.rfind(kBegin, 0) == 0) {
      if (open) bad("BEGIN inside an attempt");
      std::string_view rest = l.substr(kBegin.size());
      const auto colon = rest.find(':');
      Attempt a;
      a.line = ln;
      a.typology = std::string(trim(rest.substr(0, colon)));
      if (colon != std::string_view::npos) a.header = std::string(trim(rest.substr(colon + 1)));
      if (a.typology.empty()) bad("no typology");
      attempts.push_back(std::move(a));
      open = true;
    } else if (l.rfind(kEnd, 0) == 0) {
      if (!open) bad("END outside an attempt");
      if (trim(l.substr(kEnd.size())) != attempts.back().typology) bad("END does not match its BEGIN");
      open = false;
    } else if (trim(l).empty()) {
      // blank lines separate attempts
    } else {
      if (!open) bad("a transaction outside an attempt");
      Row r;
      r.line = ln;
      r.attempt = attempts.size() - 1;
      r.text = l;
      if (!split_fields(l, r.f)) bad("expected 11 comma-separated fields");
      try {
        parse_timestamp(r.f[0]);
      } catch (const std::invalid_argument& e) {
        bad(e.what());
      }
      attempts.back().rows.push_back(rows.size());
      rows.push_back(std::move(r));
    }
  });
  if (open) throw std::runtime_error("patterns: the last attempt has no END");

  // 2. The CSV, rule "text": every data row whose text is a listed line.
  std::vector<std::uint32_t> eid_of_row(m);
  for (std::uint64_t e = 0; e < m; ++e) {
    if (g.file_row[e] >= m) throw std::runtime_error("gt-join: file_row out of range");
    eid_of_row[g.file_row[e]] = std::uint32_t(e);
  }
  std::vector<Group> groups;
  std::unordered_map<std::string_view, std::size_t> by_text;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    auto [it, fresh] = by_text.try_emplace(rows[i].text, groups.size());
    if (fresh) groups.emplace_back();
    groups[it->second].listed.push_back(i);
  }
  std::uint64_t data_rows = 0;
  bool header = true;
  auto data_lines = [&](auto&& f) {
    data_rows = 0;
    header = true;
    for_each_line(csv_text, [&](std::size_t, std::string_view l) {
      if (header) {
        if (l != kCsvHeader) throw std::runtime_error("csv line 1: unexpected header");
        header = false;
        return;
      }
      if (l.empty()) return;  // parse_csv allows one, after the final newline
      if (data_rows >= m)
        throw std::runtime_error("gt-join: the CSV has more data rows than the graph's m = " + std::to_string(m));
      f(data_rows, l);
      ++data_rows;
    });
  };
  data_lines([&](std::uint64_t r, std::string_view l) {
    if (auto it = by_text.find(l); it != by_text.end()) groups[it->second].eids.push_back(eid_of_row[r]);
  });
  if (data_rows != m)
    throw std::runtime_error("gt-join: the CSV has " + std::to_string(data_rows) + " data rows, the graph m = " +
                             std::to_string(m) + " (not the CSV the graph was prepared from?)");
  std::vector<std::size_t> deferred;
  for (Group& gr : groups) {
    if (gr.eids.empty()) deferred.insert(deferred.end(), gr.listed.begin(), gr.listed.end());
    else resolve(gr, Rule::text, rows);
  }

  // 3. Rule "fields" for the lines no CSV row matches as text.
  if (!deferred.empty()) {
    std::sort(deferred.begin(), deferred.end());
    std::vector<Group> fgroups;
    std::vector<FieldKey> fkeys;
    std::unordered_map<std::string, std::vector<std::size_t>> by_primary;  // -> fgroups
    for (std::size_t i : deferred) {
      FieldKey k;
      if (!field_key(rows[i].f, k)) continue;  // an amount that is not a decimal: stays unmatched
      std::size_t gi = 0;
      while (gi < fkeys.size() && !(fkeys[gi] == k)) ++gi;
      if (gi == fkeys.size()) {
        fkeys.push_back(k);
        fgroups.emplace_back();
        by_primary[primary_key(rows[i].f)].push_back(gi);
      }
      fgroups[gi].listed.push_back(i);
    }
    data_lines([&](std::uint64_t r, std::string_view l) {
      if (by_text.count(l)) return;  // a row that matches a listed line as text belongs to that line
      Fields f;
      if (!split_fields(l, f)) return;
      const auto it = by_primary.find(primary_key(f));
      if (it == by_primary.end()) return;
      FieldKey k;
      if (!field_key(f, k)) return;
      for (std::size_t gi : it->second)
        if (fkeys[gi] == k) fgroups[gi].eids.push_back(eid_of_row[r]);
    });
    for (Group& gr : fgroups)
      if (!gr.eids.empty()) resolve(gr, Rule::fields, rows);
  }

  // 4. Attempts, splits, and the summary.
  const std::uint64_t b_fv = g.splits.b_fv, train = g.splits.b_test;
  auto split_of = [&](std::uint64_t e) { return e < b_fv ? "fit" : e < train ? "val" : "test"; };
  std::set<std::uint32_t> mapped, ambiguous_cands;
  std::map<std::string, std::uint64_t> rows_by_status = {{"unique", 0}, {"resolved-duplicate", 0}, {"ambiguous", 0},
                                                         {"unmatched", 0}};
  std::map<std::string, std::uint64_t> rows_by_rule = {{"text", 0}, {"fields", 0}, {"none", 0}};
  std::uint64_t label0 = 0;
  for (const Row& r : rows) {
    ++rows_by_status[status_name(r.status)];
    ++rows_by_rule[rule_name(r.rule)];
    if (r.eid >= 0) {
      mapped.insert(std::uint32_t(r.eid));
      label0 += g.label[std::size_t(r.eid)] == 0;
    }
    ambiguous_cands.insert(r.candidates.begin(), r.candidates.end());
  }

  json jatt = json::array();
  std::map<std::string, std::uint64_t> att_by_status = {{"complete", 0}, {"partial", 0}, {"unmatched", 0}};
  json by_typ = json::object();
  for (std::size_t ai = 0; ai < attempts.size(); ++ai) {
    const Attempt& a = attempts[ai];
    std::size_t nmapped = 0;
    std::int64_t ts_min = INT64_MAX, ts_max = INT64_MIN;
    std::set<std::pair<std::string_view, std::string_view>> accounts;
    std::set<std::string> splits;
    json jrows = json::array();
    for (std::size_t ri : a.rows) {
      const Row& r = rows[ri];
      const std::int64_t t = parse_timestamp(r.f[0]);
      ts_min = std::min(ts_min, t);
      ts_max = std::max(ts_max, t);
      accounts.insert({r.f[1], r.f[2]});
      accounts.insert({r.f[3], r.f[4]});
      json jr = {{"line", r.line}, {"status", status_name(r.status)}, {"rule", rule_name(r.rule)},
                 {"eid", r.eid >= 0 ? json(r.eid) : json(nullptr)}};
      if (r.status == Status::ambiguous) jr["candidates"] = r.candidates;
      if (r.eid >= 0) {
        ++nmapped;
        splits.insert(split_of(std::uint64_t(r.eid)));
      }
      jrows.push_back(std::move(jr));
    }
    const std::string status = nmapped == a.rows.size() && !a.rows.empty() ? "complete" : nmapped == 0 ? "unmatched" : "partial";
    const json split = splits.empty() ? json(nullptr) : splits.size() == 1 ? json(*splits.begin()) : json("straddles");
    ++att_by_status[status];
    json& t = by_typ[a.typology];
    if (t.is_null()) t = {{"attempts", 0}, {"complete", 0}, {"partial", 0}, {"unmatched", 0}, {"rows", 0},
                          {"fit", 0}, {"val", 0}, {"test", 0}, {"straddles", 0}};
    t["attempts"] = t["attempts"].get<std::uint64_t>() + 1;
    t[status] = t[status].get<std::uint64_t>() + 1;
    t["rows"] = t["rows"].get<std::uint64_t>() + a.rows.size();
    if (!split.is_null()) t[split.get<std::string>()] = t[split.get<std::string>()].get<std::uint64_t>() + 1;
    jatt.push_back({{"id", ai},
                    {"line", a.line},
                    {"typology", a.typology},
                    {"header", a.header},
                    {"status", status},
                    {"split", split},
                    {"accounts", accounts.size()},
                    {"ts_min", a.rows.empty() ? json(nullptr) : json(ts_min)},
                    {"ts_max", a.rows.empty() ? json(nullptr) : json(ts_max)},
                    {"span", a.rows.empty() ? json(nullptr) : json(ts_max - ts_min)},
                    {"rows", std::move(jrows)}});
  }

  std::vector<std::uint32_t> unlisted;
  std::uint64_t positives = 0, unlisted_amb = 0;
  for (std::uint64_t e = 0; e < m; ++e)
    if (g.label[e]) {
      ++positives;
      if (!mapped.count(std::uint32_t(e))) {
        unlisted.push_back(std::uint32_t(e));
        unlisted_amb += ambiguous_cands.count(std::uint32_t(e));
      }
    }

  json summary = {{"attempts", attempts.size()},
                  {"listed_rows", rows.size()},
                  {"rows_by_status", rows_by_status},
                  {"rows_by_rule", rows_by_rule},
                  {"attempts_by_status", att_by_status},
                  {"attempts_by_typology", by_typ},
                  {"positives", positives},
                  {"mapped_eids", mapped.size()},
                  {"mapped_eids_with_label_0", label0},
                  {"unlisted_positives", unlisted.size()},
                  {"unlisted_positives_that_are_ambiguous_candidates", unlisted_amb}};
  return {{"splits", {{"fit", {0, b_fv}}, {"val", {b_fv, train}}, {"test", {train, m}}}},
          {"summary", summary},
          {"attempts", jatt},
          {"unlisted_positive_eids", unlisted}};
}

std::string gt_join_dump(const nlohmann::ordered_json& j) {
  std::string s = "{\n";
  bool first = true;
  for (auto it = j.begin(); it != j.end(); ++it) {
    if (!first) s += ",\n";
    first = false;
    s += " " + nlohmann::ordered_json(it.key()).dump() + ": ";
    if (it.key() == "attempts") {
      s += "[\n";
      for (std::size_t a = 0; a < it->size(); ++a) {
        nlohmann::ordered_json head = (*it)[a];
        const nlohmann::ordered_json rows = head["rows"];
        head.erase("rows");
        std::string h = head.dump();
        h.pop_back();  // the closing brace
        s += "  " + h + ", \"rows\": [\n";
        for (std::size_t r = 0; r < rows.size(); ++r) s += "   " + rows[r].dump() + (r + 1 < rows.size() ? ",\n" : "\n");
        s += "  ]}" + std::string(a + 1 < it->size() ? ",\n" : "\n");
      }
      s += " ]";
    } else if (it->is_array() && it->size() > 16) {
      s += "[\n";
      for (std::size_t i = 0; i < it->size(); ++i) {
        if (i % 16 == 0) s += "  ";
        s += (*it)[i].dump();
        if (i + 1 < it->size()) s += i % 16 == 15 ? ",\n" : ", ";
      }
      s += "\n ]";
    } else {
      std::string v = it->dump(1);
      for (std::size_t p = 0; (p = v.find('\n', p)) != std::string::npos; p += 2) v.insert(p + 1, " ");
      s += v;
    }
  }
  return s + "\n}\n";
}

}  // namespace blz

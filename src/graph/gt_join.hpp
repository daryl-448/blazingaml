#pragma once
// Ground-truth join (SPEC v2 §3.5): the laundering attempts listed in HI-Small_Patterns.txt,
// joined to the stream's eids.
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

namespace blz {

// What the join needs from the prepared stream.
struct GtGraph {
  std::span<const std::uint32_t> file_row;  // eid -> 0-based data row of the CSV
  std::span<const std::int32_t> ts;
  std::span<const std::uint8_t> label;
};

// Joins the attempts of `patterns_text` (Patterns.txt) to the rows of `csv_text` (the CSV the
// graph was prepared from) and through file_row to eids.
//
// - A listed transaction line matches the CSV rows with identical text (rule "text"; a trailing
//   CR is ignored). With none, a field-wise match is tried (rule "fields") against the CSV rows
//   that match no listed line as text: timestamp, banks and accounts as strings, both amounts
//   parsed as decimals, both currencies and the format.
// - Identical lines form a duplicate group: the CSV rows that match, and the listed lines that
//   match them. A group of one CSV row and one listed line is `unique`. A group with as many
//   CSV rows as listed lines (> 1) maps the k-th listed line (file order of Patterns.txt) to the
//   k-th CSV row in stream order (`resolved-duplicate`). Any other group is `ambiguous`, and its
//   lines carry the candidate eids. A line with no match is `unmatched`.
// - An attempt is `complete` when every row is unique or resolved, `unmatched` when none is,
//   and `partial` otherwise. Its split is fit, val or test when all its mapped eids lie in that
//   split (SPEC v2 §3.3), else `straddles` (null with no mapped eid). Its span is max ts - min ts
//   over every listed row, and its account count the distinct (bank, account) pairs they touch.
// - Unlisted positives are the positive eids that no unique or resolved row maps to.
//
// Throws std::runtime_error on a malformed Patterns.txt (with its line number), a CSV header
// other than the expected one, or a CSV whose data-row count differs from the graph's m.
nlohmann::ordered_json gt_join(std::string_view csv_text, std::string_view patterns_text, const GtGraph& g);

// The join as text: valid JSON with one line per attempt header, per row and per 16 unlisted
// eids, so that the committed file diffs well.
std::string gt_join_dump(const nlohmann::ordered_json& j);

}  // namespace blz

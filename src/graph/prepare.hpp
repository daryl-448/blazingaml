#pragma once
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "blz/edges.hpp"
#include "io/csv.hpp"
#include "io/dataset.hpp"

namespace blz {

// The stream from the file-order table (SPEC §3.2-3.3, v3 §6.1-§6.2), consumed one column at a
// time, so at most one column is held twice: the epoch is midnight (no time zone) of the
// earliest timestamp's day and ts counts int32 seconds from it (std::runtime_error when the range
// does not fit); a stable sort by timestamp (D-4) whose order is file_row; node ids by first
// appearance in stream order, source before destination (D-3); sorted dictionaries.
EdgeTable build_edge_table(RawTable&& raw);

// SPEC v3 §6.1's counts of a stream with these splits, named as kInvariantNames: edges, nodes,
// self_loops (equal node ids, so the exact (bank, account) key), distinct_timestamps, positives,
// b_fv, b_test and the positives of fit, val and test; with `concatenated_keys`, the distinct
// bank + account strings (SPEC v2 §4.4 R1: equal to nodes iff no two pairs collide).
Invariants count_invariants(const EdgeTable& t, const Splits& s, bool concatenated_keys);

// SPEC v3 §6.1: the stream's accounts that no node-table row names (unresolved_accounts) and the
// node-table rows, (bank, account), that name no account of the stream (unmatched_node_rows),
// with up to `examples` of each as "bank,account".
struct NodeTableCounts {
  std::uint64_t unresolved_accounts = 0, unmatched_node_rows = 0;
  std::vector<std::string> unresolved_examples, unmatched_examples;
};
NodeTableCounts resolve_node_tables(const EdgeTable& t, std::span<const std::pair<std::string, std::string>> rows,
                                    std::size_t examples = 10);

// One message per difference between the counts and the pinned values, including a count that
// is not pinned and a pinned value that is not counted (empty = pass).
std::vector<std::string> check_invariants(const Invariants& got, const Invariants& pinned);

// Whether file_row is a permutation of 0..m-1 under which the stream is the stable sort of the
// CSV rows by timestamp: for consecutive eids, ts increases, or ts ties and file_row increases.
// Returns "" if so, else the first violation.
std::string check_file_row(std::span<const std::uint32_t> file_row, std::span<const std::int32_t> ts);

}  // namespace blz

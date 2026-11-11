#pragma once
#include <span>
#include <string>
#include <vector>

#include "blz/edges.hpp"
#include "io/csv.hpp"

namespace blz {

// Stable sort by timestamp (D-4), dense node ids by first appearance in stream order, source
// before destination (D-3), and sorted attribute dictionaries (SPEC §3.2-3.3).
EdgeTable build_edge_table(const RawTable& raw);

struct StreamCounts {
  std::uint64_t edges = 0, nodes = 0, positives = 0, self_loops = 0;
  std::uint64_t train_positives = 0, test_positives = 0;
  // SPEC v2 §3.3: the fit/validation boundary and the positives on each side.
  std::uint64_t fit_val_boundary = 0, fit_positives = 0, val_positives = 0;
  // SPEC v2 §4.4 R1: distinct node keys when a key is the bank string and the account string
  // concatenated with no separator (IBM Multi-GNN's key). Equal to `nodes` iff no two (bank,
  // account) pairs collide under concatenation.
  std::uint64_t concatenated_keys = 0;
};
StreamCounts count_stream(const EdgeTable& t);

// The SPEC §3.5 invariants for HI-Small, with the v2 additions (§3.3, §3.5, R1): the fit /
// validation counts, file_row a permutation of the CSV rows that reproduces the stable sort, and
// concatenated node keys as distinct as the (bank, account) pairs. Returns one message per
// violation (empty = pass).
std::vector<std::string> check_hi_small_invariants(const EdgeTable& t);

// Whether file_row is a permutation of 0..m-1 under which the stream is the stable sort of the
// CSV rows by timestamp: for consecutive eids, ts increases, or ts ties and file_row increases.
// Returns "" if so, else the first violation.
std::string check_file_row(std::span<const std::uint32_t> file_row, std::span<const std::int32_t> ts);

}  // namespace blz

#pragma once
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
};
StreamCounts count_stream(const EdgeTable& t);

// The SPEC §3.5 invariants for HI-Small. Returns one message per violation (empty = pass).
std::vector<std::string> check_hi_small_invariants(const EdgeTable& t);

}  // namespace blz

#pragma once
// T4 of SPEC v3 §4.1, checked in L2 (§4.3): the contracts of the index primitives
// (include/blz/primitives.hpp) on the index that GraphIndex builds, each against a naive definition
// over the edge list, for every graph of an exhaustive bound, both self-loop modes, both directions.
#include <array>
#include <cstdint>
#include <string>

#include "graph/exhaustive.hpp"
#include "graph/index.hpp"

namespace blz {

// Each contract, in the terms of x's list in one direction: the edges e with src e = x (out) or
// dst e = x (in), in eid order, without self-loops under SelfLoops::drop; an entry's neighbour is
// the other endpoint. Queries take every node x (and y), every timestamp of the graph and its
// neighbours t - 1 and t + 1 as time bounds (plus INT64_MIN, the generated code's "no lower bound",
// and kNoUpper), and every eid 0..m plus kAnyEid as the eid bound.
enum class Contract {
  list,               // off, nbr, ts, eid: x's entries are x's list
  prev_same,          // the previous entry of x's list with the same neighbour, or -1
  range,              // range(x, lo, hi, e): the entries with lo <= ts <= hi and eid <= e, contiguous
  head_range,         // head_range(x, hi, e): the same from x's first entry, no lower bound
  distinct_first,     // distinct_first(j, lo): no entry in [lo, j) has j's neighbour
  distinct_first_ts,  // distinct_first_ts(j, lo): no earlier entry with j's neighbour has ts >= lo
  pair_exists,        // pair_exists(x, y, lo, hi, e): the earliest ts of an entry with neighbour y in those bounds, or kNoTs
  pair_latest,        // pair_latest(x, y, lo, hi, e): the latest one, or kNoTs
};
inline constexpr std::size_t kContracts = 8;
const char* contract_name(Contract c);

struct IndexContracts {
  std::uint64_t graphs = 0;
  std::array<std::uint64_t, kContracts> checks{};      // queries compared, per contract
  std::array<std::uint64_t, kContracts> violations{};  // of them, answers that differ
  // The first violation of the graph with the smallest index, in the code's terms.
  std::uint64_t first_graph = UINT64_MAX;
  std::string first;
};

// Graph g (canonical index `id`) in self-loop mode `mode`, both directions: adds its checks and
// violations to `res`, and describes its first violation if res has none yet.
void check_index_contracts(const EdgeList& g, std::uint64_t id, SelfLoops mode, IndexContracts& res);
// Every graph of `gs` in both self-loop modes, in parallel over graphs.
IndexContracts check_index_contracts(const ExhaustiveGraphs& gs);

}  // namespace blz

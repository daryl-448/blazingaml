#pragma once
// Dataset descriptors (SPEC v3 §6.1): configs/datasets/<id>.yaml, what `blz prepare` reads.
#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "blz/edges.hpp"
#include "common/sha256.hpp"
#include "io/csv.hpp"

namespace blz {

// Counts by name, as a descriptor pins them and bench/p15_counts.py prints them.
using Invariants = std::map<std::string, std::uint64_t>;

// Every invariant a descriptor may pin (SPEC v3 §6.1). concatenated_keys belongs to a schema with
// `concatenated_keys`, the last two to a dataset with node tables.
inline constexpr std::array<std::string_view, 13> kInvariantNames = {
    "edges",         "nodes",         "self_loops",     "distinct_timestamps", "positives",
    "b_fv",          "b_test",        "fit_positives",  "val_positives",       "test_positives",
    "concatenated_keys", "unresolved_accounts", "unmatched_node_rows"};

// A table of account attributes, keyed by its bank and account columns (TransXion's person.csv
// and merchant.csv).
struct NodeTable {
  std::string file, bank_column, account_column;
};

struct DatasetDescriptor {
  std::string file;          // the descriptor's path
  std::string id;            // an identifier, stored in the .blzg
  std::string transactions;  // the CSV; paths are relative to the working directory
  const CsvSchema* schema = nullptr;
  SplitRule split = SplitRule::legacy_hi_small;
  std::string patterns;  // Patterns.txt, or empty
  std::vector<NodeTable> node_tables;
  Invariants invariants;
  Sha256Digest sha256{};  // of the descriptor's bytes
};

// Keys: id, transactions, schema (ibm | transxion), split (legacy_hi_small | temporal_60_20_20),
// invariants (a map of kInvariantNames to non-negative integers), and optionally patterns and
// node_tables (a list of {file, bank, account}). Throws std::runtime_error, located at
// file:line:column, on an unknown, duplicate or missing key or a bad value.
DatasetDescriptor parse_dataset(std::string_view yaml_text, const std::string& file);
DatasetDescriptor load_dataset(const std::string& path);

}  // namespace blz

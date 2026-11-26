// Dataset descriptors (SPEC v3 §6.1): configs/datasets/<id>.yaml.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <string>

#include "io/dataset.hpp"

using namespace blz;
using Catch::Matchers::ContainsSubstring;

namespace {

const std::string kGood =
    "id: tiny\n"
    "transactions: dataset/tx.csv\n"
    "schema: transxion\n"
    "split: temporal_60_20_20\n"
    "node_tables:\n"
    "  - {file: dataset/person.csv, bank: bank, account: bank_account_number}\n"
    "invariants:\n"
    "  edges: 3029170\n"
    "  unresolved_accounts: 0\n";

std::string error(const std::string& yaml) {
  try {
    parse_dataset(yaml, "d.yaml");
  } catch (const std::runtime_error& e) {
    return e.what();
  }
  return "";
}

}  // namespace

TEST_CASE("dataset descriptors: a valid one") {
  const DatasetDescriptor d = parse_dataset(kGood, "d.yaml");
  CHECK(d.id == "tiny");
  CHECK(d.transactions == "dataset/tx.csv");
  CHECK(d.schema == &kTransxionSchema);
  CHECK(d.split == SplitRule::temporal_60_20_20);
  CHECK(d.patterns.empty());
  REQUIRE(d.node_tables.size() == 1);
  CHECK(d.node_tables[0].file == "dataset/person.csv");
  CHECK(d.node_tables[0].account_column == "bank_account_number");
  CHECK(d.invariants == Invariants{{"edges", 3029170}, {"unresolved_accounts", 0}});
  CHECK(d.sha256 == sha256(kGood.data(), kGood.size()));
}

TEST_CASE("dataset descriptors: unknown keys and bad values are located errors") {
  CHECK_THAT(error(kGood + "colour: red\n"), ContainsSubstring("d.yaml:10:1: unknown key 'colour'"));
  CHECK_THAT(error(kGood + "  edgs: 1\n"), ContainsSubstring("unknown key 'edgs' in invariants"));
  CHECK_THAT(error(kGood + "  nodes: -1\n"), ContainsSubstring("invariant nodes must be a non-negative integer"));
  CHECK_THAT(error(kGood + "  nodes: 1.5\n"), ContainsSubstring("invariant nodes must be a non-negative integer"));
  auto replace = [](std::string s, const std::string& from, const std::string& to) {
    return s.replace(s.find(from), from.size(), to);
  };
  CHECK_THAT(error(replace(kGood, "schema: transxion", "schema: csv")), ContainsSubstring("schema must be ibm or transxion"));
  CHECK_THAT(error(replace(kGood, "split: temporal_60_20_20", "split: random")),
             ContainsSubstring("split must be legacy_hi_small or temporal_60_20_20"));
  CHECK_THAT(error(replace(kGood, "id: tiny\n", "")), ContainsSubstring("needs 'id'"));
  CHECK_THAT(error(replace(kGood, "id: tiny", "id: a-b")), ContainsSubstring("'id' must be letters"));
  CHECK_THAT(error(replace(kGood, "bank: bank,", "bank: bank, key: x,")), ContainsSubstring("unknown key 'key' in a node table"));
  CHECK_THAT(error(replace(kGood, ", account: bank_account_number", "")), ContainsSubstring("a node table needs 'account'"));
  CHECK_THAT(error(kGood.substr(0, kGood.find("invariants:"))), ContainsSubstring("needs 'invariants'"));
  CHECK_THAT(error("id: [1\n"), ContainsSubstring("d.yaml"));
}

TEST_CASE("dataset descriptors: the repository's four parse") {
  for (const char* id : {"hi_small", "li_small", "hi_medium", "transxion"}) {
    const DatasetDescriptor d = load_dataset(std::string(BLZ_SOURCE_DIR) + "/configs/datasets/" + id + ".yaml");
    INFO(id);
    CHECK(d.id == id);
    CHECK(d.invariants.count("edges") == 1);
    CHECK(d.invariants.count("concatenated_keys") == (d.schema == &kIbmSchema));
    CHECK(d.invariants.count("unresolved_accounts") == !d.node_tables.empty());
  }
  const DatasetDescriptor hs = load_dataset(std::string(BLZ_SOURCE_DIR) + "/configs/datasets/hi_small.yaml");
  CHECK(hs.split == SplitRule::legacy_hi_small);
  CHECK(hs.invariants.at("b_fv") == 3'047'216);  // SPEC v2 §3.3
}

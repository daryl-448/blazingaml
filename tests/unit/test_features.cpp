#include <catch2/catch_test_macros.hpp>
#include <vector>

#include "runtime/features.hpp"

using namespace blz;

TEST_CASE("bin_index: bin edges, open last bin, drops below b_0 and below 2") {
  const std::vector<std::uint32_t> bins = {2, 3, 5, 8};
  CHECK(bin_index(bins, 0) == -1);
  CHECK(bin_index(bins, 1) == -1);
  CHECK(bin_index(bins, 2) == 0);
  CHECK(bin_index(bins, 3) == 1);
  CHECK(bin_index(bins, 4) == 1);
  CHECK(bin_index(bins, 5) == 2);
  CHECK(bin_index(bins, 7) == 2);
  CHECK(bin_index(bins, 8) == 3);
  CHECK(bin_index(bins, 1000000) == 3);
  const std::vector<std::uint32_t> from1 = {1, 2, 3};  // min size 2 even when bin 1 exists
  CHECK(bin_index(from1, 1) == -1);
  CHECK(bin_index(from1, 2) == 1);
  const std::vector<std::uint32_t> from4 = {4, 6};
  CHECK(bin_index(from4, 3) == -1);
  CHECK(bin_index(from4, 4) == 0);
}

TEST_CASE("add_saturating") {
  std::uint32_t c = UINT32_MAX - 1;
  CHECK_FALSE(add_saturating(c));
  CHECK(c == UINT32_MAX);
  CHECK(add_saturating(c));
  CHECK(c == UINT32_MAX);
}

TEST_CASE("column names") {
  const std::vector<std::uint32_t> bins = {2, 3, 5};
  CHECK(column_names(Block::sg, bins) ==
        std::vector<std::string>{"sg[2,3)", "sg[3,5)", "sg[5,inf)"});
}

TEST_CASE("feature config parsing") {
  auto cfg = parse_feature_config(
      "k_max: 7\nblocks:\n  sg: {window: 60, bins: [2, 4]}\n  fan_in: {window: 5, bins: [3]}\n", "t");
  CHECK(cfg.k_max == 7);
  REQUIRE(cfg[Block::sg]);
  CHECK(cfg[Block::sg]->window == 60);
  CHECK(cfg[Block::sg]->bins == std::vector<std::uint32_t>{2, 4});
  CHECK(cfg[Block::fan_in]->bins == std::vector<std::uint32_t>{3});
  CHECK_FALSE(cfg[Block::tcyc]);

  CHECK_THROWS(parse_feature_config("blocks:\n  foo: {window: 1, bins: [2]}\n", "t"));
  CHECK_THROWS(parse_feature_config("blocks:\n  sg: {window: 0, bins: [2]}\n", "t"));
  CHECK_THROWS(parse_feature_config("blocks:\n  sg: {window: 1, bins: [3, 3]}\n", "t"));
  CHECK_THROWS(parse_feature_config("blocks:\n  sg: {window: 1, bins: []}\n", "t"));
  CHECK_THROWS(parse_feature_config("blocks:\n  sg: {window: 1, bins: [2], lower: open}\n", "t"));
  CHECK_THROWS(parse_feature_config("k_max: 1\nblocks:\n  sg: {window: 1, bins: [2]}\n", "t"));
  CHECK_THROWS(parse_feature_config("kmax: 3\nblocks:\n  sg: {window: 1, bins: [2]}\n", "t"));
  CHECK_THROWS(parse_feature_config("blocks: {}\n", "t"));
}

TEST_CASE("committed configs load (SPEC §4.8)") {
  const std::string dir = std::string(BLZ_SOURCE_DIR) + "/configs/";
  const auto def = load_feature_config(dir + "features_default.yaml");
  CHECK(def.k_max == 10);
  std::size_t cols = 0;
  for (Block b : kAllBlocks) {
    REQUIRE(def[b]);
    cols += def[b]->bins.size();
    CHECK(def[b]->window == (b == Block::sg ? 21600 : 86400));
  }
  CHECK(cols == 67);
  const auto par = load_feature_config(dir + "features_parity.yaml");
  for (Block b : kAllBlocks) {
    REQUIRE(par[b]);
    CHECK(par[b]->window == def[b]->window);
    CHECK(par[b]->bins.size() == 32);
    CHECK(par[b]->bins.front() == 2);
    CHECK(par[b]->bins.back() == 33);
  }
}

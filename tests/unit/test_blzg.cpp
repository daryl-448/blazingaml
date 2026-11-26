#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

#include "fixtures.hpp"
#include "io/blzg.hpp"

using namespace blz;

template <class T, class U>
static std::vector<T> vec(std::span<const U> s) { return {s.begin(), s.end()}; }

static std::vector<std::string> strs(const StringTable& t) {
  std::vector<std::string> v;
  for (std::size_t i = 0; i < t.size(); ++i) v.emplace_back(t[i]);
  return v;
}

namespace {

DatasetInfo info(const EdgeTable& t, SplitRule rule = SplitRule::legacy_hi_small) {
  return {"fixture", &kIbmSchema, rule, compute_splits(rule, t.ts), sha256("descriptor", 10)};
}

std::vector<char> file_bytes(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), {}};
}

}  // namespace

TEST_CASE(".blzg round trip") {
  const EdgeTable t = prepare_text(kFixtureCsv);
  const std::string path = test_tmp("rt.blzg");
  write_blzg(path, t, info(t, SplitRule::temporal_60_20_20));
  BlzgView v(path);
  CHECK(v.m() == t.m());
  CHECK(v.n() == t.n());
  CHECK(v.csv_sha256() == t.csv_sha256);
  CHECK(vec<std::uint32_t>(v.src) == t.src);
  CHECK(vec<std::uint32_t>(v.dst) == t.dst);
  CHECK(vec<std::int32_t>(v.ts) == t.ts);
  CHECK(vec<std::uint8_t>(v.label) == t.label);
  CHECK(vec<double>(v.amount_paid) == t.amount_paid);
  CHECK(vec<double>(v.amount_received) == t.amount_received);
  CHECK(vec<std::uint8_t>(v.pay_ccy) == t.pay_ccy);
  CHECK(vec<std::uint8_t>(v.recv_ccy) == t.recv_ccy);
  CHECK(vec<std::uint8_t>(v.format) == t.format);
  CHECK(vec<std::uint32_t>(v.file_row) == t.file_row);
  CHECK(strs(v.ccy_dict) == t.ccy_dict);
  CHECK(strs(v.format_dict) == t.format_dict);
  CHECK(strs(v.node_bank) == t.node_bank);
  CHECK(strs(v.node_account) == t.node_account);
  CHECK(reinterpret_cast<std::uintptr_t>(v.amount_paid.data()) % 64 == 0);
  // The dataset section (SPEC v3 P15).
  CHECK(v.epoch() == t.epoch);
  CHECK(v.dataset().id == "fixture");
  CHECK(v.dataset().schema == &kIbmSchema);
  CHECK(v.dataset().split == SplitRule::temporal_60_20_20);
  CHECK(v.splits() == compute_splits(SplitRule::temporal_60_20_20, t.ts));
  CHECK(v.dataset().descriptor_sha256 == sha256("descriptor", 10));
}

TEST_CASE(".blzg rejects malformed files") {
  const EdgeTable t = prepare_text(kFixtureCsv);
  const std::string path = test_tmp("bad.blzg");
  write_blzg(path, t, info(t));
  const std::vector<char> bytes = file_bytes(path);
  auto write = [&](const std::vector<char>& b) {
    std::ofstream(path, std::ios::binary | std::ios::trunc).write(b.data(), std::streamsize(b.size()));
  };
  auto truncated = bytes;
  truncated.resize(bytes.size() - 1);
  write(truncated);
  CHECK_THROWS(BlzgView(path));
  auto magic = bytes;
  magic[0] = 'X';
  write(magic);
  CHECK_THROWS(BlzgView(path));
  auto badnode = bytes;
  std::uint32_t big = 99;
  std::memcpy(badnode.data() + 64, &big, 4);  // src[0]
  write(badnode);
  CHECK_THROWS(BlzgView(path));
  CHECK_THROWS(BlzgView(test_tmp("missing.blzg")));
}

TEST_CASE(".blzg version 3: file_row and the splits are checked, versions 1 and 2 are rejected") {
  const EdgeTable t = prepare_text(kFixtureCsv);
  const std::string path = test_tmp("v3.blzg");
  write_blzg(path, t, info(t));
  const std::vector<char> bytes = file_bytes(path);
  std::uint32_t version = 0;
  std::memcpy(&version, bytes.data() + 4, 4);
  CHECK(version == 3);
  for (std::uint32_t old : {1u, 2u}) {
    auto b = bytes;
    std::memcpy(b.data() + 4, &old, 4);
    std::ofstream(path, std::ios::binary | std::ios::trunc).write(b.data(), std::streamsize(b.size()));
    try {
      BlzgView v(path);
      FAIL("a version " << old << " file loaded");
    } catch (const std::runtime_error& e) {
      CHECK(std::string(e.what()).find("re-run `blz prepare") != std::string::npos);
    }
  }
  // A file_row that is not a permutation, and one that does not reproduce the stable sort.
  auto bad = t;
  bad.file_row[0] = bad.file_row[1];
  write_blzg(path, bad, info(t));
  CHECK_THROWS(BlzgView(path));
  bad = t;
  std::swap(bad.file_row[0], bad.file_row[1]);  // eids 0 and 1 tie at ts 0
  write_blzg(path, bad, info(t));
  CHECK_THROWS(BlzgView(path));
  // Splits that are not the rule's on the timestamps.
  DatasetInfo moved = info(t);
  moved.splits.b_fv = 1;
  write_blzg(path, t, moved);
  CHECK_THROWS(BlzgView(path));
  moved.splits = {3, 2};
  CHECK_THROWS_AS(write_blzg(path, t, moved), std::invalid_argument);
}

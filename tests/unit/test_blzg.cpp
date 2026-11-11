#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

#include "fixtures.hpp"
#include "graph/prepare.hpp"
#include "io/blzg.hpp"

using namespace blz;

template <class T, class U>
static std::vector<T> vec(std::span<const U> s) { return {s.begin(), s.end()}; }

static std::vector<std::string> strs(const StringTable& t) {
  std::vector<std::string> v;
  for (std::size_t i = 0; i < t.size(); ++i) v.emplace_back(t[i]);
  return v;
}

TEST_CASE(".blzg round trip") {
  EdgeTable t = build_edge_table(parse_csv(kFixtureCsv));
  t.csv_sha256 = sha256(kFixtureCsv.data(), kFixtureCsv.size());
  const std::string path = test_tmp("rt.blzg");
  write_blzg(path, t);
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
}

TEST_CASE(".blzg rejects malformed files") {
  EdgeTable t = build_edge_table(parse_csv(kFixtureCsv));
  const std::string path = test_tmp("bad.blzg");
  write_blzg(path, t);
  std::vector<char> bytes;
  {
    std::ifstream in(path, std::ios::binary);
    bytes.assign(std::istreambuf_iterator<char>(in), {});
  }
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

TEST_CASE(".blzg version 2: file_row is checked, version 1 is rejected (SPEC v2 §3.5, D-31)") {
  EdgeTable t = build_edge_table(parse_csv(kFixtureCsv));
  const std::string path = test_tmp("v2.blzg");
  write_blzg(path, t);
  std::vector<char> bytes;
  {
    std::ifstream in(path, std::ios::binary);
    bytes.assign(std::istreambuf_iterator<char>(in), {});
  }
  std::uint32_t version = 0;
  std::memcpy(&version, bytes.data() + 4, 4);
  CHECK(version == 2);
  auto v1 = bytes;
  version = 1;
  std::memcpy(v1.data() + 4, &version, 4);
  std::ofstream(path, std::ios::binary | std::ios::trunc).write(v1.data(), std::streamsize(v1.size()));
  try {
    BlzgView v(path);
    FAIL("a version 1 file loaded");
  } catch (const std::runtime_error& e) {
    CHECK(std::string(e.what()).find("re-run `blz prepare`") != std::string::npos);
  }
  // A file_row that is not a permutation, and one that does not reproduce the stable sort.
  auto bad = t;
  bad.file_row[0] = bad.file_row[1];
  write_blzg(path, bad);
  CHECK_THROWS(BlzgView(path));
  bad = t;
  std::swap(bad.file_row[0], bad.file_row[1]);  // eids 0 and 1 tie at ts 0
  write_blzg(path, bad);
  CHECK_THROWS(BlzgView(path));
}

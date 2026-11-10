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

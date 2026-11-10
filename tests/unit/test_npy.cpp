#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <fstream>
#include <vector>

#include "common/npy.hpp"
#include "fixtures.hpp"

TEST_CASE(".npy int64 writer") {
  const std::vector<std::int64_t> data = {0, 1, 2, 3, 4, 5, 6, -7};
  const std::string path = test_tmp("x.npy");
  blz::write_npy_i64(path, data, 2, 4);
  std::ifstream in(path, std::ios::binary);
  std::vector<char> b(std::istreambuf_iterator<char>(in), {});
  REQUIRE(b.size() > 10);
  CHECK(std::memcmp(b.data(), "\x93NUMPY\x01\x00", 8) == 0);
  const std::size_t hlen = std::uint8_t(b[8]) | (std::size_t(std::uint8_t(b[9])) << 8);
  CHECK((10 + hlen) % 64 == 0);
  const std::string dict(b.data() + 10, hlen);
  CHECK(dict.find("'descr': '<i8'") != std::string::npos);
  CHECK(dict.find("'shape': (2, 4)") != std::string::npos);
  CHECK(dict.back() == '\n');
  REQUIRE(b.size() == 10 + hlen + data.size() * 8);
  std::vector<std::int64_t> got(data.size());
  std::memcpy(got.data(), b.data() + 10 + hlen, data.size() * 8);
  CHECK(got == data);
  CHECK_THROWS(blz::write_npy_i64(path, data, 3, 4));
}

TEST_CASE(".npy uint32 writer and reader round trip") {
  const std::vector<std::uint32_t> data = {0, 1, 2, 3, 4, 5, 0xffffffffu, 7, 8};
  const std::string path = test_tmp("u.npy");
  blz::write_npy_u32(path, data, 3, 3);
  blz::NpyU32View v(path);
  CHECK(v.rows() == 3);
  CHECK(v.cols() == 3);
  CHECK(std::vector<std::uint32_t>(v.data().begin(), v.data().end()) == data);
  CHECK(v.row(2)[1] == 7);
  blz::write_npy_i64(path, std::vector<std::int64_t>{1, 2}, 1, 2);
  CHECK_THROWS(blz::NpyU32View(path));  // wrong dtype
}

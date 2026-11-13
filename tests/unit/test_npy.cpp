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

TEST_CASE(".npy uint8 reader, 1-D and 2-D (blz coverage --pred)") {
  // What numpy.save writes for a 1-D uint8 array.
  auto write = [](const std::string& path, const std::string& shape, const std::vector<std::uint8_t>& data) {
    std::string dict = "{'descr': '|u1', 'fortran_order': False, 'shape': " + shape + ", }";
    dict.append((64 - (10 + dict.size() + 1) % 64) % 64, ' ');
    dict += '\n';
    std::string h = std::string("\x93NUMPY\x01\x00", 8) + char(dict.size() & 0xff) + char(dict.size() >> 8) + dict;
    std::ofstream out(path, std::ios::binary);
    out.write(h.data(), std::streamsize(h.size()));
    out.write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size()));
  };
  const std::vector<std::uint8_t> data = {255, 0, 1, 1, 255};
  const std::string path = test_tmp("p.npy");
  write(path, "(5,)", data);
  blz::NpyU8View v(path);
  CHECK(v.rows() == 5);
  CHECK(v.cols() == 1);
  CHECK(std::vector<std::uint8_t>(v.data().begin(), v.data().end()) == data);
  write(path, "(5, 1)", data);
  CHECK(blz::NpyU8View(path).rows() == 5);
  write(path, "(6,)", data);
  CHECK_THROWS(blz::NpyU8View(path));  // data size does not match the shape
  CHECK_THROWS(blz::NpyU32View(path));  // wrong dtype
}

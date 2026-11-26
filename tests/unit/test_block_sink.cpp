// SPEC v3 §5.1, H7: the block sinks. The cell hashes equal FNV-1a over the bytes, a stream's hash
// continued across chunks equals the whole block's, and an unfinished .npy stream leaves no file.
#include <catch2/catch_test_macros.hpp>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "common/npy.hpp"
#include "fixtures.hpp"
#include "runtime/block_sink.hpp"

using namespace blz;

namespace {

// Cells that reach every path of fnv1a_cells: runs of zero cells of every length around 8 (the
// 32-byte step), small values, the byte and short boundaries, and random 32-bit values.
std::vector<std::uint32_t> test_cells(std::uint64_t seed, std::size_t n) {
  std::vector<std::uint32_t> v(n);
  std::uint64_t x = seed * 0x9e3779b97f4a7c15ULL + 1;
  for (std::size_t i = 0; i < n;) {
    x = x * 6364136223846793005ULL + 1442695040888963407ULL;
    const std::uint32_t r = std::uint32_t(x >> 33);
    switch (r % 6) {
      case 0: {  // a zero run of 0..20 cells
        for (std::size_t k = r / 6 % 21; k > 0 && i < n; --k) v[i++] = 0;
        break;
      }
      case 1: v[i++] = r % 256; break;
      case 2: v[i++] = std::array<std::uint32_t, 6>{255, 256, 65535, 65536, 0xffffff, UINT32_MAX}[r / 6 % 6]; break;
      case 3: v[i++] = r % 65536; break;
      default: v[i++] = r;
    }
  }
  return v;
}

std::vector<double> test_doubles(std::size_t n) {
  std::vector<double> v = {0.0, -0.0, 1.0, 3.0, 0.5, 1e-310, -2.75, 1e300, 86400.0, 0.1, 1.0 / 3.0,
                           std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()};
  std::uint64_t x = 7;
  while (v.size() < n) {
    x = x * 6364136223846793005ULL + 1442695040888963407ULL;
    v.push_back((x >> 60) < 4 ? 0.0 : (x >> 60) < 8 ? double(x >> 50) : double(x >> 11) * 0x1.0p-53);
  }
  return v;
}

std::string slurp(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  std::ostringstream s;
  s << in.rdbuf();
  return s.str();
}

}  // namespace

TEST_CASE("fnv1a_cells: equals FNV-1a over the bytes, from any start and with any split") {
  for (std::uint64_t seed = 0; seed < 20; ++seed) {
    const std::vector<std::uint32_t> v = test_cells(seed, 400 + seed * 37);
    const std::uint64_t want = fnv1a_bytes(v.data(), v.size() * sizeof(std::uint32_t));
    std::uint64_t sum = 0, want_sum = 0;
    for (std::uint32_t c : v) want_sum += c;
    CHECK(fnv1a_cells(v, kFnv1aBasis, sum) == want);
    CHECK(sum == want_sum);
    for (std::size_t cut = 0; cut <= v.size(); cut += 1 + seed) {  // continued across a split
      const std::span<const std::uint32_t> all(v);
      std::uint64_t s2 = 0;
      const std::uint64_t h = fnv1a_cells(all.first(cut), kFnv1aBasis, s2);
      CHECK(fnv1a_cells(all.subspan(cut), h, s2) == want);
      CHECK(s2 == want_sum);
    }
  }
  std::uint64_t sum = 0;
  CHECK(fnv1a_cells(std::span<const std::uint32_t>(), kFnv1aBasis, sum) == kFnv1aBasis);
  const std::vector<double> d = test_doubles(500);
  CHECK(fnv1a_cells(d, kFnv1aBasis) == fnv1a_bytes(d.data(), d.size() * sizeof(double)));
  CHECK(fnv1a(test_cells(3, 100)) == fnv1a_bytes(test_cells(3, 100).data(), 400));
}

TEST_CASE("block streams: the tally of chunks equals the whole block's, at every chunk size") {
  const std::uint32_t cols = 3;
  const std::vector<std::uint32_t> v = test_cells(11, 3 * 101);
  std::uint64_t sum = 0;
  for (std::uint32_t c : v) sum += c;
  for (std::uint64_t chunk : {1, 2, 3, 7, 100, 101, 500}) {
    INFO("chunk " << chunk);
    HashSink sink;
    std::unique_ptr<BlockStream> s = sink.open({"b", Cell::u32, 101, cols});
    std::uint64_t parts = 0;
    for (std::uint64_t a = 0; a < 101; a += chunk, ++parts)
      s->put(std::span<const std::uint32_t>(v).subspan(a * cols, std::min<std::uint64_t>(chunk, 101 - a) * cols), 2);
    const StreamTally t = s->finish();
    CHECK(t.fnv1a == fnv1a_bytes(v.data(), v.size() * 4));
    CHECK(t.rows == 101);
    CHECK(t.chunks == parts);
    CHECK(t.instances == sum);
    CHECK(t.saturated == 2 * parts);
  }
  const std::vector<double> d = test_doubles(4 * 52);
  HashSink sink;
  std::unique_ptr<BlockStream> s = sink.open({"vstats", Cell::f64, 4, 52});
  for (std::size_t r = 0; r < 4; ++r) s->put(std::span<const double>(d).subspan(r * 52, 52));
  const StreamTally t = s->finish();
  CHECK(t.fnv1a == fnv1a_bytes(d.data(), d.size() * sizeof(double)));
  CHECK(t.instances == 0);
}

TEST_CASE("block streams: a partial row, the wrong cells, too many or too few rows are errors") {
  HashSink sink;
  const std::vector<std::uint32_t> v(6, 1);
  const std::vector<double> d(6, 1.0);
  std::unique_ptr<BlockStream> s = sink.open({"b", Cell::u32, 2, 3});
  CHECK_THROWS_AS(s->put(std::span<const std::uint32_t>(v).first(4)), std::logic_error);
  CHECK_THROWS_AS(s->put(std::span<const double>(d)), std::logic_error);
  s->put(std::span<const std::uint32_t>(v).first(3));
  CHECK_THROWS_AS(s->finish(), std::logic_error);  // one of two rows
  CHECK_THROWS_AS(s->put(std::span<const std::uint32_t>(v)), std::logic_error);  // three of two
  s->put(std::span<const std::uint32_t>(v).first(3));
  s->finish();
  CHECK_THROWS_AS(s->put(std::span<const std::uint32_t>(v).first(3)), std::logic_error);
  CHECK_THROWS_AS(s->finish(), std::logic_error);
}

TEST_CASE("npy sink: a finished stream is write_npy's file; an unfinished one leaves nothing") {
  const std::string dir = test_tmp("npy_sink");
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const std::vector<std::uint32_t> v = test_cells(5, 2 * 9);
  NpySink sink(dir);
  {
    std::unique_ptr<BlockStream> s = sink.open({"fan_in", Cell::u32, 9, 2});
    for (std::size_t r = 0; r < 9; r += 4)
      s->put(std::span<const std::uint32_t>(v).subspan(r * 2, std::min<std::size_t>(4, 9 - r) * 2));
    CHECK(std::filesystem::exists(dir + "/fan_in.npy.tmp"));
    CHECK(!std::filesystem::exists(dir + "/fan_in.npy"));
    CHECK(s->finish().fnv1a == fnv1a(v));
  }
  write_npy_u32(dir + "/want.npy", v, 9, 2);
  CHECK(slurp(dir + "/fan_in.npy") == slurp(dir + "/want.npy"));
  const NpyU32View view(dir + "/fan_in.npy");
  CHECK(view.rows() == 9);
  CHECK(view.cols() == 2);
  const std::vector<double> d = test_doubles(3 * 52);
  {
    std::unique_ptr<BlockStream> s = sink.open({"vstats", Cell::f64, 3, 52});
    s->put(std::span<const double>(d));
    s->finish();
  }
  write_npy_f64(dir + "/want_f64.npy", d, 3, 52);
  CHECK(slurp(dir + "/vstats.npy") == slurp(dir + "/want_f64.npy"));
  {  // aborted after one of two chunks: the tmp file goes, and an earlier file of that name stays
    std::unique_ptr<BlockStream> s = sink.open({"lcyc", Cell::u32, 9, 2});
    s->put(std::span<const std::uint32_t>(v).first(8));
    CHECK(std::filesystem::exists(dir + "/lcyc.npy.tmp"));
  }
  CHECK(!std::filesystem::exists(dir + "/lcyc.npy.tmp"));
  CHECK(!std::filesystem::exists(dir + "/lcyc.npy"));
  { std::unique_ptr<BlockStream> s = sink.open({"fan_in", Cell::u32, 9, 2}); }
  CHECK(slurp(dir + "/fan_in.npy") == slurp(dir + "/want.npy"));
  CHECK(!std::filesystem::exists(dir + "/fan_in.npy.tmp"));
}

TEST_CASE("memory sink: keeps a finished stream's rows, not an unfinished one's") {
  MemorySink sink;
  const std::vector<std::uint32_t> v = test_cells(9, 12);
  {
    std::unique_ptr<BlockStream> s = sink.open({"sg", Cell::u32, 4, 3});
    s->put(std::span<const std::uint32_t>(v).first(6));
    s->put(std::span<const std::uint32_t>(v).subspan(6));
    s->finish();
  }
  CHECK(sink.u32.at("sg") == v);
  { std::unique_ptr<BlockStream> s = sink.open({"tcyc", Cell::u32, 4, 3}); }
  CHECK(!sink.u32.contains("tcyc"));
}

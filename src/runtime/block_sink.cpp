#include "runtime/block_sink.hpp"

#include <bit>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <system_error>

#include "common/npy.hpp"

namespace blz {

namespace {

static_assert(std::endian::native == std::endian::little, "FNV-1a over the cells' bytes in memory order");

constexpr std::uint64_t kPrime = 1099511628211ULL;
constexpr std::uint64_t prime_pow(int k) {
  std::uint64_t r = 1;
  for (int i = 0; i < k; ++i) r *= kPrime;
  return r;
}
// h after k zero bytes is h * prime^k.
constexpr std::uint64_t kPow[] = {prime_pow(0), prime_pow(1), prime_pow(2), prime_pow(3),
                                  prime_pow(4), prime_pow(5), prime_pow(6), prime_pow(7), prime_pow(8)};
constexpr std::uint64_t kPow32 = prime_pow(32);

// Eight zero cells, 32 bytes, at c.
bool zero32(const void* c) {
  std::uint64_t w[4];
  std::memcpy(w, c, sizeof w);
  return (w[0] | w[1] | w[2] | w[3]) == 0;
}

}  // namespace

std::uint64_t fnv1a_bytes(const void* data, std::size_t bytes, std::uint64_t h) {
  const auto* b = static_cast<const unsigned char*>(data);
  for (std::size_t i = 0; i < bytes; ++i) h = (h ^ b[i]) * kPrime;
  return h;
}

std::uint64_t fnv1a_cells(std::span<const std::uint32_t> cells, std::uint64_t h, std::uint64_t& sum) {
  const std::uint32_t* c = cells.data();
  const std::size_t n = cells.size();
  std::uint64_t s = 0;
  for (std::size_t i = 0; i < n;) {
    std::uint32_t v = c[i];
    if (v == 0 && i + 8 <= n && zero32(c + i)) {
      h *= kPow32;
      i += 8;
      continue;
    }
    ++i;
    s += v;
    if (v < 0x100u) {  // the byte, then 3 zero bytes
      h = (h ^ v) * kPow[4];
    } else if (v < 0x10000u) {
      h = (h ^ (v & 0xffu)) * kPrime;
      h = (h ^ (v >> 8)) * kPow[3];
    } else {
      for (int k = 0; k < 4; ++k, v >>= 8) h = (h ^ (v & 0xffu)) * kPrime;
    }
  }
  sum += s;
  return h;
}

std::uint64_t fnv1a_cells(std::span<const double> cells, std::uint64_t h) {
  for (const double d : cells) {
    std::uint64_t w = std::bit_cast<std::uint64_t>(d);
    if (w == 0) {
      h *= kPow[8];
      continue;
    }
    const int low = std::countr_zero(w) / 8;  // zero bytes first in memory: an integral value's mantissa
    h *= kPow[low];
    w >>= 8 * low;
    for (int k = low; k < 8; ++k, w >>= 8) h = (h ^ (w & 0xffu)) * kPrime;
  }
  return h;
}

void BlockStream::take(std::size_t cells, Cell cell) {
  if (finished_) throw std::logic_error("row stream " + spec_.name + ": put after finish");
  if (cell != spec_.cell) throw std::logic_error("row stream " + spec_.name + ": wrong cell type");
  if (spec_.cols == 0 || cells % spec_.cols != 0)
    throw std::logic_error("row stream " + spec_.name + ": a chunk of partial rows");
  if (tally_.rows + cells / spec_.cols > spec_.rows)
    throw std::logic_error("row stream " + spec_.name + ": more rows than " + std::to_string(spec_.rows));
  tally_.rows += cells / spec_.cols;
  ++tally_.chunks;
}

void BlockStream::put(std::span<const std::uint32_t> cells, std::uint64_t saturated) {
  take(cells.size(), Cell::u32);
  tally_.fnv1a = fnv1a_cells(cells, tally_.fnv1a, tally_.instances);
  tally_.saturated += saturated;
  write(cells.data(), cells.size_bytes());
}

void BlockStream::put(std::span<const double> cells) {
  take(cells.size(), Cell::f64);
  tally_.fnv1a = fnv1a_cells(cells, tally_.fnv1a);
  write(cells.data(), cells.size_bytes());
}

StreamTally BlockStream::finish() {
  if (finished_) throw std::logic_error("row stream " + spec_.name + ": finished twice");
  if (tally_.rows != spec_.rows)
    throw std::logic_error("row stream " + spec_.name + ": " + std::to_string(tally_.rows) + " of " +
                           std::to_string(spec_.rows) + " rows");
  commit();
  finished_ = true;
  return tally_;
}

namespace {

class NpyStream final : public BlockStream {
 public:
  NpyStream(const StreamSpec& spec, const std::string& dir)
      : BlockStream(spec), path_(dir + "/" + spec.name + ".npy"), tmp_(path_ + ".tmp") {
    f_ = std::fopen(tmp_.c_str(), "wb");
    if (!f_) throw std::runtime_error("cannot write " + tmp_);
    const std::string h = npy_header(spec.cell == Cell::u32 ? "<u4" : "<f8", spec.rows, spec.cols);
    write(h.data(), h.size());
  }
  ~NpyStream() override {
    if (f_) std::fclose(f_);
    if (!committed_) std::remove(tmp_.c_str());
  }

 private:
  void write(const void* data, std::size_t bytes) override {
    if (bytes > 0 && std::fwrite(data, 1, bytes, f_) != bytes) throw std::runtime_error("write failed: " + tmp_);
  }
  void commit() override {
    const int rc = std::fclose(f_);
    f_ = nullptr;
    if (rc != 0) throw std::runtime_error("write failed: " + tmp_);
    std::filesystem::rename(tmp_, path_);
    committed_ = true;
  }

  std::string path_, tmp_;
  std::FILE* f_ = nullptr;
  bool committed_ = false;
};

class DropStream final : public BlockStream {
 public:
  explicit DropStream(const StreamSpec& spec) : BlockStream(spec) {}

 private:
  void write(const void*, std::size_t) override {}
  void commit() override {}
};

class MemoryStream final : public BlockStream {
 public:
  MemoryStream(const StreamSpec& spec, MemorySink& sink) : BlockStream(spec), sink_(sink) {}

 private:
  void write(const void* data, std::size_t bytes) override {
    const auto* b = static_cast<const char*>(data);
    bytes_.insert(bytes_.end(), b, b + bytes);
  }
  void commit() override {
    if (spec().cell == Cell::u32) {
      std::vector<std::uint32_t>& v = sink_.u32[spec().name];
      v.resize(bytes_.size() / sizeof(std::uint32_t));
      if (!bytes_.empty()) std::memcpy(v.data(), bytes_.data(), bytes_.size());  // an empty block has no data pointer
    } else {
      std::vector<double>& v = sink_.f64[spec().name];
      v.resize(bytes_.size() / sizeof(double));
      if (!bytes_.empty()) std::memcpy(v.data(), bytes_.data(), bytes_.size());  // an empty block has no data pointer
    }
  }

  MemorySink& sink_;
  std::vector<char> bytes_;
};

}  // namespace

std::unique_ptr<BlockStream> NpySink::open(const StreamSpec& spec) { return std::make_unique<NpyStream>(spec, dir_); }
std::unique_ptr<BlockStream> HashSink::open(const StreamSpec& spec) { return std::make_unique<DropStream>(spec); }
std::unique_ptr<BlockStream> MemorySink::open(const StreamSpec& spec) {
  return std::make_unique<MemoryStream>(spec, *this);
}

}  // namespace blz

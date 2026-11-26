#pragma once
// Where a mining run's rows go (SPEC v3 §5.1, H7): each block's rows arrive in eid order, a chunk
// of whole rows at a time, at a stream the sink opens for the block. Every stream keeps the same
// tally (the manifest's output_fnv1a, continued across chunks, so it equals the whole block's hash);
// the sinks differ only in what they do with the bytes: write a .npy file, keep them in memory, or
// drop them (the bounded host sink of §5.1).
#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace blz {

// FNV-1a 64 over the bytes, continuing from `h`. The cell forms give the same value as the bytes
// in memory order (little-endian), but multiply by FNV's prime to a power over a run of zero bytes
// and over a small cell's zero high bytes: on HI-Small clean_default's 670 MB of rows, 0.07 s
// instead of 0.69 s (D-78). fnv1a_cells also adds the cells into `sum`.
inline constexpr std::uint64_t kFnv1aBasis = 1469598103934665603ULL;
std::uint64_t fnv1a_bytes(const void* data, std::size_t bytes, std::uint64_t h = kFnv1aBasis);
std::uint64_t fnv1a_cells(std::span<const std::uint32_t> cells, std::uint64_t h, std::uint64_t& sum);
std::uint64_t fnv1a_cells(std::span<const double> cells, std::uint64_t h);
inline std::uint64_t fnv1a(std::span<const std::uint32_t> cells) {
  std::uint64_t sum = 0;
  return fnv1a_cells(cells, kFnv1aBasis, sum);
}

// What a stream carries: `rows` rows of `cols` cells, uint32 for a pattern block, float64 for R8's
// vstats. `name` is the block's name, or "vstats".
enum class Cell { u32, f64 };
struct StreamSpec {
  std::string name;
  Cell cell = Cell::u32;
  std::uint64_t rows = 0;
  std::uint32_t cols = 0;
};

// Every row a stream took.
struct StreamTally {
  std::uint64_t fnv1a = kFnv1aBasis;  // over the data bytes in row order
  std::uint64_t rows = 0, chunks = 0;
  // uint32 cells: their sum (a histogram instance adds 1 to one cell, a count block's cell is its
  // count, R7), so a run that did less work shows; and the emits lost to saturation (R10), as the
  // miner reported them. Both 0 for vstats.
  std::uint64_t instances = 0, saturated = 0;
};

// One block's rows. Destroying a stream before finish() discards its rows (an aborted run, R11).
class BlockStream {
 public:
  BlockStream(const BlockStream&) = delete;
  BlockStream& operator=(const BlockStream&) = delete;
  virtual ~BlockStream() = default;

  const StreamSpec& spec() const { return spec_; }
  // The next whole rows, in eid order, of the spec's cell type; `saturated` is their lost emits.
  // std::logic_error on the wrong type, a partial row, more rows than the spec, or after finish().
  void put(std::span<const std::uint32_t> cells, std::uint64_t saturated = 0);
  void put(std::span<const double> cells);
  // After the last row: std::logic_error unless every row came.
  StreamTally finish();

 protected:
  explicit BlockStream(StreamSpec spec) : spec_(std::move(spec)) {}

 private:
  virtual void write(const void* data, std::size_t bytes) = 0;
  virtual void commit() = 0;
  void take(std::size_t cells, Cell cell);

  StreamSpec spec_;
  StreamTally tally_;
  bool finished_ = false;
};

class BlockSink {
 public:
  virtual ~BlockSink() = default;
  virtual std::unique_ptr<BlockStream> open(const StreamSpec& spec) = 0;
};

// <dir>/<name>.npy ('<u4' or '<f8', rows x cols): written to <name>.npy.tmp, renamed at finish, and
// removed when the stream is destroyed unfinished. `dir` must exist.
class NpySink final : public BlockSink {
 public:
  explicit NpySink(std::string dir) : dir_(std::move(dir)) {}
  std::unique_ptr<BlockStream> open(const StreamSpec& spec) override;

 private:
  std::string dir_;
};

// The bounded host sink of SPEC v3 §5.1: hashes, counts and drops.
class HashSink final : public BlockSink {
 public:
  std::unique_ptr<BlockStream> open(const StreamSpec& spec) override;
};

// Keeps each finished stream's rows by name (tests, and fuzz-diff through the chunk driver); a
// stream destroyed unfinished leaves nothing. Opening a name again replaces its rows at finish.
class MemorySink final : public BlockSink {
 public:
  std::unique_ptr<BlockStream> open(const StreamSpec& spec) override;
  std::map<std::string, std::vector<std::uint32_t>> u32;
  std::map<std::string, std::vector<double>> f64;
};

}  // namespace blz

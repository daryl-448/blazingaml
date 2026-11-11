#include "io/blzg.hpp"

#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#include "graph/prepare.hpp"

namespace blz {
namespace {

constexpr std::size_t kAlign = 64;
constexpr std::size_t kHeader = 64;

class Writer {
 public:
  explicit Writer(const std::string& path) : f_(std::fopen(path.c_str(), "wb"), &std::fclose) {
    if (!f_) throw std::runtime_error("cannot write " + path);
  }
  void bytes(const void* p, std::size_t n) {
    if (n && std::fwrite(p, 1, n, f_.get()) != n) throw std::runtime_error("write failed");
    pos_ += n;
  }
  template <class T>
  void pod(const T& v) { bytes(&v, sizeof v); }
  void align() {
    static const char zeros[kAlign] = {};
    bytes(zeros, (kAlign - pos_ % kAlign) % kAlign);
  }
  template <class T>
  void array(const std::vector<T>& v) {
    align();
    bytes(v.data(), v.size() * sizeof(T));
  }
  void strings(const std::vector<std::string>& v) {
    align();
    pod(std::uint64_t(v.size()));
    std::uint64_t off = 0;
    pod(off);
    for (auto& s : v) pod(off += s.size());
    for (auto& s : v) bytes(s.data(), s.size());
  }
  void close_at(std::size_t offset, std::uint64_t value) {
    if (std::fseek(f_.get(), long(offset), SEEK_SET) != 0) throw std::runtime_error("seek failed");
    if (std::fwrite(&value, sizeof value, 1, f_.get()) != 1) throw std::runtime_error("write failed");
    if (std::fclose(f_.release()) != 0) throw std::runtime_error("close failed");
  }
  std::size_t pos() const { return pos_; }

 private:
  std::unique_ptr<FILE, int (*)(FILE*)> f_;
  std::size_t pos_ = 0;
};

class Reader {
 public:
  Reader(const char* base, std::size_t size) : base_(base), size_(size), pos_(kHeader) {}
  const char* take(std::size_t n, const char* what) {
    if (pos_ > size_ || n > size_ - pos_) throw std::runtime_error(std::string("blzg truncated in ") + what);
    const char* p = base_ + pos_;
    pos_ += n;
    return p;
  }
  void align() { pos_ += (kAlign - pos_ % kAlign) % kAlign; }
  template <class T>
  std::span<const T> array(std::uint64_t count, const char* what) {
    align();
    if (count > size_ / sizeof(T)) throw std::runtime_error(std::string("blzg truncated in ") + what);
    return {reinterpret_cast<const T*>(take(count * sizeof(T), what)), std::size_t(count)};
  }
  StringTable strings(const char* what) {
    align();
    std::uint64_t count;
    std::memcpy(&count, take(8, what), 8);
    StringTable t;
    if (count >= size_ / 8) throw std::runtime_error(std::string("blzg truncated in ") + what);
    t.offsets = {reinterpret_cast<const std::uint64_t*>(take((count + 1) * 8, what)),
                 std::size_t(count + 1)};
    if (t.offsets[0] != 0) throw std::runtime_error(std::string("blzg bad offsets in ") + what);
    for (std::size_t i = 0; i < count; ++i)
      if (t.offsets[i + 1] < t.offsets[i])
        throw std::runtime_error(std::string("blzg bad offsets in ") + what);
    t.chars = take(t.offsets[count], what);
    return t;
  }
  std::size_t pos() const { return pos_; }

 private:
  const char* base_;
  std::size_t size_, pos_;
};

}  // namespace

void write_blzg(const std::string& path, const EdgeTable& t) {
  const std::uint64_t m = t.m(), n = t.n();
  for (std::size_t len : {t.dst.size(), t.ts.size(), t.label.size(), t.amount_paid.size(),
                          t.amount_received.size(), t.pay_ccy.size(), t.recv_ccy.size(),
                          t.format.size(), t.file_row.size()})
    if (len != m) throw std::invalid_argument("write_blzg: edge arrays differ in length");
  if (t.node_account.size() != n) throw std::invalid_argument("write_blzg: node tables differ");

  const std::string tmp = path + ".tmp";
  Writer w(tmp);
  w.bytes("BLZG", 4);
  w.pod(kBlzgVersion);
  w.pod(m);
  w.pod(n);
  w.bytes(t.csv_sha256.data(), 32);
  const std::size_t size_at = w.pos();
  w.pod(std::uint64_t(0));  // file size, patched below
  w.array(t.src);
  w.array(t.dst);
  w.array(t.ts);
  w.array(t.label);
  w.array(t.amount_paid);
  w.array(t.amount_received);
  w.array(t.pay_ccy);
  w.array(t.recv_ccy);
  w.array(t.format);
  w.array(t.file_row);
  w.strings(t.ccy_dict);
  w.strings(t.format_dict);
  w.strings(t.node_bank);
  w.strings(t.node_account);
  w.close_at(size_at, w.pos());
  if (std::rename(tmp.c_str(), path.c_str()) != 0) throw std::runtime_error("rename failed: " + path);
}

BlzgView::BlzgView(const std::string& path) : file_(path) {
  const char* p = file_.data();
  const std::size_t size = file_.size();
  std::uint32_t version;
  std::uint64_t file_size;
  if (size < kHeader || std::memcmp(p, "BLZG", 4) != 0) throw std::runtime_error(path + ": not a .blzg file");
  std::memcpy(&version, p + 4, 4);
  if (version == 1)
    throw std::runtime_error(path + ": .blzg version 1 (v1, no file_row); this build reads version " +
                             std::to_string(kBlzgVersion) + ": re-run `blz prepare` to rebuild the cache");
  if (version != kBlzgVersion)
    throw std::runtime_error(path + ": unsupported .blzg version " + std::to_string(version));
  std::memcpy(&m_, p + 8, 8);
  std::memcpy(&n_, p + 16, 8);
  std::memcpy(sha_.data(), p + 24, 32);
  std::memcpy(&file_size, p + 56, 8);
  if (file_size != size) throw std::runtime_error(path + ": size mismatch (truncated?)");

  Reader r(p, size);
  src = r.array<std::uint32_t>(m_, "src");
  dst = r.array<std::uint32_t>(m_, "dst");
  ts = r.array<std::int32_t>(m_, "ts");
  label = r.array<std::uint8_t>(m_, "label");
  amount_paid = r.array<double>(m_, "amount_paid");
  amount_received = r.array<double>(m_, "amount_received");
  pay_ccy = r.array<std::uint8_t>(m_, "pay_ccy");
  recv_ccy = r.array<std::uint8_t>(m_, "recv_ccy");
  format = r.array<std::uint8_t>(m_, "format");
  file_row = r.array<std::uint32_t>(m_, "file_row");
  ccy_dict = r.strings("ccy_dict");
  format_dict = r.strings("format_dict");
  node_bank = r.strings("node_bank");
  node_account = r.strings("node_account");
  if (node_bank.size() != n_ || node_account.size() != n_)
    throw std::runtime_error(path + ": node table size != n");
  if (r.pos() != size) throw std::runtime_error(path + ": trailing bytes");
  for (std::uint64_t e = 0; e < m_; ++e)
    if (src[e] >= n_ || dst[e] >= n_) throw std::runtime_error(path + ": node id out of range");
  if (const std::string err = check_file_row(file_row, ts); !err.empty()) throw std::runtime_error(path + ": " + err);
}

}  // namespace blz

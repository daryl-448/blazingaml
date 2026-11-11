#pragma once
#include <cstdint>
#include <span>
#include <vector>

#include "blz/primitives.hpp"

namespace blz {

// Owns the T-CSR and P-CSR arrays of both directions (SPEC §5.1, §5.2), built once from an edge
// stream in stream order, and exposes them as a GraphView. The edge arrays themselves are not
// copied: the view points into `src`, `dst` and `ts`, which must outlive the index.
class GraphIndex {
 public:
  GraphIndex(std::span<const std::uint32_t> src, std::span<const std::uint32_t> dst,
             std::span<const std::int32_t> ts, std::uint64_t n);
  GraphIndex(const GraphIndex&) = delete;
  GraphIndex& operator=(const GraphIndex&) = delete;

  const GraphView& view() const { return view_; }
  std::uint64_t bytes() const;  // index arrays only

 private:
  struct TCsr {
    std::vector<std::uint32_t> off, nbr, eid;
    std::vector<std::int32_t> ts, prev_same;
  };
  struct PCsr {
    std::vector<std::uint32_t> off, nbr, run, run_eid;
    std::vector<std::int32_t> run_ts;
  };
  static void build_tcsr(TCsr& t, std::span<const std::uint32_t> key, std::span<const std::uint32_t> nbr,
                         std::span<const std::int32_t> ts, std::uint64_t n);
  static void build_pcsr(PCsr& p, const TCsr& t, std::uint64_t n);

  TCsr out_, in_;
  PCsr pout_, pin_;
  GraphView view_;
};

}  // namespace blz

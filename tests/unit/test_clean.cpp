// The clean family's runtime (SPEC v3 §3.2): count blocks (R7) and the loop-free index (R1), on
// hand-checked graphs.
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <vector>

#include "blz/registry.hpp"

using namespace blz;

TEST_CASE("count rows: an emit adds its size, min_size aside; a sum past UINT32_MAX clamps and loses one emit") {
  std::uint32_t row = 0;
  RowSink s{&row, nullptr, 1, true};
  s.emit(0, 2);
  CHECK(row == 0);
  s.emit(1, 2);
  CHECK(row == 1);
  s.emit(5, 2);
  CHECK(row == 6);
  CHECK(s.saturated == 0);
  row = UINT32_MAX - 3;
  s.emit(3, 2);
  CHECK(row == UINT32_MAX);
  CHECK(s.saturated == 0);
  s.emit(1, 2);
  CHECK(row == UINT32_MAX);
  CHECK(s.saturated == 1);
  row = UINT32_MAX - 1;
  s.emit(7, 2);
  CHECK(row == UINT32_MAX);
  CHECK(s.saturated == 2);
}

TEST_CASE("histogram rows: an emit below min_size or b_0 is dropped") {
  const std::uint32_t bins[] = {2, 4};
  std::uint32_t row[2] = {};
  RowSink s{row, bins, 2};
  for (std::int64_t size : {0, 1, 2, 3, 4, 9}) s.emit(size, 2);
  CHECK(row[0] == 2);
  CHECK(row[1] == 2);
  s.emit(3, 4);  // a spec's min_size above 2 (SPEC §6.6)
  CHECK(row[0] == 2);
}

#pragma once
// Metamorphic tests (SPEC v2 §3.4): transformations of a fuzz graph (§13.3 of v1) under which
// every row must stay the same. They catch semantic mistakes that backend parity cannot, because
// every backend would share them.
#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "runtime/profile.hpp"

namespace blz {

// The tests of §3.4.
enum class MetaTest { prefix_closure, relabeling, time_shift, visibility_invariance, order_relaxation };
inline constexpr std::array<MetaTest, 5> kMetaTests = {MetaTest::prefix_closure, MetaTest::relabeling,
                                                       MetaTest::time_shift, MetaTest::visibility_invariance,
                                                       MetaTest::order_relaxation};
const char* meta_test_name(MetaTest t);

// Whether test `t` runs for `prof` on `backend`: prefix closure needs trigger credit;
// visibility invariance needs a tcyc or tchain block (every backend runs both visibilities since P10);
// order relaxation needs an sg_ord block.
bool meta_test_applies(MetaTest t, const Profile& prof, Backend backend);

struct MetamorphicResult {
  std::uint64_t graphs = 0, edges = 0;
  // [test][block]: rows that differ from the untransformed graph's, summed over the graphs (and,
  // for prefix closure, over its cuts; for cuda, over the thresholds).
  std::array<std::array<std::uint64_t, kAllBlocks.size()>, kMetaTests.size()> mismatched_rows{};
  // [test]: rows compared, summed the same way.
  std::array<std::uint64_t, kMetaTests.size()> rows_compared{};
  std::vector<std::uint64_t> bad_seeds;  // ascending
};

// Runs every test on the graphs of seeds [s0, s1), for every block of `prof` with its patterns
// and k_max, each graph's own window, and `bins` (each block's own bins when `bins` is empty), on
// one backend (for cuda, once per light/heavy threshold in `thresholds`; the other backends ignore
// it). Per graph:
//   - prefix closure: mining the first k edges gives the same rows for them as mining the whole
//     graph, at two cuts k drawn from the seed (under timestamp visibility, k is moved forward
//     to the next timestamp boundary); for trigger credit only (participants credit is not
//     causal);
//   - relabeling: a seeded permutation of the node ids leaves every row unchanged;
//   - time shift: adding a seeded constant in [1, 2^30] to every timestamp leaves every row
//     unchanged;
//   - visibility invariance: tcyc and tchain mined under the other visibility (stream <->
//     timestamp) give the same rows (SPEC v2 §6.1, §8 S5);
//   - order relaxation: per row, sg_ord's instance count (the sum over its bins) is at most that
//     of sg_gather mined alone at sg_ord's window and bins, by the same backend (SPEC v2 §6.3).
// check_profile_backend(prof, backend) must hold; this throws otherwise.
MetamorphicResult metamorphic(const Profile& prof, Backend backend, std::uint64_t s0, std::uint64_t s1,
                              std::span<const std::uint32_t> bins, std::span<const std::uint32_t> thresholds);

}  // namespace blz

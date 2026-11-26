// Trigger selections (SPEC §8, SPEC v2 §11): fit/val, sample:<pct>:<seed>, and the rows of a
// sample equal to the full run's rows at its eids on the oracle and the CPU backend.
#include <catch2/catch_test_macros.hpp>
#include <vector>

#include "graph/fuzz_gen.hpp"
#include "graph/index.hpp"
#include "oracle/oracle.hpp"
#include "runtime/cpu_backend.hpp"
#include "runtime/profile.hpp"
#include "runtime/triggers.hpp"

using namespace blz;

TEST_CASE("triggers: the named selections") {
  // m = 10, train = 8, floor(0.6 m) = 6 continues ts 5, so b_fv = 8 = train: val is empty.
  const std::vector<std::int32_t> ts = {0, 1, 2, 3, 4, 5, 5, 5, 8, 9};
  const Splits sp = compute_splits(SplitRule::legacy_hi_small, ts);
  auto range = [&](const std::string& s) {
    const TriggerSel t = select_triggers(s, ts.size(), sp);
    return std::pair{t.first, t.last};
  };
  CHECK(range("all") == std::pair<std::uint64_t, std::uint64_t>{0, 10});
  CHECK(range("train") == std::pair<std::uint64_t, std::uint64_t>{0, 8});
  CHECK(range("test") == std::pair<std::uint64_t, std::uint64_t>{8, 10});
  CHECK(range("fit") == std::pair<std::uint64_t, std::uint64_t>{0, 8});
  CHECK(range("val") == std::pair<std::uint64_t, std::uint64_t>{8, 8});
  CHECK(range("range:2:5") == std::pair<std::uint64_t, std::uint64_t>{2, 5});
  // The splits come from the caller: here temporal_60_20_20's on a stream with a tie at each cut.
  const std::vector<std::int32_t> ts2 = {0, 1, 2, 3, 4, 5, 5, 7, 7, 9};
  const Splits sp2 = compute_splits(SplitRule::temporal_60_20_20, ts2);
  CHECK(select_triggers("fit", 10, sp2).last == 7);
  CHECK(select_triggers("val", 10, sp2).first == 7);
  CHECK(select_triggers("val", 10, sp2).rows() == 2);
  CHECK(select_triggers("test", 10, sp2).first == 9);
  // --edges 5 keeps the whole stream's splits: fit fits, train and test do not.
  CHECK(select_triggers("fit", 5, Splits{3, 8}).last == 3);
  CHECK_THROWS(select_triggers("train", 5, Splits{3, 8}));
  CHECK_THROWS(select_triggers("test", 5, Splits{3, 8}));
  CHECK_THROWS(select_triggers("range:5:2", 10, sp));
  CHECK_THROWS(select_triggers("range:0:11", 10, sp));
  CHECK_THROWS(select_triggers("range:1", 10, sp));
  CHECK_THROWS(select_triggers("sample:0:1", 10, sp));
  CHECK_THROWS(select_triggers("sample:101:1", 10, sp));
  CHECK_THROWS(select_triggers("sample:1x:1", 10, sp));
  CHECK_THROWS(select_triggers("sample:nan:1", 10, sp));
  CHECK_THROWS(select_triggers("sample:5:-1", 10, sp));
  CHECK_THROWS(select_triggers("everything", 10, sp));
}

TEST_CASE("triggers: a sample is reproducible, independent of m, and near its percentage") {
  const std::uint64_t m = 100'000;
  const TriggerSel a = select_triggers("sample:5:42", m, {});
  const TriggerSel b = select_triggers("sample:5:42", m, {});
  CHECK(a.sampled);
  CHECK(a.eids == b.eids);
  CHECK(a.rows() == a.eids.size());
  CHECK(a.rows() > 4'500);
  CHECK(a.rows() < 5'500);
  CHECK(std::is_sorted(a.eids.begin(), a.eids.end()));
  CHECK(a.first == a.eids.front());
  CHECK(a.last == a.eids.back() + 1u);
  CHECK(a.eid(3) == a.eids[3]);
  // A shorter stream selects the same eids below its m.
  const TriggerSel s = select_triggers("sample:5:42", 50'000, {});
  std::vector<std::uint32_t> head;
  for (std::uint32_t e : a.eids)
    if (e < 50'000) head.push_back(e);
  CHECK(s.eids == head);
  CHECK(select_triggers("sample:5:43", m, {}).eids != a.eids);
  CHECK(select_triggers("sample:100:7", m, {}).rows() == 100'000);
  CHECK(select_triggers("sample:0.5:1", m, {}).json()["percent"] == 0.5);
}

TEST_CASE("triggers: sampled rows equal the full run's rows at those eids (oracle, cpu)") {
  const Profile prof = registry_profile();
  for (std::uint64_t seed = 0; seed < 60; ++seed) {
    const FuzzGraph fg = fuzz_graph(seed);
    const std::uint64_t m = fg.src.size();
    const Oracle oracle({fg.src, fg.dst, fg.ts, fg.n});
    const GraphIndex index(fg.src, fg.dst, fg.ts, fg.n);
    const TriggerSel sel = select_triggers("sample:30:" + std::to_string(seed), m, {});
    TriggerSel all;
    all.last = m;
    for (const ProfileBlock& pb : prof.blocks) {
      const BlockParams p{fg.window, {2, 3, 4, 6, 9}};
      const std::size_t nb = p.bins.size();
      std::vector<std::uint32_t> full(m * nb, 0), o(sel.rows() * nb, 0), c(sel.rows() * nb, 0);
      oracle.mine(pb.block, p, prof.k_max, 0, m, full.data());
      oracle.mine(pb.block, p, prof.k_max, sel.eids, o.data());
      mine_cpu(index.view(), pb.patterns, p, Visibility::stream, sel, Schedule::dynamic, c.data());
      std::vector<std::uint32_t> want;
      for (std::uint32_t e : sel.eids) want.insert(want.end(), full.begin() + e * nb, full.begin() + (e + 1) * nb);
      INFO("seed " << seed << " block " << block_name(pb.block));
      CHECK(o == want);
      CHECK(c == want);
    }
  }
}

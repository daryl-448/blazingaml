---
name: verify-codegen
description: "L2, L3 and T4 of SPEC_V3 §4.3 (targets T1, T2, T4): the real compiled code that blzc generates, and every optimised path, against the profile's oracle on every graph within a bound (blz fuzz-diff --bound, verify/codegen.sh), RapidCheck properties past that bound (tests/unit/test_l3.cpp, l3_random), the index primitives' and ReachScratch::search's contracts (blz index-contracts), and the code-mutant runner (verify/mutants.py). Use when a change touches a pattern spec (patterns/, patterns/clean/), blzc (src/dsl, src/ir, src/codegen), include/blz, the index builder, a backend or driver (CPU, CUDA, hybrid, fusion, packing, vstats, the chunk driver and its sinks), an oracle, a profile that the checks use, or the checks themselves; at P14, P16 and P17 phase exits; and for every speed change (SPEC_V3 §4.4 V-code, V-diff, V-mut)."
---

# verify-codegen: L2, L3 and T4 (T1, T2, T4)

**Properties.**
- T1: every generated `<pattern>_trigger`, as compiled, gives its family's oracle's rows: clean
  specs the clean oracle's, legacy specs the legacy oracle's.
- T2: every optimised path gives the plain path's rows. The paths are the sweep lowering,
  reachability pruning, scan fusion, packed D2H, the hybrid, CUDA's light and heavy kernels, the
  `blzc --disable` build, and range mining through the chunk driver (H7): each backend's rows,
  vstats and row hash, mined in chunks of any size, equal the oracle's for the whole selection.
- T4: the index primitives meet their contracts, and `ReachScratch::search` meets the `Search`
  contract that L1's pruning proofs assume.

**Harm if broken.** A wrong row changes a feature silently. A hash on one dataset shows it only if
the bug fires there, and the classifier then trains on it.

**Tools.** Exhaustive enumeration in compiled C++ (L2), which runs the real, unmodified generated
code on every graph within a bound. RapidCheck at commit 2c3c436 inside Catch2 v3 (L3). CBMC
needed five lowerings of the real headers and ESBMC ran out of memory at m = 3. Enumeration covered
CBMC's m = 5 domain about 60× faster, on the code as it ships (SPEC_V3 §4.2).

## 1. Does my change touch covered code?

Sync commit: `3a02e81`. At it, the whole suite in section 2 passed and every mutant of
`verify/mutants.py` was caught where recorded. The chunk driver and its sinks (P15, from b8bfa99)
joined the record at 6edf4ab. There the chunked checks of verify/codegen.sh, ctest and L3 passed,
and the driver's mutants were caught where recorded (SPEC_V3 D-83, D-87). The full suite has not
run since 3a02e81, so the sync commit stays there. Run:

```
git diff --stat 3a02e81 -- patterns include/blz src/dsl src/ir src/codegen \
  src/graph/index.cpp src/graph/index.hpp src/graph/exhaustive.cpp src/graph/index_contracts.cpp \
  src/runtime/{cpu_backend,cuda_backend,hybrid,fusion,widen,triggers,registration,profile,fuzz_diff}.* \
  src/runtime/{chunk_driver,block_sink}.* \
  src/runtime/*vstats* src/oracle tests/snapshots tests/profiles configs/profiles/clean_all.yaml \
  tests/unit/test_l3.cpp verify/codegen.sh verify/mutants.py CMakeLists.txt tests/CMakeLists.txt
```

This compares the sync commit with the working tree, so it lists uncommitted edits too. Don't
write `3a02e81..HEAD`: that compares commits only and misses a change that isn't committed yet.
If it lists nothing, the verification still describes the code. If it lists files, find them in
the record:

| Code | Covered by |
|---|---|
| The 13 legacy specs, patterns/{deg_in,deg_out,fan_in,fan_out,fan_oneway_out,fan_total,gs,lcyc,sg_gather,sg_ordered,sg_scatter,tchain,tcyc}.yaml | L2 profiles tests/profiles/l2_legacy_stream.yaml and l2_legacy_tsvis.yaml (every legacy pattern, one profile per visibility), against src/oracle/oracle.cpp through profile_oracle.cpp |
| The 13 clean specs, patterns/clean/clean_*.yaml, and R8's vertex statistics | configs/profiles/clean_all.yaml (every clean pattern and vstats), against src/oracle/clean_oracle.cpp; vstats compared bit for bit |
| blzc: src/dsl/*, src/ir/{lower,plan,compile}.cpp, src/codegen/{cpu,cuda,registry}_emitter.cpp | T1 through what they generate, build/<preset>/generated/, which must equal tests/snapshots/ (ctest `blzc_snapshots`, `blzc_snapshots_disable_*`) |
| include/blz/primitives.hpp: `TCsrView::range`, `head_range`, `distinct_first`, `distinct_first_ts`, `first_ts_at_least`; `PCsrView::pair_exists`, `pair_latest`; `count_distinct` | T4 (`blz index-contracts`: `range`, `head_range`, `distinct_first`, `distinct_first_ts`, `pair_exists`, `pair_latest`), and T1 end to end |
| src/graph/index.cpp: `GraphIndex`, `build_tcsr`, `build_pcsr`, `SelfLoops::drop` | T4 contracts `list` and `prev_same`, both self-loop modes |
| include/blz/reach.hpp: `ReachScratch::search`, `keep`; `tcyc_trigger_reach` | T4 contract `search`; L2 cpu on the legacy profiles (`reach-pruned rows`); L3 k_max |
| include/blz/sweep.hpp `sweep_count`, the generated `*_cpu` sweep drivers | L2 cpu (`sweep rows`) |
| include/blz/registry.hpp: `RowSink::emit`, `RowSink::over`, `BlockOut` | L2 rows and the limit-1 stop eids; L3's R10 properties (saturation at UINT32_MAX) |
| src/runtime/cpu_backend.cpp, triggers.cpp, registration.cpp, profile.cpp | L2 cpu, both schedules (`cpu dynamic`, `cpu sorted`) |
| src/runtime/cuda_backend.cu (light and heavy kernels, `pack_kernel`, staging), include/blz/cuda_kernels.cuh, fusion.cpp, widen.cpp, generated `*_cuda.cu` and `fused_*_cuda.cu` | L2 on rel-cuda: cuda at H = 0, 4, all; `--fuse --pack`; limit 1. L3's packing properties (2 and 4 bytes, the tail cells) |
| src/runtime/hybrid.cpp | L2 hybrid, every H × C in {0, 4, all} |
| src/runtime/vstats.cpp, clean_vstats.cpp, clean_vstats_cuda.cu | L2 clean_all, bit for bit |
| `BLZ_DISABLE_OPTS=find_start,distinct_first,degree_order,count_only,sweep` (build/rel-noopt) | L2 rel-noopt: the unoptimised generated code against the oracle |
| Memory and overflow | L2 dev (ASan + UBSan): cpu with each graph alone, and T4 |
| src/runtime/chunk_driver.cpp: `ChunkDriver::run`, `run_block` (the chunked path and D-79's sweep path), `run_fused`, `run_vstats`, `rows_buffer`; triggers.cpp `slice`; block_sink.cpp: `BlockStream::put`, `finish`, `fnv1a_cells`, `MemorySink`, `HashSink` (`blz measure`'s hash path: the same streams, writing nothing) | L2 through the chunk driver (`blz fuzz-diff --chunk-rows`): `cpu-chunks`, `cuda-chunks`, `cuda-fuse-pack-chunks`, `hybrid-chunks` in verify/codegen.sh and ctest `l2_smoke_chunks_*`, each stream's rows, vstats and tally (rows, puts, FNV-1a, instances) against the oracle's at every selection; L3 chunks; the unit tests in test_chunk_driver.cpp and test_block_sink.cpp; the model in verify/lean/Blz/Chunks.lean (`/verify-algorithms`) |

The harness is trusted code too. src/graph/exhaustive.{hpp,cpp} defines the domain. src/runtime/
fuzz_diff.{hpp,cpp} does the comparison, the path hits and the counterexamples. Then come
src/graph/index_contracts.{hpp,cpp}, the `fuzz-diff` and `index-contracts` commands in
src/apps/blz.cpp, tests/unit/test_l3.cpp, verify/codegen.sh and verify/mutants.py. A change there
needs the mutant run (section 2), which shows the checks can still fail.

## 2. Run it

Build first: `cmake --build build/<preset>` for each build you run. `nvcc` is /opt/cuda/bin/nvcc,
which the rel-cuda preset already names. Check `uptime` and `free -g` before a timed run: the
desktop is shared, and other jobs move the seconds but not the verdicts.

**Quick, on every change that touches covered code** (the times are E20's, at the sync commit):

```
ctest --preset rel-cuda       # 630 s at 3a02e81 (E20), 573 s at 024be12 with the chunked tests
```

Without a GPU, or for a CPU-only change, `ctest --preset rel` (130 s at 3a02e81) runs the same checks
without the CUDA ones. Label `fast` holds the L2 smoke checks: every graph of m=4,n=4,t=3 (29,119 graphs; every block has
nonzero rows) on the three profiles, plain and at a work limit of 1, plus cuda, `--fuse --pack` and
the hybrid in CUDA builds. The `l2_smoke_chunks_*` tests run the backends through the chunk driver
in chunks of 1, 2 and 3 triggers: cpu at that bound on the three profiles, and in CUDA builds cuda,
`--fuse --pack` and the hybrid at m=3,n=4,t=3 and H = C = 2 on clean_all. It also holds T4 with `search` at that
bound, every L3 property at its fixed seed, and the snapshot tests.

Run the quick check first on every change, an edit to an existing algorithm's code included. If it
fails, the change is wrong, and the full run can't reverse that verdict. To confirm that a failure
comes from your change and was not there before: at the sync commit the whole suite passed
(results/e20_verification.md). If `git diff --stat 3a02e81 HEAD` lists no covered code, your edit
is the only code that changed since.

**Full, before you commit a new algorithm, lowering, optimised path or check, and at phase exits**
(about 108 min at 3a02e81, E20):

```
verify/codegen.sh && ctest --test-dir build/rel-cuda -L verify -R l3_random && verify/mutants.py --escape
```

It stops at the first part that fails. The parts are L2 on every build present (51 min), L3 at
fresh seeds with ten times the cases (26 s on rel, 90 s on rel-cuda), and V-mut, where every code mutant must be
caught (53 min with --escape). `bench/e20.py run` runs the same parts with L1, the fixed seeds on each build
and a load sampler, and `bench/e20.py report` writes results/e20_verification.{json,md}.

- `verify/codegen.sh [--smoke] [rel] [rel-cuda] [dev] [rel-noopt]` runs the checks of each build
  present, at bounds A = m=5,n=5,t=4 (3,105,502 graphs, 61,785,528 trigger checks) and B =
  m=6,n=4,t=3, and prints one table. Its reports go to build/verify/codegen.{json,log}, and it exits
  1 if any check fails. `ctest -L verify` in a preset build runs it for that build.
- Its `*-chunks` checks run a backend through the chunk driver (`fuzz-diff --chunk-rows`, D-83),
  over every trigger, the range [m/4, m − m/5) of each union and, on cpu, a 40 % sample. They
  compare every stream's rows, vstats and tally (rows, ⌈rows / k⌉ puts, FNV-1a, instances) with the
  oracle's at those triggers. rel runs `cpu-chunks` on the three profiles, in chunks of 1, 2 and 3
  at SMOKE and of 3 and 1000 at A. rel-cuda runs `cuda-chunks`, `cuda-fuse-pack-chunks` and
  `hybrid-chunks` at H = C = 4: in chunks of 1000 at A on the three profiles, and in chunks of 1, 2
  and 3 at CHUNKS_GPU = m=4,n=4,t=3 on clean_all (--smoke: m=3,n=4,t=3 at H = C = 2). They add 5.5
  min to rel and 28 min to rel-cuda (measured at 024be12; the GPU's chunks of 1 to 3 at
  CHUNKS_GPU take 322 to 452 s each, as a put costs the GPU about 25 us).
  `--only=GLOB` runs the checks whose name matches, for example `verify/codegen.sh
  --only='*-chunks/*' rel rel-cuda`.
- `verify/mutants.py` plants each one-line mutant of its list in a scratch worktree
  (build/mutants/src, never the main tree) and walks its build's ladder of checks, cheapest first.
  It records the check and layer that caught it, the counterexample's size and the seconds in
  build/mutants/report.json. `verify/mutants.py NAME` runs one mutant, and `--list` prints them all.
  `--escape` also confirms that each L3-only mutant passes the full-bound L2 checks that run its
  path. `--keep` keeps the scratch builds for a faster rerun. The chunk driver's 8 mutants are
  caught by `l2-smoke-cpu-chunks` and `l2-chunks-cuda-fuse-pack`, rungs that come after the direct
  smoke rungs. Those 8, the 5 mutants caught by `l3` and the baseline took 952 s with builds at
  b5b70af.
- One check by hand: `build/rel/blz fuzz-diff --bound m=5,n=5,t=4 --profile <profile>
  [--backend cuda|hybrid] [--fuse --pack] [--batch-edges 1 --work-limit 1] [--chunk-rows 1,2,3]`,
  or `build/rel/blz index-contracts --bound m=5,n=5,t=4`. Exit 2 means a mismatch or a violation.
- One L3 property: `build/rel/tests/blz_unit_tests "L3 k_max: cycles and chains of K_MAX - 1 to
  K_MAX + 1 edges on every profile and path"`. `blz_unit_tests "[l3]" --list-tests` lists them.

## 3. Read a counterexample

All three print the graph in the code's terms: `eid: u->v@t`, in stream order, with node ids
renumbered by first appearance and the first timestamp 0.

**fuzz-diff** (L2, and L3 through the same text):

```
counterexample: graph 1940 of the bound (batch 3), m = 4, n = 3
  block sg_ord, window 1, path cpu dynamic
  edges (eid: u->v@t): 0: 0->0@0  1: 0->1@0  2: 0->2@0  3: 1->2@1
  trigger eid 3 (1->2@1): got [1 0 0 0 0 0 0 0 0], want [0 0 0 0 0 0 0 0 0]
```

Read it as: under l2_legacy_stream, at window 1, the CPU's dynamic schedule gave trigger 3 one
sg_ord instance in the first bin, and the legacy oracle gave none. `got` is the backend and `want`
the oracle, one cell per bin of the profile. A count block has one cell. This came from the mutant
`order_keep_le`, which wrote legacy sg_ord's strict order test as `<=`. The edges 0->1@0 and 1->2@1
are not in strict order through node 1 within window 1.

- `path` names the run: `cpu dynamic` or `cpu sorted`, `+ reach` with pruning, `cuda H=<h>`,
  `hybrid H=<h> C=<c>`, then `fused at windows a,b` and `packed`. A path that fails alone points at
  its optimisation. All paths failing together points at the per-trigger code or a primitive.
- `work limit: got 0, want completed (stop eids)` is R11. The backend stopped the lcyc search, and
  the oracle finished. Rows are not compared then, only the stop eids. "Inconclusive" counts in the
  summary are not failures.
- vstats prints the columns that differ with 17 digits. Any bit counts (AC-V3-1).
- A chunked run's `path` adds the chunk size and the selection: `cpu dynamic chunks of 2, all`,
  or `range:a:b` or `sample:40:<case>`. Its trigger eid is the selection's eid of the row. A
  mismatch that is not a row prints as a note in place of got and want, at the selection's first
  trigger. `stream tally: fnv1a 362932690540399283, want 10211252084931458275` means the stream's
  hash, rows, puts or instances differ from the oracle's rows at those triggers (the mutant
  `hash_restarts`). `the chunk driver threw: row stream lcyc: 0 of 1 rows` is a `BlockStream`
  guard or `slice`'s precondition (`chunk_last_dropped`). A note is the driver's or a sink's bug,
  not a backend's.
- `(fails only inside its batch: no graph of the batch fails alone)`: see section 4.

**index-contracts** (T4):

```
graph 0 (n = 1; edges, eid: u->v@t: 0: 0->0@0), self-loops kept, out-lists: pair_latest:
  pair_latest(0, 0, 0, 0, 0) = kNoTs, want 0
graph 27 (n = 2; edges, eid: u->v@t: 0: 0->1@0  1: 1->0@1), self-loops kept, in-lists: search:
  trigger 1 (1->0@1), stream visibility, window 1: after search(in, 1, 1, 0, 0, 1, 9),
  keep(0, -1) is false; want ld >= 0, as ldJ after 9 rounds gives (node ld: 0 0, 1 1)
```

The first is a primitive call with its arguments `(x, y, lo, hi, max_eid)` and the naive answer.
The second is the search: the call exactly as `tcyc_trigger_reach` makes it `(U, t, lo, hi,
max_eid, hops)`, the node whose `keep` failed, and ldJ's latest departure times. ldJ is Prune.lean's
round-by-round search, transcribed. Here edge 0->1@0 leaves node 0 towards U = 1 at time 0, so
`keep(0, -1)` must hold. (This came from `reach_range_lo`, which made the search skip edges at
`lo`.)

**L3** (RapidCheck), here from the mutant `reach_rounds_from_1`, which runs the pruning search one
round short:

```
Falsifiable after 7 tests and 9 shrinks
(anonymous namespace)::Case:
  n 10, m 10, windows 9, bins (each block's own), cuda H 1 (runs 8), hybrid C 1, work limit 100000000
  edges (eid: u->v@t): 0: 0->1@0  1: 1->2@1  2: 2->3@2  ...  9: 9->0@9
profile l2_legacy_stream
    counterexample: case 0, m = 10, n = 10
      block tcyc, window 9, path cpu dynamic reach
      trigger eid 9 (9->0@9): got [0 0 0 0 0 0 0 0 0], want [0 0 0 0 0 0 0 0 1]
195 shrinks tried
rerun: BLZ_L3_SEED=4599894285215703917 BLZ_L3_SCALE=1 blz_unit_tests "L3 k_max: ..."
```

The shrunk case comes first, then the profile and the fuzz-diff text above. Here a 10-cycle at
K_MAX is lost on the pruned path only. Shrinking stops 60 s after it starts, at the smallest failing
case so far, and the failure says so. Run the rerun line from build/<preset>/tests.

## 4. Real bug, or a modelling or bound error?

Each counterexample is a handful of edges. Work the rule of SPEC_V3 §3.2 (clean) or the legacy
rule (SPEC_V2, frozen) out by hand on it first.

- **The rule agrees with `want`**: the generated code, a primitive or a driver is wrong. That is a
  real bug. The path, and any T4 violation at the same time, says where. Report it in the code's
  terms, and do not change the oracle or the bound to make it pass.
- **The rule agrees with `got`**: the oracle is wrong, which is a transcription bug in
  src/oracle/clean_oracle.cpp or oracle.cpp. Fix the clean oracle from the rule's text. The legacy
  oracle is frozen by AC-V3-0's hashes, so changing it is a decision for the user.
- **`fails only inside its batch`**: no member fails alone. L2 packs graphs into disjoint unions
  (src/graph/exhaustive.hpp, `GraphUnion`). That is valid because every row depends only on the
  trigger's weakly connected component. Either the bug needs a graph's position in the arrays (node
  0, the arrays' ends), or a change broke locality: a row now reads beyond the component, such as a
  global count. Rerun with `--batch-edges 1` to tell them apart. Broken locality is a harness
  question, not a code bug, until you show a single graph that fails. A chunked run is the
  exception: a driver bug that needs two chunks or more (a hash restarted per put, vstats sliced
  from the start) cannot fail on a graph of one trigger, so it fails only inside its batch. That
  is a real bug in the driver, at the path's chunk size.
- **The domain's assumptions.** Canonical graphs start at ts 0 and list each node relabelling once.
  That is sound only while rows depend on timestamps through t − t' alone, and on node ids only
  through ties and P-CSR order inside a component. A rule that reads absolute time or ids breaks
  the domain: change src/graph/exhaustive.cpp first.
- **A T4 violation** compares a primitive with the naive definition in
  src/graph/index_contracts.cpp. If the primitive's contract changed on purpose, change the naive
  definition and the L1 contract (`/verify-algorithms`) together. `search` compares with ldJ. If
  the search was changed on purpose to keep less, Prune.lean's proofs must change first.
- **Path hits.** A pass only covers what ran. Each fuzz-diff summary prints the reference's nonzero
  rows per block and the rows each optimised path computed: `sweep rows`, `reach-pruned rows`,
  fused and packed rows, the light, heavy and hybrid triggers. T4 prints `search: ldJ reached N
  nodes`. A chunked run prints `chunk driver:` streams, puts, the streams of more than one put
  (with the block that has fewest), the sweep streams mined once and put in chunks, the fused
  streams, and the runs over a range and a sample. Its `sweep rows` are the sweep streams' rows; the
  CPU's chunked runs are dynamic without pruning, and the hybrid's are sorted with pruning.
  Two zeros are expected. Reach-pruned rows are 0 on clean_all, because pruning is legacy
  only (clean tcyc has `cpu_reach` false). Packed rows at 2 and 4 bytes are 0 in L2, because no cell
  passes 255 at m ≤ 6. Chunked: sample runs are 0 on cuda and the hybrid (they take an eid range),
  sweep streams 0 on cuda, and the hybrid's heavy triggers 0 at H = C = 4 (its GPU gets only
  triggers below C). Any other zero means the check did not run that path, and verify/codegen.sh
  fails it as `FAIL (<what is missing>)`: a block with no rows or stop eids compared, a limit-1 run
  where no graph reached the limit, a zero on a path the check asks for, or a chunked check with a
  block or vstats that never got a stream of more than one put.
- **Bound errors.** L2's bound cannot reach k_max-length cycles and chains (10 edges), R10's
  saturation, packing at 2 and 4 bytes, int32 extremes or batch positions. Those are L3's, and its
  generators take their ranges from the spec constants (K_MAX, the bins). A mutant that escapes L2
  and L3 means a missing generator, not a bigger number in verify/codegen.sh. Add the property.
- **An L3 failure at a random seed** is real unless the generator built an input that breaks the
  stream's invariants (ts non-decreasing in eid order, node ids < n). Check the shrunk case against
  them. Then rerun with the printed seed.
- **verify/mutants.py**: `ESCAPED` means no check sees that bug, which is a coverage gap: add or
  widen a check. `MOVED` means another rung caught it first. It is a record update once you know
  why, for example a new cheaper check. `NOT ESCAPED` means a full-bound L2 check caught an
  L3-only mutant: drop that check from its `escapes`. `BAD` means the mutant's `old` text is gone,
  it does not build, or a check failed without a mismatch (a crash, a timeout, a setup error): read
  its log in build/mutants/logs before you count it as caught.
  Find the same line in the new code and update the entry; deleting it is not the fix.

## 5. Update the verification for an intended change

- **A new clean pattern.**
  - Write the spec in patterns/clean/ and implement its rule in src/oracle/clean_oracle.cpp from
    SPEC_V3's text.
  - Add the block to configs/profiles/clean_all.yaml, the clean test profile (every pattern, §4.4).
  - Update the snapshots with the command in tests/CMakeLists.txt, after reviewing the diff.
  - Run `verify/codegen.sh --smoke rel`, then confirm that the new block has nonzero rows at bound
    A. Its smallest instance plus one edge must fit in A (m ≤ 5); if not, raise A in
    verify/codegen.sh, which mutants.py reads.
  - If the pattern has k_max-long instances, extend L3's k_max generator.
  - Add a mutant of its new code path to `MUTANTS` in verify/mutants.py with the rung that catches
    it, run `verify/mutants.py <name>`, and add a row to section 1.
- **A new optimised path** (a hillclimb change under SPEC_V3 §4.4):
  - Make fuzz-diff run it next to the plain path (src/runtime/fuzz_diff.cpp) and count the rows it
    computes in a `FuzzDiff` path-hit field that blz.cpp prints.
  - Add its check to verify/codegen.sh if it needs another build or flag, and a rung to `LADDERS`
    in verify/mutants.py.
  - Plant at least one mutant in it and record what catches it. If L2's bound cannot reach its
    edge cases, add an L3 property.
  - CUDA-only paths are checked on a CUDA build. Planner-gated paths are checked with the profile
    that enables them.
- **A changed rule.** Change SPEC_V3.md first, with the user's approval. Then change the oracle and
  the generator, and the L1 theorem if a shortcut depends on the rule. Legacy outputs stay frozen
  (AC-V3-0).
- **A new index primitive.** Add a `Contract` with a naive definition to
  src/graph/index_contracts.{hpp,cpp}, and a mutant of it.
- **A new bound.** Edit the variables at the top of verify/codegen.sh, and `BLZ_L2_SMOKE` in
  tests/CMakeLists.txt for the smoke tests. mutants.py reads codegen.sh's values.
- **Moving the sync commit.** Move it only after the full run in section 2 passes at the new
  commit, every mutant is caught where recorded, and section 1's record matches the code. Then edit
  the hash in section 1. Until then, leave the old hash. It tells the next agent where to start.

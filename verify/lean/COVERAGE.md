# L1 coverage: what the proofs justify, under which contracts, checked by what

L1 (SPEC v3 §4.3) proves T3 (§4.1) for all sizes: each algorithmic shortcut the generated code takes
over the literal rule of §3.2 gives the rule's answer. `verify/algorithms.sh` builds this project
from scratch and fails on `sorryAx`; `verify/algorithms.sh --mutants` also runs the 36 proof
mutants in `mutants.py` (ctest `verify_algorithms`, label `verify`). Toolchain: Lean 4.36.0, core
only.

Each theorem names the C++ it justifies three ways: the generator (`src/ir/plan.cpp` and
`src/codegen/cpu_emitter.cpp`), the generated function, and the snapshot of that function in
`tests/snapshots/`. The `blzc_snapshots` test (label fast) fails when the generated text changes, and
that is the signal to revisit this file. Each contract the proofs assume is a hypothesis of the
theorem, and the last section names the executable check that verifies it on the real code.

## The model and what it leaves out

- Timestamps, windows and bounds are unbounded integers (`Int`). The code's int64 bound arithmetic
  and int32 timestamps are L3's (`L3 arithmetic: timestamps at the int32 limits and windows from 1
  to INT32_MAX`).
- One node sequence at a time. A hop is its timestamp list, the times of the visible edges between
  two consecutive nodes in list order. The DFS that enumerates simple node sequences (distinct
  nodes, U and V excluded) is T1's: L2 (`verify/codegen.sh`) and L3 cover it, L1 does not.
- A primitive's answer is a function with its contract as a hypothesis (`IsEarliest`, `IsLatest`,
  `Search`). It is not a model of the binary searches. `Scan.lean` derives the hop contracts from
  the T4 contracts of `range`, `distinct_first` and `list`. `IsEarliest.unique` and
  `IsLatest.unique` show that any answer meeting a contract equals the reference answer.
- CUDA's heavy-kernel lane stride and the work estimate and schedule change only who runs a
  binding, not which bindings exist. L2 covers them (rel-cuda, both schedules), L1 does not.

## The shortcuts, and where each is proved

| Shortcut over the literal rule | Code | Proved by |
|---|---|---|
| One greedy binding per hop, not every witness | tcyc hops (`for_all`, distinct) | `earliest_witness` and its clean and legacy forms |
| The closing hop is a `pair_exists` probe | tcyc `close_k` stages | `earliest_witness` (its `first` is pair_exists's contract) |
| R2's open lower bound as `ts >= t - delta + 1`, R4's `<=` as `max_ts(lo, ts_h)` | clean tcyc and tchain bounds | `clean_tcyc_earliest_witness`, `clean_tchain_latest_witness` |
| Q5's `<` as `ts_h + 1` and `ts_h - 1`, and the trigger's own time excluded by `min_ts(t - 1, vis_t)` | legacy tcyc and tchain bounds | `legacy_tcyc_earliest_witness`, `legacy_tchain_latest_witness` |
| R3's `ts <= t` folded into each range's `hi` (`vis_t`) | every clean range | the clean forms (hi = t) |
| The redundant `max` with lo and `min` with vis_t | `max_ts(t - delta + 1, ts_h)`, `min_ts(ts_h, vis_t)` | modelled literally (`greedy`, `latestGreedy`), so no separate lemma is needed |
| Backward search with the latest witness per hop | tchain (`pair_latest`) | `latest_witness` and its clean and legacy forms |
| tchain never tests `pair_latest` for kNoTs | `ts_h = pair_latest(...)` after a scanned entry | `tchain_hop_isLatest` |
| A distinct-first scan binds each neighbour once, at its earliest time in range | every distinct scan | `scanBinding_isEarliest`, `scanBinding_eq_earliestIn` |
| `count_distinct` counts distinct neighbours | gs, fan and degree counts | `countDistinct_eq` |
| sg_ord compares one earliest and one latest witness, not every pair, in both scan branches | `intermediates` stage | `order_test` and its clean (R6b) and legacy forms |
| No `y ∉ {s, w}` filter on sg_ord's intermediates under R1 | clean_sg_ord | out of L1: R1 (no self-loops in the clean index) means y ≠ s, w on any kept path. Checked by L2 clean_all |
| gs: `count_distinct` twice, the a = 0 and b = 0 gate, A's `ts <= t` folded into the range, no `!= x` filters under R1 | clean_gs, gs | `gsRow_eq`, `clean_gs_row`, `legacy_gs_row` |
| One forward sweep per list instead of a count per trigger, with incremental `count` and `seen` | fan and degree drivers | `Sweep.sweep_correct` |
| Stream visibility as list position `<= j` instead of `eid <= e` | sweep under stream visibility | `Sweep.sweep_correct` (`heid`) |
| Clearing `seen` over the last window only | `sweep_count`'s clearing loop | `Sweep.clear_zero` |
| Reachability pruning of the trigger and of each binding (legacy only) | `tcyc_trigger_reach` | `root_kept`, `bindings_kept`, `greedy_bindings_kept` |
| `range`'s binary searches, `pair_exists`'s first-entry visibility, `prev_same` | include/blz/primitives.hpp | T4, not L1: `blz index-contracts` |

## Theorems

Names are in namespace `Blz`. Each module's docstring gives the statement in full.

### Hops.lean: the rule

`Witness o lo t hops ts` is the rule for one node sequence: ts picks one time per hop, lo ≤ t₁, and
t₁ R t₂ R … R t_k R t. R is ≤ for R4 (o = `le`) and < for Q5 (o = `lt`). The module has no main
theorem. Its lemmas are about `Chain`.

### EarliestWitness.lean: tcyc

- **`earliest_witness`**: for either order, any bounds and any number of hops, a witness exists iff
  the greedy binds every hop. The greedy takes the earliest time at least the previous hop's
  (strictly later for Q5), and at most hi.
  - Contracts: `hfirst` (each hop's answer is the earliest in [b, hi]), and `hhi` (x R t iff x ≤ hi).
- **`clean_tcyc_earliest_witness`**: window edges with t − δ < t₁ ≤ … ≤ t_k ≤ t exist iff the loop
  nest binds every hop from `t − δ + 1`, each later one from `max(t − δ + 1, h)`, all at most t.
  - Justifies the generator's `plan_stage` (for_all and exists stages), `inclusive` (src/ir/plan.cpp),
    and `CpuEmitter::binding` and `CpuEmitter::check` (src/codegen/cpu_emitter.cpp).
  - Generated function: `clean_tcyc_trigger`. Snapshot: tests/snapshots/clean_tcyc.hpp.
  - Contract: `hfirst`.
- **`legacy_tcyc_earliest_witness`**: window edges with t − δ ≤ t₁ < … < t_k < t exist iff the loop
  nest binds every hop from `t − δ`, each later one from `max(t − δ, h + 1)`, all at most
  `min(t − 1, vis_t)`.
  - Generated function: `tcyc_trigger`. Snapshot: tests/snapshots/tcyc.hpp.
  - Contracts: `hfirst`, and `hvis` (t − 1 ≤ vis_t).

### LatestWitness.lean: tchain, the mirror

- **`latest_witness`**: for either order, a witness exists iff the backward greedy binds every hop.
  The greedy walks the hops from the trigger back and takes the latest time at most the next hop's
  (strictly earlier for Q5), capped at vis.
  - Contracts: `hlast` (each hop's answer is the latest in [lo, c]), `hc` and `hcv`.
- **`clean_tchain_latest_witness`**: R4's chain exists iff the loop nest, from the trigger back,
  binds hop 1 at most t and each earlier hop at most `min(h, t)`, all from `t − δ + 1`.
  - Generated function: `clean_tchain_trigger`. Snapshot: tests/snapshots/clean_tchain.hpp.
  - Generator: `plan_stage` (`latest_witness`) and `CpuEmitter::binding`.
  - Contract: `hlast`.
- **`legacy_tchain_latest_witness`**: Q5's chain exists iff the loop nest binds hop 1 at most
  `min(t − 1, vis_t)` and each earlier hop at most `min(h − 1, vis_t)`, all from `t − δ`.
  - Generated function: `tchain_trigger`. Snapshot: tests/snapshots/tchain.hpp.
  - Contracts: `hlast`, `hvis`.

### OrderTest.lean: sg_ord

- **`order_test`**: comparing the earliest s → y time with the latest y → w time decides whether
  some pair of edges in [lo, hi] is in order.
  - Covers both scan branches. Scanning S.out, `sw` is the scanned entry and `pw` is pair_latest.
    Scanning W.in, `sw` is pair_latest and `pw` is pair_exists.
  - Contracts: `hA : IsEarliest` and `hB : IsLatest`.
- **`clean_sg_ord_order_test`**: R6b, with the window t − δ < t' ≤ t and t₂ ≤ t₃.
  - Generated function: `clean_sg_ord_trigger`. Snapshot: tests/snapshots/clean_sg_ord.hpp.
  - Generator: `plan_stage` (intersection with order) and `CpuEmitter::intersection_loop`.
- **`legacy_sg_ord_order_test`**: legacy sg_ordered, with the closed window t − δ ≤ t' ≤ vis_t and
  t₂ < t₃.
  - Generated function: `sg_ordered_trigger`. Snapshot: tests/snapshots/sg_ordered.hpp.

### Scan.lean: the hop contracts from T4's

- **`countDistinct_eq`**: `count_distinct` over a range is the number of distinct neighbours that
  pass keep.
  - Justifies `count_distinct` in include/blz/primitives.hpp and `CpuEmitter::check` (count stages).
  - Contract: `distinct_first`, the model `firsts`.
- **`scanBinding_isEarliest`**: a distinct-first scan of `range(x, b, hi, me)` binds neighbour y at
  the earliest time of hop x → y in [b, hi]. That is `IsEarliest`, the `hfirst` of the tcyc
  theorems.
  - Contracts: `range` and `distinct_first` (the model), and `hs` (x's list sorted by ts).
- **`tchain_hop_isLatest`**: tchain's answer for a hop meets `IsLatest`. The answer is none when the
  scan does not visit y, otherwise pair_latest's.
  - Contract: `hpl` (pair_latest's).
- **`scanBinding_eq_earliestIn`** and **`tchainAnswer_eq_latestIn`**: those answers equal the
  reference answers `earliestIn` and `latestIn`.

### Gs.lean: R7

- **`gsRow_eq`**: the code's gate (0 when a = 0 or b = 0) changes nothing: the row is min(a, b).
- **`clean_gs_row`**: a self-loop trigger's row is 0 (R1). Otherwise the row is min(|A|, |B|).
  - A is the distinct sources s ∉ {x, w} of window edges s → x with ts ≤ t, and B is the distinct
    sinks of window edges x → d.
  - Generated function: `clean_gs_trigger`. Snapshot: tests/snapshots/clean_gs.hpp.
  - Contracts: `range`, `distinct_first`, `hin` and `hout` (no self-loop entries in x's lists, and
    eid ≤ kAnyEid).
- **`legacy_gs_row`**: the same with legacy visibility, ts < t on A's side, and the explicit
  `!= x` filters.
  - Generated function: `gs_trigger`. Snapshot: tests/snapshots/gs.hpp.

### Sweep.lean: the sweep lowering

- **`Sweep.sweep_correct`**: for every trigger j of a node's list, the count `sweep_count` emits
  equals the per-trigger plan's count over `range(x, lower(t_j), vis_t, vis_e)`.
  - Covers the edge count and the distinct count, under stream and timestamp visibility.
  - Justifies `sweep_count<Distinct>` (include/blz/sweep.hpp) and `CpuEmitter::sweep_driver`.
  - Generated functions: the `*_cpu` drivers in tests/snapshots/{clean_,}{fan,deg}_{in,out}_cpu.cpp.
  - Contracts: `hsorted` (ts sorted along the list), `heid` (eid strictly increasing along the
    list), and `hlower` (lower is monotone).
- **`Sweep.clear_zero`**: the clearing loop leaves `seen` all zero, so the next list starts from the
  empty state.
  - Contract: none beyond the invariant `sweep_correct` establishes.

### Prune.lean: reachability pruning, legacy strict order only

Pruning is planned only for strict-order chains (legacy tcyc), and P14 does not extend it to R4. The
mutant "prune: extended to R4's ≤ greedy" shows why: under ≤, two hops at the same time fail the
binding check ld[c] > h.

- **`root_kept`**: if the trigger closes a simple cycle V → … → U with a strict witness of at most R
  hops, then V passes the root check, `keep(V, t − δ − 1)`, that is ld[V] ≥ t − δ.
- **`bindings_kept`**: every binding (c_j, t_j) of any such witness passes `keep(c_j, t_j)`.
- **`greedy_bindings_kept`**: every binding the legacy greedy makes passes the check. So pruning
  never cuts a branch that emits.
- **`ldJ_search`**: a plain round-by-round search meets the `Search` contract, so the contract can
  be met.
- **`one_round_cuts`**: with one round fewer than hops, a real cycle is cut. So the round bound is
  needed.
- Justifies `ReachScratch::search` and `ReachScratch::keep` (include/blz/reach.hpp) and the
  reach-pruned loop nest from `CpuEmitter`.
  - Generated function: `tcyc_trigger_reach`. Snapshot: tests/snapshots/tcyc_cpu.cpp.
  - Called with `hops = 9`, hi = `min_ts(t - 1, vis_t)`.
  - Contracts: `Search` (init, keep, relax) and, for `greedy_bindings_kept`, `hfirst`.
  - The C++ runs 9 rounds. The proofs read ld only up to round R = 9, so `Search` is needed
    only for rounds below 9. Later rounds are the search continued, which never runs.

## Contracts and their executable checks

| Lean hypothesis | What it says of the C++ | Executable check |
|---|---|---|
| `hfirst` for a closing hop | `pair_exists(x, y, lo, hi, e)` is the earliest ts of an entry with neighbour y in those bounds, or kNoTs | `blz index-contracts`, contract `pair_exists`: ctest `l2_smoke_index_contracts` (fast), and `rel/index-contracts/m=5,n=5,t=4` and `dev/index-contracts/m=4,n=4,t=3` in verify/codegen.sh. Also the unit test "primitives: range, distinct_first, pair_exists, pair_latest and the counts equal naive scans" |
| `hfirst` for a scanned hop | derived in Scan.lean from `range`, `distinct_first` and `hs` | `blz index-contracts`, contracts `range`, `distinct_first` and `distinct_first_ts` (find_start disabled), with the checks above |
| `hlast`, `hpl` | `pair_latest(x, y, lo, hi, e)` is the latest such ts, or kNoTs | `blz index-contracts`, contract `pair_latest` |
| `hs`, `hsorted` | a node's list is sorted by ts | `blz index-contracts`, contract `list` (x's entries in eid order), together with GraphIndex rejecting an unsorted stream (src/graph/index.cpp:20; unit test "index: rejects an unsorted stream and out-of-range node ids") |
| `heid` | a node's list is strictly increasing in eid | `blz index-contracts`, contract `list` |
| `hin`, `hout` (R1) | the clean index has no self-loop entries; an eid fits in uint32 | `blz index-contracts`, contract `list` under SelfLoops::drop (it checks both modes); eid ≤ kAnyEid holds by the type `std::uint32_t` |
| `hhi`, `hc`, `hcv`, `hvis` | vis_t is t (timestamp visibility) or kNoUpper (stream visibility) | the generated prologue `vis_t = p.ts_visibility ? t : kNoUpper` (`blzc_snapshots`); clean profiles pin timestamp visibility (E403, src/runtime/profile.cpp; unit test "profiles: a clean profile, its count blocks and E403 both ways (SPEC v3 §3.2, R7)") |
| `hlower` | every driver's lower(t) is t − δ + 1 or t − δ | the generated `lower` lambdas (`blzc_snapshots`) |
| `Search` (init, keep, relax) | ReachScratch's frontier search gives ld[U] ≥ t, never lowers an ld, and relaxes every visible edge x → y in [lo, hi] with x ≠ y and ts < ld[y] in the next round | **no direct check; this is a gap.** The end-to-end consequence is checked: fuzz-diff's cpu backend runs the reach-pruned path against the oracle at bounds A and B (the `rel/cpu/<profile>/<bound>` checks of verify/codegen.sh, which report `reach-pruned rows`). Unit tests: "reach: latest departure times towards the target, hop bound, window, visibility" and "reach: hand-checked tcyc rows are the same with and without pruning" |

## Known gaps

- **ReachScratch meets `Search`**: this is argued in reach.hpp's comment, not proved or checked
  directly. In-round reads of raised values only raise ld, and `x == target` is skipped, which is
  harmless since ld[U] = t is already maximal for edges that must have ts < t. L2's pruned paths
  check the consequence. A direct check would compare `ld` after r rounds with the Jacobi rounds
  of `ldJ`, a later unit's job.
- **The DFS over node sequences** (T1) and the CUDA-only lowering are L2's and L3's, not L1's.
- **Machine width**: L1's integers are unbounded. L3 arithmetic covers the int32 timestamps and int64
  bounds.
- **sg_ord's count `len_I`** is not one theorem. `order_test` decides each intermediate y, and
  Scan's `firsts_nodup` and `mem_nbr_firsts` say the distinct-first scan visits each neighbour
  in range once. Composing them into |{y | some pair in order}| is left to L2's clean_all and
  legacy checks.
- **sg_ord's missing y ∉ {s, w} filter** rests on R1 through the clean index. L2 clean_all checks it,
  but L1 does not model the intersection's element set.

---
name: verify-algorithms
description: "L1 of SPEC_V3 §4.3 (target T3): the Lean 4 proofs in verify/lean that the generated loop nests' shortcuts (greedy earliest/latest witness, sg_ord's order test, distinct-first scans, gs's counts, the sweep lowering, legacy reachability pruning) give the literal rule's answer for all sizes. Use when a change touches blzc's planner or CPU emitter (src/ir/plan.cpp, src/codegen/cpu_emitter.cpp), the snapshots in tests/snapshots that COVERAGE.md names, include/blz/{primitives,sweep,reach}.hpp, a rule of SPEC_V3 §3.2 (R2-R7) or legacy Q5, or verify/lean itself; at P14, P16 and P17 phase exits; and for any new algorithm a speed change brings (SPEC_V3 §4.4 V-proof)."
---

# verify-algorithms: the L1 proofs (T3)

**Property.** Each shortcut the generated code takes over the literal rule of SPEC_V3 §3.2 (or the
legacy rule) gives the rule's answer, for every graph size, under contracts that the executable
checks verify on the real C++.

**Harm if broken.** A shortcut that drops one witness drops a row silently: tests on small or random
graphs see it only by chance, and the classifier's features change.

**Tool.** Lean 4.36.0, core only, no Mathlib (D-51). Proofs are kernel-checked, so there is no solver
timeout. The models are hand transcriptions of the generated code. Lean does not read C++, so the
coverage record below is how a proof stays tied to the code.

## 1. Does my change touch covered code?

Sync commit: `7871cfe`. At it, COVERAGE.md matched the code: `blzc_snapshots` passed, so the
generated text is the text the models describe; `verify/algorithms.sh --mutants` passed; and
`blz index-contracts` checked every contract the proofs assume. Run:

```
git diff --stat 7871cfe -- src/ir/plan.cpp src/codegen/cpu_emitter.cpp include/blz/primitives.hpp \
  include/blz/sweep.hpp include/blz/reach.hpp tests/snapshots verify/lean SPEC_V3.md patterns
```

This compares the sync commit with the working tree, so uncommitted edits are listed too. Don't
write `7871cfe..HEAD`, which compares commits only and misses a change before it is committed.
If it lists nothing, the proofs still describe the code. If it lists files, read the matching rows
of `verify/lean/COVERAGE.md`. The record, in short:

| Theorem (namespace `Blz`) | Module | Generated function, snapshot | Generator |
|---|---|---|---|
| `clean_tcyc_earliest_witness`, `legacy_tcyc_earliest_witness` (on `earliest_witness`) | EarliestWitness.lean | `clean_tcyc_trigger`, `tcyc_trigger`: tests/snapshots/clean_tcyc.hpp, tcyc.hpp | `plan_stage`, `inclusive` (plan.cpp); `CpuEmitter::binding`, `check` |
| `clean_tchain_latest_witness`, `legacy_tchain_latest_witness` (on `latest_witness`) | LatestWitness.lean | `clean_tchain_trigger`, `tchain_trigger`: clean_tchain.hpp, tchain.hpp | `plan_stage` (`latest_witness`), `CpuEmitter::binding` |
| `clean_sg_ord_order_test`, `legacy_sg_ord_order_test` (on `order_test`) | OrderTest.lean | `clean_sg_ord_trigger`, `sg_ordered_trigger`: clean_sg_ord.hpp, sg_ordered.hpp | `CpuEmitter::intersection_loop` |
| `countDistinct_eq`, `scanBinding_isEarliest`, `tchain_hop_isLatest`, `scanBinding_eq_earliestIn`, `tchainAnswer_eq_latestIn` | Scan.lean | every distinct scan; `count_distinct` | include/blz/primitives.hpp; `CpuEmitter::check` |
| `gsRow_eq`, `clean_gs_row`, `legacy_gs_row` | Gs.lean | `clean_gs_trigger`, `gs_trigger`: clean_gs.hpp, gs.hpp | count stages |
| `Sweep.sweep_correct`, `Sweep.clear_zero` | Sweep.lean | the `*_cpu` drivers in {clean_,}{fan,deg}_{in,out}_cpu.cpp | `sweep_count` (include/blz/sweep.hpp), `CpuEmitter::sweep_driver` |
| `root_kept`, `bindings_kept`, `greedy_bindings_kept`, `ldJ_search`, `one_round_cuts` | Prune.lean (legacy strict order only) | `tcyc_trigger_reach`: tcyc_cpu.cpp | `ReachScratch::search`, `keep` (include/blz/reach.hpp) |

Every theorem takes its contracts on the code as hypotheses (`hfirst`, `hlast`, `hpl`, `hs`,
`hsorted`, `heid`, `hin`, `hout`, `hhi`, `hvis`, `hlower`, `Search`). COVERAGE.md's last table names
the executable check of each: `blz index-contracts` (contracts `range`, `distinct_first`,
`distinct_first_ts`, `pair_exists`, `pair_latest`, `list`, `search`), `blzc_snapshots`, and E403.
Those checks belong to `/verify-codegen`.

There are two ways a model goes stale, and `verify/algorithms.sh` notices neither, because Lean
does not read C++:
- **The generated text changed.** ctest `blzc_snapshots` fails (label fast). After a reviewed
  snapshot update, open COVERAGE.md for each changed snapshot and compare the generated bounds and
  loop order with the Lean definition that models them: `greedy` and `latestGreedy` with their bound
  arguments, `firsts`, `Sweep.advLo` and `advHi`, `relaxRound`.
- **A hand-written header changed**: include/blz/sweep.hpp, reach.hpp or primitives.hpp. No snapshot
  changes, and L1 passes whatever the header says. Compare the edited lines with their model by
  hand. The models' docstrings quote the C++ they transcribe: `Sweep.enter`, `leave` and `clear` for
  `sweep_count`, `relaxRound` for `ReachScratch::search`, Scan.lean for the primitives. Then run
  `/verify-codegen`, which runs the real header.

A green `verify/algorithms.sh` after a C++ edit proves only that the proofs still check. It says
nothing about whether the C++ still matches them.

## 2. Run it

Quick, on every change that touches the files above (4 s):

```
verify/algorithms.sh
```

Full, at phase exits and after any change to verify/lean, once the quick form passes (30 s). It
stops at the build, so it runs no mutants while the build fails:

```
verify/algorithms.sh --mutants
```

It builds a fresh copy of verify/lean in a temporary directory from an empty `.lake`. It fails on
any build error, on `declaration uses sorry`, and on any main theorem whose axioms go beyond
`propext`, `Classical.choice` and `Quot.sound` (Axioms.lean lists the 24 main theorems). With
`--mutants` it then runs the 37 proof mutants of verify/lean/mutants.py. Each must fail to build at
the declarations recorded for it. ctest runs the full form as `verify_algorithms` (label verify)
when lake is found at configure time.

The toolchain lives in `~/.cache/blz-fv/lean/elan`. When it is missing, the script prints the two
install commands (elan, then `elan toolchain install leanprover/lean4:v4.36.0`, about 3 GB).

## 3. Read a failure

- **A build error** prints `error: Blz/<Module>.lean:<line>:<col>: ...`. The declaration that
  contains the line is the one that no longer holds. Lean's message gives the unsolved goal: the
  hypotheses in scope, then `⊢` and what was left to prove. Read the goal as a claim about the code:
  for example `⊢ x < t` inside `greedy_sound` says the greedy took a time the rule forbids.
- **`Missing cases:`** under a line of a theorem: a `match` in the proof can no longer cover every
  input, because the model now gives another answer there. Each listed case is an input shape, and
  together they are the counterexample's shape. For example, `none, (some _), ...` in `order_test`
  means: no s → y witness, one y → w witness, and the model now answers true where no ordered pair
  exists.
- **`L1 FAILED: a declaration uses sorry`** or **`... depends on sorryAx`**: someone left a proof
  unfinished. That is never a pass.
- **The axioms check fails** when a theorem was renamed or removed without editing Axioms.lean, or
  when it now uses another axiom.
- **mutants.py** prints a line per mutant:
  - `KILLED` is the pass, with the declarations that failed and whether the first failure is the
    false claim itself (`claim`) or a brittle proof step before it (`step`).
  - `SURVIVED`: the mutant builds, so the proofs did not notice a wrong model, contract or claim.
  - `MOVED`: it fails, but at other declarations than recorded.
  - `BAD`: its `old` text is not in the module exactly once.

## 4. Real bug, or a modelling error?

The rule side of every theorem (`Witness`, `Chain` and `Picks` in Hops.lean, and R7's sets in Gs.lean)
is SPEC_V3 §3.2 or legacy Q5, written literally. The code side (`greedy`, `latestGreedy`, `scanBinding`,
`sweep`, `ldJ`) is a hand model of the generated text. When a proof breaks after a change:

1. Find which side moved. Compare the code side's definition with the snapshot named in COVERAGE.md,
   bound by bound and loop by loop. If section 1's diff lists only verify/lean, the C++ is the
   code that passed at the sync commit, so the snapshot is the reference. An edited model that no
   longer matches it is a model error: restore the model. Don't re-prove a claim that has become
   false, and skip `/verify-codegen`, since its code did not change.
2. **The model no longer matches the code.** First find out whether the new code computes the same
   set. Run `/verify-codegen`'s quick check (L2 on the real code), or copy verify/lean to a scratch
   directory, transcribe the new line into the model, and run `lake build` there. If either shows a
   difference, the change is a bug, so go to step 3's report. Don't re-prove. Only a change that
   computes the same set (a renamed bound, a reordered loop, a new emitter idiom) is a modelling
   update. For that, change the definition to the new text, keep the theorem's rule side
   unchanged, and re-prove.
3. **The model matches the code and the proof cannot be repaired**: the new shortcut may be unsound.
   Look for a concrete case first, with `verify/codegen.sh --smoke` or a hand graph. A failed proof
   is not yet a counterexample. If the case exists, it is a real bug: report it in the code's terms.
4. **Never weaken the rule side or drop a contract to make a proof go through.** A theorem that
   proves less than the code needs is a silent gap. If the rule itself changed, change SPEC_V3.md
   first, with the user's approval (its §3.5 decisions are the user's).
5. A `MOVED` mutant after a proof refactor is a record update: check that it still fails at the
   claim it makes false, then edit `where` in mutants.py. A `SURVIVED` mutant means the proofs lost
   strength. Find which hypothesis or lemma now makes the false claim provable, and do not just
   delete the mutant.

The edges of the model are stated in COVERAGE.md and are not bugs. Integers are unbounded (machine
width is L3's). The DFS over node sequences and the CUDA lane schedule are L2's and L3's. Pruning is
proved for the legacy strict order only.

## 5. Update the verification for an intended change

- **A new shortcut or algorithm** (for example a hillclimb change under SPEC_V3 §4.4):
  - Add a module in verify/lean/Blz that states the literal rule, the code's model and the
    equivalence for all sizes, with every assumption on the C++ as a hypothesis. Import it in Blz.lean.
  - Add `#print axioms` lines for its main theorems to Axioms.lean.
  - Add at least one proof mutant per main theorem to mutants.py (a wrong bound, a dropped contract
    clause, or a swapped comparison), each with `where` and `claim`. Run `--mutants` and copy the
    printed declarations into the record.
  - Add rows to COVERAGE.md: the generated function, its snapshot, the generator function, and each
    new contract with the executable check that verifies it on the code. A contract without a check
    is a gap. Write it under Known gaps, or add the check to `blz index-contracts`
    (src/graph/index_contracts.cpp), as the `search` contract was for `Search`.
- **A changed rule**: change the rule side to the new text, then re-prove the code side. Update the
  clean and legacy forms separately, since legacy outputs are frozen (AC-V3-0).
- **Pruning extended to R4**: the mutant "prune: extended to R4's ≤ greedy" shows that the present
  binding check `ld[c] > h` is unsound under ≤. A new check needs a new theorem.
- **Moving the sync commit.** Move it only after `blzc_snapshots` passes, `verify/algorithms.sh
  --mutants` passes, and COVERAGE.md's rows match the snapshots at the new commit. Then edit the
  hash in section 1 of this file. If you skipped the check, leave the old hash: it tells the next
  agent where to start.

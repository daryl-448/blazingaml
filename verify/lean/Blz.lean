import Blz.Hops
import Blz.EarliestWitness
import Blz.LatestWitness
import Blz.OrderTest
import Blz.Scan
import Blz.Gs
import Blz.Sweep
import Blz.Prune
import Blz.Chunks

/-!
# L1: the generated code's shortcuts against SPEC v3 §3.2's rules (T3), for all sizes

Each module proves one shortcut that the generated loop nests take over the literal rule, with
the contracts it assumes of the C++ as explicit hypotheses. COVERAGE.md maps every main theorem
to the C++ it justifies and every contract to the executable check that verifies it on the real
code. `verify/algorithms.sh` builds this library from scratch and fails if any theorem depends on
`sorryAx`; `--mutants` runs the proof mutants.

- `Hops`: hop lists, the window and the two orders (R4 ≤, legacy Q5 <).
- `EarliestWitness`: tcyc's one greedy binding per hop finds a witness iff one exists.
- `LatestWitness`: the mirror, tchain's backward search with pair_latest.
- `OrderTest`: sg_ord compares one earliest with one latest witness (R6b).
- `Scan`: a distinct-first scan meets the hop contracts; count_distinct counts distinct
  neighbours.
- `Gs`: gs's two counts, its gate and its folded order test give min(|A|, |B|) (R7).
- `Sweep`: the sweep lowering emits the per-trigger count, edge and distinct, both visibilities.
- `Prune`: reachability pruning never cuts a cycle; legacy strict order only.
- `Chunks`: the chunk driver's range mining (H7): a selection mined in chunks gives the whole
  selection's rows and hash.
-/

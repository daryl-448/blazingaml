import Blz.EarliestWitness

/-!
# Reachability pruning is sound, for the legacy strict order only (SPEC v2 §8 S6, D-43)

**Legacy only.** The planner applies reachability pruning to strict-order chains, which in the
library is the legacy `tcyc` (`tcyc_trigger_reach` in tests/snapshots/tcyc_cpu.cpp); clean tcyc
has `cpu_reach` false. P14 does not extend pruning to R4, so this module proves it for Q5's strict
order (`Order.lt`) and nothing else.

Per trigger U → V at t, `ReachScratch::search` (include/blz/reach.hpp) runs a backward search from
U over the in-lists, `hops` = 9 rounds, and gives each node x its latest departure time ld[x]. The
pruned loop nest then skips the trigger unless `keep(V, t - delta - 1)` (ld[V] ≥ t − δ) and skips a
binding of node c at hop time h unless `keep(c, h)` (ld[c] > h).

Model. E is the trigger's search set: the visible edges with ts in [lo, t − 1], lo = t − δ. A
witness over hop lists drawn from E is the same as one over all visible edges, since the legacy
witness already confines every pick to [lo, t − 1]. The search produces arrays ld 0, ld 1, …, one
per round. The proof uses only the **`Search` contract** below; `ldJ`, a plain round-by-round
search, meets it. reach.hpp's frontier and in-round updates are not modelled, so that the C++
search meets `Search` is a contract on the code. Ported from the planning prototype
(~/.cache/blz-fv/prototypes-2026-11-22/lean/BlzFv/Prune.lean) onto `Witness .lt`.
-/
namespace Blz

structure Edge where
  src : Nat
  dst : Nat
  ts : Int
deriving DecidableEq, Repr

/-- One hop's timestamp list: the edges a → b of E. -/
def hopTs (E : List Edge) (a b : Nat) : List Int :=
  (E.filter fun e => e.src == a && e.dst == b).map (·.ts)

/-- The hop lists of a node sequence c_0, c_1, …, c_n (n hops). -/
def hopLists (E : List Edge) : List Nat → List (List Int)
  | a :: b :: cs => hopTs E a b :: hopLists E (b :: cs)
  | _ => []

/-- `x` reaches `U` by a walk of at most `m` edges whose times strictly increase, the first after
`h`, all before `t`. Nodes may repeat; self-loops are not hops. -/
inductive Reach (E : List Edge) (U : Nat) (t : Int) : Nat → Nat → Int → Prop
  | here {m : Nat} {h : Int} : h < t → Reach E U t m U h
  | step {m x y : Nat} {h τ : Int} :
      ⟨x, y, τ⟩ ∈ E → x ≠ y → h < τ → Reach E U t m y τ → Reach E U t (m + 1) x h

/-- **Contract** of the search, round by round; nothing else is used. ld r x is node x's latest
departure time after r rounds. -/
structure Search (E : List Edge) (U : Nat) (t : Int) (ld : Nat → Nat → Int) : Prop where
  init : t ≤ ld 0 U
  keep : ∀ r x, ld r x ≤ ld (r + 1) x
  relax : ∀ r e, e ∈ E → e.src ≠ e.dst → e.ts < ld r e.dst → e.ts ≤ ld (r + 1) e.src

variable {E : List Edge} {U : Nat} {t : Int} {ld : Nat → Nat → Int}

theorem Search.mono (S : Search E U t ld) {x : Nat} {r : Nat} : ∀ {R}, r ≤ R → ld r x ≤ ld R x
  | 0, h => by rw [Nat.le_zero.1 h]; exact Int.le_refl _
  | R + 1, h => by
    rcases Nat.lt_or_eq_of_le h with h | h
    · exact Int.le_trans (S.mono (Nat.le_of_lt_succ h)) (S.keep R x)
    · rw [h]; exact Int.le_refl _

/-- After m rounds, every node that reaches U within m edges after h has ld > h. -/
theorem Search.reach (S : Search E U t ld) : ∀ {m x h}, Reach E U t m x h → h < ld m x
  | m, _, _, .here hh => Int.lt_of_lt_of_le hh (Int.le_trans S.init (S.mono (Nat.zero_le m)))
  | m + 1, _, _, .step (y := y) (τ := τ) hE hne hlt hr => by
    have := S.relax m ⟨_, y, τ⟩ hE hne (S.reach hr)
    simp only at this
    omega

theorem mem_hopTs {a b : Nat} {τ : Int} (h : τ ∈ hopTs E a b) : (⟨a, b, τ⟩ : Edge) ∈ E := by
  simp only [hopTs, List.mem_map, List.mem_filter, Bool.and_eq_true, beq_iff_eq] at h
  obtain ⟨⟨s, d, τ'⟩, ⟨hm, rfl, rfl⟩, rfl⟩ := h
  exact hm

/-- A binding of `x` at hop time `h` whose rest of the path to U has a strict witness can reach U
within the rest's length. -/
theorem reach_of_witness :
    ∀ {cs : List Nat} {x : Nat} {h : Int} {ts : List Int},
      (x :: cs).Nodup → (x :: cs).getLast? = some U → h < t →
      Witness .lt (h + 1) t (hopLists E (x :: cs)) ts → Reach E U t cs.length x h
  | [], x, _, _, _, hend, ht, _ => by
    simp at hend; subst hend; exact .here ht
  | c :: cs, x, h, ts, hnd, hend, ht, hw => by
    cases ts with
    | nil => exact absurd hw.1 (by simp [hopLists, Picks])
    | cons τ ts =>
      obtain ⟨hτ, hlo, hhi, hrest⟩ := Witness.uncons hw
      have hxc : x ≠ c := fun e => by simp [e] at hnd
      have hnd' : (c :: cs).Nodup := (List.nodup_cons.1 hnd).2
      have hend' : (c :: cs).getLast? = some U := by simpa using hend
      simp only [Order.rel] at hhi
      exact .step (mem_hopTs hτ) hxc (by omega) (reach_of_witness hnd' hend' hhi hrest)

/-- **Pruning keeps the trigger (root check).** If the trigger closes a simple cycle
V → c_1 → … → U with a strict witness and at most R hops, V passes the root check ld[V] ≥ lo. -/
theorem root_kept (S : Search E U t ld) {V : Nat} {cs : List Nat} {lo : Int} {ts : List Int}
    {R : Nat} (hlo : lo ≤ t) (hnd : (V :: cs).Nodup) (hend : (V :: cs).getLast? = some U)
    (hR : cs.length ≤ R) (hw : Witness .lt lo t (hopLists E (V :: cs)) ts) : lo ≤ ld R V := by
  have hw' : Witness .lt (lo - 1 + 1) t (hopLists E (V :: cs)) ts := by
    rw [Int.sub_add_cancel]; exact hw
  have := S.reach (reach_of_witness hnd hend (by omega) hw')
  have := S.mono (x := V) hR
  omega

/-- **Pruning keeps every binding.** Along that cycle, every binding (c_j, t_j) of any strict
witness passes the binding check ld[c_j] > t_j. -/
theorem bindings_kept (S : Search E U t ld) {R : Nat} :
    ∀ {V : Nat} {cs : List Nat} {lo : Int} {ts : List Int},
      (V :: cs).Nodup → (V :: cs).getLast? = some U → cs.length ≤ R →
      Witness .lt lo t (hopLists E (V :: cs)) ts → ∀ p ∈ cs.zip ts, p.2 < ld R p.1
  | _, [], _, _, _, _, _, _, p, hp => by simp at hp
  | V, c :: cs, lo, ts, hnd, hend, hR, hw, p, hp => by
    cases ts with
    | nil => simp at hp
    | cons τ ts =>
      obtain ⟨_, _, hhi, hrest⟩ := Witness.uncons hw
      have hnd' : (c :: cs).Nodup := (List.nodup_cons.1 hnd).2
      have hend' : (c :: cs).getLast? = some U := by simpa using hend
      have hR' : cs.length ≤ R := by simp at hR; omega
      simp only [Order.rel] at hhi
      rcases List.mem_cons.1 hp with rfl | hp
      · have := S.reach (reach_of_witness hnd' hend' hhi hrest)
        have := S.mono (x := c) hR'
        simp only
        omega
      · exact bindings_kept S hnd' hend' hR' hrest p hp

/-- **The code's bindings are kept.** The legacy loop nest binds the greedy's times
(`legacy_tcyc_earliest_witness`); when the greedy closes the cycle, every one of its bindings
passes the binding check, so pruning never cuts a branch that emits. Contracts: `Search`, and
`hfirst` as in `EarliestWitness.lean`. -/
theorem greedy_bindings_kept (S : Search E U t ld) {first : List Int → Int → Int → Option Int}
    (hfirst : ∀ L b hi, IsEarliest L b hi (first L b hi)) {R : Nat} {V : Nat} {cs : List Nat}
    {lo : Int} {ts : List Int} (hnd : (V :: cs).Nodup) (hend : (V :: cs).getLast? = some U)
    (hR : cs.length ≤ R)
    (hg : greedy .lt first lo (t - 1) lo (hopLists E (V :: cs)) = some ts) :
    ∀ p ∈ cs.zip ts, p.2 < ld R p.1 :=
  bindings_kept S hnd hend hR
    (greedy_sound hfirst (fun x => by show x < t ↔ _; omega) (Int.le_refl lo) hg)

/-! ### A concrete search that meets the contract: Jacobi rounds, U and self-loops skipped -/

def latestOf (p : Edge → Bool) (a : Int) (E : List Edge) : Int :=
  E.foldl (fun acc e => if p e then max acc e.ts else acc) a

theorem latestOf_ge {p : Edge → Bool} :
    ∀ {E : List Edge} {a : Int}, a ≤ latestOf p a E ∧ ∀ e ∈ E, p e = true → e.ts ≤ latestOf p a E
  | [], a => ⟨Int.le_refl a, by simp⟩
  | e :: E, a => by
    have ih := @latestOf_ge p E (if p e then max a e.ts else a)
    have hstep : latestOf p a (e :: E) = latestOf p (if p e then max a e.ts else a) E := rfl
    rw [hstep]
    refine ⟨Int.le_trans (by split <;> omega) ih.1, fun e' he' hp => ?_⟩
    rcases List.mem_cons.1 he' with rfl | he'
    · exact Int.le_trans (by simp [hp]; omega) ih.1
    · exact ih.2 e' he' hp

/-- One round: ld'[x] = max(ld[x], latest ts of an edge x → y, y ≠ x, with ts < ld[y]). -/
def relaxRound (E : List Edge) (U : Nat) (ld : Nat → Int) (x : Nat) : Int :=
  if x = U then ld x
  else latestOf (fun e => e.src == x && e.src != e.dst && decide (e.ts < ld e.dst)) (ld x) E

def ldJ (E : List Edge) (U : Nat) (t bot : Int) : Nat → Nat → Int
  | 0 => fun x => if x = U then t else bot
  | r + 1 => relaxRound E U (ldJ E U t bot r)

theorem ldJ_U {bot : Int} : ∀ r, ldJ E U t bot r U = t
  | 0 => by simp [ldJ]
  | r + 1 => by simp [ldJ, relaxRound, ldJ_U r]

/-- The contract can be met: a plain round-by-round search meets `Search`. -/
theorem ldJ_search {bot : Int} (hE : ∀ e ∈ E, e.ts < t) : Search E U t (ldJ E U t bot) where
  init := by simp [ldJ]
  keep r x := by
    show _ ≤ relaxRound E U _ x
    unfold relaxRound; split
    · exact Int.le_refl _
    · exact latestOf_ge.1
  relax r e he hne hlt := by
    show _ ≤ relaxRound E U _ e.src
    unfold relaxRound; split
    · rename_i h; rw [h, ldJ_U]; exact Int.le_of_lt (hE e he)
    · exact latestOf_ge.2 e he (by simp [hne, hlt])

/-! ### The pruning can fail: one round fewer than the hops cuts a real cycle (D-43's mutant) -/

/-- Trigger U = 0 → V = 1 at t = 3; cycle 1 → 2 → 0 at times 1, 2 inside [lo, t − 1] = [0, 2]. -/
def E2 : List Edge := [⟨1, 2, 1⟩, ⟨2, 0, 2⟩]

theorem two_rounds_keep : 0 ≤ ldJ E2 0 3 (-1) 2 1 ∧ 1 < ldJ E2 0 3 (-1) 2 2 := by decide

theorem one_round_cuts :
    (∃ ts, Witness .lt 0 3 (hopLists E2 [1, 2, 0]) ts) ∧ ¬ 0 ≤ ldJ E2 0 3 (-1) 1 1 := by
  refine ⟨⟨[1, 2], ?_⟩, by decide⟩
  refine ⟨by simp [hopLists, hopTs, E2, Picks], by simp, ?_⟩
  simp [Chain, Order.rel]

end Blz

import Blz.Scan

/-!
# The sweep lowering: one forward pass gives every trigger's count (SPEC v2 §7.1, D-29)

`sweep_count<Distinct>` (include/blz/sweep.hpp), which the generated fan and degree drivers call
(tests/snapshots/{clean_,}{fan,deg}_{in,out}_cpu.cpp), walks one node's list once. The entries of
the list are the triggers whose dst (in-lists) or src (out-lists) is that node. For trigger j it
moves the window's upper end `hi` forward over the entries that j sees, and its lower end `lo`
forward over the entries older than `lower(t_j)`, keeping the window's count as entries enter and
leave: an edge count, or a distinct count with per-neighbour counts `seen`. Then it emits `count`.

The per-trigger plan (`clean_fan_in_trigger` and the others) counts instead, from scratch, the
entries of `range(x, lower(t_j), vis_t, vis_e)` that pass keep: `count_edges`, or
`count_distinct`, which is the number of distinct neighbours (`countDistinct_eq`). This module
proves that the pass emits exactly that count for every trigger, under both visibilities, for both
counts, and that the clearing loop leaves `seen` all zero for the next node.

The list is modelled by its arrays, positions 0 … b − 1 being off[x] … off[x + 1] − 1.
-/
namespace Blz.Sweep

/-- One node's list and the pattern's parameters, as `sweep_count` reads them. -/
structure Cfg where
  /-- The list's length, off[x + 1] − off[x]. -/
  b : Nat
  ts : Nat → Int
  nbr : Nat → Nat
  eid : Nat → Nat
  /-- keep(x, ·): whether an entry's neighbour enters the count (the element filters). -/
  keep : Nat → Bool
  /-- lower(t): the window's inclusive lower bound for a trigger at t. -/
  lower : Int → Int
  /-- ts_visibility. -/
  tsVis : Bool
  /-- The template's `Distinct`. -/
  distinct : Bool

/-- The pass's state: the window [lo, hi), its count, and the per-neighbour counts. -/
structure State where
  lo : Nat
  hi : Nat
  count : Int
  seen : Nat → Nat

variable (c : Cfg)

/-- One turn of the upper loop's body: `if (keep(x, l.nbr[hi])) { Distinct ? count +=
seen[l.nbr[hi]]++ == 0 : ++count; }`, then `++hi`. -/
def enter (s : State) : State :=
  let y := c.nbr s.hi
  { lo := s.lo, hi := s.hi + 1,
    count := if c.keep y then
        (if c.distinct then s.count + (if s.seen y = 0 then 1 else 0) else s.count + 1)
      else s.count,
    seen := if c.keep y ∧ c.distinct then (fun z => if z = y then s.seen z + 1 else s.seen z)
      else s.seen }

/-- One turn of the lower loop's body: `if (keep(x, l.nbr[lo])) { Distinct ? count -=
--seen[l.nbr[lo]] == 0 : --count; }`, then `++lo`. -/
def leave (s : State) : State :=
  let y := c.nbr s.lo
  let seen' := fun z => if z = y then s.seen z - 1 else s.seen z
  { lo := s.lo + 1, hi := s.hi,
    count := if c.keep y then
        (if c.distinct then s.count - (if seen' y = 0 then 1 else 0) else s.count - 1)
      else s.count,
    seen := if c.keep y ∧ c.distinct then seen' else s.seen }

/-- The upper loop's test for trigger j at position k: `hi < b && (ts_visibility ? l.ts[hi] <= t :
hi <= j)`. -/
def enters (j k : Nat) : Bool :=
  decide (k < c.b) && (if c.tsVis then decide (c.ts k ≤ c.ts j) else decide (k ≤ j))

/-- `for (; hi < b && …; ++hi) …` -/
def advHi (j : Nat) (s : State) : State :=
  if _h : enters c j s.hi = true then advHi j (enter c s) else s
termination_by c.b - s.hi
decreasing_by
  simp only [enter, enters, Bool.and_eq_true, decide_eq_true_eq] at *
  omega

/-- `for (; lo < hi && l.ts[lo] < lo_ts; ++lo) …` -/
def advLo (lts : Int) (s : State) : State :=
  if _h : s.lo < s.hi ∧ c.ts s.lo < lts then advLo lts (leave c s) else s
termination_by s.hi - s.lo
decreasing_by simp only [leave]; omega

/-- Trigger j's turn: both loops, then the emit reads `count`. -/
def step (j : Nat) (s : State) : State := advLo c (c.lower (c.ts j)) (advHi c j s)

/-- The state after triggers 0, …, j − 1, starting from the empty window with `seen` all zero
(the vector's initial value, and what `clear_zero` leaves for the next node). -/
def after : Nat → State
  | 0 => ⟨0, 0, 0, fun _ => 0⟩
  | j + 1 => step c j (after j)

/-- Whether position k is in trigger j's window as the per-trigger plan reads it: the entries of
`range(x, lower(t_j), vis_t, vis_e)`, that is lower(t_j) ≤ ts and, under stream visibility,
eid ≤ eid_j, under timestamp visibility, ts ≤ t_j. -/
def inWindow (j k : Nat) : Bool :=
  decide (c.lower (c.ts j) ≤ c.ts k) &&
    (if c.tsVis then decide (c.ts k ≤ c.ts j) else decide (c.eid k ≤ c.eid j))

/-- The neighbours that trigger j's per-trigger plan counts, as `countDistinct_eq` states them. -/
def windowNbrs (j : Nat) : List Nat :=
  (((List.range c.b).filter (inWindow c j)).map c.nbr).filter c.keep

/-- Trigger j's count in the per-trigger plan: `count_distinct` or `count_edges` over its range. -/
def spec (j : Nat) : Int :=
  if c.distinct then ndistinct (windowNbrs c j) else (windowNbrs c j).length

/-! ### The count invariant: count and seen describe the window [lo, hi) -/

/-- The kept neighbours of the window [lo, hi). -/
def win (lo hi : Nat) : List Nat := ((List.range' lo (hi - lo)).map c.nbr).filter c.keep

structure CountInv (s : State) : Prop where
  lo_le : s.lo ≤ s.hi
  hi_le : s.hi ≤ c.b
  count : s.count = if c.distinct then (ndistinct (win c s.lo s.hi) : Int) else (win c s.lo s.hi).length
  seen : c.distinct = true → ∀ y, s.seen y = (win c s.lo s.hi).count y

theorem ndistinct_snoc : ∀ (l : List Nat) (y : Nat),
    ndistinct (l ++ [y]) = ndistinct l + if y ∈ l then 0 else 1
  | [], y => by simp [ndistinct]
  | a :: as, y => by
    have ih := ndistinct_snoc as y
    simp only [List.cons_append, ndistinct, List.mem_append, List.mem_cons]
    by_cases ha : a ∈ as
    · simp only [ha, true_or, ↓reduceIte, ih]
      by_cases hy : y ∈ as
      · simp [hy]
      · have : y ≠ a := fun e => hy (e ▸ ha)
        simp [hy, this]
    · by_cases hay : a = y
      · subst hay; simp [ha, ih]
      · have : ¬ y = a := fun e => hay e.symm
        simp only [ha, hay, ↓reduceIte, ih, this, false_or]
        by_cases hy : y ∈ as <;> simp [hy] <;> omega

theorem win_enter (lo hi : Nat) (h : lo ≤ hi) :
    win c lo (hi + 1) = win c lo hi ++ (if c.keep (c.nbr hi) then [c.nbr hi] else []) := by
  simp only [win]
  rw [show hi + 1 - lo = (hi - lo) + 1 by omega, List.range'_concat]
  rw [show lo + 1 * (hi - lo) = hi by omega]
  simp only [List.map_append, List.map_cons, List.map_nil, List.filter_append]
  split <;> simp_all

theorem win_leave (lo hi : Nat) (h : lo < hi) :
    win c lo hi = (if c.keep (c.nbr lo) then [c.nbr lo] else []) ++ win c (lo + 1) hi := by
  simp only [win]
  rw [show hi - lo = (hi - (lo + 1)) + 1 by omega, List.range'_succ]
  simp only [List.map_cons, List.filter_cons]
  split <;> simp_all

theorem CountInv.enter {s : State} (hs : CountInv c s) (hb : s.hi < c.b) :
    CountInv c (Sweep.enter c s) := by
  have hw := win_enter c s.lo s.hi hs.lo_le
  have hc := hs.count
  refine ⟨by simp only [Sweep.enter]; have := hs.lo_le; omega, by simp only [Sweep.enter]; omega, ?_, ?_⟩
  · simp only [Sweep.enter]
    rw [hw, hc]
    by_cases hk : c.keep (c.nbr s.hi) = true
    · by_cases hd : c.distinct = true
      · rw [hs.seen hd (c.nbr s.hi)]
        simp only [hk, hd, ↓reduceIte, ndistinct_snoc]
        by_cases hm : c.nbr s.hi ∈ win c s.lo s.hi
        · have := List.count_pos_iff.2 hm
          have : (win c s.lo s.hi).count (c.nbr s.hi) ≠ 0 := by omega
          simp [hm, this]
        · have : (win c s.lo s.hi).count (c.nbr s.hi) = 0 := List.count_eq_zero.2 hm
          simp [hm, this]
      · simp [hk, hd]
    · simp [hk]
  · intro hd y
    simp only [Sweep.enter]
    rw [hw]
    by_cases hk : c.keep (c.nbr s.hi) = true
    · simp only [hk, hd, and_self, ↓reduceIte, List.count_append, List.count_singleton, beq_iff_eq]
      rw [hs.seen hd]
      by_cases hy : y = c.nbr s.hi
      · subst hy; simp
      · simp [hy, Ne.symm hy]
    · simp [hk, hs.seen hd]

theorem CountInv.leave {s : State} (hs : CountInv c s) (hlt : s.lo < s.hi) :
    CountInv c (Sweep.leave c s) := by
  have hw := win_leave c s.lo s.hi hlt
  have hc := hs.count
  rw [hw] at hc
  refine ⟨by simp only [Sweep.leave]; omega, by simp only [Sweep.leave]; exact hs.hi_le, ?_, ?_⟩
  · simp only [Sweep.leave]
    by_cases hk : c.keep (c.nbr s.lo) = true
    · by_cases hd : c.distinct = true
      · have hseen := hs.seen hd (c.nbr s.lo)
        rw [hw] at hseen
        simp only [hk, ↓reduceIte, List.singleton_append, List.count_cons_self] at hc hseen
        simp only [hk, hd, ↓reduceIte, ndistinct] at hc ⊢
        rw [hseen]
        by_cases hm : c.nbr s.lo ∈ win c (s.lo + 1) s.hi
        · have := List.count_pos_iff.2 hm
          simp only [hm, ↓reduceIte] at hc
          rw [hc]
          have : (win c (s.lo + 1) s.hi).count (c.nbr s.lo) ≠ 0 := by omega
          simp [this]
        · have : (win c (s.lo + 1) s.hi).count (c.nbr s.lo) = 0 := List.count_eq_zero.2 hm
          simp only [hm, ↓reduceIte] at hc
          rw [hc]
          simp [this]
      · simp only [hk, hd, ↓reduceIte, List.singleton_append, List.length_cons] at hc ⊢
        simp only [Bool.false_eq_true, ↓reduceIte] at hc ⊢
        omega
    · simp only [hk, Bool.false_eq_true, ↓reduceIte, List.nil_append] at hc ⊢
      exact hc
  · intro hd y
    have hseen := hs.seen hd y
    rw [hw] at hseen
    simp only [Sweep.leave]
    by_cases hk : c.keep (c.nbr s.lo) = true
    · simp only [hk, hd, and_self, ↓reduceIte, List.singleton_append, List.count_cons, beq_iff_eq]
        at hseen ⊢
      by_cases hy : y = c.nbr s.lo
      · subst hy; simp at hseen ⊢; omega
      · simp [hy, Ne.symm hy] at hseen ⊢; exact hseen
    · simp only [hk, Bool.false_eq_true, ↓reduceIte, List.nil_append, false_and] at hseen ⊢
      exact hseen

/-! ### The loops -/

/-- Whether trigger j sees position k, in the loop's own terms: `k <= j` (stream) or
`ts[k] <= t_j` (timestamp). -/
def sees (j k : Nat) : Prop := if c.tsVis then c.ts k ≤ c.ts j else k ≤ j

section
variable (hsorted : ∀ i k, i ≤ k → k < c.b → c.ts i ≤ c.ts k)
include hsorted

/-- What trigger j sees is a prefix of the list. -/
theorem sees_prefix {j k k' : Nat} (hk : k < c.b) (hkk : k' ≤ k) (h : sees c j k) : sees c j k' := by
  unfold sees at *
  split at h
  · rename_i hv; simp only [hv, ↓reduceIte]; exact Int.le_trans (hsorted k' k hkk hk) h
  · rename_i hv; simp only [hv, Bool.false_eq_true, ↓reduceIte]; omega

theorem advHi_spec (j : Nat) : ∀ (s : State), CountInv c s → (∀ k < s.hi, sees c j k) →
    CountInv c (advHi c j s) ∧ (advHi c j s).lo = s.lo ∧
      (∀ k < c.b, k < (advHi c j s).hi ↔ sees c j k)
  | s, hs, hsee => by
    rw [advHi]
    split
    · rename_i h
      have h' := h
      simp only [enters, Bool.and_eq_true, decide_eq_true_eq] at h'
      have hsee' : ∀ k < (enter c s).hi, sees c j k := by
        intro k hk
        simp only [enter] at hk
        rcases Nat.lt_succ_iff_lt_or_eq.1 hk with hk | rfl
        · exact hsee k hk
        · unfold sees; split <;> simp_all
      have := advHi_spec j (enter c s) (hs.enter c h'.1) hsee'
      exact ⟨this.1, this.2.1.trans (by simp [enter]), this.2.2⟩
    · rename_i h
      refine ⟨hs, rfl, fun k hk => ⟨hsee k, fun hkj => ?_⟩⟩
      apply Classical.byContradiction
      intro hn
      have hle : s.hi ≤ k := Nat.le_of_not_lt hn
      have hhi : s.hi < c.b := Nat.lt_of_le_of_lt hle hk
      have := sees_prefix c hsorted hk hle hkj
      apply h
      simp only [enters, Bool.and_eq_true, decide_eq_true_eq, hhi, true_and]
      unfold sees at this; split at this <;> simp_all
termination_by s => c.b - s.hi
decreasing_by simp only [enter, enters, Bool.and_eq_true, decide_eq_true_eq] at *; omega

theorem advLo_spec (lts : Int) : ∀ (s : State), CountInv c s → (∀ k < s.lo, c.ts k < lts) →
    CountInv c (advLo c lts s) ∧ (advLo c lts s).hi = s.hi ∧
      (∀ k < (advLo c lts s).lo, c.ts k < lts) ∧
      (∀ k, (advLo c lts s).lo ≤ k → k < (advLo c lts s).hi → lts ≤ c.ts k)
  | s, hs, hold => by
    rw [advLo]
    split
    · rename_i h
      have hold' : ∀ k < (leave c s).lo, c.ts k < lts := by
        intro k hk
        simp only [leave] at hk
        rcases Nat.lt_succ_iff_lt_or_eq.1 hk with hk | rfl
        · exact hold k hk
        · exact h.2
      have := advLo_spec lts (leave c s) (hs.leave c h.1) hold'
      exact ⟨this.1, this.2.1.trans (by simp [leave]), this.2.2⟩
    · rename_i h
      refine ⟨hs, rfl, hold, fun k hlk hkh => ?_⟩
      have hlt : s.lo < s.hi := Nat.lt_of_le_of_lt hlk hkh
      have : lts ≤ c.ts s.lo := Int.le_of_not_gt fun hc => h ⟨hlt, hc⟩
      exact Int.le_trans this (hsorted s.lo k hlk (Nat.lt_of_lt_of_le hkh hs.hi_le))
termination_by s => s.hi - s.lo
decreasing_by simp only [leave]; omega

/-- After trigger j's turn: the window is [lo, hi) with every position below hi seen by j, every
position below lo older than lower(t_j), and every position in between not older. -/
structure PosInv (j : Nat) (s : State) : Prop where
  hi : ∀ k < c.b, k < s.hi ↔ sees c j k
  lo : ∀ k < s.lo, c.ts k < c.lower (c.ts j)
  mid : ∀ k, s.lo ≤ k → k < s.hi → c.lower (c.ts j) ≤ c.ts k

theorem after_inv (hlower : ∀ a a', a ≤ a' → c.lower a ≤ c.lower a') :
    ∀ j, j < c.b → CountInv c (after c (j + 1)) ∧ PosInv c j (after c (j + 1))
  | j, hj => by
    -- before trigger j: what is entered is seen by j, what has left is older than lower(t_j)
    have pre : CountInv c (after c j) ∧ (∀ k < (after c j).hi, sees c j k) ∧
        (∀ k < (after c j).lo, c.ts k < c.lower (c.ts j)) := by
      cases j with
      | zero => exact ⟨⟨Nat.le_refl _, Nat.zero_le _, by simp [win, ndistinct, after], by simp [win, after]⟩,
          by simp [after], by simp [after]⟩
      | succ i =>
        obtain ⟨hc, hp⟩ := after_inv hlower i (by omega)
        have hti : c.ts i ≤ c.ts (i + 1) := hsorted i (i + 1) (by omega) hj
        refine ⟨hc, fun k hk => ?_, fun k hk => Int.lt_of_lt_of_le (hp.lo k hk) (hlower _ _ hti)⟩
        have := (hp.hi k (Nat.lt_of_lt_of_le hk hc.hi_le)).1 hk
        unfold sees at this ⊢
        split at this
        · rename_i hv; simp only [hv, ↓reduceIte]; exact Int.le_trans this hti
        · rename_i hv; simp only [hv, Bool.false_eq_true, ↓reduceIte]; omega
    obtain ⟨hc, hsee, hold⟩ := pre
    obtain ⟨hc1, hlo1, hhi1⟩ := advHi_spec c hsorted j (after c j) hc hsee
    obtain ⟨hc2, hhi2, hlo2, hmid2⟩ :=
      advLo_spec c hsorted (c.lower (c.ts j)) (advHi c j (after c j)) hc1 (by rw [hlo1]; exact hold)
    refine ⟨hc2, ⟨fun k hk => ?_, hlo2, hmid2⟩⟩
    show k < (step c j (after c j)).hi ↔ _
    simp only [step]; rw [hhi2]; exact hhi1 k hk
end

/-! ### The theorem -/

theorem range_filter_eq {P : Nat → Bool} {lo hi b : Nat} (hlh : lo ≤ hi) (hhb : hi ≤ b)
    (hP : ∀ k < b, P k = true ↔ lo ≤ k ∧ k < hi) :
    (List.range b).filter P = List.range' lo (hi - lo) := by
  have split : List.range b = List.range' 0 lo ++ List.range' lo (hi - lo) ++ List.range' hi (b - hi) := by
    have e : b = lo + ((hi - lo) + (b - hi)) := by omega
    rw [List.range_eq_range']
    conv => lhs; rw [e]
    rw [← List.range'_append, ← List.range'_append]
    simp only [Nat.one_mul, Nat.zero_add, List.append_assoc]
    rw [show lo + (hi - lo) = hi by omega]
  rw [split, List.filter_append, List.filter_append]
  rw [List.filter_eq_nil_iff.2 (fun k hk => ?_), List.filter_eq_self.2 (fun k hk => ?_),
    List.filter_eq_nil_iff.2 (fun k hk => ?_)]
  · simp
  · rw [List.mem_range'] at hk; obtain ⟨i, hi, rfl⟩ := hk; rw [hP _ (by omega)]; omega
  · rw [List.mem_range'] at hk; obtain ⟨i, hi, rfl⟩ := hk; rw [hP _ (by omega)]; omega
  · rw [List.mem_range'] at hk; obtain ⟨i, hi, rfl⟩ := hk; rw [hP _ (by omega)]; omega

/-- **The sweep equals the per-trigger plan.** For every trigger j of the node's list, the count
that `sweep_count` emits is the per-trigger count over trigger j's range: the edge count, or the
distinct count, under stream or timestamp visibility. Contracts: the list is sorted by ts
(`list` and src/graph/index.cpp:20) and strictly increasing in eid (`list`), and lower is
monotone (it is t − δ + 1 or t − δ in every generated driver). -/
theorem sweep_correct (hsorted : ∀ i k, i ≤ k → k < c.b → c.ts i ≤ c.ts k)
    (heid : ∀ i k, i < k → k < c.b → c.eid i < c.eid k)
    (hlower : ∀ a a', a ≤ a' → c.lower a ≤ c.lower a') :
    ∀ j < c.b, (after c (j + 1)).count = spec c j := by
  intro j hj
  obtain ⟨hc, hp⟩ := after_inv c hsorted hlower j hj
  have hwin : (List.range c.b).filter (inWindow c j) =
      List.range' (after c (j + 1)).lo ((after c (j + 1)).hi - (after c (j + 1)).lo) := by
    apply range_filter_eq hc.lo_le hc.hi_le
    intro k hk
    have hhi := hp.hi k hk
    simp only [inWindow, Bool.and_eq_true, decide_eq_true_eq]
    unfold sees at hhi
    constructor
    · rintro ⟨hlow, hvis⟩
      refine ⟨Nat.le_of_not_lt fun hkl => ?_, ?_⟩
      · have := hp.lo k hkl; omega
      · rw [hhi]
        split at hvis
        · rename_i hv; simp only [hv, ↓reduceIte]; exact of_decide_eq_true hvis
        · rename_i hv; simp only [hv]
          have := of_decide_eq_true hvis
          exact Nat.le_of_not_lt fun hjk => by have := heid j k hjk hk; omega
    · rintro ⟨hlk, hkh⟩
      refine ⟨hp.mid k hlk hkh, ?_⟩
      have := hhi.1 hkh
      split
      · rename_i hv; simp only [hv, ↓reduceIte] at this; exact decide_eq_true this
      · rename_i hv; simp only [hv] at this
        apply decide_eq_true
        rcases Nat.lt_or_eq_of_le this with h | rfl
        · exact Nat.le_of_lt (heid k j h hj)
        · exact Nat.le_refl _
  rw [hc.count, spec, windowNbrs, hwin]
  rfl

/-! ### Clearing `seen` for the next node -/

/-- `seen` after `for (k = lo; k < hi; ++k) seen[l.nbr[k]] = 0;`. -/
def clear (s : State) : Nat → Nat :=
  (List.range' s.lo (s.hi - s.lo)).foldl (fun f k => fun z => if z = c.nbr k then 0 else f z) s.seen

theorem foldl_clear (ks : List Nat) (f : Nat → Nat) (z : Nat) :
    ks.foldl (fun f k => fun z => if z = c.nbr k then 0 else f z) f z =
      if z ∈ ks.map c.nbr then 0 else f z := by
  induction ks generalizing f with
  | nil => simp
  | cons k ks ih =>
    simp only [List.foldl_cons, ih, List.map_cons, List.mem_cons]
    by_cases h1 : z ∈ ks.map c.nbr
    · simp [h1]
    · by_cases h2 : z = c.nbr k <;> simp [h1, h2]

/-- **The clearing loop leaves `seen` all zero**, so the next node's pass starts as `after 0`
does: it walks the last window, the only positions whose neighbours `seen` counts. -/
theorem clear_zero {s : State} (hs : CountInv c s) (hd : c.distinct = true) : ∀ y, clear c s y = 0 := by
  intro y
  rw [clear, foldl_clear]
  split
  · rfl
  · rename_i h
    rw [hs.seen hd, List.count_eq_zero]
    intro hy
    exact h (List.mem_filter.1 hy).1

end Blz.Sweep

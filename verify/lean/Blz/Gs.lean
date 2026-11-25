import Blz.Scan

/-!
# gs: R7's sets through two distinct counts, a gate and a folded order test

`clean_gs_trigger` (tests/snapshots/clean_gs.hpp) and `gs_trigger` (tests/snapshots/gs.hpp), for
a trigger x → w at t. R7 (V2 S5, D-42): A = the distinct sources s ∉ {x, w} with s → x in the
window and ts ≤ t (ties included, R4; legacy ts < t); B = fan_out's set, the distinct sinks of the
window edges x → d; gs = min(|A|, |B|). The code takes three shortcuts:

- |A| and |B| are `count_distinct` over one range each (`countDistinct_eq`).
- A's order test ts ≤ t is the range's upper bound `vis_t` (clean) or `min_ts(t - 1, vis_t)`
  (legacy), and the clean code drops the s ≠ x and d ≠ x filters: the clean index has no
  self-loops (R1).
- It returns before the sinks stage when a = 0 and emits min(a, b) only when b ≠ 0; the count row
  starts at 0 and takes at most one emit (E403, `one_instance`).
-/
namespace Blz

/-- `kAnyEid` (include/blz/primitives.hpp): the eid bound of timestamp visibility. -/
def kAnyEid : Nat := 4294967295

/-- The row `clean_gs_trigger` leaves for a = |A|'s count and b = |B|'s count. -/
def gsRow (a b : Nat) : Nat := if a = 0 then 0 else if b = 0 then 0 else min a b

/-- The gate changes nothing: the row is min(a, b). -/
theorem gsRow_eq (a b : Nat) : gsRow a b = min a b := by
  unfold gsRow; split <;> (try split) <;> omega

/-- The neighbours of the entries of L that pass p. -/
def nbrsWhere (L : List Entry) (p : Entry → Prop) [DecidablePred p] : List Nat :=
  (L.filter fun e => decide (p e)).map (·.nbr)

theorem filter_range_keep {L : List Entry} {b hi : Int} {me : Nat} {keep : Nat → Bool}
    {p : Entry → Prop} [DecidablePred p]
    (h : ∀ e ∈ L, (keep e.nbr = true ∧ inRange b hi me e = true) ↔ p e) :
    ((L.filter (inRange b hi me)).map (·.nbr)).filter keep = nbrsWhere L p := by
  rw [List.filter_map, List.filter_filter, nbrsWhere]
  congr 1
  apply List.filter_congr
  intro e he
  rw [Bool.eq_iff_iff, decide_eq_true_iff, ← h e he]
  simp only [Function.comp_apply, Bool.and_eq_true]

/-- **R7, clean gs.** For a trigger x → w at t, with the window t − δ < t' ≤ t and A's order
test t' ≤ t written out: the row is 0 for a self-loop trigger (R1), else min(|A|, |B|).
Contracts: `range` and `distinct_first` (through `countDistinct_eq`), no self-loop entries in x's
lists (`list` under SelfLoops::drop), every eid at most kAnyEid. -/
theorem clean_gs_row (Lin Lout : List Entry) (x w : Nat) (δ t : Int)
    (hin : ∀ e ∈ Lin, e.nbr ≠ x ∧ e.eid ≤ kAnyEid) (hout : ∀ e ∈ Lout, e.nbr ≠ x ∧ e.eid ≤ kAnyEid) :
    (if x = w then 0 else
      gsRow (countDistinct (fun s => !(s == w)) (Lin.filter (inRange (t - δ + 1) t kAnyEid)))
        (countDistinct (fun _ => true) (Lout.filter (inRange (t - δ + 1) t kAnyEid)))) =
    (if x = w then 0 else
      min (ndistinct (nbrsWhere Lin fun e => (t - δ < e.ts ∧ e.ts ≤ t) ∧ e.ts ≤ t ∧ e.nbr ≠ x ∧ e.nbr ≠ w))
        (ndistinct (nbrsWhere Lout fun e => t - δ < e.ts ∧ e.ts ≤ t))) := by
  split
  · rfl
  rw [gsRow_eq, countDistinct_eq, countDistinct_eq]
  congr 2
  · refine filter_range_keep fun e he => ?_
    have := hin e he
    simp only [inRange, Bool.and_eq_true, decide_eq_true_eq, Bool.not_eq_true', beq_eq_false_iff_ne]
    omega
  · refine filter_range_keep fun e he => ?_
    have := hout e he
    simp only [inRange, Bool.and_eq_true, decide_eq_true_eq, true_and]
    omega

/-- **Legacy gs** (V2 S5, D-42; self-loops kept, so the code filters them). The theorem gives the
size `gs_trigger` emits, 0 meaning no emit, not the row: the legacy block is a histogram, whose sink
bins that size and drops one below the first bin (|A| = |B| = 1 adds nothing). The binning is L2's.
Visible means
ts ≤ vis_t and eid ≤ me (vis_t = t, me = kAnyEid under timestamp visibility; vis_t = kNoUpper,
me = the trigger's eid under stream visibility). A's order test is ts < t, the range's bound
min(t − 1, vis_t). Contracts: `range` and `distinct_first`. -/
theorem legacy_gs_row (Lin Lout : List Entry) (x w : Nat) (δ t vis_t : Int) (me : Nat) :
    (if x = w then 0 else
      gsRow (countDistinct (fun s => !(s == x || s == w)) (Lin.filter (inRange (t - δ + 1) (min (t - 1) vis_t) me)))
        (countDistinct (fun d => !(d == x)) (Lout.filter (inRange (t - δ + 1) vis_t me)))) =
    (if x = w then 0 else
      min (ndistinct (nbrsWhere Lin fun e =>
            (e.ts ≤ vis_t ∧ e.eid ≤ me) ∧ t - δ < e.ts ∧ e.ts < t ∧ e.nbr ≠ x ∧ e.nbr ≠ w))
        (ndistinct (nbrsWhere Lout fun e => (e.ts ≤ vis_t ∧ e.eid ≤ me) ∧ t - δ < e.ts ∧ e.nbr ≠ x))) := by
  split
  · rfl
  rw [gsRow_eq, countDistinct_eq, countDistinct_eq]
  congr 2
  · refine filter_range_keep fun e _ => ?_
    simp only [inRange, Bool.and_eq_true, decide_eq_true_eq, Bool.not_eq_true', Bool.or_eq_false_iff,
      beq_eq_false_iff_ne]
    omega
  · refine filter_range_keep fun e _ => ?_
    simp only [inRange, Bool.and_eq_true, decide_eq_true_eq, Bool.not_eq_true', beq_eq_false_iff_ne]
    omega

end Blz

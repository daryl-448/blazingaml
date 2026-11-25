import Blz.EarliestWitness
import Blz.LatestWitness

/-!
# sg_ord's order test: one earliest and one latest witness instead of every pair (R6b)

`clean_sg_ord_trigger` (tests/snapshots/clean_sg_ord.hpp) and `sg_ordered_trigger`
(tests/snapshots/sg_ordered.hpp) keep an intermediate y of the pair (s, w) when an edge s → y at
t₂ and an edge y → w at t₃ in the window have t₂ ≤ t₃ (clean, R6b) or t₂ < t₃ (legacy). The code
never compares pairs: it compares min(e2), the earliest s → y, with max(e3), the latest y → w.
It scans the side with the smaller window and probes the other, so the two values come from
different primitives in the two branches:

- scanning S.out: min(e2) is the scanned entry (distinct-first, so the earliest in range) and
  max(e3) is `pair_latest(W.in, y)`;
- scanning W.in: max(e3) is `pair_latest(W.in, y)` and min(e2) is `pair_exists(S.out, y)`.

In both, a y the scan does not visit has no edge on the scanned side and is not kept. So both
branches compute `orderTest o ea la` with ea meeting `IsEarliest` for the s → y list and la meeting
`IsLatest` for the y → w list, over the same bounds [lo, hi]; the theorem covers both.
-/
namespace Blz

/-- The code's test: kept when both witnesses exist and min(e2) R max(e3) (`sw <= pw` or
`pw <= sw`, `<` in the legacy code). -/
def orderTest (o : Order) : Option Int → Option Int → Bool
  | some a, some c => decide (o.rel a c)
  | _, _ => false

/-- R6b's condition for one intermediate y: an edge s → y at a and an edge y → w at c, both in
[lo, hi], with a R c. A is the s → y list, B the y → w list. -/
def OrderedPair (o : Order) (A B : List Int) (lo hi : Int) : Prop :=
  ∃ a ∈ A, ∃ c ∈ B, (lo ≤ a ∧ a ≤ hi) ∧ (lo ≤ c ∧ c ≤ hi) ∧ o.rel a c

/-- **The order test, both orders, both scan branches.** Comparing the earliest s → y with the
latest y → w decides whether some pair is in order. Contracts: `hA` (the scanned entry, or
pair_exists) and `hB` (pair_latest). -/
theorem order_test {o : Order} {A B : List Int} {lo hi : Int} {ea la : Option Int}
    (hA : IsEarliest A lo hi ea) (hB : IsLatest B lo hi la) :
    orderTest o ea la = true ↔ OrderedPair o A B lo hi := by
  constructor
  · intro h
    match ea, la, hA, hB, h with
    | some a, some c, ⟨ha, hla, hah, _⟩, ⟨hc, hlc, hch, _⟩, h =>
      exact ⟨a, ha, c, hc, ⟨hla, hah⟩, ⟨hlc, hch⟩, of_decide_eq_true h⟩
  · rintro ⟨a, ha, c, hc, ⟨hla, hah⟩, ⟨hlc, hch⟩, hac⟩
    match ea, la, hA, hB with
    | none, _, hA, _ => have := hA a ha hla; omega
    | some _, none, _, hB => have := hB c hc hlc; omega
    | some a', some c', ⟨_, _, _, hmin⟩, ⟨_, _, _, hmax⟩ =>
      have h1 := hmin a ha hla hah
      have h2 := hmax c hc hlc hch
      exact decide_eq_true (Order.rel_of_le_rel h1 (Order.rel_of_rel_le hac h2))

/-- **R6b, clean_sg_ord**: with the window t − δ < t' ≤ t (lo = t − δ + 1, hi = vis_t = t), the
test keeps y iff s → y at t₂ and y → w at t₃ exist in the window with t₂ ≤ t₃. -/
theorem clean_sg_ord_order_test {A B : List Int} {δ t : Int} {ea la : Option Int}
    (hA : IsEarliest A (t - δ + 1) t ea) (hB : IsLatest B (t - δ + 1) t la) :
    orderTest .le ea la = true ↔
      ∃ t₂ ∈ A, ∃ t₃ ∈ B, (t - δ < t₂ ∧ t₂ ≤ t) ∧ (t - δ < t₃ ∧ t₃ ≤ t) ∧ t₂ ≤ t₃ := by
  rw [order_test hA hB]
  constructor <;> rintro ⟨a, ha, c, hc, ⟨h1, h2⟩, ⟨h3, h4⟩, h5⟩ <;>
    exact ⟨a, ha, c, hc, ⟨by omega, h2⟩, ⟨by omega, h4⟩, h5⟩

/-- **Legacy sg_ordered**: with the closed window t − δ ≤ t' ≤ vis_t (vis_t is t, or kNoUpper
under stream visibility), the test keeps y iff such edges exist with t₂ < t₃. -/
theorem legacy_sg_ord_order_test {A B : List Int} {δ t vis_t : Int} {ea la : Option Int}
    (hA : IsEarliest A (t - δ) vis_t ea) (hB : IsLatest B (t - δ) vis_t la) :
    orderTest .lt ea la = true ↔
      ∃ t₂ ∈ A, ∃ t₃ ∈ B, (t - δ ≤ t₂ ∧ t₂ ≤ vis_t) ∧ (t - δ ≤ t₃ ∧ t₃ ≤ vis_t) ∧ t₂ < t₃ :=
  order_test hA hB

/-- Two s → y edges at 1 and 4 and one y → w edge at 3: some pair is in order (1 ≤ 3), so y is
kept, though the latest s → y (4) is after it. -/
example : orderTest .le (earliestIn [4, 1] 0 5) (latestIn [3] 0 5) = true := by decide

end Blz

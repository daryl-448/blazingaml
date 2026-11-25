import Blz.Hops

/-!
# Earliest witness: tcyc's forward greedy (R4, and the legacy Q5)

`clean_tcyc_trigger` (tests/snapshots/clean_tcyc.hpp) and `tcyc_trigger` (tests/snapshots/tcyc.hpp)
bind each hop of a node sequence once, at its earliest timestamp in range, instead of trying every
assignment. Hop 1 is a distinct-first scan of `range(V, lo, hi, vis_e)`; hop i + 1 scans from
`max_ts(lo, h_i)` (clean) or `max_ts(lo, h_i + 1)` (legacy); the closing hop into U is
`pair_exists` with the same bounds. Every hop's upper bound is `vis_t` = t (clean; the clean
profiles pin timestamp visibility, E403) or `min_ts(t - 1, vis_t)` (legacy).

The theorem: for any node sequence, an assignment that R4 (or Q5) accepts exists exactly when every
hop of this greedy finds a timestamp. It assumes only each hop's answer contract, `IsEarliest`;
the hop lists may be in any order.
-/
namespace Blz

/-- The lower bound the code puts on the hop after one at time x: x (clean, `max_ts(t - delta + 1,
ts_h)`) or x + 1 (legacy, `max_ts(t - delta, ts_h + 1)`). -/
def Order.next : Order → Int → Int
  | .le, x => x
  | .lt, x => x + 1

theorem Order.rel_iff_next {o : Order} {a b : Int} : o.rel a b ↔ o.next a ≤ b := by
  cases o <;> simp only [Order.rel, Order.next] <;> omega

/-- **Contract** of one hop's answer in tcyc, for hop list L, lower bound b and upper bound hi:
the earliest timestamp of L in [b, hi], or none when L has none there. It is the contract of
`pair_exists` (the closing hop), and of the binding that a distinct-first scan of
`range(c, b, hi, vis_e)` makes for its neighbour (every other hop; `Scan.lean` derives it from the
`range`, `distinct_first` and `list` contracts). Checked by `blz index-contracts`. -/
def IsEarliest (L : List Int) (b hi : Int) : Option Int → Prop
  | none => ∀ x ∈ L, b ≤ x → hi < x
  | some y => y ∈ L ∧ b ≤ y ∧ y ≤ hi ∧ ∀ x ∈ L, b ≤ x → x ≤ hi → y ≤ x

/-- The bindings tcyc's loop nest makes along one node sequence, hop by hop: the hop's answer
`first L b hi`, with b = lo for hop 1 and `max lo (o.next h)` after a hop at h, and `hi` for every
hop. `none` when a hop has no answer: the loop nest never reaches the sequence's emit. -/
def greedy (o : Order) (first : List Int → Int → Int → Option Int) (lo hi : Int) :
    Int → List (List Int) → Option (List Int)
  | _, [] => some []
  | b, L :: Ls =>
    match first L b hi with
    | none => none
    | some x => (greedy o first lo hi (max lo (o.next x)) Ls).map (x :: ·)

variable {o : Order}

theorem Witness.mono_lo {lo lo' t : Int} {hops : List (List Int)} {ts : List Int}
    (h : Witness o lo' t hops ts) (hl : lo ≤ lo') : Witness o lo t hops ts :=
  ⟨h.1, fun x hx => Int.le_trans hl (h.2.1 x hx), h.2.2⟩

theorem witness_nil {lo t : Int} : Witness o lo t [] [] := ⟨trivial, by simp, trivial⟩

/-- Add a first hop at x in front of a witness for the rest from `o.next x`. -/
theorem Witness.cons {b t x : Int} {L : List Int} {Ls : List (List Int)} {ts : List Int}
    (hx : x ∈ L) (hb : b ≤ x) (hxt : o.rel x t) (h : Witness o (o.next x) t Ls ts) :
    Witness o b t (L :: Ls) (x :: ts) := by
  refine ⟨⟨hx, h.1⟩, fun y hy => by simp at hy; omega, ?_⟩
  cases ts with
  | nil => exact hxt
  | cons y ys => exact ⟨Order.rel_iff_next.2 (h.2.1 y rfl), h.2.2⟩

/-- The first pick of a witness, and the witness for the rest from `o.next x`. -/
theorem Witness.uncons {b t x : Int} {L : List Int} {Ls : List (List Int)} {ts : List Int}
    (h : Witness o b t (L :: Ls) (x :: ts)) :
    x ∈ L ∧ b ≤ x ∧ o.rel x t ∧ Witness o (o.next x) t Ls ts := by
  obtain ⟨hp, hb, hc⟩ := h
  refine ⟨hp.1, hb x rfl, Chain.head_rel hc, hp.2, ?_, ?_⟩
  · cases ts with
    | nil => simp
    | cons y ys => intro z hz; simp at hz; subst hz; exact Order.rel_iff_next.1 hc.1
  · cases ts with
    | nil => trivial
    | cons y ys => exact hc.2

/-- Peel the first hop: a witness is a first pick x plus a witness for the rest from `o.next x`. -/
theorem witness_cons {b t : Int} {L : List Int} {Ls : List (List Int)} :
    (∃ ts, Witness o b t (L :: Ls) ts) ↔
      ∃ x ∈ L, b ≤ x ∧ o.rel x t ∧ ∃ ts, Witness o (o.next x) t Ls ts := by
  constructor
  · rintro ⟨_ | ⟨x, ts⟩, hw⟩
    · simp [Witness, Picks] at hw
    · obtain ⟨hx, hb, hxt, hrest⟩ := hw.uncons
      exact ⟨x, hx, hb, hxt, ts, hrest⟩
  · rintro ⟨x, hx, hb, hxt, ts, hw⟩
    exact ⟨x :: ts, Witness.cons hx hb hxt hw⟩

section
variable {first : List Int → Int → Int → Option Int}
  (hfirst : ∀ L b hi, IsEarliest L b hi (first L b hi)) {lo hi t : Int}
  (hhi : ∀ x, o.rel x t ↔ x ≤ hi)
include hfirst hhi

/-- The greedy's bindings are a witness. -/
theorem greedy_sound : ∀ {hops : List (List Int)} {b : Int} {ts : List Int}, lo ≤ b →
    greedy o first lo hi b hops = some ts → Witness o b t hops ts
  | [], _, ts, _, h => by simp [greedy] at h; subst h; exact witness_nil
  | L :: Ls, b, ts, hlo, h => by
    simp only [greedy] at h
    have hf := hfirst L b hi
    split at h
    · simp at h
    · rename_i x hx
      rw [hx] at hf
      obtain ⟨hxL, hbx, hxhi, -⟩ := hf
      obtain ⟨ts', hg, rfl⟩ := Option.map_eq_some_iff.1 h
      have hnext : max lo (o.next x) = o.next x := by
        have := Order.rel_iff_next (o := o) (a := x) (b := o.next x)
        cases o <;> simp only [Order.next] at * <;> omega
      rw [hnext] at hg
      have hw := greedy_sound (Int.le_trans hlo (Int.le_trans hbx (by cases o <;> simp only [Order.next] <;> omega))) hg
      exact Witness.cons hxL hbx ((hhi x).2 hxhi) hw

/-- Every hop of the greedy answers when a witness exists. -/
theorem greedy_complete : ∀ {hops : List (List Int)} {b : Int}, lo ≤ b →
    (∃ ts, Witness o b t hops ts) → (greedy o first lo hi b hops).isSome
  | [], _, _, _ => by simp [greedy]
  | L :: Ls, b, hlo, hw => by
    obtain ⟨x, hxL, hbx, hxt, ts, hrest⟩ := witness_cons.1 hw
    have hxhi := (hhi x).1 hxt
    simp only [greedy]
    have hf := hfirst L b hi
    split
    · rename_i hnone; rw [hnone] at hf; have := hf x hxL hbx; omega
    · rename_i y hy
      rw [hy] at hf
      obtain ⟨-, hby, -, hmin⟩ := hf
      have hyx := hmin x hxL hbx hxhi
      have hnext : max lo (o.next y) = o.next y := by
        cases o <;> simp only [Order.next] <;> omega
      rw [hnext, Option.isSome_map]
      refine greedy_complete (by cases o <;> simp only [Order.next] <;> omega) ⟨ts, hrest.mono_lo ?_⟩
      cases o <;> simp only [Order.next] <;> omega

/-- **Earliest witness, both orders.** Hop lists in any order, any number of hops, any bounds: an
assignment with lo ≤ t_1 and t_1 R … R t_k R t exists iff the greedy binds every hop, given that
each hop's answer meets `IsEarliest` and the hops' upper bound hi is the last time R allows
before t. -/
theorem earliest_witness (hops : List (List Int)) :
    (∃ ts, Witness o lo t hops ts) ↔ (greedy o first lo hi lo hops).isSome := by
  constructor
  · exact greedy_complete hfirst hhi (Int.le_refl lo)
  · intro h
    obtain ⟨ts, hts⟩ := Option.isSome_iff_exists.1 h
    exact ⟨ts, greedy_sound hfirst hhi (Int.le_refl lo) hts⟩
end

/-- **R4, clean_tcyc** (`clean_tcyc_trigger`, tests/snapshots/clean_tcyc.hpp). For a node sequence
whose k hops offer the timestamp lists `hops`: window edges at t − δ < t_1 ≤ t_2 ≤ … ≤ t_k ≤ t exist
iff the generated loop nest binds every hop, hop 1 from t − δ + 1 and each later one from
max(t − δ + 1, h), all at most t. Contract: `hfirst` (pair_exists, and range with distinct_first). -/
theorem clean_tcyc_earliest_witness {first : List Int → Int → Int → Option Int}
    (hfirst : ∀ L b hi, IsEarliest L b hi (first L b hi)) (δ t : Int) (hops : List (List Int)) :
    (∃ ts, Picks ts hops ∧ (∀ t₁ ∈ ts.head?, t - δ < t₁) ∧ Chain (· ≤ ·) ts t) ↔
      (greedy .le first (t - δ + 1) t (t - δ + 1) hops).isSome := by
  rw [← earliest_witness (o := .le) (lo := t - δ + 1) (hi := t) (t := t) hfirst (fun _ => Iff.rfl) hops]
  constructor <;> rintro ⟨ts, hp, hlo, hc⟩ <;>
    exact ⟨ts, hp, fun x hx => by have := hlo x hx; omega, hc⟩

/-- **Q5, legacy tcyc** (`tcyc_trigger`, tests/snapshots/tcyc.hpp). Window edges at
t − δ ≤ t_1 < t_2 < … < t_k < t exist iff the loop nest binds every hop, hop 1 from t − δ and each
later one from max(t − δ, h + 1), all at most min(t − 1, vis_t), where vis_t is t (timestamp
visibility) or kNoUpper (stream visibility). -/
theorem legacy_tcyc_earliest_witness {first : List Int → Int → Int → Option Int}
    (hfirst : ∀ L b hi, IsEarliest L b hi (first L b hi)) (δ t vis_t : Int) (hvis : t - 1 ≤ vis_t)
    (hops : List (List Int)) :
    (∃ ts, Picks ts hops ∧ (∀ t₁ ∈ ts.head?, t - δ ≤ t₁) ∧ Chain (· < ·) ts t) ↔
      (greedy .lt first (t - δ) (min (t - 1) vis_t) (t - δ) hops).isSome := by
  rw [← earliest_witness (o := .lt) (lo := t - δ) (hi := min (t - 1) vis_t) (t := t) hfirst
    (fun x => by show x < t ↔ _; omega) hops]
  rfl

/-! ### A reference answer that meets the contract, and two spot checks -/

/-- The earliest timestamp of L in [b, hi], by a plain scan: shows `IsEarliest` can be met. -/
def earliestIn : List Int → Int → Int → Option Int
  | [], _, _ => none
  | x :: xs, b, hi =>
    match earliestIn xs b hi with
    | none => if b ≤ x ∧ x ≤ hi then some x else none
    | some y => if b ≤ x ∧ x ≤ hi ∧ x < y then some x else some y

theorem earliestIn_spec : ∀ L b hi, IsEarliest L b hi (earliestIn L b hi)
  | [], _, _ => by simp [earliestIn, IsEarliest]
  | x :: xs, b, hi => by
    have ih := earliestIn_spec xs b hi
    unfold earliestIn
    split
    · rename_i h; rw [h] at ih
      split
      · rename_i hx
        refine ⟨List.mem_cons_self .., hx.1, hx.2, fun z hz hbz hzh => ?_⟩
        rcases List.mem_cons.1 hz with rfl | hz
        · exact Int.le_refl _
        · have := ih z hz hbz; omega
      · rename_i hx
        intro z hz hbz
        rcases List.mem_cons.1 hz with rfl | hz
        · omega
        · exact ih z hz hbz
    · rename_i y h; rw [h] at ih
      obtain ⟨hy, hby, hyh, hmin⟩ := ih
      split
      · rename_i hx
        refine ⟨List.mem_cons_self .., hx.1, hx.2.1, fun z hz hbz hzh => ?_⟩
        rcases List.mem_cons.1 hz with rfl | hz
        · exact Int.le_refl _
        · have := hmin z hz hbz hzh; omega
      · rename_i hx
        refine ⟨List.mem_cons_of_mem _ hy, hby, hyh, fun z hz hbz hzh => ?_⟩
        rcases List.mem_cons.1 hz with rfl | hz
        · omega
        · exact hmin z hz hbz hzh

/-- The contract fixes the answer: any two answers that meet it agree. So the theorems cover every
hop the code answers, by scan or by probe. -/
theorem IsEarliest.unique {L : List Int} {b hi : Int} :
    ∀ {r r' : Option Int}, IsEarliest L b hi r → IsEarliest L b hi r' → r = r'
  | none, none, _, _ => rfl
  | none, some _, h, ⟨hy, hb, hh, _⟩ => by have := h _ hy hb; omega
  | some _, none, ⟨hy, hb, hh, _⟩, h => by have := h _ hy hb; omega
  | some _, some _, ⟨hy, hb, hh, hm⟩, ⟨hy', hb', hh', hm'⟩ => by
    have := hm _ hy' hb' hh'; have := hm' _ hy hb hh; congr 1; omega

/-- A 3-cycle whose two hops share the trigger's minute t = 5, window (0, 5]: R4 counts it, and
the legacy strict order drops it (A19, Q5). -/
example : (greedy .le earliestIn 1 5 1 [[5], [5]]).isSome = true := by decide
example : (greedy .lt earliestIn 0 4 0 [[5], [5]]).isSome = false := by decide

end Blz

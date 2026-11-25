import Blz.Hops

/-!
# Latest witness: tchain's backward greedy (R4, and the legacy Q5)

`clean_tchain_trigger` (tests/snapshots/clean_tchain.hpp) and `tchain_trigger`
(tests/snapshots/tchain.hpp) walk a chain c_0 → … → c_k = U → V from the trigger backwards. Hop 1 is
the edge into U, from a distinct-first scan of `range(U.in, lo, cap, vis_e)` with cap = `vis_t` = t
(clean) or `min_ts(t - 1, vis_t)` (legacy); its time is `pair_latest` over the same bounds. Each
earlier hop takes the same from cap `min_ts(h, vis_t)` (clean) or `min_ts(h - 1, vis_t)` (legacy),
where h is the time just bound.

The theorem: an assignment R4 (or Q5) accepts exists exactly when every hop of this backward greedy
finds a timestamp. The spec is the same `Witness` as the forward greedy's, over the hops in chain
order; the code visits them reversed. It assumes only each hop's answer contract, `IsLatest`.
-/
namespace Blz

/-- The upper bound the code puts on the hop before one at time y: y (clean, `min_ts(ts_h,
vis_t)`) or y − 1 (legacy, `min_ts(ts_h - 1, vis_t)`). -/
def Order.prev : Order → Int → Int
  | .le, y => y
  | .lt, y => y - 1

theorem Order.rel_iff_prev {o : Order} {a b : Int} : o.rel a b ↔ a ≤ o.prev b := by
  cases o <;> simp only [Order.rel, Order.prev] <;> omega

/-- **Contract** of one hop's answer in tchain, for hop list L, lower bound lo and cap c: the
latest timestamp of L in [lo, c], or none when L has none there. The code's answer is none when the
scan of `range(·, lo, c, vis_e)` does not visit the neighbour, else `pair_latest` with the same
bounds (`Scan.lean`, `tchain_hop_isLatest`). Checked by `blz index-contracts` (`pair_latest`). -/
def IsLatest (L : List Int) (lo c : Int) : Option Int → Prop
  | none => ∀ x ∈ L, lo ≤ x → c < x
  | some y => y ∈ L ∧ lo ≤ y ∧ y ≤ c ∧ ∀ x ∈ L, lo ≤ x → x ≤ c → x ≤ y

/-- The bindings tchain's loop nest makes along one chain, hops listed from the trigger backwards:
each hop's answer `last L lo c`, with c the cap of hop 1 and `min (o.prev h) vis` after a hop
bound at h. `none` when a hop has no answer. -/
def latestGreedy (o : Order) (last : List Int → Int → Int → Option Int) (lo vis : Int) :
    Int → List (List Int) → Option (List Int)
  | _, [] => some []
  | c, L :: Ls =>
    match last L lo c with
    | none => none
    | some y => (latestGreedy o last lo vis (min (o.prev y) vis) Ls).map (y :: ·)

variable {o : Order}

/-- Add a last hop at y behind a witness that ends at y. -/
theorem Witness.snoc {lo y t : Int} {hops : List (List Int)} {L : List Int} {ts : List Int}
    (h : Witness o lo y hops ts) (hy : y ∈ L) (hlo : lo ≤ y) (hyt : o.rel y t) :
    Witness o lo t (hops ++ [L]) (ts ++ [y]) := by
  refine ⟨picks_snoc.2 ⟨h.1, hy⟩, fun x hx => ?_, chain_snoc.2 ⟨h.2.2, hyt⟩⟩
  cases ts with
  | nil => simp at hx; omega
  | cons a as => exact h.2.1 x (by simpa using hx)

/-- Peel the last hop: a witness is a last pick y plus a witness for the rest that ends at y. -/
theorem witness_snoc {lo t : Int} {hops : List (List Int)} {L : List Int} :
    (∃ ts, Witness o lo t (hops ++ [L]) ts) ↔
      ∃ y ∈ L, lo ≤ y ∧ o.rel y t ∧ ∃ ts, Witness o lo y hops ts := by
  constructor
  · rintro ⟨ts, hp, hlo, hc⟩
    obtain ⟨ts', y, rfl⟩ : ∃ ts' y, ts = ts' ++ [y] := by
      have hne : ts ≠ [] := fun h => by simpa [h] using Picks.length_eq hp
      exact ⟨ts.dropLast, ts.getLast hne, (List.dropLast_concat_getLast hne).symm⟩
    obtain ⟨hp', hy⟩ := picks_snoc.1 hp
    obtain ⟨hc', hyt⟩ := chain_snoc.1 hc
    refine ⟨y, hy, ?_, hyt, ts', hp', fun x hx => hlo x (by cases ts' <;> simp_all), hc'⟩
    cases ts' with
    | nil => exact hlo y (by simp)
    | cons a as =>
      exact Int.le_trans (hlo a (by simp)) (Chain.le_end hc' a (List.mem_cons_self ..))
  · rintro ⟨y, hy, hlo, hyt, ts, hw⟩
    exact ⟨ts ++ [y], hw.snoc hy hlo hyt⟩

section
variable {last : List Int → Int → Int → Option Int}
  (hlast : ∀ L lo c, IsLatest L lo c (last L lo c)) {lo vis : Int}
include hlast

/-- The backward greedy's bindings, reversed into chain order, are a witness. -/
theorem latestGreedy_sound : ∀ {hopsRev : List (List Int)} {c t : Int} {ts : List Int},
    c ≤ vis → (∀ x, o.rel x t ↔ x ≤ c) →
    latestGreedy o last lo vis c hopsRev = some ts → Witness o lo t hopsRev.reverse ts.reverse
  | [], _, _, _, _, _, h => by
    simp [latestGreedy] at h; subst h; exact ⟨trivial, by simp, trivial⟩
  | L :: Ls, c, t, ts, hcv, hc, h => by
    simp only [latestGreedy] at h
    have hl := hlast L lo c
    split at h
    · simp at h
    · rename_i y hy
      rw [hy] at hl
      obtain ⟨hyL, hloy, hyc, -⟩ := hl
      obtain ⟨ts', hg, rfl⟩ := Option.map_eq_some_iff.1 h
      have hprev : min (o.prev y) vis = o.prev y := by
        cases o <;> simp only [Order.prev] <;> omega
      rw [hprev] at hg
      have hw := latestGreedy_sound (by cases o <;> simp only [Order.prev] <;> omega)
        (fun x => Order.rel_iff_prev) hg
      simpa using hw.snoc hyL hloy ((hc y).2 hyc)

/-- Every hop of the backward greedy answers when a witness exists. -/
theorem latestGreedy_complete : ∀ {hopsRev : List (List Int)} {c t : Int},
    c ≤ vis → (∀ x, o.rel x t ↔ x ≤ c) →
    (∃ ts, Witness o lo t hopsRev.reverse ts) → (latestGreedy o last lo vis c hopsRev).isSome
  | [], _, _, _, _, _ => by simp [latestGreedy]
  | L :: Ls, c, t, hcv, hc, hw => by
    rw [List.reverse_cons] at hw
    obtain ⟨y', hy'L, hloy', hy't, ts, hrest⟩ := witness_snoc.1 hw
    have hy'c := (hc y').1 hy't
    simp only [latestGreedy]
    have hl := hlast L lo c
    split
    · rename_i hnone; rw [hnone] at hl; have := hl y' hy'L hloy'; omega
    · rename_i y hy
      rw [hy] at hl
      obtain ⟨-, -, hyc, hmax⟩ := hl
      have hy'y := hmax y' hy'L hloy' hy'c
      have hprev : min (o.prev y) vis = o.prev y := by
        cases o <;> simp only [Order.prev] <;> omega
      rw [hprev, Option.isSome_map]
      exact latestGreedy_complete (by cases o <;> simp only [Order.prev] <;> omega)
        (fun x => Order.rel_iff_prev) ⟨ts, hrest.1, hrest.2.1, Chain.mono_end hrest.2.2 hy'y⟩

/-- **Latest witness, both orders.** For hops in chain order (hop 1 leaves c_0, the last enters
U): an assignment with lo ≤ t_1 and t_1 R … R t_k R t exists iff the backward greedy, run over the
reversed hops from cap c, binds every hop, given that each hop's answer meets `IsLatest`, c is the
last time R allows before t, and c ≤ vis. -/
theorem latest_witness {t c : Int} (hc : ∀ x, o.rel x t ↔ x ≤ c) (hcv : c ≤ vis)
    (hops : List (List Int)) :
    (∃ ts, Witness o lo t hops ts) ↔ (latestGreedy o last lo vis c hops.reverse).isSome := by
  constructor
  · intro h
    exact latestGreedy_complete hlast hcv hc (by simpa using h)
  · intro h
    obtain ⟨ts, hts⟩ := Option.isSome_iff_exists.1 h
    exact ⟨ts.reverse, by simpa using latestGreedy_sound hlast hcv hc hts⟩
end

/-- **R4, clean_tchain** (`clean_tchain_trigger`, tests/snapshots/clean_tchain.hpp). For a chain
whose k hops offer the timestamp lists `hops` (in chain order): window edges at
t − δ < t_1 ≤ … ≤ t_k ≤ t exist iff the generated loop nest, from the trigger backwards, binds
every hop, hop 1 at most t and each earlier one at most min(h, t), all from t − δ + 1. Contract:
`hlast` (pair_latest, after a distinct-first scan of the same range). -/
theorem clean_tchain_latest_witness {last : List Int → Int → Int → Option Int}
    (hlast : ∀ L lo c, IsLatest L lo c (last L lo c)) (δ t : Int) (hops : List (List Int)) :
    (∃ ts, Picks ts hops ∧ (∀ t₁ ∈ ts.head?, t - δ < t₁) ∧ Chain (· ≤ ·) ts t) ↔
      (latestGreedy .le last (t - δ + 1) t t hops.reverse).isSome := by
  rw [← latest_witness (o := .le) (lo := t - δ + 1) (vis := t) (t := t) (c := t) hlast
    (fun _ => Iff.rfl) (Int.le_refl t) hops]
  constructor <;> rintro ⟨ts, hp, hlo, hc⟩ <;>
    exact ⟨ts, hp, fun x hx => by have := hlo x hx; omega, hc⟩

/-- **Q5, legacy tchain** (`tchain_trigger`, tests/snapshots/tchain.hpp). Window edges at
t − δ ≤ t_1 < … < t_k < t exist iff the loop nest binds every hop backwards, hop 1 at most
min(t − 1, vis_t) and each earlier one at most min(h − 1, vis_t), all from t − δ; vis_t is t
(timestamp visibility) or kNoUpper (stream visibility). -/
theorem legacy_tchain_latest_witness {last : List Int → Int → Int → Option Int}
    (hlast : ∀ L lo c, IsLatest L lo c (last L lo c)) (δ t vis_t : Int) (hvis : t - 1 ≤ vis_t)
    (hops : List (List Int)) :
    (∃ ts, Picks ts hops ∧ (∀ t₁ ∈ ts.head?, t - δ ≤ t₁) ∧ Chain (· < ·) ts t) ↔
      (latestGreedy .lt last (t - δ) vis_t (min (t - 1) vis_t) hops.reverse).isSome := by
  rw [← latest_witness (o := .lt) (lo := t - δ) (vis := vis_t) (t := t) (c := min (t - 1) vis_t)
    hlast (fun x => by show x < t ↔ _; omega) (by omega) hops]
  rfl

/-! ### A reference answer that meets the contract, and two spot checks -/

/-- The latest timestamp of L in [lo, c], by a plain scan: shows `IsLatest` can be met. -/
def latestIn : List Int → Int → Int → Option Int
  | [], _, _ => none
  | x :: xs, lo, c =>
    match latestIn xs lo c with
    | none => if lo ≤ x ∧ x ≤ c then some x else none
    | some y => if lo ≤ x ∧ x ≤ c ∧ y < x then some x else some y

theorem latestIn_spec : ∀ L lo c, IsLatest L lo c (latestIn L lo c)
  | [], _, _ => by simp [latestIn, IsLatest]
  | x :: xs, lo, c => by
    have ih := latestIn_spec xs lo c
    unfold latestIn
    split
    · rename_i h; rw [h] at ih
      split
      · rename_i hx
        refine ⟨List.mem_cons_self .., hx.1, hx.2, fun z hz hlz hzc => ?_⟩
        rcases List.mem_cons.1 hz with rfl | hz
        · exact Int.le_refl _
        · have := ih z hz hlz; omega
      · rename_i hx
        intro z hz hlz
        rcases List.mem_cons.1 hz with rfl | hz
        · omega
        · exact ih z hz hlz
    · rename_i y h; rw [h] at ih
      obtain ⟨hy, hly, hyc, hmax⟩ := ih
      split
      · rename_i hx
        refine ⟨List.mem_cons_self .., hx.1, hx.2.1, fun z hz hlz hzc => ?_⟩
        rcases List.mem_cons.1 hz with rfl | hz
        · exact Int.le_refl _
        · have := hmax z hz hlz hzc; omega
      · rename_i hx
        refine ⟨List.mem_cons_of_mem _ hy, hly, hyc, fun z hz hlz hzc => ?_⟩
        rcases List.mem_cons.1 hz with rfl | hz
        · omega
        · exact hmax z hz hlz hzc

/-- The contract fixes the answer: any two answers that meet it agree. -/
theorem IsLatest.unique {L : List Int} {lo c : Int} :
    ∀ {r r' : Option Int}, IsLatest L lo c r → IsLatest L lo c r' → r = r'
  | none, none, _, _ => rfl
  | none, some _, h, ⟨hy, hl, hc, _⟩ => by have := h _ hy hl; omega
  | some _, none, ⟨hy, hl, hc, _⟩, h => by have := h _ hy hl; omega
  | some _, some _, ⟨hy, hl, hc, hm⟩, ⟨hy', hl', hc', hm'⟩ => by
    have := hm _ hy' hl' hc'; have := hm' _ hy hl hc; congr 1; omega

/-- A 3-edge chain c_0 → c_1 → U → V whose hops are both in the trigger's minute t = 5: R4 counts
it, the legacy strict order does not. Hops are listed from the trigger backwards. -/
example : (latestGreedy .le latestIn 1 5 5 [[5], [5]]).isSome = true := by decide
example : (latestGreedy .lt latestIn 0 5 4 [[5], [5]]).isSome = false := by decide

end Blz

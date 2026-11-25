/-!
# Hops, the two hop orders, and one node sequence's witness

SPEC_V3.md §3.2 R2–R4 and the legacy order of §3.1 Q5. tcyc and tchain fix a node sequence
c_0 → c_1 → … → c_k and ask whether its hops have edges at times that respect the window and the
order. Hop i offers the timestamps of the visible edges c_{i−1} → c_i (its parallel edges) as a
list in any order, duplicates allowed. Nothing here assumes a sorted list.
-/
namespace Blz

/-- The two hop orders: `le` is R4's non-decreasing order (clean profiles), `lt` the strict order
that the legacy profiles keep (Q5). -/
inductive Order where
  | le
  | lt
deriving DecidableEq, Repr

/-- `o.rel a b`: a hop at time a may come before a hop, or the trigger, at time b. -/
def Order.rel : Order → Int → Int → Prop
  | .le, a, b => a ≤ b
  | .lt, a, b => a < b

instance (o : Order) (a b : Int) : Decidable (o.rel a b) := by
  cases o <;> unfold Order.rel <;> infer_instance

/-- `Picks ts hops`: ts takes one timestamp from each hop's list, in order. -/
def Picks : List Int → List (List Int) → Prop
  | [], [] => True
  | x :: ts, L :: Ls => x ∈ L ∧ Picks ts Ls
  | _, _ => False

/-- `Chain R [t_1, …, t_k] t`: t_1 R t_2 R … R t_k R t. -/
def Chain (R : Int → Int → Prop) : List Int → Int → Prop
  | [], _ => True
  | [x], t => R x t
  | x :: y :: ys, t => R x y ∧ Chain R (y :: ys) t

/-- One node sequence's witness: a time t_i from hop i's list for each hop, with lo ≤ t_1 and
t_1 R t_2 R … R t_k R t, where R is `o.rel` and t the trigger's time. The clean rule (R2–R4) is
o = le with lo = t − δ + 1, that is t − δ < t_1 over the integers; the legacy rule (Q5) is o = lt
with lo = t − δ. The theorems' clean and legacy forms restate it with R4's own symbols. -/
def Witness (o : Order) (lo t : Int) (hops : List (List Int)) (ts : List Int) : Prop :=
  Picks ts hops ∧ (∀ x ∈ ts.head?, lo ≤ x) ∧ Chain o.rel ts t

theorem Order.le_of_rel {o : Order} {a b : Int} (h : o.rel a b) : a ≤ b := by
  cases o <;> simp only [Order.rel] at h <;> omega

theorem Order.rel_of_rel_le {o : Order} {a b c : Int} (h : o.rel a b) (hbc : b ≤ c) : o.rel a c := by
  cases o <;> simp only [Order.rel] at * <;> omega

theorem Order.rel_of_le_rel {o : Order} {a b c : Int} (hab : a ≤ b) (h : o.rel b c) : o.rel a c := by
  cases o <;> simp only [Order.rel] at * <;> omega

/-- The first time of a chain is related to its end. -/
theorem Chain.head_rel {o : Order} : ∀ {x : Int} {ts : List Int} {t : Int},
    Chain o.rel (x :: ts) t → o.rel x t
  | _, [], _, h => h
  | _, _ :: _, _, h => Order.rel_of_le_rel (Order.le_of_rel h.1) (Chain.head_rel h.2)

/-- Every time of a chain is at most its end. -/
theorem Chain.le_end {o : Order} : ∀ {ts : List Int} {t : Int},
    Chain o.rel ts t → ∀ x ∈ ts, x ≤ t
  | [], _, _, _, hx => by simp at hx
  | [_], _, h, _, hx => by simp at hx; subst hx; exact Order.le_of_rel h
  | _ :: y :: ys, _, h, _, hx => by
    rcases List.mem_cons.1 hx with rfl | hx
    · exact Int.le_trans (Order.le_of_rel h.1) (Chain.le_end h.2 y (List.mem_cons_self ..))
    · exact Chain.le_end h.2 _ hx

/-- A chain to t is a chain to any later end. -/
theorem Chain.mono_end {o : Order} : ∀ {ts : List Int} {t t' : Int},
    Chain o.rel ts t → t ≤ t' → Chain o.rel ts t'
  | [], _, _, _, _ => trivial
  | [_], _, _, h, ht => Order.rel_of_rel_le h ht
  | _ :: _ :: _, _, _, h, ht => ⟨h.1, Chain.mono_end h.2 ht⟩

/-- Peel the last time: a chain to t ending in x is a chain to x followed by x R t. -/
theorem chain_snoc {R : Int → Int → Prop} : ∀ {ts : List Int} {x t : Int},
    Chain R (ts ++ [x]) t ↔ Chain R ts x ∧ R x t
  | [], _, _ => by simp [Chain]
  | [_], _, _ => by simp [Chain]
  | a :: b :: rest, x, t => by
    show (R a b ∧ Chain R (b :: rest ++ [x]) t) ↔ (R a b ∧ Chain R (b :: rest) x) ∧ R x t
    rw [chain_snoc (ts := b :: rest)]
    exact ⟨fun ⟨h1, h2, h3⟩ => ⟨⟨h1, h2⟩, h3⟩, fun ⟨⟨h1, h2⟩, h3⟩ => ⟨h1, h2, h3⟩⟩

theorem Picks.length_eq : ∀ {ts : List Int} {hops : List (List Int)},
    Picks ts hops → ts.length = hops.length
  | [], [], _ => rfl
  | _ :: _, _ :: _, h => by simp [Picks.length_eq h.2]

/-- Peel the last hop's pick. -/
theorem picks_snoc : ∀ {ts : List Int} {hops : List (List Int)} {x : Int} {L : List Int},
    Picks (ts ++ [x]) (hops ++ [L]) ↔ Picks ts hops ∧ x ∈ L
  | [], [], _, _ => by simp [Picks]
  | [], _ :: hs, _, _ => by cases hs <;> simp [Picks]
  | _ :: _, [], _, _ => by simp [Picks]
  | _ :: ts, _ :: hs, _, _ => by
    simp only [List.cons_append, Picks]
    rw [picks_snoc]
    exact ⟨fun ⟨a, b, c⟩ => ⟨⟨a, b⟩, c⟩, fun ⟨⟨a, b⟩, c⟩ => ⟨a, b, c⟩⟩

end Blz

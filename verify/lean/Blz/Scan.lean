import Blz.EarliestWitness
import Blz.LatestWitness

/-!
# Distinct-first scans: each neighbour once, at its earliest timestamp

Every pattern stage that walks a T-CSR range with `if (!distinct_first(j, r.lo)) continue;`
(include/blz/primitives.hpp; tcyc's and tchain's hops, sg_ord's sources and intermediates, gs's
counts, fan's `count_distinct`) relies on three contracts of the index, each checked by
`blz index-contracts`:

- `range`: the positions of `range(x, b, hi, me)` hold exactly x's entries with b ≤ ts ≤ hi and
  eid ≤ me, in list order. Here: the scanned entries are `L.filter (inRange b hi me)`.
- `distinct_first`: distinct_first(j, r.lo) holds iff no entry of the range before j has j's
  neighbour. Here: `firsts`, which visits exactly those entries.
- `list`: x's entries are its edges in eid order. With the stream sorted by ts (`GraphIndex`
  throws otherwise, src/graph/index.cpp:20), the list is sorted by ts. Only the earliest-timestamp
  lemma uses this.

From them: the scan visits each neighbour of the range once (`firsts_nodup`, `mem_nbr_firsts`),
at its earliest timestamp in range (`firsts_earliest`), so `count_distinct` is the number of
distinct neighbours (`countDistinct_eq`), a hop's binding meets `IsEarliest`
(`scanBinding_isEarliest`) and tchain's hop answer meets `IsLatest` (`tchain_hop_isLatest`).
-/
namespace Blz

/-- One entry of a node's T-CSR list (`TCsrView`): its neighbour, timestamp and eid. -/
structure Entry where
  nbr : Nat
  ts : Int
  eid : Nat
deriving DecidableEq, Repr

/-- Whether an entry lies in `range(x, b, hi, me)`: b ≤ ts ≤ hi and eid ≤ me. -/
def inRange (b hi : Int) (me : Nat) (e : Entry) : Bool :=
  decide (b ≤ e.ts) && decide (e.ts ≤ hi) && decide (e.eid ≤ me)

/-- The entries a distinct-first scan visits, distinct_first read as its contract: those whose
neighbour no earlier entry of the range has. `pre` holds the earlier entries' neighbours. -/
def firstsFrom : List Nat → List Entry → List Entry
  | _, [] => []
  | pre, e :: es =>
    if e.nbr ∈ pre then firstsFrom (e.nbr :: pre) es else e :: firstsFrom (e.nbr :: pre) es

/-- The entries a distinct-first scan of the range R visits. -/
def firsts (R : List Entry) : List Entry := firstsFrom [] R

/-- `count_distinct(l, r, keep)`: the visited entries whose neighbour passes keep. -/
def countDistinct (keep : Nat → Bool) (R : List Entry) : Nat :=
  (firsts R).countP fun e => keep e.nbr

/-- The number of distinct values of l, |{y | y ∈ l}|. -/
def ndistinct : List Nat → Nat
  | [] => 0
  | y :: ys => if y ∈ ys then ndistinct ys else ndistinct ys + 1

theorem mem_nbr_firstsFrom : ∀ {pre : List Nat} {R : List Entry} {y : Nat},
    y ∈ (firstsFrom pre R).map (·.nbr) ↔ y ∈ R.map (·.nbr) ∧ y ∉ pre
  | _, [], _ => by simp [firstsFrom]
  | pre, e :: es, y => by
    unfold firstsFrom
    by_cases h : e.nbr ∈ pre
    · rw [ite_eq_left h, mem_nbr_firstsFrom]
      simp only [List.map_cons, List.mem_cons, not_or]
      constructor
      · rintro ⟨h1, -, h3⟩; exact ⟨Or.inr h1, h3⟩
      · rintro ⟨h1 | h1, h3⟩
        · subst h1; exact absurd h h3
        · exact ⟨h1, fun h4 => h3 (h4 ▸ h), h3⟩
    · rw [ite_eq_right h]
      simp only [List.map_cons, List.mem_cons]
      rw [mem_nbr_firstsFrom]
      simp only [List.mem_cons, not_or]
      constructor
      · rintro (h1 | ⟨h1, -, h3⟩)
        · subst h1; exact ⟨Or.inl rfl, h⟩
        · exact ⟨Or.inr h1, h3⟩
      · rintro ⟨h1 | h1, h3⟩
        · exact Or.inl h1
        · by_cases h4 : y = e.nbr
          · exact Or.inl h4
          · exact Or.inr ⟨h1, h4, h3⟩

theorem mem_firstsFrom : ∀ {pre : List Nat} {R : List Entry} {e : Entry},
    e ∈ firstsFrom pre R → e ∈ R
  | _, [], _, h => by simp [firstsFrom] at h
  | pre, a :: as, e, h => by
    unfold firstsFrom at h
    split at h
    · exact List.mem_cons_of_mem _ (mem_firstsFrom h)
    · rcases List.mem_cons.1 h with rfl | h
      · exact List.mem_cons_self ..
      · exact List.mem_cons_of_mem _ (mem_firstsFrom h)

theorem nbr_not_mem_pre {pre : List Nat} {R : List Entry} {e : Entry}
    (h : e ∈ firstsFrom pre R) : e.nbr ∉ pre :=
  (mem_nbr_firstsFrom.1 (List.mem_map_of_mem h)).2

theorem firstsFrom_nodup : ∀ {pre : List Nat} {R : List Entry},
    ((firstsFrom pre R).map (·.nbr)).Nodup
  | _, [] => by simp [firstsFrom]
  | pre, e :: es => by
    unfold firstsFrom
    split
    · exact firstsFrom_nodup
    · rw [List.map_cons, List.nodup_cons]
      exact ⟨fun h => (mem_nbr_firstsFrom.1 h).2 (List.mem_cons_self ..), firstsFrom_nodup⟩

/-- On a list sorted by ts, a visited entry has the earliest timestamp of its neighbour's entries. -/
theorem firstsFrom_earliest : ∀ {pre : List Nat} {R : List Entry} {e : Entry},
    R.Pairwise (fun a c => a.ts ≤ c.ts) → e ∈ firstsFrom pre R →
    ∀ e' ∈ R, e'.nbr = e.nbr → e.ts ≤ e'.ts
  | _, [], _, _, h, _, _, _ => by simp [firstsFrom] at h
  | pre, a :: as, e, hs, h, e', he', hn => by
    rw [List.pairwise_cons] at hs
    unfold firstsFrom at h
    have rest : e ∈ firstsFrom (a.nbr :: pre) as → e.ts ≤ e'.ts := fun h => by
      have hne := nbr_not_mem_pre h
      rcases List.mem_cons.1 he' with rfl | he'
      · exact absurd (hn ▸ List.mem_cons_self ..) hne
      · exact firstsFrom_earliest hs.2 h e' he' hn
    split at h
    · exact rest h
    · rcases List.mem_cons.1 h with rfl | h
      · rcases List.mem_cons.1 he' with rfl | he'
        · exact Int.le_refl _
        · exact hs.1 e' he'
      · exact rest h

theorem mem_nbr_firsts {R : List Entry} {y : Nat} :
    y ∈ (firsts R).map (·.nbr) ↔ y ∈ R.map (·.nbr) := by
  simp [firsts, mem_nbr_firstsFrom]

theorem firsts_nodup {R : List Entry} : ((firsts R).map (·.nbr)).Nodup := firstsFrom_nodup

theorem firsts_earliest {R : List Entry} {e : Entry} (hs : R.Pairwise (fun a c => a.ts ≤ c.ts))
    (h : e ∈ firsts R) : ∀ e' ∈ R, e'.nbr = e.nbr → e.ts ≤ e'.ts :=
  firstsFrom_earliest hs h

/-! ### Counting distinct values -/

/-- l with each value kept at its last occurrence. -/
def dedupLast : List Nat → List Nat
  | [] => []
  | y :: ys => if y ∈ ys then dedupLast ys else y :: dedupLast ys

theorem mem_dedupLast : ∀ {l : List Nat} {y : Nat}, y ∈ dedupLast l ↔ y ∈ l
  | [], _ => by simp [dedupLast]
  | a :: as, y => by
    unfold dedupLast
    split
    · rename_i h
      rw [mem_dedupLast, List.mem_cons]
      exact ⟨Or.inr, fun h' => h'.elim (fun e => e ▸ h) id⟩
    · rw [List.mem_cons, mem_dedupLast, List.mem_cons]

theorem dedupLast_nodup : ∀ {l : List Nat}, (dedupLast l).Nodup
  | [] => by simp [dedupLast]
  | a :: as => by
    unfold dedupLast
    split
    · exact dedupLast_nodup
    · rename_i h; exact List.nodup_cons.2 ⟨fun h' => h (mem_dedupLast.1 h'), dedupLast_nodup⟩

theorem length_dedupLast : ∀ {l : List Nat}, (dedupLast l).length = ndistinct l
  | [] => rfl
  | a :: as => by
    unfold dedupLast ndistinct
    split <;> simp [length_dedupLast]

/-- ndistinct l is the length of any duplicate-free list with l's members. -/
theorem ndistinct_eq_length {N l : List Nat} (hN : N.Nodup) (hm : ∀ y, y ∈ N ↔ y ∈ l) :
    ndistinct l = N.length := by
  rw [← length_dedupLast]
  exact ((List.perm_ext_iff_of_nodup dedupLast_nodup hN).2
    fun y => by rw [mem_dedupLast, hm]).length_eq

/-- **count_distinct counts distinct neighbours**: over a range R, the number of distinct
neighbours that pass keep. Contracts: `distinct_first` (the scan is `firsts`). -/
theorem countDistinct_eq (keep : Nat → Bool) (R : List Entry) :
    countDistinct keep R = ndistinct ((R.map (·.nbr)).filter keep) := by
  have hlen : countDistinct keep R = (((firsts R).map (·.nbr)).filter keep).length := by
    rw [countDistinct, List.countP_eq_length_filter, List.filter_map, List.length_map]
    rfl
  rw [hlen, ndistinct_eq_length (firsts_nodup.sublist List.filter_sublist)]
  intro y
  simp only [List.mem_filter, mem_nbr_firsts]

/-! ### A hop's binding -/

/-- Hop x → y's timestamp list: the timestamps of x's visible entries (eid ≤ me) with neighbour y,
in list order. -/
def hopList (L : List Entry) (me y : Nat) : List Int :=
  (L.filter fun e => e.nbr == y && decide (e.eid ≤ me)).map (·.ts)

/-- The binding a distinct-first scan of `range(x, b, hi, me)` makes for neighbour y: the
timestamp of the visited entry with neighbour y, or none when the scan visits none. -/
def scanBinding (L : List Entry) (b hi : Int) (me y : Nat) : Option Int :=
  ((firsts (L.filter (inRange b hi me))).find? (·.nbr == y)).map (·.ts)

theorem mem_hopList {L : List Entry} {me y : Nat} {x : Int} :
    x ∈ hopList L me y ↔ ∃ e ∈ L, e.nbr = y ∧ e.eid ≤ me ∧ e.ts = x := by
  simp [hopList, and_assoc]

/-- The scan visits neighbour y iff hop x → y has a timestamp in [b, hi]. No sortedness needed. -/
theorem scanBinding_isSome {L : List Entry} {b hi : Int} {me y : Nat} :
    (scanBinding L b hi me y).isSome ↔ ∃ x ∈ hopList L me y, b ≤ x ∧ x ≤ hi := by
  simp only [scanBinding, Option.isSome_map]
  constructor
  · intro h
    obtain ⟨e, he⟩ := Option.isSome_iff_exists.1 h
    have heR := mem_firsts (List.mem_of_find?_eq_some he)
    have hny := List.find?_some he
    simp only [List.mem_filter, inRange, Bool.and_eq_true, decide_eq_true_eq] at heR
    exact ⟨e.ts, mem_hopList.2 ⟨e, heR.1, by simpa using hny, heR.2.2, rfl⟩, heR.2.1.1, heR.2.1.2⟩
  · rintro ⟨x, hx, hbx, hxh⟩
    obtain ⟨e, heL, hny, hme, rfl⟩ := mem_hopList.1 hx
    have hR : e ∈ L.filter (inRange b hi me) := by
      simp [List.mem_filter, inRange, heL, hbx, hxh, hme]
    obtain ⟨e', he', hn'⟩ := List.mem_map.1 (mem_nbr_firsts.2 (List.mem_map_of_mem (f := (·.nbr)) hR))
    cases hf : (firsts (L.filter (inRange b hi me))).find? (·.nbr == y) with
    | some _ => rfl
    | none =>
      have := List.find?_eq_none.1 hf e' he'
      simp [hn', hny] at this
where
  mem_firsts {R : List Entry} {e : Entry} (h : e ∈ firsts R) : e ∈ R := mem_firstsFrom h

/-- **A distinct-first scan binds each neighbour at its earliest timestamp in range**: the
binding meets `IsEarliest` for the hop's list, which tcyc's greedy assumes. Contracts: `range`,
`distinct_first` (the model), and `hs`, x's list sorted by ts (`list` and index.cpp:20). -/
theorem scanBinding_isEarliest {L : List Entry} (hs : L.Pairwise (fun a c => a.ts ≤ c.ts))
    (b hi : Int) (me y : Nat) : IsEarliest (hopList L me y) b hi (scanBinding L b hi me y) := by
  cases h : scanBinding L b hi me y with
  | none =>
    intro x hx hbx
    have : ¬ (scanBinding L b hi me y).isSome := by simp [h]
    rw [scanBinding_isSome] at this
    exact Int.lt_of_not_ge fun hxh => this ⟨x, hx, hbx, hxh⟩
  | some x =>
    simp only [scanBinding, Option.map_eq_some_iff] at h
    obtain ⟨e, hf, rfl⟩ := h
    have hef := List.mem_of_find?_eq_some hf
    have hny : e.nbr = y := by simpa using List.find?_some hf
    have heR := mem_firstsFrom hef
    have hsR := hs.sublist (List.filter_sublist (p := inRange b hi me))
    simp only [List.mem_filter, inRange, Bool.and_eq_true, decide_eq_true_eq] at heR
    refine ⟨mem_hopList.2 ⟨e, heR.1, hny, heR.2.2, rfl⟩, heR.2.1.1, heR.2.1.2, ?_⟩
    intro z hz hbz hzh
    obtain ⟨e', he'L, hny', hme', rfl⟩ := mem_hopList.1 hz
    refine firsts_earliest hsR hef e' ?_ (hny'.trans hny.symm)
    simp [List.mem_filter, inRange, he'L, hbz, hzh, hme']

/-- tchain's answer for one hop: none when the scan of `range(x, lo, c, me)` does not visit y,
else pair_latest's answer `pl` over the same bounds. -/
def tchainAnswer (L : List Entry) (lo c : Int) (me y : Nat) (pl : Option Int) : Option Int :=
  if (scanBinding L lo c me y).isSome then pl else none

/-- **tchain's hop answer meets `IsLatest`**, which tchain's backward greedy assumes. The code
never tests pair_latest's answer for kNoTs: a visited neighbour has an entry in range, so the
contract makes it a timestamp. Contracts: `range`, `distinct_first`, and `hpl`, pair_latest's. -/
theorem tchain_hop_isLatest {L : List Entry} {lo c : Int} {me y : Nat} {pl : Option Int}
    (hpl : IsLatest (hopList L me y) lo c pl) :
    IsLatest (hopList L me y) lo c (tchainAnswer L lo c me y pl) := by
  unfold tchainAnswer
  split
  · exact hpl
  · rename_i h
    rw [scanBinding_isSome] at h
    exact fun x hx hlx => Int.lt_of_not_ge fun hxc => h ⟨x, hx, hlx, hxc⟩

/-- So every hop the tcyc loop nest binds by a scan is `earliestIn` of its hop list, the answer
the greedy of `EarliestWitness.lean` takes. -/
theorem scanBinding_eq_earliestIn {L : List Entry} (hs : L.Pairwise (fun a c => a.ts ≤ c.ts))
    (b hi : Int) (me y : Nat) : scanBinding L b hi me y = earliestIn (hopList L me y) b hi :=
  (scanBinding_isEarliest hs b hi me y).unique (earliestIn_spec _ _ _)

/-- And every hop tchain binds is `latestIn` of its hop list. -/
theorem tchainAnswer_eq_latestIn {L : List Entry} {lo c : Int} {me y : Nat} {pl : Option Int}
    (hpl : IsLatest (hopList L me y) lo c pl) :
    tchainAnswer L lo c me y pl = latestIn (hopList L me y) lo c :=
  (tchain_hop_isLatest hpl).unique (latestIn_spec _ _ _)

end Blz

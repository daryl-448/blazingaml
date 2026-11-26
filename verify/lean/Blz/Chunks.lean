/-!
# Range mining: a selection mined in chunks gives the whole selection's rows (SPEC v3 H7)

`ChunkDriver` (src/runtime/chunk_driver.cpp, D-77) mines each block of a run for a selection of
n rows, `chunk_rows` triggers at a time, and puts the rows on the block's stream in eid order. H7's
obligation is the range-split identity: rows depend only on the trigger, so the chunks' rows,
concatenated, are the whole selection's. This module proves it for the driver's loops as written:

- `chunksFrom` is the loop `for (a = start; a < n; a += k) { b = a + min(k, n - a); … }`, which
  `run_block`, `run_fused` and `run_vstats` use. Its chunks cover 0 … n − 1 once each, in order,
  the last partial chunk included (`chunks_cover`), each a valid argument of `slice` (`mem_chunks`),
  and there are ⌈n / k⌉ of them, the stream's `chunks` tally (`length_chunks`).
- `slice` is `slice(sel, a, b)` (src/runtime/triggers.cpp): a range's sub-range, or a sample's eids
  a … b − 1 (`slice_eids`).
- `runChunked` is `run_fused` (each slot's stream) and `run_vstats`: one backend call per chunk.
  `runBlock` is `run_block`, whose part size is `chunk_rows`, or the whole selection for a block
  with a CPU sweep plan (D-79), whose one buffer is then put on the stream in chunks.
  Concatenated, their puts are the whole selection's cells (`runChunked_flatten`,
  `runBlock_flatten`).
- `BlockStream::put` continues FNV-1a from the stream's last value, so the stream's hash is the
  hash of the concatenation, the whole block's `output_fnv1a` (`stream_hash`, `runBlock_hash`).

The backend's answer is a function with its contract as a hypothesis (D-63): `Local mine row` says
the cells it writes for a part are its triggers' rows in row order, `row e` for the trigger e,
whatever the part. That is H7's premise. L2's chunked runs check it on the real code (COVERAGE.md).
-/
namespace Blz.Chunks

/-! ### The chunk loop -/

/-- The chunks [a, b) of `for (a = start; a < n; a += k) { b = a + min(k, n - a); … }`, from
`start` = a. The C++ never runs it with k = 0: the driver's constructor rejects `chunk_rows == 0`,
and a sweep block's part size is max(n, 1). -/
def chunksFrom (k n a : Nat) : List (Nat × Nat) :=
  if _h : a < n ∧ 0 < k then (a, a + min k (n - a)) :: chunksFrom k n (a + k) else []
termination_by n - a

/-- The chunks of rows 0 … n − 1. -/
def chunks (k n : Nat) : List (Nat × Nat) := chunksFrom k n 0

theorem chunksFrom_of_ge {k n a : Nat} (h : n ≤ a) : chunksFrom k n a = [] := by
  rw [chunksFrom]; split
  · omega
  · rfl

/-- Each chunk is a non-empty part of [start, n) of at most k rows: `slice(sel, a, b)`'s
precondition a < b ≤ rows, so the driver never makes it throw. -/
theorem mem_chunksFrom {k n : Nat} : ∀ {a : Nat} {p : Nat × Nat}, p ∈ chunksFrom k n a →
    a ≤ p.1 ∧ p.1 < p.2 ∧ p.2 ≤ n ∧ p.2 - p.1 ≤ k := by
  intro a p hp
  rw [chunksFrom] at hp
  split at hp
  · next h =>
    rcases List.mem_cons.1 hp with rfl | hp
    · simp only; omega
    · have := mem_chunksFrom hp
      omega
  · simp at hp
termination_by a => n - a

theorem mem_chunks {k n : Nat} {p : Nat × Nat} (hp : p ∈ chunks k n) :
    p.1 < p.2 ∧ p.2 ≤ n ∧ p.2 - p.1 ≤ k :=
  (mem_chunksFrom hp).2

/-- The telescope: when F joins adjacent parts (F a b ++ F b c = F a c), the chunks' F, in loop
order, give F start n. -/
theorem chunksFrom_flatMap {β : Type} (F : Nat → Nat → List β)
    (hF : ∀ a b c, a ≤ b → b ≤ c → F a b ++ F b c = F a c) {k n : Nat} (hk : 0 < k) :
    ∀ a, a ≤ n → (chunksFrom k n a).flatMap (fun p => F p.1 p.2) = F a n := by
  intro a ha
  rw [chunksFrom]
  split
  · next h =>
    rw [List.flatMap_cons]
    by_cases hn : a + k < n
    · rw [chunksFrom_flatMap F hF hk (a + k) (by omega), show a + min k (n - a) = a + k by omega]
      exact hF a (a + k) n (by omega) (by omega)
    · rw [chunksFrom_of_ge (by omega), show a + min k (n - a) = n by omega]
      simp
  · next h =>
    have hn : a = n := by omega
    subst hn
    have h2 := congrArg List.length (hF a a a (Nat.le_refl a) (Nat.le_refl a))
    rw [List.length_append] at h2
    simp only [List.flatMap_nil]
    exact (List.length_eq_zero_iff.1 (by omega)).symm
termination_by a => n - a

/-- **Cover.** The chunks of 0 … n − 1, concatenated, are the rows 0 … n − 1: each row once, in
order, with no gap and no overlap, and the last partial chunk kept. -/
theorem chunks_cover {k n : Nat} (hk : 0 < k) :
    (chunks k n).flatMap (fun p => List.range' p.1 (p.2 - p.1)) = List.range n := by
  rw [chunks, chunksFrom_flatMap (fun a b => List.range' a (b - a)) ?_ hk 0 (Nat.zero_le n),
    List.range_eq_range']
  · rfl
  · intro a b c hab hbc
    have := @List.range'_append a (b - a) (c - b) 1
    rw [show a + 1 * (b - a) = b by omega, show b - a + (c - b) = c - a by omega] at this
    exact this

theorem length_chunksFrom {k n : Nat} (hk : 0 < k) :
    ∀ a, (chunksFrom k n a).length = (n - a + k - 1) / k := by
  intro a
  rw [chunksFrom]
  split
  · next h =>
    rw [List.length_cons, length_chunksFrom hk (a + k)]
    by_cases hn : a + k < n
    · rw [show n - a + k - 1 = (n - (a + k) + k - 1) + k by omega, Nat.add_div_right _ hk]
    · rw [show n - (a + k) + k - 1 = k - 1 by omega, Nat.div_eq_of_lt (by omega)]
      exact (Nat.div_eq_of_lt_le (by omega) (by omega)).symm
  · next h =>
    rw [List.length_nil, show n - a = 0 by omega, Nat.zero_add, Nat.div_eq_of_lt (by omega)]
termination_by a => n - a

/-- The stream's `chunks` tally for a run of n rows: ⌈n / k⌉ puts. -/
theorem length_chunks {k n : Nat} (hk : 0 < k) : (chunks k n).length = (n + k - 1) / k := by
  rw [chunks, length_chunksFrom hk 0, Nat.sub_zero]

/-! ### Selections and their slices -/

/-- A `TriggerSel`: the eid range [first, last), or a sample's eids. -/
inductive Sel where
  | range (first last : Nat)
  | sample (eids : List Nat)

/-- `TriggerSel::rows()`. -/
def Sel.rows : Sel → Nat
  | .range f l => l - f
  | .sample es => es.length

/-- Row r's trigger, `TriggerSel::eid(r)`, for r = 0 … rows − 1. -/
def Sel.eids : Sel → List Nat
  | .range f l => List.range' f (l - f)
  | .sample es => es

theorem Sel.length_eids (s : Sel) : s.eids.length = s.rows := by
  cases s <;> simp [Sel.eids, Sel.rows]

/-- `slice(sel, a, b)`: `part.first = sel.first + a; part.last = sel.first + b`, or a sample's
`part.eids.assign(sel.eids.begin() + a, sel.eids.begin() + b)`. -/
def slice : Sel → Nat → Nat → Sel
  | .range f _, a, b => .range (f + a) (f + b)
  | .sample es, a, b => .sample ((es.drop a).take (b - a))

theorem take_range' (s n j : Nat) (h : j ≤ n) : (List.range' s n).take j = List.range' s j := by
  have := @List.range'_append s j (n - j) 1
  rw [show j + (n - j) = n by omega] at this
  rw [← this, List.take_append_of_le_length (by simp), List.take_of_length_le (by simp)]

/-- A slice's rows are the selection's rows a … b − 1, as the driver's chunk [a, b) needs. -/
theorem slice_eids (s : Sel) {a b : Nat} (hab : a ≤ b) (hb : b ≤ s.rows) :
    (slice s a b).eids = (s.eids.drop a).take (b - a) := by
  cases s with
  | range f l =>
    simp only [slice, Sel.eids, Sel.rows] at hb ⊢
    rw [List.drop_range', take_range' _ _ _ (by omega)]
    congr 1 <;> omega
  | sample es => rfl

/-! ### The backend's contract, and the parts' cells -/

/-- The contract on a backend's `mine` (D-63, H7): the cells it writes for a part are the rows of
the part's triggers in row order, `row e` for trigger e. A row depends only on its trigger and the
whole graph, not on the part. -/
def Local {α : Type} (mine : Sel → List α) (row : Nat → List α) : Prop :=
  ∀ s, mine s = s.eids.flatMap row

/-- Rows a … b − 1 of a list of triggers, as cells. -/
def part {α : Type} (row : Nat → List α) (L : List Nat) (a b : Nat) : List α :=
  ((L.drop a).take (b - a)).flatMap row

theorem part_append {α : Type} (row : Nat → List α) (L : List Nat) {a b c : Nat} (hab : a ≤ b)
    (hbc : b ≤ c) : part row L a b ++ part row L b c = part row L a c := by
  simp only [part]
  rw [← List.flatMap_append, show c - a = (b - a) + (c - b) by omega, List.take_add, List.drop_drop,
    show a + (b - a) = b by omega]

/-- Dropping i rows of `cols` cells each. -/
theorem flatMap_drop {α : Type} (row : Nat → List α) {cols : Nat} (hcols : ∀ e, (row e).length = cols) :
    ∀ (L : List Nat) (i : Nat), (L.flatMap row).drop (i * cols) = (L.drop i).flatMap row
  | [], i => by simp
  | x :: L, 0 => by simp
  | x :: L, i + 1 => by
    have h1 : (row x).length = cols := hcols x
    rw [List.flatMap_cons, Nat.succ_mul, Nat.add_comm, ← List.drop_drop,
      List.drop_append_of_le_length (by omega), List.drop_of_length_le (l := row x) (by omega),
      List.nil_append, flatMap_drop row hcols L i, List.drop_succ_cons]

/-- Taking j rows of `cols` cells each. -/
theorem flatMap_take {α : Type} (row : Nat → List α) {cols : Nat} (hcols : ∀ e, (row e).length = cols) :
    ∀ (L : List Nat) (j : Nat), (L.flatMap row).take (j * cols) = (L.take j).flatMap row
  | [], j => by simp
  | x :: L, 0 => by simp
  | x :: L, j + 1 => by
    have h1 : (row x).length = cols := hcols x
    rw [List.flatMap_cons, Nat.succ_mul, Nat.add_comm, List.take_add,
      List.take_append_of_le_length (by omega), List.take_of_length_le (l := row x) (by omega),
      List.drop_append_of_le_length (by omega), List.drop_of_length_le (l := row x) (by omega), List.nil_append,
      flatMap_take row hcols L j, List.take_succ_cons, List.flatMap_cons]

theorem flatMap_congr_mem {β α : Type} {l : List β} {f g : β → List α} (h : ∀ x ∈ l, f x = g x) :
    l.flatMap f = l.flatMap g := by
  rw [List.flatMap_def, List.flatMap_def, List.map_congr_left h]

theorem flatten_flatMap {β α : Type} (f : β → List (List α)) :
    ∀ l : List β, (l.flatMap f).flatten = l.flatMap fun x => (f x).flatten
  | [] => rfl
  | x :: l => by rw [List.flatMap_cons, List.flatten_append, flatten_flatMap f l, List.flatMap_cons]

/-- The whole selection's rows. -/
theorem part_all {α : Type} (row : Nat → List α) (L : List Nat) : part row L 0 L.length = L.flatMap row := by
  simp [part]

/-- Chunk [c, e) of part [a, b)'s buffer, read at `buf + (c - a) * cols`, is rows c … e − 1. -/
theorem part_put {α : Type} (row : Nat → List α) {cols : Nat} (hcols : ∀ e, (row e).length = cols)
    (L : List Nat) {a b c e : Nat} (hac : a ≤ c) (hce : c ≤ e) (heb : e ≤ b) :
    ((part row L a b).drop ((c - a) * cols)).take ((e - c) * cols) = part row L c e := by
  simp only [part]
  rw [flatMap_drop row hcols, flatMap_take row hcols, List.drop_take, List.drop_drop, List.take_take,
    show a + (c - a) = c by omega, show min (e - c) (b - a - (c - a)) = e - c by omega]

/-! ### The driver's runs -/

/-- `run_fused` (one slot's stream, `mine` being that slot's answer) and `run_vstats`: per chunk
[a, b), the backend's cells for `slice(sel, a, b)`, put on the stream in loop order. -/
def runChunked {α : Type} (mine : Sel → List α) (k : Nat) (sel : Sel) : List (List α) :=
  (chunks k sel.rows).map fun p => mine (slice sel p.1 p.2)

/-- `run_block`: per part [a, b) of `part_rows` = P rows (`chunk_rows`, or for a block with a CPU
sweep plan the whole selection, D-79), the backend's buffer `buf` for `slice(sel, a, b)`; then per
chunk [c, e) of [a, b) of `chunk_rows` = k rows, the put of `buf + (c - a) * cols`,
(e − c) · cols cells. -/
def runBlock {α : Type} (mine : Sel → List α) (cols P k : Nat) (sel : Sel) : List (List α) :=
  (chunks P sel.rows).flatMap fun p =>
    (chunksFrom k p.2 p.1).map fun q =>
      ((mine (slice sel p.1 p.2)).drop ((q.1 - p.1) * cols)).take ((q.2 - q.1) * cols)

/-- A part's cells under the contract. -/
theorem mine_slice {α : Type} {mine : Sel → List α} {row : Nat → List α} (hmine : Local mine row)
    (sel : Sel) {a b : Nat} (hab : a ≤ b) (hb : b ≤ sel.rows) :
    mine (slice sel a b) = part row sel.eids a b := by
  rw [hmine, slice_eids sel hab hb, part]

/-- **The range-split identity, `run_fused` and `run_vstats`.** Under the contract, the chunks'
cells concatenated are the whole selection's: what one call for the whole selection writes. -/
theorem runChunked_flatten {α : Type} {mine : Sel → List α} {row : Nat → List α} {k : Nat}
    (hk : 0 < k) (hmine : Local mine row) (sel : Sel) :
    (runChunked mine k sel).flatten = mine sel := by
  rw [runChunked, ← List.flatMap_def,
    flatMap_congr_mem (g := fun p => part row sel.eids p.1 p.2) fun p hp =>
      have := mem_chunks hp
      mine_slice hmine sel (by omega) this.2.1,
    chunks, chunksFrom_flatMap _ (fun _ _ _ => part_append row sel.eids) hk 0 (Nat.zero_le _),
    ← Sel.length_eids, part_all, hmine]

/-- **The range-split identity, `run_block`.** For any part size P ≥ 1 (`chunk_rows`, or the whole
selection for a sweep block, D-79) and chunk size k ≥ 1, under the contract and with `cols` cells
per row, the puts concatenated are the whole selection's cells. -/
theorem runBlock_flatten {α : Type} {mine : Sel → List α} {row : Nat → List α} {cols P k : Nat}
    (hP : 0 < P) (hk : 0 < k) (hmine : Local mine row) (hcols : ∀ e, (row e).length = cols) (sel : Sel) :
    (runBlock mine cols P k sel).flatten = mine sel := by
  rw [runBlock, flatten_flatMap,
    flatMap_congr_mem (g := fun p => part row sel.eids p.1 p.2) fun p hp => by
      have hp' := mem_chunks hp
      rw [← List.flatMap_def,
        flatMap_congr_mem (g := fun q => part row sel.eids q.1 q.2) fun q hq => by
          have hq' := mem_chunksFrom hq
          rw [mine_slice hmine sel (by omega) hp'.2.1]
          exact part_put row hcols sel.eids hq'.1 (by omega) hq'.2.2.1,
        chunksFrom_flatMap _ (fun _ _ _ => part_append row sel.eids) hk p.1 (by omega)],
    chunks, chunksFrom_flatMap _ (fun _ _ _ => part_append row sel.eids) hP 0 (Nat.zero_le _),
    ← Sel.length_eids, part_all, hmine]

/-! ### The stream's hash -/

/-- One byte of FNV-1a 64: `h = (h ^ b) * prime` (`fnv1a_bytes`, src/runtime/block_sink.cpp). -/
def fnvStep (h : UInt64) (b : UInt8) : UInt64 := (h ^^^ b.toUInt64) * 1099511628211

/-- FNV-1a 64 over bytes, from the value h. -/
def fnv1a (h : UInt64) (bs : List UInt8) : UInt64 := bs.foldl fnvStep h

/-- `BlockStream::put`'s hash over a stream's puts, from h0 (`kFnv1aBasis`): each put continues
from the stream's value, `tally_.fnv1a = fnv1a_cells(cells, tally_.fnv1a, …)`. -/
def streamHash {α : Type} (cellHash : UInt64 → List α → UInt64) (h0 : UInt64) (puts : List (List α)) :
    UInt64 :=
  puts.foldl (fun h c => cellHash h c) h0

/-- **The hash continued across chunks.** Under the contract that `fnv1a_cells` is FNV-1a over the
cells' bytes in memory order (`enc`, D-78), the stream's hash is FNV-1a over the concatenation. -/
theorem stream_hash {α : Type} {cellHash : UInt64 → List α → UInt64} {enc : α → List UInt8}
    (hcells : ∀ h cs, cellHash h cs = fnv1a h (cs.flatMap enc)) :
    ∀ (h0 : UInt64) (puts : List (List α)), streamHash cellHash h0 puts = fnv1a h0 (puts.flatten.flatMap enc)
  | h0, [] => rfl
  | h0, c :: cs => by
    simp only [streamHash, List.foldl_cons] at *
    rw [← streamHash, stream_hash hcells, hcells, fnv1a, fnv1a, ← List.foldl_append, ← List.flatMap_append,
      List.flatten_cons, fnv1a]

/-- **A block's `output_fnv1a`.** The stream's hash after `run_block`'s puts is the hash of the whole
selection's cells: chunked, swept or whole, the same value (`blz mine`'s manifest, `blz measure`). -/
theorem runBlock_hash {α : Type} {mine : Sel → List α} {row : Nat → List α} {cols P k : Nat}
    {cellHash : UInt64 → List α → UInt64} {enc : α → List UInt8}
    (hP : 0 < P) (hk : 0 < k) (hmine : Local mine row) (hcols : ∀ e, (row e).length = cols)
    (hcells : ∀ h cs, cellHash h cs = fnv1a h (cs.flatMap enc)) (h0 : UInt64) (sel : Sel) :
    streamHash cellHash h0 (runBlock mine cols P k sel) = fnv1a h0 ((mine sel).flatMap enc) := by
  rw [stream_hash hcells, runBlock_flatten hP hk hmine hcols]

end Blz.Chunks

# BlazingAML replication — implementation SPEC (v1)

Status: draft for review · Date: 2026-11-10 · Scope: HI-Small, C++20 (OpenMP + CUDA), no FraudGT.

This document specifies a C++ re-implementation of *BlazingAML* (arXiv:2604.12241, digested in
`BlazingAML.md`). It is normative: MUST / SHOULD / MAY carry their RFC 2119 meaning.

Every factual claim is tagged with its source:

- **[P]** the paper, via `BlazingAML.md` (section numbers are the digest's, e.g. [P §5.3]).
- **[G]** GFP documentation or the GFP paper (Blanuša et al., arXiv:2402.08593v2), or the snapml
  1.17.2 docstring.
- **[X]** a probe experiment run against snapml 1.17.2 `GraphFeaturePreprocessor` on 2026-11-10.
  Appendix A lists every case.
- **[D]** a decision made by this spec. Each has an ID (`D-n`) so it can be revisited.

---

## 1. Goals, scope, success criteria

### 1.1 Goals

- **G1. Features.** Mine per-transaction pattern-count features on HI-Small for four groups:
  Fan, Degree, Cycle and Scatter-Gather (SG). The values MUST equal those of IBM GFP, which the
  paper names as ground truth ("identical in value") [P §1, §7.4].
- **G2. Compiler.** All production features MUST come from code that a pattern compiler generates
  from declarative YAML specs. The compiler emits C++/OpenMP and CUDA [P §5].
- **G3. Accuracy.** Reproduce the HI-Small row of Table 2: XGB-only → +Fan → +Degree → +Cycle → +SG
  [P §7.1], and the Table 3 confusion matrix [P §7.2].
- **G4. Speed.** Measure mining speed against GFP on the same hardware, per pattern, with
  CPU thread scaling and GPU runs. These are analogues of Figs. 6–8 [P §7.3].
- **G5. Auditability.** Every deviation from the paper is listed (Appendix B), and every GFP
  behaviour we rely on is backed by a reproducible probe (Appendix A).

### 1.2 In scope for v1

- Data: `dataset/HI-Small_Trans.csv` only.
- Patterns: fan-in, fan-out, degree-in, degree-out, scatter-gather and temporal cycle.
- Backends: oracle (naive C++), generated CPU (OpenMP) and generated CUDA (sm_75, sm_80).
- Classifier: XGBoost, called through its C API from C++.
- GFP: a Python harness (`tools/gfp/`) used only as a parity oracle and timing baseline.

### 1.3 Out of scope for v1 (non-goals)

- FraudGT and Fig. 12 throughput comparison.
- LI-\*, Medium and Large datasets, and Trovares.
- Multi-GPU.
- Stack, bipartite and gather-scatter features.
- DSL ops `union`, `difference` and `differentiate`. The keywords are reserved and rejected (§6.9).
- A live streaming ingestion service; v1 mines offline with streaming-equivalent semantics (§4.1).
- GNN classifiers.

Stretch items are listed in §16 (P6). They MUST NOT block v1 acceptance.

### 1.4 Acceptance criteria (definition of done for v1)

| ID | Criterion |
|---|---|
| AC1 | For every v1 pattern, generated-CPU and generated-CUDA features are **bit-identical** to the oracle. This MUST hold on HI-Small and on ≥ 10,000 random fuzz graphs (§13.3). |
| AC2 | Oracle features equal GFP in stream mode (batch size 1, §11.1) on HI-Small, for every column of fan_in, fan_out, deg_in, deg_out, sg and tcyc. GFP is fed `ts + 1` (D-12, [X A33]). The one exception allowed by design is tcyc instances longer than `K_MAX` (DG-1). Any other mismatch fails AC2 unless it is (a) reproduced as a minimal probe, (b) explained in Appendix A, and (c) accepted through a new `D-n` decision. |
| AC3 | The five HI-Small F1 values of Table 2 and the Table 3 confusion matrix are produced with the pre-declared default classifier config (§10.2). They are reported next to the paper's numbers. No choice is tuned on the test split. |
| AC4 | Per-pattern mining time, thread-scaling curves (1–12 threads, Ryzen 5 3600), GPU times (GTX 1650 SUPER, A30) and speedup versus GFP on the same machine are reported in `results/` (§12). |
| AC5 | The compiler rejects every case in the invalid-spec suite (§13.4), E101–E401, with the specified error code and a `file:line:col` location. E402 needs the feature config, so it is checked by a runtime registration test instead (§13.1). |

---

## 2. Hardware and toolchain

### 2.1 Machines

| Name | CPU | GPU | Memory | Role |
|---|---|---|---|---|
| `local` | Ryzen 5 3600: 6C/12T, Zen 2, AVX2, 32 MB L3 | GTX 1650 SUPER: Turing **sm_75**, 4 GB | 16 GB total, often ≈ 3 GB free | Development, CPU scaling, GFP parity, small-GPU runs |
| `a30` | JarvisLabs instance (vCPU count: record at runtime) | A30: Ampere **sm_80**, 24 GB HBM2 | Record at runtime | GPU benchmarks; CPU runs only if it has > 12 vCPUs |
| `cpu-cloud` | — | — | — | Not provisioned. The design MUST NOT assume it. |

Paper hardware for contrast: 2× Xeon 8380 (160 hardware threads) and an A40. Absolute numbers are
therefore not comparable; only ratios on the same machine are reported [P §6].

### 2.2 Toolchain

- **C++20.** Minimum GCC 12 or Clang 16, because JarvisLabs images ship older compilers. Code MUST
  NOT depend on GCC-16-only features. Local host compiler: GCC 16.2.
- **CMake ≥ 3.28 and Ninja.** Local: CMake 4.4.5 and Ninja 1.13.2. CMake 4 rejects dependencies
  that declare `cmake_minimum_required` below 3.5; if a fetched dependency hits this, pass
  `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`.
- **OpenMP** (libgomp).
- **CUDA ≥ 12.4.** Local: CUDA 13.4.3 in `/opt/cuda`. Its host-compiler check accepts GCC ≤ 16, so
  the default GCC 16.2 is the host compiler. A C++20 test kernel for sm_75 was compiled and run
  successfully on the GTX 1650 SUPER on 2026-11-10. `nvcc` may be missing from `PATH` in
  non-login shells. The build MUST find CUDA through `CMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc`
  (or `CUDAToolkit_ROOT`), not through `PATH`. `CMAKE_CUDA_ARCHITECTURES=75;80`.
- **Dependencies** come through CMake `FetchContent`, each pinned to an exact tag in
  `cmake/deps.cmake`:
  - `yaml-cpp` (spec parsing);
  - `xgboost` (C API, built as a static, CPU-only library, `USE_CUDA=OFF` [D-13]);
  - `Catch2` v3 (tests);
  - `nlohmann_json` (results files);
  - `CLI11` (command-line parsing).
- **Python (tools only).** The `tools/gfp/` uv project pins Python 3.12 and `snapml==1.17.2`.
  System Python is 3.14, which snapml does not support.

### 2.3 Build options

| Option | Default | Effect |
|---|---|---|
| `BLZ_ENABLE_CUDA` | `OFF` | Builds the CUDA backend. |
| `BLZ_NATIVE` | `ON` | Adds `-march=native`. Benchmark result files record the flags used. |
| `BLZ_SANITIZE` | `OFF` | ASan + UBSan, for test builds. |
| `BLZ_WITH_XGBOOST` | `ON` | Builds `blz train` against XGBoost. Off only in the A30 runbook (D-15). |

Presets (`CMakePresets.json`): `dev` (Debug, sanitizers), `rel` (Release, CPU), `rel-cuda`.

---

## 3. Data

### 3.1 Input

`HI-Small_Trans.csv` has this header [P §12]:

```
Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering
```

The header has two columns both named `Account`; parse columns by position.

### 3.2 Parsing and normalization

- **Timestamp.** Format `YYYY/MM/DD HH:MM`. Convert to `int32` seconds since `2022/09/01 00:00`,
  naive with no time zone [D-1]. Valid range: 0 to 1,527,480 (`2022/09/18 16:18`).
- **Node key.** The pair (bank string, account string), compared as **raw strings** with leading
  zeros kept (for example `010`) [D-2].
- **Node id.** A dense `uint32`, assigned in order of first appearance while walking the stream
  order (§3.3). Within a row the source is checked before the destination [D-3].
- **Kept attributes**, unused by the v1 default classifier:
  - `amount_paid` and `amount_received` (`double`);
  - `pay_ccy` and `recv_ccy` (`uint8` dictionary codes, dictionary sorted by string);
  - `format` (`uint8` dictionary code);
  - `label` (`uint8`).
- **Parser.** MUST stream the file through `mmap` with a hand-written field scanner; no iostreams
  per row. A malformed row is a hard error that reports the line number.

### 3.3 Stream order and edge ids

- Sort rows **stably** by timestamp. Ties keep file order [P §12, D-4].
- Edge id `eid` = position in this order, from 0 to m−1. "Stream order" means increasing `eid`.
- Because the sort is stable, `eid₁ < eid₂ ⇒ ts₁ ≤ ts₂`. The data structures in §5 rely on this.

### 3.4 Split

- Train is `eid < ⌊0.8·m⌋ = 4,062,676`; test is the remaining **1,015,669** edges [P §6, §12].
- Mining runs over the whole stream. Windows look backward only (§4), so a test edge's features use
  only edges that precede it. In the default trigger mode this means no label or future leakage.

### 3.5 Invariants (asserted by `blz prepare`; a failure aborts)

| Quantity | Expected |
|---|---|
| Edges | 5,078,345 |
| Nodes | 515,088 |
| Positives (`Is Laundering = 1`) | 5,177 |
| Self-loops | 591,212 |
| Test positives | 1,797 |
| Train positives | 3,380 |

### 3.6 Binary cache (`.blzg`)

`blz prepare` writes one little-endian file:

- A header: magic `BLZG`, `u32` version, `u64 m`, `u64 n`, and the SHA-256 of the source CSV.
- The edge arrays in stream order, structure-of-arrays: `src u32[m]`, `dst u32[m]`, `ts i32[m]`,
  `label u8[m]`, and the attribute arrays.
- The node key table.

The file is meant to be `mmap`ed. All other commands read `.blzg`, never the CSV. The file is
written to `data/`, which is git-ignored. `blz export-edges` writes `edges.npy`
(`int64[m,4]`: eid, src, dst, ts) for the GFP harness.

---

## 4. Feature semantics (normative)

Every implementation (oracle, generated CPU, generated CUDA) MUST produce exactly what this section
defines. The defaults reproduce GFP stream mode. The evidence is in Appendix A.

### 4.1 Trigger, visibility, window

- **Trigger.** Each edge `e = (u → v, t, id)` is processed as a *trigger*. Its features are the
  histogram of the pattern instances that `e` completes. This is "credit = trigger" [X A5, D-5].
- **Visibility.** `Vis(e) = {e' : id' ≤ id}`: the trigger plus everything before it in stream
  order. Same-timestamp edges that come earlier in the file are visible; later ones are not
  [X A4, A15, D-6].
- **Window.** For a pattern P with span δ_P, `W_P(e) = {e' ∈ Vis(e) : t' ∈ L_P(t) }`. The lower
  bound depends on the pattern:
  - fan and degree are **open**: `t' > t − δ` [X A2, A3];
  - SG and tcyc are **closed**: `t' ≥ t − δ` [X A7, A20].
- **Self-loop triggers** (`u = v`) emit nothing for every pattern [X A13, A14].
- **Equivalence to streaming.** Running offline over the full sorted stream with these rules gives
  exactly the result of inserting edges one at a time in stream order and mining each on arrival.
  Triggers are therefore independent, which makes the computation embarrassingly parallel over
  triggers.

`N_in(x)` and `N_out(x)` below denote neighbour **node sets** restricted to the edges of the
window `W_P(e)`.

### 4.2 Histograms

- Each output **block** (fan_in, fan_out, deg_in, deg_out, sg, tcyc) has bins
  `b_0 < b_1 < … < b_{k−1}`.
- An instance of size S increments bin i when `b_i ≤ S < b_{i+1}`. The last bin is open-ended.
  Sizes below `b_0` are dropped [G].
- Every pattern has **min size 2**: smaller instances are never emitted, whatever the bins
  [X A1, A21, A22].
- Counts are `uint32` with saturating addition. A global saturation counter is reported; it is
  expected to be 0.
- Column names follow `<block>[<b_i>,<b_{i+1}>)` and `<block>[<b_last>,inf)`.
- Block order follows GFP: fan_in, fan_out, deg_in, deg_out, sg, tcyc [G].

### 4.3 Fan

For a trigger `e = (u→v)` with `u ≠ v`:

- `fan_in(e) = |{x ∈ N_in(v) : x ≠ v}|`. This includes u.
- `fan_out(e) = |{y ∈ N_out(u) : y ≠ u}|`. This includes v.

Each block emits one instance of that size when the size is ≥ 2 [X A1, A16].

### 4.4 Degree

For a trigger `e = (u→v)` with `u ≠ v`:

- `deg_in(e)` = the number of **edges** `(x→v) ∈ W` with `x ≠ v`. Parallel edges count
  separately, and e itself is included.
- `deg_out(e)` is symmetric over u's out-edges.

One instance is emitted when the count is ≥ 2 [X A1, A16].

### 4.5 Scatter-gather (sg)

An SG instance is a pair (s, w) with `s ≠ w`, a source s and a sink w. A trigger `e = (u→v)` with
`u ≠ v` completes these instances:

- **Gather role** (e is an intermediate → sink edge): for each `s ∈ N_in(u) \ {u, v}`, the
  instance (s, w = v) with `I = (N_out(s) ∩ N_in(w)) \ {w}` [X A10, A12, A14, A17].
- **Scatter role** (e is a source → intermediate edge): for each `w ∈ N_out(v) \ {u, v}`, the
  instance (s = u, w) with `I = (N_out(s) ∩ N_in(w)) \ {s}` [X A6, A16, A30–A32].

Each instance with `|I| ≥ 2` emits size `|I|`, counted as distinct nodes.

**The exclusion depends on the role.** It removes the pattern endpoint that the trigger touches:
w in the gather role, s in the scatter role. The other endpoint can appear in I through its own
self-loop. In the gather role, s→s plus s→w puts s in I [X A10]. In the scatter role, s→w plus w→w
puts w in I [X A30].

The two roles never produce the same (s, w) pair: gather pairs have w = v, and scatter pairs have
w ≠ v. The SG count is therefore the sum of both roles. There is **no** temporal order inside an
SG instance, only the window [X A6].

### 4.6 Temporal cycle (tcyc)

For a trigger `e = (u→v, t)` with `u ≠ v`, count the distinct node sequences
`(c_0 = v, c_1, …, c_{k−1} = u)` with k ≥ 2 and all c_i pairwise distinct (a simple cycle).
A sequence counts when there exist edges `(c_{i−1} → c_i) ∈ W` for i = 1..k−1 with timestamps

    t − δ ≤ t_1 < t_2 < … < t_{k−1} < t

The instance size is k, the number of cycle edges including e.

- Parallel edges do **not** multiply the count; each node sequence counts once [X A23, A25].
- Ties never form a cycle, because the order is strict [X A19].
- Self-loops are not cycles [X A22].
- **Earliest-witness rule.** For a fixed node sequence, a valid timestamp assignment exists if and
  only if greedily taking, at every hop, the earliest edge later than the previous hop succeeds.
  All implementations MAY rely on this.
- **Length cap.** `K_MAX` caps k (default 10 [D-7]). It is a compile-time constant of the spec. P0
  measures the tcyc length distribution under GFP, which has no cap [X A26], and decision gate
  DG-1 (§17) may raise it.

### 4.7 Variant knobs (not defaults)

| Knob | Default | Alternative | Stage |
|---|---|---|---|
| `visibility` | `stream` | `timestamp`: `Vis(e) = {e' : t' ≤ t}`. This reads Fig. 5's `break if e.t > t` literally. | P6 |
| `credit` | `trigger` | `participants`: every edge of an instance gets +1 and each instance is counted once (Fig. 5's "Update sg_tx features"); CPU only. | P6 |
| `lc-cycle` | off | Unordered simple cycles within the window, length ≤ 10 (the paper's Fig. 5c has no order). | P6 |

### 4.8 Default parameters

| Block | δ (s) | Lower bound | Bins |
|---|---:|---|---|
| fan_in, fan_out | 86,400 | open | [2,3,4,5,6,8,10,13,16,20,30,50] |
| deg_in, deg_out | 86,400 | open | [2,3,4,5,6,8,10,13,16,20,30,50] |
| sg | 21,600 | closed | [2,3,4,5,6,8,10,13,16,20] |
| tcyc | 86,400 | closed | [2,3,4,5,6,7,8,9,10] |

- The windows come from GFP's AML setup: six hours for SG, one day for everything else [G, D-8].
  The paper gives no δ [P §10].
- The bins are [D-9]. They are fine-grained up to about 16–20, because the IBM generator's
  patterns reach about that size (`HI-Small_Patterns.txt`: "Max 16-degree Fan-Out",
  "Max 10 hops"). The result is 67 pattern columns.
- Parity runs (§11.1) use different bins, `[2,3,…,33]`, with an open last bin.

All values live in `configs/features_default.yaml`. None are hard-coded.

---

## 5. Graph data structures

All arrays are structure-of-arrays, `uint32` positions, built once by `blz` after loading `.blzg`.
Self-loops are **kept** in every index, and specs filter them explicitly (§6.8). This is what
reproduces GFP's SG self-loop behaviour (§4.5).

### 5.1 T-CSR (temporal CSR, one per direction)

- For each node x, `off[x]..off[x+1]` holds x's out-edges (in-edges for the in-direction), sorted
  by eid, which also sorts them by ts.
- Per entry: `nbr u32`, `ts i32`, `eid u32`, and `prev_same i32`. `prev_same` is the position of
  the previous entry in the **same list** with the same `nbr`, or −1.
- **Distinct-first test.** In a scanned range `[lo, hi)`, entry j is the first (earliest) occurrence
  of its neighbour exactly when `prev_same[j] < lo`. This gives:
  - distinct iteration without hash sets or per-thread marker arrays, on both CPU and GPU;
  - each distinct neighbour's **earliest witness** for free.

### 5.2 P-CSR (pair index, one per direction)

- For each node x, its distinct neighbours sorted by node id.
- Each pair (x, y) points to a ts-sorted run of that pair's `(ts, eid)` values.
- Used for existence probes: "is there an edge x→y with ts in [a, b] (or open variants) and
  eid ≤ E?" This takes two binary searches.

### 5.3 Primitives (`include/blz/primitives.hpp`, all `BLZ_HD` = `__host__ __device__` inline)

| Primitive | Meaning |
|---|---|
| `range(dir, x, lo_ts, lo_strict, hi)` | Returns `[lo, hi)` positions. The lower end comes from a lower_bound on ts (the paper's `find_start`). The upper end is one of `eid ≤ E` (stream visibility), `ts ≤ T` (timestamp visibility) or `ts < T` (strict-before, for ordered hops). |
| `distinct_first(dir, j, lo)` | `prev_same[j] < lo` |
| `pair_exists(dir, x, y, lo_ts, lo_strict, hi)` | P-CSR probe. Returns the earliest witness ts, or `NONE`. |
| `count_distinct(dir, x, range, exclude)` / `count_edges(...)` | Fan and degree counts in O(window). |

As implemented (P4), every bound is inclusive, because the planner already normalizes strict
bounds (§7.2), and the direction is the view the call is made on (`g.out`, `g.in`, `g.pout`,
`g.pin`):

- `TCsrView::range(x, lo_ts, hi_ts, max_eid)`: the entries with `lo_ts ≤ ts ≤ hi_ts` and
  `eid ≤ max_eid`, by binary search; `hi_ts = kNoUpper` skips the time search on the upper end.
- `TCsrView::distinct_first(j, lo)`.
- `PCsrView::pair_exists(x, y, lo_ts, hi_ts, max_eid)`: the earliest witness ts, or `kNoTs`.
- `count_distinct(view, range, keep)` / `count_edges(view, range, keep)`: `keep(nbr)` is the
  element filter, which generalizes `exclude`. P5 adds two defaulted arguments, `first` and
  `stride`, so that the lanes of a CUDA warp can split one count (positions `lo + first`,
  `lo + first + stride`, …; `distinct_first` still tests against `lo`).

`src/graph/index.cpp` builds both directions of T-CSR and P-CSR: 268 MB on HI-Small, in 0.4 s.
All primitives have unit tests against naive scans (§13.1).

### 5.4 Memory budget

On HI-Small the indexes take ≈ 300 MB: T-CSR 2 × m × 16 B ≈ 162 MB, plus P-CSR. Feature blocks are
`uint32`, about 1.36 GB for 67 columns. Rules:

- `blz mine` MUST mine **one block at a time** and write each block to its own `.npy`.
- `blz train` MUST stream the blocks into an XGBoost QuantileDMatrix in row batches (§10.2), so
  that peak RSS stays ≤ 3 GB on `local`. Peak RSS is measured with `/usr/bin/time -v` and recorded.
  Measured in P2: 2.35 GB for `all` (69 columns), most of it the mapped `.npy` pages.

---

## 6. Pattern DSL

The DSL keeps the paper's vocabulary: `pattern`, `time_window`-like windows, `stages`, `id`, `op`,
`src`, `dst`, `dst_var`, and `constraints` with `skip_if` / `break_if` [P §5.3]. It fixes the
inconsistencies listed in [P §9], and it adds the minimum needed for exact semantics: named edge
variables, `emit`, `exists`, `count`, `repeat` and `gate`.

### 6.1 File structure

```yaml
version: 1
pattern: <ident>              # unique pattern name
block: <ident>                # output histogram block; patterns that share a block are summed
consts: {K_MAX: 10}           # optional compile-time integers
window: {span: <param>, lower: open|closed}   # span is a runtime int parameter (seconds)
visibility: stream            # stream | timestamp (timestamp: P6)
trigger: {edge: e0, src: <Var>, dst: <Var>}   # binds two node vars; `t` means e0.t
constraints:                  # trigger-level; only skip_if is allowed here
  - skip_if: <pred>
stages: [<stage>, ...]
```

### 6.2 Expressions

```
pred  := pred 'or' pred | pred 'and' pred | 'not' pred | '(' pred ')' | cmp
cmp   := node ('=='|'!=') node | node ('in'|'not in') '[' nodelist ']'
       | texpr ('<'|'<='|'>'|'>=') texpr | iexpr ('<'|'<='|'>'|'>='|'=='|'!=') iexpr
node  := Var | Var '[' iconst ']'           # indexed vars come only from repeat expansion
texpr := 't' | edgevar '.t' | texpr ('+'|'-') iexpr
iexpr := INT | param | const | 'len(' SetVar ')' | IntVar | iexpr ('+'|'-') iexpr
```

Identifiers: node vars are capitalized (`U`, `C3`), edge vars are `e<digits>` or lowercase names,
and params / consts are as declared.

### 6.3 Stage kinds

There are two kinds of stage:

- **Binding stages** open a loop level. Every later stage runs once per binding.
- **Check stages** run once per current binding and do not open a level. With `gate: true`, the
  later stages run only when the check passed.

| op | Kind | Fields | Meaning |
|---|---|---|---|
| `for_all` | binding | `src: X.in\|X.out`, `edge`, `dst_var`, `multiplicity: distinct\|edges` (default distinct) | Iterates the neighbours of X within the window and bounds. `distinct` binds each node once, with its earliest witness edge. |
| `intersection` | binding or check | `src: A.dir`, `dst: B.dir`, `edges: [ea, eb]`, plus either `dst_var` (binding) or `output: SetVar` + `element` (check) | Neighbour-set intersection over distinct nodes. |
| `exists` | check | `src: X.dir`, `node: Y`, `edge` | Is Y in X's neighbours within the bounds? Exposes the earliest witness as `edge.t`. |
| `count` | check | `src: X.dir`, `edge`, `element`, `multiplicity`, `output: IntVar` | The size of the filtered neighbour set or edge multiset. |

`element` names the per-element node variable of a check stage. It is visible only in that
stage's own constraints and is never bound for later stages. It is optional; a stage without
element filters can omit it.

Every stage MAY carry `emit: {size: <iexpr>, min_size: <int>}` (§6.6) and `constraints`.

**Scope** (fixed in P3). Every name (const, span parameter, trigger variables, and each stage's
`edge`, `edges`, `dst_var`, `element` and `output`) is declared exactly once per spec; there is no
shadowing (E102). What a stage makes visible:

| Stage | Own constraints see | Later stages see |
|---|---|---|
| `for_all`, binding `intersection` | its scanned edge(s), `dst_var` | `dst_var`; its edge(s) as earliest witnesses |
| `exists` | its edge | its edge as a witness, **only with `gate: true`** (otherwise there may be no witness) |
| `count`, check `intersection` | its scanned edge(s), `element` | `output` (not visible in its own constraints) |

`emit` sees what later stages see, plus a binding stage's `dst_var`; it never sees `element`.

**Gate.** A check passes when `exists` finds a witness, or when `count` / a check
`intersection` produces a non-zero result. `gate` is not allowed on binding stages (E101).

### 6.4 Constraints: bounds versus filters

The compiler classifies each constraint of a stage:

1. **Time bound.** The predicate has the form `ex.t OP τ` or `τ OP ex.t`, where `ex` is an edge
   variable scanned by *this* stage and τ does not mention `ex`.
   - A lower bound (`skip_if: ex.t <= τ` or `< τ`) becomes the scan's start, computed by binary
     search. This is the paper's `find_start`, which is always applied; the paper's printed loops
     ignore it [P §9.7, B-4].
   - An upper bound (`break_if: ex.t > τ` or `>= τ`, or the equivalent `skip_if`) becomes the
     scan's end.
   - `break_if` with `<` or `<=` on a scanned edge is error **E201** (non-monotone break).
2. **Node filter.** The predicate mentions only node variables, ints, params and `len()`. It is
   applied as `skip_if`. A node-only `break_if` is error **E202**. Two clarifications from P3:
   the trigger's own time (`t`, `e0.t`) is exact, so a filter MAY use it; and any `break_if`
   that bounds no edge scanned by its stage is E202.
3. Anything else, such as a mix of a scanned edge's time with a node test inside one predicate, or
   two scanned edges of one intersection compared to each other, is error **E203**. Cross-operand
   ordering inside an intersection is P6.

The **window** and **visibility** bounds are added to every scanned edge automatically.

A top-level `or` is split first: `skip_if: a or b` is `skip_if: a` plus `skip_if: b`, and the same
holds for `break_if`. Each part is then classified on its own. Trigger-level constraints have no
scanned edge, so they are always node filters.

**Witness direction rule.** An earlier stage's witness time (`ej.t` with j bound earlier) MAY
appear only as a *lower* bound of a later scan. Otherwise the earliest-witness rule (§4.6) is not
exact. A violation is error **E204**. This includes a witness time inside a node filter, and a
witness lower bound on an aggregating scan (`count`, a check `intersection`, a `for_all` with
`multiplicity: edges`), whose result would depend on which witness was taken. Witness lower bounds
are allowed on `exists`, a distinct `for_all` and a binding `intersection`.

### 6.5 `repeat`

`repeat` is compile-time unrolling:

```yaml
- repeat: {var: k, from: 1, to: K_MAX - 2}
  stages: [ ... ]
```

The template stages are inlined once per k, in order. Every string in the template is
substituted:

- **Arithmetic.** `{a}` is replaced by the decimal value of `a`, an integer expression over the
  repeat variable, consts and `+ - *`. For example, with k = 3, `hop_{k}` → `hop_3`,
  `C{k-1}.out` → `C2.out`, and `"{k+2}"` → `5`. A value that consists only of one `{…}` becomes an
  integer.
- **Prefixed range.** `P{a..b}` is an identifier prefix followed by a range. It expands to the list
  item sequence `Pa, Pa+1, …, Pb`, so the prefix repeats on every item. With k = 3,
  `[U, V, C{1..k-1}]` → `[U, V, C1, C2]`. An empty range (b < a) removes the item and its
  separating comma.

Templated values and predicates MUST be quoted YAML strings (§6.8), because `{` `}` `[` `]` `,`
are flow indicators. The unrolled depth after expansion MUST be ≤ 16 (error **E301**). The depth
counts binding stages, so tcyc's depth is `K_MAX − 2` and `K_MAX` ≤ 18. E301 also covers a
`repeat` or range of more than 1,024 values and more than 1,024 stages after expansion.

The implementation (P3) substitutes `{…}` in every scalar outside `consts`, with the consts and
the enclosing repeat variables in scope, so a const can also be used as `{K_MAX}` outside a
template. Repeats nest. A YAML list item that is exactly `P{a..b}` expands into separate items.
`tests/specs/expanded/` holds the expected expansion of `tcyc.yaml` at `K_MAX = 5` (§13.4).

### 6.6 `emit`

`emit` runs once per binding (binding stage) or once per evaluation that passes (check stage).
`size` is evaluated, and if `size ≥ max(min_size, 2)` the trigger's row in `block` gets +1 at
`bin(size)`. `min_size` defaults to 2 (§4.2).

### 6.7 Static checks

| Code | Check |
|---|---|
| E101 | YAML schema error: unknown key, missing required key, wrong type. |
| E102 | Unknown or duplicate identifier; use before definition. Also two specs of one `blzc` run that define the same `pattern` name (P4: it names generated functions and registry entries). |
| E103 | Type error (node, time, int and set are kinds that do not mix). |
| E201–E204 | Constraint classification (§6.4). |
| E206 | A `family: clean` spec (SPEC v3 §3.2) whose window lower bound is not `open` (R2). At the `lower:` value. |
| E207 | A `family: clean` spec that does not pin `visibility: timestamp` (R3). At the `visibility:` value, or at `family:` when the spec has none. |
| E301 | Unrolled depth > 16. |
| E401 | Reserved op (`union`, `difference`, `differentiate`): "not supported in v1". |
| E402 | `emit` or `block` refers to a block that is missing from the feature config. Checked at registration, at runtime start. |
| E501 | Code generation (P4): a spec the front end accepts but the v1 backends cannot generate. These are `visibility: timestamp` (a P6 variant), a `pattern` name with an index, and two names that map to the same C++ identifier (`C[3]` and `C_3`). |

Diagnostics MUST include `file:line:col`, which yaml-cpp marks provide, and the error code.
Inside a single-line predicate that expansion did not rewrite, the column points at the offending
token. Otherwise it points at the start of the scalar.

### 6.8 v1 pattern library (`patterns/*.yaml`)

```yaml
# patterns/fan_in.yaml            (fan_out.yaml: mirror, using U.out and Y != U)
version: 1
pattern: fan_in
block: fan_in
window: {span: delta_fan, lower: open}
trigger: {edge: e0, src: U, dst: V}
constraints: [{skip_if: "U == V"}]
stages:
  - id: in_nbrs
    op: count
    src: V.in
    edge: e1
    element: X
    multiplicity: distinct
    output: n
    constraints: [{skip_if: "X == V"}]
    emit: {size: n}
```

```yaml
# patterns/deg_in.yaml            (deg_out.yaml: mirror)
version: 1
pattern: deg_in
block: deg_in
window: {span: delta_deg, lower: open}
trigger: {edge: e0, src: U, dst: V}
constraints: [{skip_if: "U == V"}]
stages:
  - id: in_edges
    op: count
    src: V.in
    edge: e1
    element: X
    multiplicity: edges
    output: n
    constraints: [{skip_if: "X == V"}]
    emit: {size: n}
```

```yaml
# patterns/sg_gather.yaml — the trigger is an intermediate→sink edge (the paper's Fig. 5a role)
version: 1
pattern: sg_gather
block: sg
window: {span: delta_sg, lower: closed}
trigger: {edge: e0, src: X, dst: W}
constraints: [{skip_if: X == W}]
stages:
  - id: sources
    op: for_all
    src: X.in
    edge: e1
    dst_var: S
    constraints: [{skip_if: "S in [X, W]"}]
  - id: intermediates
    op: intersection
    src: S.out
    dst: W.in
    edges: [e2, e3]
    output: I
    element: Y
    constraints: [{skip_if: "Y == W"}]    # gather role excludes the sink (§4.5)
    emit: {size: "len(I)"}
```

```yaml
# patterns/sg_scatter.yaml — the trigger is a source→intermediate edge (the paper's Fig. 4a role)
version: 1
pattern: sg_scatter
block: sg
window: {span: delta_sg, lower: closed}
trigger: {edge: e0, src: S, dst: X}
constraints: [{skip_if: S == X}]
stages:
  - id: sinks
    op: for_all
    src: X.out
    edge: e1
    dst_var: W
    constraints: [{skip_if: "W in [S, X]"}]
  - id: intermediates
    op: intersection
    src: S.out
    dst: W.in
    edges: [e2, e3]
    output: I
    element: Y
    constraints: [{skip_if: "Y == S"}]    # scatter role excludes the source (§4.5, A30–A32)
    emit: {size: "len(I)"}
```

```yaml
# patterns/tcyc.yaml — temporal cycles of length 2..K_MAX that the trigger U→V closes
version: 1
pattern: tcyc
block: tcyc
consts: {K_MAX: 10}
window: {span: delta_cyc, lower: closed}
trigger: {edge: e0, src: U, dst: V}
constraints: [{skip_if: "U == V"}]
stages:
  - id: close_2                          # the cycle V→U→V
    op: exists
    src: V.out
    node: U
    edge: c2
    constraints: [{break_if: "c2.t >= t"}]
    emit: {size: 2}
  - id: hop_1                            # first hop: only the window bounds it from below
    op: for_all
    src: V.out
    edge: h1
    dst_var: C1
    constraints:
      - break_if: "h1.t >= t"
      - skip_if: "C1 in [U, V]"
  - id: close_3
    op: exists
    src: C1.out
    node: U
    edge: c3
    constraints: [{skip_if: "c3.t <= h1.t"}, {break_if: "c3.t >= t"}]
    emit: {size: 3}
  - repeat: {var: k, from: 2, to: "K_MAX - 2"}
    stages:
      - id: "hop_{k}"
        op: for_all
        src: "C{k-1}.out"
        edge: "h{k}"
        dst_var: "C{k}"
        constraints:
          - skip_if: "h{k}.t <= h{k-1}.t"   # strictly after the previous hop
          - break_if: "h{k}.t >= t"
          - skip_if: "C{k} in [U, V, C{1..k-1}]"
      - id: "close_{k+2}"
        op: exists
        src: "C{k}.out"
        node: U
        edge: "c{k+2}"
        constraints:
          - skip_if: "c{k+2}.t <= h{k}.t"
          - break_if: "c{k+2}.t >= t"
        emit: {size: "{k+2}"}
```

Predicates are always written as quoted strings. That is required wherever a predicate contains a
flow indicator (`[ ] { } ,`), and the library quotes every predicate for consistency.

Hop k binds C_k, and `close_{k+2}` then emits a cycle of k + 2 edges (the trigger, V→C_1, …,
C_k→U). Excluding U from the bindings is safe, because the cycle that would close there is the one
`close_{k+1}` already counted.

`tcyc.yaml` needs `K_MAX ≥ 3`: `hop_1` and `close_3` are outside the repeat, so at `K_MAX = 2`
it would still count 3-cycles.

### 6.9 Mapping from the paper's YAML

| Paper [P §5.3] | This DSL | Why |
|---|---|---|
| `src` / `dst` / `dst_var` / `constraints` | Same names | Fidelity. |
| `e0.start = find_Start(t−δ, L)` but loops start at the list head | `window:` plus automatic `find_start` | Fixes [P §9.7] and the digest note in §5.3. |
| The edge of a scan is implicit (`e0`, `e1`) | Named with `edge:` | Constraints need to refer to it. |
| `sg_tx.add((e1,e2))` then `len(sg_tx) > 2` | `output: I` with `emit {size: len(I)}`, min 2, over distinct nodes | GFP parity; resolves [P §9.6] (edges vs nodes). |
| 4-cycle with `src: N0.out_neigh` | `tcyc` family with N1.out-style chaining | Resolves [P §9.1–9.4]. |
| Ops `union`, `difference`, `differentiate` | Reserved (E401). `differentiate` behaviour = `skip_if` node filters | Unused by any evaluated feature [P §8]. |

Appendix B lists the full set of deviations.

---

## 7. Compiler (`blzc`)

### 7.1 Pipeline

1. **Parse.** yaml-cpp, keeping positions.
2. **Expand.** Unroll `repeat`; substitute consts.
3. **Resolve.** Scopes: trigger, then each binding level.
4. **Type-check.**
5. **Classify constraints** (§6.4).
6. **Plan** (§7.3).
7. **Lower** to IR (§7.2).
8. **Emit** the CPU and CUDA sources (§7.4, §7.5).
9. **Write** the generated sources and their registry entry.

`blzc` is a host tool built first. CMake runs it over `patterns/*.yaml` at build time
(`add_custom_command`) and writes into `${build}/generated/`. Generated files are not committed.
They are covered by snapshot tests instead (§13.4).

### 7.2 IR

```
Pattern   { name, block, window{span_param, lower_open}, visibility,
            trigger{e0, src_var, dst_var}, trigger_filters[], levels[] }
Level     { binding: Stage | ROOT, checks[]: Stage }   // checks run inside this level
Stage     { id, op, binding, operands[]: Operand, bind_var?, output_var?, element_var?,
            probe_node?, multiplicity, node_filters[]: Pred, gate, emit?, exports[], notes[] }
Operand   { node_var, dir(in|out), edge_var, lo[]: TimeBound, hi[]: TimeBound,
            lo_incl[], hi_incl[]: TExpr, access }
TimeBound { expr: TExpr, strict: bool, from_window: bool }
```

The time bounds live on the operand (P3). An intersection has two scanned edges, each with its
own bounds. The planner rewrites them in inclusive form, since timestamps are integers: strict
lower x becomes x + 1 and strict upper x becomes x − 1. It drops duplicates, so the scan range is
`ts ≥ max(lo_incl)`, `ts ≤ min(hi_incl)`, plus visibility. Implementation: `src/ir/ir.hpp`;
`blzc --dump-ir` prints it, and `tests/specs/ir/` holds the library's dumps.

### 7.3 Planner decisions (each is recorded as a comment in the generated code)

- **Index choice.**
  - Scans (`for_all`, `count`, the scanned side of `intersection`) use T-CSR with distinct-first.
  - Probes (`exists`, the probed side of `intersection`) use P-CSR.
- **Intersection side.** With no cross-operand constraint (always true in v1), the generated code
  computes both window sizes at runtime, scans the smaller side and probes the larger one. This is
  the paper's degree-based ordering [P §5.4].
- **No materialization.** When a set is used only through `len()`, the code counts instead of
  materializing it [P §5.4].
- **Effective bounds.** Multiple lower bounds become one start computed with `max`. Upper bounds
  are folded into the range end when they are time or eid thresholds; this avoids a per-element
  `break`.

### 7.4 CPU backend (OpenMP)

The emitter produces two pieces:

- `template<class Sink> BLZ_HD void <pattern>_trigger(const GraphView&, const Params&, uint32_t e, Sink&)`
  — the loop nest for a single trigger, built only from §5.3 primitives. It is shared verbatim with
  CUDA.
- `void <pattern>_cpu(const GraphView&, const Params&, std::span<const uint32_t> triggers, BlockOut&)`
  — an OpenMP driver: `#pragma omp parallel for schedule(dynamic, chunk)` over triggers. Each
  trigger writes only its own row, so no atomics are needed.

Scheduling option `--schedule`:

- `dynamic` (default): chunk = 256.
- `sorted`: triggers ordered by estimated work, descending. The estimate is the first scan's window
  size, computed with two binary searches. This is the paper's degree-based balancing [P §5.4].

As implemented (P4):

- **Files.** For each spec `<stem>.yaml`, `blzc` writes `<stem>.hpp` and `<stem>_cpu.cpp`.
  `<stem>.hpp` holds `<pattern>_trigger` and `<pattern>_work`, both `BLZ_HD`, plus the declaration
  of `<pattern>_cpu`. `<stem>_cpu.cpp` holds the OpenMP driver. The driver takes `chunk` as a
  fifth argument.
- **Names.** Every DSL name gets a kind prefix in C++: `v_` for nodes, `ts_` for witness times,
  `i_` for count outputs, `len_` for set sizes, `p_` for parameters, and `r_` / `j_` for ranges
  and loop indices. A DSL name therefore cannot collide with the fixed locals `g`, `p`, `e`, `t`
  and `out`.
- **Lowering.** A binding stage opens a loop. Its level's checks run inside it in spec order, and
  the next level nests inside them. A node filter is a `continue`, or for an element a rejected
  element. A failed gate skips the rest of its level: `continue`, or `return` at the root. A
  check stage emits only when it passed (`exists` found a witness, `count` or a check
  intersection is non-zero), and the emit runs when `size ≥ max(min_size, 2)`.
- **Intersection.** Both ranges are computed, and one loop scans the smaller one through a
  `TCsrView&` and probes the other through a `PCsrView&`, so the body is emitted once.
- **Work estimate.** `<pattern>_work` is the size of the plan's first scan, from the trigger
  node, using only the bounds that do not depend on a binding (t, parameters and consts). The
  result is a superset of the real range.
- **Sorted schedule.** A stable counting sort on the work estimate (descending, ties in eid
  order) and chunk 16. Its cost, the estimates plus the sort, is part of the mining time.
- **Blocks with several patterns.** The patterns of a block (`sg_gather`, `sg_scatter`) run one
  after the other over the same trigger list and add into the same rows.
- **Start-up check.** E402 for every registered pattern against the feature config. The
  config's `k_max` must also equal the compiled `K_MAX` of every pattern that declares it,
  because `K_MAX` is a compile-time constant (§4.6).

Expected shape of the generated code for `sg_gather`, for illustration only:

```cpp
// generated from patterns/sg_gather.yaml — do not edit
template <class Sink> BLZ_HD inline void sg_gather_trigger(const GraphView& g, const Params& p, uint32_t e, Sink& out) {
  const uint32_t X = g.src[e], W = g.dst[e]; const int32_t t = g.ts[e];
  if (X == W) return;
  const int32_t lo_t = t - p.delta_sg;                                   // closed lower bound
  const Range A = g.in.range(X, lo_t, /*strict*/false, Upper::eid(e));  // stage `sources`
  for (uint32_t i = A.lo; i < A.hi; ++i) {
    if (!g.in.distinct_first(i, A.lo)) continue;
    const uint32_t S = g.in.nbr[i];
    if (S == X || S == W) continue;
    // stage `intermediates`: |S.out ∩ W.in \ {W}|, plan: scan smaller, probe larger, count only
    const Range B = g.out.range(S, lo_t, false, Upper::eid(e));
    const Range C = g.in.range(W, lo_t, false, Upper::eid(e));
    uint32_t n = 0;
    if (B.size() <= C.size()) { for (uint32_t j = B.lo; j < B.hi; ++j) {
        if (!g.out.distinct_first(j, B.lo)) continue; const uint32_t Y = g.out.nbr[j];
        if (Y == W) continue; n += g.pout.exists(Y, W, lo_t, false, Upper::eid(e)); } }
    else { /* mirror: scan C (W.in), probe S.out via pout.exists(S, Y, ...) */ }
    out.emit(Block::sg, n);                                              // min-size and bins handled in emit
  }
}
```

### 7.5 CUDA backend

- **Same body.** The `<pattern>_trigger` body is compiled as device code; this is why it is
  `BLZ_HD`.
- **Light kernel.** Thread-per-trigger.
- **Heavy kernel.** Warp-per-trigger. The emitter derives it mechanically for any spec:
  - **Root-level check stages** (those before the first binding stage, such as tcyc's `close_2`)
    run once, on lane 0.
  - **The outermost binding stage** has its scan positions split across the 32 lanes with a
    stride. `distinct_first` is a per-position test, so the split is safe. Each lane runs every
    stage nested inside its own bindings and accumulates their emits into lane-local bin counts.
    Lanes own disjoint bindings, so their instances are disjoint, and the bin counts are summed
    across the warp.
  - **A spec with no binding stage** (fan and degree, whose only stage is a root-level `count`):
    the lanes split the count scan, the **raw count** is summed across the warp, and lane 0
    performs the single emit. Partial counts are never binned.
  - Lane 0 writes the row.
- **Bucketing.** A work-estimate kernel computes the first scan's window size per trigger. Triggers
  are then partitioned into light and heavy lists at threshold `H`. The default is 32; P5 tunes it
  per GPU and records it.
- **Memory.** Graph arrays are copied once and reused across blocks. Each block's output buffer is
  allocated, filled and copied back in turn.
- **Timing.** Host-to-device copies, kernel time (cudaEvents) and device-to-host copies are timed
  separately (§12.1).
- **Targets.** sm_75 and sm_80 from the same source. No architecture-specific intrinsics beyond
  warp shuffles.

As implemented (P5):

- **Files.** For each spec, `blzc` also writes `<stem>_cuda.cu`. It is always generated and
  snapshotted, and compiled only with `BLZ_ENABLE_CUDA`. It holds `<pattern>_trigger_warp`, the
  heavy kernel's body, and two host launchers, `<pattern>_cuda_bucket` and `<pattern>_cuda`. They
  instantiate the kernel templates of `include/blz/cuda_kernels.cuh` with the pattern's light body
  (`<pattern>_trigger`, the very text the OpenMP driver runs), its heavy body and its work
  estimate. `<stem>.hpp` declares the launchers.
- **Light kernel.** 128 threads per block. Each thread runs one trigger of the light list and
  emits straight into the trigger's row through `RowSink`.
- **Heavy kernel.** 8 warps per block, one trigger per warp. The emitter's warp mode changes only
  the root level and the first binding stage:
  - The first binding stage's scan (`for_all` or a binding `intersection`) visits positions
    `lo + lane`, `lo + lane + 32`, …. `distinct_first` still tests against `lo`.
  - With no binding stage, the root count named by the plan's split stage uses the strided
    `count_distinct` / `count_edges` (§5.3) and `cuda::warp_sum`, before anything reads it.
  - A root check that has no gate and that no other stage reads (tcyc's `close_2`) runs on lane 0
    only. Any other root check is evaluated by every lane: its inputs are the same in every lane,
    so its result and its gate are too, with no broadcast. It emits on lane 0 only. This extends
    "root checks run once, on lane 0" to root checks whose results later stages read.
  - Emits go to the warp's 64-bit histogram in shared memory (shared `atomicAdd`). After the
    body, the lanes add the histogram into the row (lane i takes bins i, i + 32, …), saturating at
    `UINT32_MAX` and counting the lost emits. This equals `RowSink`'s one-at-a-time saturation.
- **Bucketing.** Per pattern, a kernel writes `flags[i] = work(trigger i) ≥ H`. Then
  `cub::DevicePartition::Flagged` puts the heavy triggers first, in eid order, and the light ones
  after them, in reverse eid order. The heavy count is read back to size the heavy launch.
  H = 0 (every trigger heavy) and H = `all` (`UINT32_MAX`, every trigger light) skip the
  estimates.
- **Memory.** `DeviceGraph` (`src/runtime/cuda_backend.cu`) copies the edge arrays and both
  directions of T-CSR and P-CSR into one allocation, from pageable memory: 329 MB on HI-Small.
  `CudaMiner` owns the scratch buffers (output, trigger ids, flags, partition, CUB temp storage)
  and reuses them across blocks. `blz mine` and `blz bench` copy each block back into pinned host
  memory.
- **Timing.** cudaEvents on the miner's stream measure:
  - the kernel time: output zeroing + bucketing (including the heavy-count read-back) + light +
    heavy, summed over the block's patterns (the host's timing bookkeeping between patterns is
    not in it);
  - the D2H time, separately;
  - and the host wall time of the whole call.

  The graph upload is timed on its own. `blz bench --backend cuda` takes `reps` fresh uploads
  after a warm-up, and T_BlazingAML(block) = median upload + kernels + D2H (§11.2).
- **Threshold.** `blz mine --heavy-threshold H` (default 32). `blz bench --backend cuda --heavy
  0,8,…,all` times every block at each H. `bench/e5_report.py` takes the H with the smallest
  kernel time summed over the blocks as the GPU's tuned H, and records it (`tuned`) with each
  block's own best.
- **Limits.** 1 to 768 bins per block, because a block's shared histograms (8 warps × 8 B × bins)
  must fit in 48 KB.

### 7.6 Registry

`blzc` also generates `registry.cpp`, which defines
`const PatternInfo kPatterns[] = {{name, block, params[], cpu_fn, cuda_fn}, ...}`. At start-up,
`blz` checks every pattern against the feature config, including E402.

As implemented (P4), `PatternInfo` (`include/blz/registry.hpp`) holds `name`, `block`,
`block_loc` (the `file:line:col` of the spec's `block:` value, used in E402), `file`,
`span_param` (the one runtime parameter, set from the block's `window`), `consts`, `cpu` and
`work`. P5 adds `cuda_bucket` and `cuda`, the CUDA launchers. `registry.cpp` points at them
only when `BLZ_WITH_CUDA` is defined (a `rel-cuda` build); otherwise they are null.
`registered_patterns()` returns the table in the order of `blzc`'s spec arguments.

---

## 8. Runtime (`blz`)

- **Block outputs.** Each block is written as `<out>/<block>.npy`: `uint32[m, bins]`, C order.
  A sidecar `features.json` holds column names, config, the git SHA, backend and timings.
- **`--triggers`.** `all` (default) or `train` / `test` / `range:a:b`. Every trigger always sees
  the full graph; this option only selects which rows are computed.
- **Determinism.** Outputs MUST be identical across thread counts, schedules and backends. They are
  integers, so this is checked by hash (§13.5).

---

## 9. Reference oracle (`src/oracle/`)

The oracle is hand-written straight from §4, separately from the DSL and compiler.

- **Structures.** It uses only per-node `std::vector<uint32_t>` lists of edge ids in eid order, plus
  `std::set` / `std::map` for deduplication. It MUST NOT use T-CSR, P-CSR or `prev_same`.
- **Per trigger** it:
  1. builds the window-visible in- and out-edge lists of every node it touches, using naive
     filtering;
  2. computes each definition literally:
     - fan and degree: sets and counts;
     - SG: an explicit set intersection for each s (gather role) and each w (scatter role);
     - tcyc: DFS over distinct node sequences from v. For each prefix it carries the full set of
       arrival times at the prefix's last node over every valid assignment of parallel edges, so
       the search over parallel edges is exhaustive. It deliberately does not reduce that set to
       its minimum (the greedy rule), so that the rule itself gets tested. The DFS is pruned,
       without changing the result, by each node's latest departure time towards u inside the
       window (a backward search from u).
- **Parallelism.** OpenMP over triggers is allowed. HI-Small MUST finish within 2 h on `local`.
  If it does not, the oracle MAY use the greedy rule for tcyc only after the exhaustive version has
  been validated on all fuzz graphs (§13.3) and on a 1% random trigger sample of HI-Small.

---

## 10. Classifier

### 10.1 Feature sets (the Table 2 columns, added cumulatively [P §6])

| Set | Columns |
|---|---|
| `xgb_only` | `src_id`, `dst_id` (dense node ids, as float). Per the Table 2 caption [P §9.19, D-10]. |
| `fan` | + fan_in, fan_out blocks |
| `fan_deg` | + deg_in, deg_out |
| `fan_deg_cyc` | + tcyc |
| `all` | + sg |

Columns are `src_id`, `dst_id`, then the blocks in the order the sets add them (fan_in, fan_out,
deg_in, deg_out, tcyc, sg), so each set's columns are a prefix of the next set's.

### 10.2 XGBoost (pre-declared; no tuning on test)

- **Default config [D-11].** XGBoost library defaults, as an untuned `XGBClassifier()` would use
  them. The paper's precision ≈ 0.95 / recall ≈ 0.31 profile [P §7.2] is consistent with
  unweighted defaults at a 0.5 threshold.
  - `objective=binary:logistic`, `tree_method=hist`, `max_depth=6`, `eta=0.3`,
    `num_boost_round=100`, `min_child_weight=1`, `subsample=1`, `colsample_bytree=1`, `lambda=1`,
    `max_bin=256`, `scale_pos_weight=1`, `seed=0`.
  - Decision threshold 0.5: a row is predicted laundering when its probability is **> 0.5**
    (`XGBClassifier.predict`'s rule).
  - Device: CPU [D-13].
- **Data path.** `XGQuantileDMatrixCreateFromCallback`, with row batches of 262,144 converted to
  float32 from the `.npy` blocks (memory, §5.4). The steps:
  1. Build the training matrix from train rows only.
  2. Build the test matrix with that training handle passed as `ref`, so that the test data are
     quantized with the training cut points.
  3. Train on the train split; predict on the test split.
- **Sensitivity grid (E6, reported in full, never used to pick a headline):**
  `scale_pos_weight ∈ {1, 5}`, `max_depth ∈ {6, 10}`, `num_boost_round ∈ {100, 500}`.

### 10.3 Metrics

Reported per run:

- the F1 of the laundering class, on the paper's ×100 scale;
- precision and recall;
- the confusion matrix (the Table 3 layout);
- PR-AUC, as average precision: Σₖ (Rₖ − Rₖ₋₁)·Pₖ over the distinct scores in decreasing order,
  with tied scores forming one step (sklearn's `average_precision_score`).

Each is computed in C++ (`src/ml/metrics.cpp`) and written to the results JSON.
`bench/xgb_xcheck.py` retrains a set with Python XGBoost from an in-memory `DMatrix` as an
independent check of the C API path.

---

## 11. GFP harness (`tools/gfp/`, Python 3.12, uv)

### 11.1 Parity mode

`parity.py --edges edges.npy --block {fan|degree|sg|tcyc} --tw <s> --bins 2..33 --out gfp_<block>.npy`

- Runs one GFP instance per block, with only that pattern enabled, `vertex_stats=False`,
  `num_threads=1` and `time_window=-1`.
- Calls `transform()` **once per edge in stream order**, which is batch size 1, on a fresh
  preprocessor. Batch mode changes semantics: edges in a batch see later edges, and every
  participating edge is credited [X A28]. Batch mode is therefore never used for parity.
- Never calls `fit()` or `fit_transform()`, and never sends an edge twice. snapml ≥ 1.14.1
  double-counts edges that are already in the graph [X A29].
- Sends `ts + 1` [D-12]. snapml drops every temporal cycle through a node that has an in-window
  self-loop at exactly ts 0 [X A33]. Every rule of §4 is translation-invariant and ts 0 is only
  this repo's epoch (D-1), so the shift removes the artifact and changes nothing else.
- Writes `uint32` blocks shaped like `blz`'s, so `blz compare` can diff them.
- Expected runtime is minutes, dominated by Python call overhead.

`blz compare a.npy b.npy [--graph g --tw δ] [--report r.json --name key]` prints (and with
`--report`, merges into that JSON under `key`):

- per-column mismatch counts;
- the first 20 mismatching eids with both rows;
- a mismatch histogram by trigger property: self-loop endpoint present, tie with the previous edge,
  window-boundary edge.

### 11.2 Timing mode

`bench_gfp.py` times only `transform()`, with each pattern group enabled alone. The grid:

- `num_threads ∈ {1, 2, 4, 6, 12}` on `local`, or nproc on `a30`;
- `batch ∈ {2048, 65536}`.

**Each timed run is one complete stream traversal on a fresh, empty GFP instance.** This also
applies to the warm-up. Construction and `set_params` happen outside the timer, there is no
`fit()`, and every edge is sent exactly once. Reusing an instance changes the workload [X A29].

It reports every point. `T_GFP_best` = the minimum over batch sizes at the highest thread count.
P4 runs a reduced grid, recorded as D-14.

    speedup = T_GFP_best / T_BlazingAML

- On CPU, `T_BlazingAML` is the mining time (§12.1) for that block at the given thread count.
- On GPU, it is H2D + kernels + D2H for that block, with the graph upload included.

The one-off index build (§5) is shared by all blocks. It is reported next to the speedups and is
not included in `T_BlazingAML`.

Because batch semantics differ from ours (§11.1), result tables MUST carry a caveat: the timings
compare equal work, not identical outputs.

### 11.3 Fuzz versus GFP

`fuzz_vs_gfp.py` runs `blz fuzz`, which writes the §13.3 graphs and the oracle's features
(bins 2..33, tcyc uncapped), then replays each graph into a fresh GFP in stream mode and diffs every
row. It is used in P1 to settle
the SG self-loop rules and to confirm Appendix A at scale.

### 11.4 Probe suite

`probes.py` re-runs every case of Appendix A. It writes `tests/golden/gfp_probes.json`
(edges + params → expected per-edge histograms), which is committed. The C++ tests consume it, so
the GFP-derived behaviour is checked without Python at test time. A28 (batch mode), A29
(the double-count bug) and A33 (the ts-0 artifact) document GFP itself and are not part of the
golden set. Every probe stream is in ts order, as `blz` streams are (§3.3).

---

## 12. Experiments and reporting

### 12.1 Timing method

- Load and index build are timed separately from mining.
- **Mining time** = wall clock of the trigger loop for one block, all triggers.
- 1 warm-up run, then 5 timed runs; report the median and min/max.
- CPU: `OMP_PROC_BIND=close`, `OMP_PLACES=threads`. Thread counts 1, 2, 4, 6, 8, 12; 8 and 12 use
  SMT.
- GPU: kernel time from cudaEvents, plus H2D and D2H times, plus end-to-end wall time.
- Every results JSON records:
  - git SHA and dirty flag;
  - compiler and CUDA versions and build flags;
  - `lscpu`, `nvidia-smi -q` and free memory;
  - the config hash.

### 12.2 Experiment list

| ID | What | Output |
|---|---|---|
| E0 | Dataset profile:<br>• degree and distinct-neighbour distributions<br>• window occupancy at 6 h and 24 h<br>• pair count and multiplicity<br>• timestamp-tie histogram<br>• self-loop endpoints<br>• GFP tcyc length histogram (bins 2..33) | `results/e0_profile.json` |
| E1 | Parity: oracle vs GFP stream mode (AC2); generated CPU/CUDA vs oracle (AC1) | `results/e1_parity.json` |
| E2 | Table 2 HI-Small row with the default config (`bench/e2.sh`) | `results/e2_table2.json` and `results/e2_table2.md`, next to the paper's numbers |
| E3 | Table 3 confusion matrix for `all` | in `e2` |
| E4 | CPU: per-block mining time for threads {1..12}, plus speedup vs GFP best (the Figs. 6–8 analogue) | `results/e4_cpu_<host>.json` |
| E5 | GPU: per-block time on the 1650S and A30, with the light/heavy split recorded | `results/e5_gpu_<host>.json` |
| E6 | Sensitivity: the XGB grid (§10.2); `xgb_only` + tx id [P §9.19] | `results/e6_sens.json` |

`bench/report.py` (Python, matplotlib) renders the tables and plots from the results JSON.
Plots are written to `results/plots/`.

---

## 13. Testing

### 13.1 Unit tests

- CSV field scanner, timestamp parsing and dictionaries, run on fixtures.
- Stable sort and eid assignment.
- Node-id assignment order.
- `.blzg` round-trip.
- T-CSR and P-CSR construction; `prev_same`.
- Every §5.3 primitive against a naive scan, on random lists that include ties and duplicates.
- Bin lookup: edges of bins, the open last bin, sizes below `b_0`, min size 2.
- Pattern registration against a feature config that lacks a block → E402.

### 13.2 Golden tests

`tests/golden/gfp_probes.json` (§11.4) is evaluated against the oracle, the generated CPU code and
(when built) the generated CUDA code. The match MUST be exact.

### 13.3 Differential fuzzing

The generator draws random multigraphs:

- n ∈ [3, 40] nodes, m ∈ [5, 400] edges;
- timestamps uniform in [0, T) with T ∈ [2, 60], so that ties are common;
- self-loop probability 0.1 and parallel-edge probability 0.2 (a parallel edge repeats a random
  earlier pair);
- one δ ∈ [1, T] for every block, so window boundaries are hit often;
- seed-indexed and reproducible: `src/graph/fuzz_gen.cpp`, a splitmix64 stream seeded with the
  seed, with no `<random>` distributions.

AC1 compares oracle, CPU and CUDA on ≥ 10,000 seeds. `fuzz_vs_gfp.py` compares the oracle with GFP
on ≥ 2,000 seeds in P1.

### 13.4 Compiler tests

- **Parser and expansion** (`repeat` substitution and ranges).
- **Invalid-spec suite** (`tests/specs/invalid/*.yaml`). Each file names its expected error code
  and position in a leading comment. Together they cover E101–E401.
- **Expansion fixture.** Expanding `tcyc.yaml` with `K_MAX = 5` MUST equal
  `tests/specs/expanded/tcyc_k5.yaml` (§6.5).
- **Snapshot tests.** The generated `.cpp` and `.cu` for each library spec are compared with
  `tests/snapshots/`. A changed snapshot needs an explicit `--update-snapshots`. P4 has the CPU
  files and `registry.cpp`; P5 adds the `.cu` files. The fast test `blzc_snapshots` runs
  `blzc patterns/*.yaml --snapshots tests/snapshots` in the source directory, which is also how
  the build generates code, and the same command with `--update-snapshots` rewrites them.

### 13.5 Integration tests (label `slow`, run manually)

- HI-Small, all blocks: oracle equals CPU equals CUDA.
- A run at 1 thread and a run at 12 threads give the same hash.
- The invariants of §3.5 hold.

CTest labels are `fast` (default, under 2 min) and `slow`. There is no CI remote yet; `ctest -L fast`
MUST pass before every commit.

---

## 14. CLI

```
blz prepare  --csv dataset/HI-Small_Trans.csv --out data/hi_small.blzg
blz export-edges --graph data/hi_small.blzg --out data/edges.npy
blz stats    --graph data/hi_small.blzg --out results/e0_profile.json
blz mine     --graph data/hi_small.blzg --config configs/features_default.yaml
             --backend oracle|cpu|cuda [--threads N] [--schedule dynamic|sorted]
             [--heavy-threshold H] [--blocks fan_in,...] [--triggers all|train|test|range:a:b]
             --out features/<run>/
blz compare  <a.npy> <b.npy> [--graph ... --tw δ] [--first-eid E] [--report r.json --name key]
blz fuzz     --seeds a:b [--bins 2..33] [--k-max 64] --out data/fuzz/
blz train    --graph ... --features features/<run>/ --set xgb_only|fan|fan_deg|fan_deg_cyc|all
             --xgb configs/xgb_default.yaml --out results/<run>.json
blz bench    --graph ... --config ... --backend cpu --threads 1,2,4,6,8,12 --reps 5 --out ...
             [--schedule dynamic,sorted] [--blocks ...] [--name key]
blz bench    --graph ... --config ... --backend cuda --heavy 0,8,16,32,...,all --reps 5 --out ...
             [--blocks ...] [--name key]
blz fuzz-diff --seeds a:b [--backend cpu|cuda] [--heavy 0,4,all] [--bins 2..33]
             [--report r.json --name key]
blzc         patterns/<p>.yaml... --out <dir> [--check-only | --dump-ir | --expand] [--const NAME=V]
blzc         patterns/<p>.yaml... --snapshots <dir> [--update-snapshots]
```

`blz fuzz-diff` (P4) compares the generated CPU backend with the oracle on the §13.3 graphs,
every row of every block, under both schedules (AC1). With `--backend cuda` (P5) it compares the
CUDA backend instead, once per light/heavy threshold of `--heavy`. `blz bench` merges its result
into `--out` under `--name` (default: the backend's name).

---

## 15. Repository layout

```
CMakeLists.txt  CMakePresets.json  cmake/deps.cmake
patterns/            v1 YAML specs (§6.8)
configs/             features_default.yaml, features_parity.yaml, xgb_default.yaml, xgb_grid.yaml
include/blz/         public headers: types, graph view, primitives (BLZ_HD)
src/common/          timer, logging, npy writer, json
src/io/              CSV scanner, .blzg reader/writer
src/graph/           stream sort, node map, T-CSR/P-CSR build, device mirror
src/dsl/             YAML → AST, expression parser, repeat expansion
src/ir/              resolve, type-check, constraint classification, planner
src/codegen/         cpu_emitter, cuda_emitter, code writer, registry emitter
src/runtime/         block output, binning, OpenMP driver, CUDA driver
src/oracle/          naive reference (§9)
src/ml/              XGBoost C-API wrapper, metrics
src/apps/            blz.cpp, blzc.cpp
tests/               unit/, golden/, specs/{invalid,expanded,ir}/, snapshots/, fuzz/
tools/gfp/           pyproject.toml (uv), probes.py, parity.py, bench_gfp.py, fuzz_vs_gfp.py
bench/               run scripts, report.py
results/             committed small JSON/markdown results and plots
```

`.gitignore` adds `build/`, `data/` and `features/`.

---

## 16. Milestones

Each phase lists its exit criterion.

### P0 — Foundations

- CMake presets build on `local`, including `rel-cuda` against `/opt/cuda`.
- Repository skeleton, presets and dependencies pinned.
- `blz prepare` with the §3.5 invariants passing; `.blzg`; `export-edges`.
- `blz stats` (E0, without the GFP part).
- `tools/gfp` uv project, `probes.py`, and the golden JSON committed.
- The E0 GFP tcyc length histogram.

**Exit:** the invariants pass, and the golden probes are regenerated and equal Appendix A.

### P1 — Oracle and parity (semantics freeze)

- Oracle for all v1 blocks.
- Golden tests pass.
- `fuzz_vs_gfp` reaches 2,000 seeds with no unexplained diff.
- `parity.py` runs on HI-Small for all blocks.

**Exit:** AC2. §4 and Appendix A are updated, and the semantics are **frozen**: later changes need
a new `D-n`.

**Outcome (2026-11-10).** The oracle matches the golden probes, GFP on 2,000 fuzz graphs
(403,557 edges) and GFP on every HI-Small row of all six blocks (`results/e1_parity.json`). The
one GFP divergence found, the ts-0 self-loop artifact, is A33 and is handled by D-12. HI-Small
SG is sparse (808 instances in the parity bins), so the fuzz graphs carry most of the SG
evidence. The oracle mines all of HI-Small in under 2 minutes on `local` (12 threads), with a
peak RSS of 0.8 GB. §4 is frozen.

### P2 — Classifier with oracle features

- XGBoost wrapper and metrics.
- E2 and E3 using oracle features.

**Exit:** AC3, the first Table 2 row. If `all` is outside [36.6, 56.6], i.e. off by more than
10 points from 46.6, open decision gate DG-2 (§17) before going further.

**Outcome (2026-11-10).** `blz train` (XGBoost 3.4.2 C API, QuantileDMatrix callback) and the
metrics are in; `bench/e2.sh` produces `results/e2_table2.{json,md}`. F1 ×100, paper in
parentheses: xgb_only 0.0 (11.1), fan 0.9 (23.2), fan_deg 6.1 (38.4), fan_deg_cyc 6.6 (42.6),
all 6.4 (46.6). Table 3 analogue for `all`: TP 60, FN 1,737, FP 18, TN 1,013,854 (paper 558,
1,239, 31, 1,013,841). Python XGBoost 3.4.1 from a plain `DMatrix` (`bench/xgb_xcheck.py`)
gives the same confusion counts for `all` and `xgb_only`, so the C API path is not the cause.
The model has the paper's high-precision profile (0.77) but a tenth of its recall (0.03).
`all` is outside [36.6, 56.6]: **DG-2 is open and parked** (§17); the gap is reported, not closed. Training takes under 20 s per set on
12 threads.

### P3 — DSL front end and IR

- Parser, expansion, resolve, type-check, classification, planner.
- The invalid-spec suite.
- `blzc --check-only` passes on the library.

**Exit:** AC5.

**Outcome (2026-11-11).** The front end is `src/dsl/` (YAML tree with positions, repeat
expansion, expression parser, schema check) and `src/ir/` (resolve, type-check, classification,
planner, IR dump). `blzc --check-only` accepts all seven library specs; it runs as the fast
`blzc_check_library` test. The invalid-spec suite has 53 cases (`tests/specs/invalid/`), each
checked for its code and its exact `file:line:col`, and together they cover E101–E401. E402 is
`check_registration` (`src/runtime/registration.cpp`) with a unit test; P4 wires it to the
generated registry. `tcyc.yaml` expanded at `K_MAX = 5` matches the hand-written
`tests/specs/expanded/tcyc_k5.yaml`. The planned IR of every library spec is snapshotted in
`tests/specs/ir/`. Rules this phase had to pin down are written into §6.3–§6.7 and §7.2: scope,
gate, `or` splitting, the trigger time in filters, E202/E204 coverage, and per-operand bounds.
Code generation (`blzc` without a mode flag) is P4.

### P4 — CPU backend

- Emitter, OpenMP driver, both schedules, registry, build integration.
- Snapshot tests.
- AC1 for CPU.
- E4 on `local`, with GFP timing on `local`.

**Exit:** AC1 (CPU) and E4 published.

**Outcome (2026-11-11).** For each library spec, `blzc` writes `<stem>.hpp` (the trigger
template and its work estimate, both `BLZ_HD` so P5 can reuse them) and `<stem>_cpu.cpp` (the
OpenMP driver). It also writes `registry.cpp`. The build runs `blzc` on the specs. The 15
generated files are snapshotted in `tests/snapshots/` (fast test `blzc_snapshots`).
`blz mine --backend cpu`, `blz bench` and `blz fuzz-diff` run the generated code over the shared
index (§5.3). Rules this phase pinned down are in §6.7 (E102 for duplicate pattern names, E501),
§7.4 and §7.6 ("As implemented").

**AC1 (CPU) holds** (`bench/ac1_cpu.sh`, `results/e1_parity.json`):

- On HI-Small, all six blocks equal the oracle bit for bit under two runs:
  - the default config, at 12 threads with `dynamic`;
  - the parity config, at 1 thread with `sorted`.
- On 10,000 fuzz graphs (2,053,897 edges), both schedules give 0 mismatching rows.

**E4** is in `results/e4_cpu_local.{json,md}`.

- With `dynamic`, the six blocks take 17.2 s at 1 thread and 2.67 s at 12 threads (6.4×).
  fan_out is about 63% of the total.
- `sorted` is slower for every block except tcyc (0.383 s against 0.394 s at 12 threads), so
  `dynamic` stays the default.
- GFP (snapml 1.17.2, reduced grid, D-14): `T_GFP_best` is 98.9 s for fan, 85.0 s for degree,
  31.7 s for sg and 18.0 s for tcyc.
- Speedups at 12 threads are 56× (fan), 409× (degree), 109× (sg) and 46× (tcyc). At 1 thread
  they are 9×, 73×, 16× and 6×.

Caveats (also under "Measurement notes" in `results/e4_cpu_local.md`):

- `local` is a shared desktop. At 12 threads, fan_out measured 1.21–1.70 s across runs. The table
  keeps the 1.70 s value from the full sweep, which is the conservative figure. The cores held
  about 3.8 GHz under load, so this is not throttling.
- GFP degree is bimodal on an idle machine: each run takes about 85 s or about 300 s. At batch
  2048 and 12 threads the runs took 84.9, 85.0 and 305.0 s. At batch 65536 they took about 296 s
  at 12 threads but 120 s at 4 threads. The degree speedup therefore depends on which mode the
  `T_GFP_best` runs hit.
- The GFP temp-cycle points overlapped an unrelated workload on 2 of the 12 hardware threads.
  One idle-machine run before the grid took 15.7 s at 12 threads (batch 65536), which gives
  about 40× rather than 46×. A re-run was also disturbed, so it is not used.

### P5 — CUDA backend

- Light and heavy kernels, bucketing, transfers.
- AC1 for CUDA on the 1650S.
- E5 on the 1650S and the A30, via a JarvisLabs runbook: `bench/a30.sh` copies the binary cache
  and the repo, builds `rel-cuda`, runs E5 (and E4 if the instance has > 12 vCPUs), fetches the
  results and stops the instance.

**Exit:** AC1 (CUDA) and E5 published. **v1 is done.**

**Outcome (2026-11-11).** For each library spec, `blzc` also writes `<stem>_cuda.cu`: the heavy
kernel's body (`<pattern>_trigger_warp`, derived from the same plan) and the launchers. The light
kernel runs the shared `<pattern>_trigger`. The kernel templates, the bucketing and the device
mirror are in `include/blz/cuda_kernels.cuh` and `src/runtime/cuda_backend.cu`. The `.cu`
files are in the snapshots. `blz mine`, `blz bench` and `blz fuzz-diff` take `--backend cuda`.
Rules this phase pinned down are in §5.3, §7.5 and §7.6 ("As implemented").

**AC1 (CUDA) holds on the 1650S** (`bench/ac1_cuda.sh`, `results/e1_parity.json`):

- On HI-Small, all six blocks equal the oracle bit for bit in three runs:
  - the default config at H = 32, which uses both kernels;
  - the parity config at H = 0 (every trigger on the heavy kernel);
  - the parity config at H = `all` (every trigger on the light kernel).
- On 10,000 fuzz graphs (2,053,897 edges), each mined at H = 0, 4 and `all`, 0 rows mismatch.
- The fast tests run the golden probes and 500 fuzz graphs on the GPU at three thresholds.

**E5 on the 1650S** is in `results/e5_gpu_local.{json,md}` (default config, H swept over
0, 8, …, 1024, `all`):

- The six blocks' kernels take 626 ms at the tuned H = 32 (the SPEC default), against 629 ms
  at H = 64, a tie. D2H adds 103 ms, and the graph upload (329 MB) takes 34 ms.
- Speedups over GFP best on the same machine (upload + kernels + D2H): fan 395×, degree 456×,
  sg 193× and tcyc 68× (about 60× with E4's idle-machine GFP tcyc time). Against the CPU
  backend at 12 threads they are 7.1×, 1.1×, 1.8× and 1.5×.
- Every block's output hash equals the CPU backend's in E4.
- tcyc is dominated by a few thousand heavy triggers (149 ms for 3,489 triggers): a warp splits
  only the first hop. This is what the P6 hybrid addresses.

**E5 and E4 on the A30** (`bench/a30.sh`, one run, about 35 minutes of instance time; instance
destroyed afterwards). The instance was a JarvisLabs container in IN2: an A30 with 56 SMs, driver
580, and 16 vCPUs of an AMD EPYC 7542. nvcc 12.6 came with the image; the runbook installed
g++-12. Results are in `results/e5_gpu_a30.{json,md}`, `results/e4_cpu_a30.{json,md}`,
`results/a30_env.txt`, `results/a30_run.log`, and `fuzz_cuda_vs_oracle_a30` in
`results/e1_parity.json`.

- AC1 holds on sm_80:
  - the 68 fast tests pass (no XGBoost tests, D-15);
  - 10,000 fuzz graphs at H = 0, 4 and `all` have 0 mismatching rows;
  - every E5 block's hash equals the CPU backend's on the same host.
- The six blocks' kernels take 84.4 ms at the tuned H = 64, against 84.7 ms at 32, a tie. D2H
  adds 57.5 ms and the graph upload 29 ms, so the transfers cost as much as the kernels.
- Speedups over GFP best (16 threads) on the same host are fan 1081×, degree 717×, sg 633× and
  tcyc 317×. Against the CPU backend at 16 threads they are 12.6×, 2.1×, 4.1× and 3.5×.
- E4 ran because the instance has 16 vCPUs. With `dynamic`, the six blocks take 22.3 s at 1
  thread and 1.50 s at 16 (14.8×).
- tcyc's heavy kernel still dominates its time (29 ms for 2,885 triggers).

The exit criterion is met: AC1 (CUDA) and E5 are published for both GPUs.

### P6 — Stretch (any order)

- Hybrid CPU/GPU for tcyc: heavy triggers on CPU, light ones on GPU, run concurrently [P §5.4].
- `credit=participants` and `visibility=timestamp`, with their F1 impact.
- `lc-cycle`.
- Cross-operand ordering in intersections, with an `sg_ordered` spec (Fig. 3 option 1).
- `union` / `difference`.
- Stack, bipartite and gather-scatter patterns.
- Streaming replay (`blz replay`: per-edge latency).
- Raw-attribute and vertex-statistics features.
- Other datasets.

---

## 17. Risks and decision gates

| ID | Risk / gate | Mitigation / rule |
|---|---|---|
| R1 | GFP has undocumented quirks, for example self-loops in SG (§4.5) and lc-cycle ties. | Probe- and fuzz-driven semantics (P1); AC2 requires every mismatch to be explained. |
| R2 | The paper's XGB config is unknown, so F1 may differ from 46.6. | A fixed default plus the full sensitivity grid is reported (E6). Nothing is selected on test. |
| R3 | tcyc enumeration blows up at hubs in a 24 h window. | The `K_MAX` cap and E0 profile. Sorted scheduling. The P6 hybrid. |
| R4 | `local` has little free RAM (≈ 3 GB). | Block-at-a-time mining and QuantileDMatrix streaming (§5.4). The A30 host is the fallback for training. |
| R5 | A JarvisLabs image's nvcc and GCC pair differs from `local`'s. | Code targets GCC ≥ 12 and CUDA ≥ 12.4 (§2.2). The CPU build never needs nvcc. |
| R6 | The comparison hardware is far smaller than the paper's. | Only same-machine ratios and scaling shapes are compared; no absolute claims. |
| DG-1 | E0 shows GFP tcyc cycles longer than `K_MAX` on > 0.1% of triggers. | Raise `K_MAX` to cover 99.9%, or to the maximum if that is ≤ 16. Above 16, decide between an explicit-stack lowering of `repeat` and accepting the mismatch, and record a `D-n`. |
| DG-2 | `all` F1 is off by > 10 points. | Before changing anything, check in order:<br>1. the split positives (1,797);<br>2. the feature column order;<br>3. the `credit=participants` variant (P6, pulled forward);<br>4. bins.<br>Record each result.<br>**Opened 2026-11-10 (P2: `all` = 6.4).** 1: 1,797 test positives, as expected. 2: GFP block order (sg before tcyc) gives the same confusion counts (`bench/xgb_xcheck.py`). 3: not run; it needs the P6 variant in the oracle and a new `D-n`. 4: parity bins 2..33 (194 columns, `features/oracle_parity`) give 4.6. Extra diagnostics (`bench/dg2/`, Python XGBoost, same params; test-split looks, so never a headline): IDs + raw attributes (amounts, currencies, format) 5.0; `all` + raw attributes 24.7 (TP 257, FP 31, FN 1,540); fan/degree with no window (all earlier edges; equal to the oracle on every edge of day 1) 7.9, but with 2,064 FP, the opposite of the paper's high-precision profile. **Parked 2026-11-11**: the gap is reported as is. Leads for later: GFP vertex statistics; GFP batch mode or `fit_transform` (credit to every edge plus visibility of later edges, [X A28, A29]); check 3. **Closed 2026-11-12 (SPEC v2 P8, AC-V2-2).** The pre-registered ladder (`results/dg2_ledger.md`) froze a recipe (`configs/dg2_frozen.yaml`: GFP batch pattern blocks, GFP vertex statistics, S2 raw columns, XGBoost 5/10/500; GFP-produced, not compiled) on validation only. E8 (`results/e8_table2_v2.md`) scored it once on test: 58.0, 60.6, 63.5, 63.9, 63.1 against the paper's 11.1, 23.2, 38.4, 42.6, 46.6; Table 3 TP 993, FP 360, FN 804 (precision 0.734, recall 0.553). Verdict **GAP** by overshoot (SPEC v2 D-21). `v1_default` reproduces E2 (6.4). Leads are in SPEC v2 P8. |

---

## Appendix A — GFP behaviour ledger (snapml 1.17.2, stream mode, batch size 1) [X]

Setup for every case unless stated otherwise:

- `num_threads=1`, `vertex_stats=False`;
- every pattern enabled with bins `[1..8]` and `lc-cycle_len=8`;
- tw = 100;
- edges written `src→dst@ts`, inserted in the listed order, one `transform` call each.

Only the non-empty outputs are listed. `{size: count}` gives the histogram.

| # | Edges | GFP output | Rule |
|---|---|---|---|
| A1 | 1→9@1, 2→9@2, 1→9@3, 3→9@4, 4→9@50 | @2: fan_in{2:1} deg_in{2:1}.<br>@3: fan_in{2:1} deg_in{3:1} deg_out{2:1}.<br>@4: fan_in{3:1} deg_in{4:1}.<br>@50: fan_in{4:1} deg_in{5:1}. | Fan counts distinct neighbours, including the trigger's; degree counts parallel edges; min size 2 (size 1 is never emitted, even with bin 1). |
| A2 | A1 with tw = 3 | @4: deg_in{3:1} (edges @2, @3, @4). @50: nothing. | The window drops t' ≤ t−δ: an open lower bound. |
| A3 | 1→9@0, 2→9@10 with tw = 9 / 10 / 11 | fan_in{2:1} only for tw = 11 | Fan and degree use an open lower bound. |
| A4 | 1→9@5, 2→9@5, 3→9@5 | fan_in{2:1}, then fan_in{3:1} | Same-ts edges inserted earlier are visible. |
| A5 | 0→1@1, 0→2@2, 0→3@3, 1→5@4, 2→5@5, 3→5@6 | 2→5: sg{2:1}. 3→5: sg{3:1}. The scatter edges get no sg. | Credit goes to the trigger only; size = number of intermediates; a growing pattern is re-counted at its new size. |
| A6 | 1→5@1, 2→5@2, 0→1@3, 0→2@4 | 0→2: sg{2:1} | The scatter role counts; no order inside SG. |
| A7 | 0→1@0, 0→2@5, 1→5@6, 2→5@10 with tw = 9 / 10 / 11 | sg{2:1} for tw = 10 and 11 | SG uses a closed lower bound. |
| A8 | 0→1@1, 0→1@2, 1→5@3, 2→5@4, 0→2@5, 1→5@6 | 0→2@5: sg{2:1}. 1→5@6: sg{2:1}. | Intermediates are distinct nodes; a parallel trigger re-counts the instance. |
| A9 | 0→1@1, 0→2@2, 1→0@3, 2→0@4 | no sg (tcyc{2:1} on 1→0 and on 2→0) | Requires s ≠ w. |
| A10 | 0→0@1, 0→1@2, 0→5@3, 1→5@4 | 1→5: sg{2:1} | s ∈ I when s→s and s→w exist. |
| A11 | 0→1@2, 0→5@3, 1→5@4 | no sg | Control for A10. |
| A12 | 5→5@1, 0→1@2, 0→5@3, 1→5@4 | no sg | w ∉ I. |
| A13 | 9→1@1, 9→2@2, 3→9@3, 4→9@4, 9→9@5 | 9→9: nothing | Self-loop triggers emit nothing. |
| A14 | 0→1@1, 1→5@2, 0→0@3, 0→5@4, 0→0@6 | 0→5: fan_in{2:1} fan_out{2:1} deg_in{2:1} deg_out{2:1}, no sg | In the gather role, s ≠ the trigger's source (s = u through the self-loop is excluded). Self-loops are excluded from fan and degree. |
| A15 | 0→1@5, 0→2@5, 1→5@5, 2→5@5 | 2→5: sg{2:1} | Ties are visible in stream order. |
| A16 | 1→5@1, 2→5@2, 1→6@3, 2→6@4, 0→1@5, 0→2@6 | 0→2: sg{2:2} | One instance per (s, w). |
| A17 | 0→1@1, 0→2@2, 3→1@3, 3→2@4, 1→5@5, 2→5@6 | 2→5: sg{2:2} | Gather role with several sources. |
| A18 | 10→11@1, 11→12@2, 12→10@3 | 12→10: tcyc{3:1} lcyc{3:1} | — |
| A19 | 3-cycles with ts (1,1,2), (1,2,2) and (1,1,1) | no tcyc | The order is strictly increasing. |
| A20 | 10→11@0, 11→12@5, 12→10@10 with tw = 9 / 10 / 11 | tcyc{3:1} for tw = 10 and 11 | tcyc uses a closed lower bound. |
| A21 | 10→11@1, 11→10@2 | tcyc{2:1} | 2-cycles count. |
| A22 | 10→10@1, 10→11@2 | nothing | No 1-cycles. |
| A23 | 10→11@1, 11→12@2, 10→11@3, 11→12@4, 12→10@5 | tcyc{3:1} | Distinct node cycles, not edge combinations. |
| A24 | 10→11@1, 10→13@1, 11→12@2, 13→12@2, 12→10@3 | tcyc{3:2} | Two node cycles. |
| A25 | 10→11@1, 11→12@3, 10→11@4, 12→10@5 | tcyc{3:1} | All parallel edges are searched (existence). |
| A26 | 30→31→…→35→30 at ts 1..6 | tcyc{6:1} | No small length cap. |
| A27 | 40→41@1, 41→40@2, 40→42@3, 42→40@4 | 41→40: tcyc{2:1}. 42→40: tcyc{2:1}. | Simple cycles only. |
| A28 | A5 and A1, each passed as **one** `transform()` batch to a fresh preprocessor | A5: all six edges get sg{3:1}.<br>A1: 1→9@1 gets fan_in{2:2, 3:1, 4:1}. | In batch mode, every edge of an instance found in the batch is credited, and edges see later edges of the same batch. Fan and degree instances are per-trigger snapshots, so earlier edges collect one per later trigger. Batch mode differs from stream mode and is never used for parity. |
| A29 | 8 unique edges: `fit(X)` then `transform(X)` (what `fit_transform` does), compared with a fresh `transform(X)` | 1.14.0: identical.<br>1.14.1 and later (still in 1.17.2): row 0 sg/tcyc `[4,0,0,6]` vs `[2,0,0,3]`. The exported graph still holds 8 edges. | **snapml bug, introduced in 1.14.1:** edges re-sent to `transform()` are double-counted. Bisected over 1.12.0–1.17.2; a brute-force count matches the fresh `transform()`. Never re-send edges, and never use `fit` or `fit_transform` (§11). |
| A30 | 5→5@1, 0→5@2, 1→5@3, 0→1@4 | 0→1: sg{2:1} | Scatter role: w ∈ I through w→w. |
| A31 | 0→5@2, 1→5@3, 0→1@4 | no sg | Control for A30. |
| A32 | 0→0@1, 0→5@2, 1→5@3, 0→1@4 | no sg | Scatter role: s ∉ I, even with s→s. |
| A33 | 13→13@0, 13→9@3, 9→13@4 (and 9 variants in `probes.py`) | 9→13: no tcyc. With the loop at ts 1, or every ts shifted by +1: tcyc{2:1}. | **snapml artifact:** a temporal cycle is dropped when any of its nodes other than the trigger's source u has an in-window self-loop at exactly ts 0 (also for intermediates; a loop at u, or at a node off the cycle, has no effect; other cycles of the same trigger are kept). Not reproduced by `blz`; the harness sends ts + 1 (D-12). Fan, degree and SG are unaffected (2,000 fuzz graphs). On HI-Small the shift changes no tcyc row. |

**Open items.**

- **O-2.** lc-cycle tie behaviour is irregular: (1,1,2) found, (1,2,2) not found, and the result
  depends on insertion order. Not needed for v1.

(O-1, the scatter-role exclusion, was resolved by A30–A32.)

---

## Appendix B — Deviations from the paper

| # | Paper | This spec | Reason |
|---|---|---|---|
| B-1 | SG is drawn with the trigger as the gather edge (Fig. 5a) or as the scatter edge (Fig. 4a) [P §9.5]. | SG = gather role + scatter role. | GFP parity [X A5, A6]; both figures are covered. |
| B-2 | `len(sg_tx) > 2` over edge pairs [P §5.3]. | `|I| ≥ 2` distinct intermediates. | GFP parity [X A5, A8]; resolves [P §9.6]. |
| B-3 | 4-cycle spec with unclear legs and no order [P §9.1–9.4]. | tcyc of lengths 2..K_MAX with strictly increasing timestamps. | GFP's temp-cycle [G, X]; the paper says its features equal GFP's. |
| B-4 | Printed loops ignore `e.start` [P §5.3 note]. | Every scan starts at `find_start`. | Correctness. |
| B-5 | The window lower bound is unspecified. | Per pattern: open for fan and degree, closed for SG and tcyc. | GFP parity [X A2, A3, A7, A20]. |
| B-6 | Stages output edge tuples (Fig. 5) or nodes (Alg. 1) [P §9.6]. | Node sets with earliest witnesses. | Alg. 1 plus exact existence semantics. |
| B-7 | "Update sg_tx features" may credit every edge of an instance. | Credit goes to the trigger (default); `participants` is a P6 variant. | GFP stream parity [X A5]. |
| B-8 | δ is unspecified [P §10]. | 6 h for SG and 24 h for the rest. | GFP's AML setup [G]. |
| B-9 | `break if e.t > t` (timestamp visibility). | Stream visibility (default); `timestamp` is a P6 variant. | GFP parity [X A4, A15]. |
| B-10 | 2× Xeon 8380 and an A40; six datasets; FraudGT. | Ryzen 5 3600, GTX 1650S and A30; HI-Small; no FraudGT. | Available hardware and the v1 scope. |

## Appendix C — Decisions

| ID | Decision |
|---|---|
| D-1 | Timestamps are int32 seconds since 2022-09-01 00:00, naive. |
| D-2 | Nodes are keyed by the raw (bank, account) strings. |
| D-3 | Dense node ids in order of first appearance in stream order, source before destination. |
| D-4 | Stable sort by timestamp; ties keep file order. |
| D-5 | Credit = trigger. |
| D-6 | Visibility = stream. |
| D-7 | `K_MAX = 10`, pending DG-1. |
| D-8 | Windows: SG 6 h; everything else 24 h. |
| D-9 | Bins as in §4.8. |
| D-10 | `xgb_only` uses `src_id` and `dst_id`. |
| D-11 | XGBoost library defaults, threshold 0.5. |
| D-12 | The GFP harness (parity and fuzz) sends ts + 1, to avoid the ts-0 self-loop artifact [X A33]. Our semantics keep the cycle. |
| D-13 | XGBoost is built CPU-only in every preset. Training takes < 20 s per set on `local` (P2), while the CUDA build of XGBoost takes hours there and would be rebuilt on every temporary A30 instance. |
| D-14 | The GFP timing grid (§11.2) is reduced. Both batch sizes run at the highest thread count, 3 runs each; that thread count defines `T_GFP_best`. The other thread counts use batch 65536 only, 1 run each, and there is no warm-up run. One GFP traversal of HI-Small takes 16–120 s per pattern at 12 threads (fan is the slowest), so the full grid with a warm-up and 5 runs would take about 10 h on `local`. The reduced grid takes about 1.5 h. `bench_gfp.py` flags restore the full grid. |
| D-15 | The A30 build (`bench/a30.sh`) is `rel-cuda` with `CMAKE_CUDA_ARCHITECTURES=80` and `BLZ_WITH_XGBOOST=OFF`. Nothing trains on the A30, and both XGBoost and an sm_75 pass would be rebuilt on every temporary instance (as for D-13). The source tree is shipped without `.git`; `BLZ_GIT_SHA` and `BLZ_GIT_DIRTY` carry its commit into the results. |
| D-16 | On the A30, GFP is timed only at nproc threads, with both batch sizes and 3 runs each. That is all T_GFP_best needs; D-14's scaling points are not repeated on paid time. |

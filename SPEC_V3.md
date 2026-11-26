# BlazingAML — SPEC v3: clean semantics, formal verification, hillclimbing

Status: draft for review · Date: 2026-11-22 · Scope: HI-Small, LI-Small, HI-Medium, TransXion;
C++20 (OpenMP + CUDA); `local` first, A30 campaigns with the user's approval.

This document plans v3. It builds on `SPEC.md` (v1) and `SPEC_V2.md` (v2), which stay the record
of what was built and measured. Section references without a prefix (§4.6, A28, D-8) point into
`SPEC.md`; references with a `V2` prefix (V2 §7.1, V2 D-43) point into `SPEC_V2.md`. Where they
disagree with this document, this document wins for v3 work. MUST / SHOULD / MAY carry their RFC
2119 meaning. The tags of v1 and v2 still apply ([P] paper, [G] GFP docs, [X] GFP probe, [D]
decision, [V1] measured, [W] web source, [R] recalled and unchecked), plus:

- **[M3]** measured during v3 planning (2026-11-22). The method is given inline.
- **[I]** inferred from measured counts or code, not measured itself.

---

## 0. This spec is malleable

**Agents working on v3 have full autonomy over data structures, module boundaries, optimisation
strategies and implementation sequencing.** Do not preserve a design merely because a previous
agent proposed it. That includes the designs in `SPEC.md`, `SPEC_V2.md` and this document. The
algorithm list (§5.3), the phase plan (§7) and every "how" in this spec are starting points. Replace
them when you find something better, and record the replacement as a decision (Appendix C‴) with
its evidence.

What is fixed, unless the user changes it:

1. The goals and acceptance criteria (§2).
2. The clean semantics of §3.2. They change only by a decision that names the quirk or the
   common-sense argument and the measured effect.
3. **No speed change lands without its verification (§4.4).** A faster path that is not verified
   equal to the reference is not done, whatever it measures.
4. The working rules (§2.5): no tuning on a test split, the shared-desktop timing hygiene, RAM
   limits, and A30 approval.

---

## 1. Where v2 ended

| # | Fact | Source |
|---|---|---|
| F14 | v2 core (P7–P11) and stretch S1–S7 are done. E8 closed DG-2 (verdict GAP by overshoot, frozen recipe 63.1 against the paper's 46.6). The HI-Small test split is spent. | V2 §4.6, memory of the run |
| F15 | Mining time, HI-Small `v1_default`. EPYC 7542 at 16 threads: 618 ms in all (tcyc 44 %, sg 35 %, fan and degree 21 %). 1650S T = 650 ms with fuse and pack; A30 T = 158 ms (D2H 33 %, tcyc 21 %, fan and degree 19 %, upload 18 %). | [V1] `results/e11_ablation_a30.md`, `results/e12_s7_{local,a30}.md` |
| F16 | Long windows (`dg2_wb`: fan and degree 12 h, sg 5 d, tcyc 20 d) move the weight to tcyc and sg: EPYC 16 threads sg 928 ms, tcyc 1.61 s; 1650S tcyc 1,687 ms of 2,379 ms of kernels. | [V1] `results/e12_gpu_a30.md`, `results/e12_s7_local.json` |
| F17 | The work is out of proportion to the output. HI-Small `v1_default` has 1,125 tcyc instances and 808 sg instances. Finding them costs 34.1 M range calls (tcyc) and 40.1 M range calls plus 15.8 M pair probes (sg). | [M3] operation-counting replica of the generated loop nests, which reproduces E10's instance counts and E5/E12's heavy-trigger counts exactly |
| F18 | Self-loops are about 11.6 % of HI-Small rows (591,212 under `blz prepare`'s (bank, account) key, `src/graph/prepare.cpp:134`), about 11.6 % of LI-Small, 8.0 % of HI-Medium and 39 rows of TransXion. The non-HI-Small figures are a raw `awk` survey; the prepare invariants of §6.1 use counts under the exact key. | [M3] |
| F19 | The IBM datasets have one-minute timestamps. Every row of HI-Small, LI-Small and HI-Medium shares its timestamp with another row (15,018 distinct timestamps in HI-Small). TransXion has one-second timestamps; 12.2 % of its rows share one. | [M3] |
| F20 | New datasets are in `dataset/`. LI-Small: 6,924,049 rows (Table 1 says 6,924,055 edges), 3,565 positives. HI-Medium: 31,898,238 rows (Table 1: 31,898,238), 35,230 positives. TransXion: `tx.csv` 3,029,170 rows over 2025, 47,526 accounts, 4,641 positives, 4 payment formats, 97 % CNY, plus `person.csv` (46,399) and `merchant.csv` (1,329) keyed by `bank_account_number`. | [M3] |
| F21 | v1 and v2 copy GFP behaviour that is not common sense (§3.1). The largest are file-order tie handling, per-pattern window bounds and self-loops in SG. | [M3] code sweep, Appendix A, V2 Appendix A′ |
| F22 | Four verification tools were prototyped on the same two targets before this spec was written (§4.2). | [M3] |

---

## 2. Goals, scope, acceptance

### 2.1 Goals

- **G12. Clean semantics.** A new semantic profile family, `clean`, that drops every GFP quirk in
  favour of the common-sense rule (§3). The clean profile becomes the default.
- **G13. Formal verification where it pays.** Verify the algorithms and the compiler's output where
  a bug would silently change features and tests rarely reach the case (§4). Not every module.
- **G14. Hillclimb mining speed** with new algorithms, each one verified equal to the reference
  semantics (§5).
- **G15. New datasets.** LI-Small, HI-Medium and TransXion load, mine and train (§6). HI-Medium is
  the hillclimb's target workload.

### 2.2 Out of scope

- **Streaming replay (V2 S8) is postponed.** v3 is offline mining only. The clean semantics are
  chosen so that a later replay can emit a timestamp's rows once that timestamp closes (§3.2 R3).
- FraudGT; LI-Large, HI-Large and the other Large and Medium sets; multi-GPU; GNN classifiers.
- Reopening DG-2 or rescoring the HI-Small test split.
- Unbounded verification of `blzc` itself. Section 4.3 says why.

### 2.3 Acceptance criteria

| ID | Criterion |
|---|---|
| AC-V3-0 | **No silent change to legacy outputs.** While the legacy profiles exist (§3.4), `v1_default` gives the AC-V2-0 hashes and `v1_parity` the parity hashes of `results/ac1_cpu_p11_run.log` (fan_in 231040277472725923, fan_out 6330076044337963426, deg_in 6366489652548864147, deg_out 10227089942834718195, sg 2140027548842065401, tcyc 3161311333401399936). A legacy profile resolves its blocks to the pattern specs it was frozen with (by spec hash), so a new pattern never joins a legacy block. Retiring a legacy profile is a decision with the user's approval, and its recorded results stay in `results/`. |
| AC-V3-1 | **Clean semantics.** The clean reference (§3.2) exists as an oracle written straight from the rules. The oracle, the CPU backend and the CUDA backend give bit-identical clean rows on HI-Small, LI-Small, TransXion and 10,000 fuzz graphs, including a sparse-time fuzz setting (§4.4). HI-Medium matches between the oracle and the CPU backend, mined in trigger ranges. The metamorphic suite includes tie-permutation invariance (§3.2 R3). |
| AC-V3-2 | **Quirk ledger.** Every row of §3.1 is resolved: implemented, kept with a reason, or moved to `tools/gfp`. E16 reports each change's effect on instance counts, active rows and validation F1. |
| AC-V3-3 | **Verification stack.** The tools of §4.3 are pinned and run by one command each. Two kinds of planted break are recorded separately: **proof mutations** (a wrong lemma or model, which the proof checker must reject) and **code mutations** (a one-line change to the compiled C++, which L2 or L3 must catch; an unchanged proof cannot see these). Each project-local verify skill (§4.5) was run once on a mutation of the artifact it checks and caught it: a proof mutation for the proof skill, a code mutation in the real C++ for the code skill. Every target in §4.1 has a passing check, its stated bound or scope, evidence that the check exercised the path it covers (§4.4), and a coverage record. |
| AC-V3-4 | **Hillclimb.** §5's stop predicate is met, or the run stopped under §5.6 with the remaining ideas listed. Every kept change has its §4.4 evidence. `decision.tsv` lists every attempt, kept or reverted. |
| AC-V3-5 | **Datasets.** Each new dataset passes its `blz prepare` invariants and has E17 rows (mining time, peak RSS) and E18 rows (validation F1; test F1 once, after the freeze, §6.4). |

### 2.4 Tiers

- **v3 core** = P12–P17 (§7). v3 is done when AC-V3-0 … AC-V3-5 hold.
- Anything else an agent wants to add is a stretch item that needs the user's approval first.

### 2.5 Working rules

The v2 rules stand (V2 §2.5), with these changes and additions:

- **No tuning on test, per dataset.** Each new dataset gets its own validation split. Its test split
  is scored once, after a committed freeze (§6.4). The HI-Small test split stays spent.
- **Bit-exact backends; same-machine ratios only.**
- **Shared desktop.** Check `ps --sort=-pcpu` and `free -g` before a timing run and wait for an
  idle window. Never stop the user's jobs. Log foreign load during the run. Keep peak RSS under
  about 3 GB per process on `local` unless the preflight shows more is free.
- **A30.** A campaign runs whenever a phase needs one, with the user's approval (D-59).
- **Preflight before anything large** (V2 §2.5). HI-Medium always preflights.
- **Report the gap, then move on.** When a number disagrees with the paper or with an estimate in
  this spec, record it and continue.

---

## 3. Clean semantics

### 3.1 Quirk ledger

Each row is a behaviour that v1 or v2 copied from GFP (or kept for GFP's sake) and the clean rule
that replaces it. "Where" points at the code as of `1b5123d`.

| # | GFP / v2 behaviour | Where | Clean rule | Decision (clean rule replaces the quirk, or keep the current behaviour) |
|---|---|---|---|---|
| Q1 | **Self-loops in SG.** An endpoint joins the intermediate set through its own self-loop: s ∈ I via s→s in the gather role (A10), w ∈ I via w→w in the scatter role (A30). The exclusion depends on the role. | `patterns/sg_gather.yaml:22`, `sg_scatter.yaml:22`, `src/oracle/oracle.cpp:81-91`, `src/runtime/batch.cpp:186-191` | Self-loops take no part in any pattern (R1). | Clean rule (the user's example). |
| Q2 | **Self-loops elsewhere, inconsistently.** Fan and degree drop them (A14); vertex statistics count them on both sides (A43); self-loop triggers emit nothing for patterns (A13) but get vstats rows; in batch mode self-loop triggers do search SG (A41). | every `skip_if: "U == V"` and `Y != U` filter in `patterns/*.yaml`; `src/runtime/vstats.hpp:65-93`; `src/oracle/batch_oracle.cpp:57` | One rule (R1): the mining graph has no self-loops. | Clean rule. |
| Q3 | **Window bounds differ by pattern.** Open `t' > t − δ` for fan, degree, gs and the S4 set ops; closed `t' ≥ t − δ` for sg, tcyc, lcyc and tchain (A2, A3, A7, A20). The rule is written twice. | `src/runtime/features.hpp:32-34`; each spec's `window.lower`, lowered at `src/ir/lower.cpp:531-550`; `batch.cpp:68` | One half-open window `(t − δ, t]` for every pattern (R2). | Clean rule. |
| Q4 | **Stream visibility.** A trigger sees edges with eid ≤ e, so same-timestamp rows depend on file order (A4, A15). Every IBM row shares its minute (F19). | `oracle.cpp:34-36`, `features.hpp:36-39`, `src/ir/plan.cpp:63-70`; the order is D-4's stable sort | Timestamp visibility: a trigger sees every edge with ts ≤ t (R3). | Clean rule, confirmed by the user on 2026-11-22. |
| Q5 | **Strict time order in chains.** tcyc and tchain need t_1 < t_2 < … < t, so a cycle or chain with two hops in the same minute is lost (A19). | `patterns/tcyc.yaml`, `tchain.yaml`; `oracle.cpp:173-175, 270-271` | Non-decreasing order, t_1 ≤ … ≤ t (R4). | Clean rule, confirmed by the user on 2026-11-22. |
| Q6 | **lcyc ties.** GFP decides a tie edge by its node insertion rank (A49); `blz` drops every tie edge instead (D-38). | `patterns/lcyc.yaml:20-52` (`break_if: "*.t >= t"`), `oracle.cpp:200-210` | Every visible window edge counts, ties included (R5). | Clean rule. |
| Q7 | **Unordered SG.** An SG instance has no order inside the window (A6), so the scatter role counts money forwarded before it was received. | `sg_gather.yaml`, `sg_scatter.yaml`, `oracle.cpp:66-92` | Both variants, as two blocks: `sg` stays unordered (R6a) and `sg_ord` adds the order, each intermediate receiving from s no later than it sends to w (R6b). | Both kept, the user's choice on 2026-11-22. |
| Q8 | **Scalar counts as one-hot histograms.** Fan, degree, fan_total, fan_oneway_out and gs emit one instance per trigger into bins, so a block of 12 columns carries one number, and counts 0 and 1 give the same all-zero row (A1, min size 2). | `emit: {size: n}` in those specs; `oracle.cpp:63, 336-339` | A count block emits the count as one integer column (R7). | Clean rule. |
| Q9 | **Vertex statistics' window.** v2 copies GFP's window: the node's edges with ts ≥ ts[e − 1] − W, anchored at the previous edge, where W is GFP's time_window, the largest *pattern* window (A43). (GFP's running-sum drift after eviction, A45, was never copied: v2 already computes each window exactly, D-37.) | `src/runtime/vstats.hpp:7-11, 96-98`; `blz vstats --window` | The node's visible edges in `(t − W_vs, t]` with the block's own W_vs, computed exactly (R8). | Clean rule. |
| Q10 | **Vertex statistics' guards and time base.** A variance below 10⁻⁵·mean² is reported as 0 with its skew and kurtosis (A44). Time statistics use absolute timestamps (plus D-12's +1 in parity runs only), so with mean ts near 7.6·10⁵ s the guard zeroes ts variance on any side that spans under about 40 minutes, and the threshold grows with stream time. | `vstats.hpp:35-37, 54, 70, 78` | Time statistics are over ages t − ts'. No threshold guard; the only zeroing is R8's rule for equal values and a computed m2 of 0. | Clean rule. |
| Q11 | **GFP batch mode and `credit: participants`.** Rows depend on batch boundaries (A28, A34–A42, D-36), with `k_max: 33` to match GFP's uncapped cycles. | `src/runtime/batch.*`, `src/oracle/batch_oracle.cpp`, `src/runtime/profile.*` (batch fields), `configs/profiles/{dg2_g1,batch_b4,batch_b32}.yaml` | Not part of clean. Out of the default build and the clean path: deleted, or isolated behind legacy tooling (§3.4, P12). | Clean rule. Done in P12: deleted (D-55). |
| Q12 | **GFP harness leaks into the runtime.** `VstatsParams::ts_offset` and `blz vstats --ts-offset` (D-12's +1), `blz fuzz --vstats` hard-coding 1.0, the fuzz generator aimed at GFP's variance guard, batch-only fuzz and coverage plumbing. | `vstats.hpp:32`, `src/apps/blz.cpp:549, 605, 707-768, 1607-1692, 1788-1799`, `src/runtime/fuzz_gen.hpp:20-23` | GFP-only workarounds live in `tools/gfp` or are deleted with their feature. | Clean rule. Done in P12 (D-55, D-56). `fuzz_amounts` is kept, with its reason in D-56. |
| Q13 | **Input range fixed to HI-Small.** Epoch 2022-09-01 and `kMaxTimestamp = 1,527,480` are hard-coded. | `src/io/csv.cpp:18, 46`, `src/io/csv.hpp:29` | A dataset descriptor gives the schema; the epoch is the first row's day; the range is checked against int32 (§6.1). | Clean rule. |
| Q14 | **GFP's bin rule** (first bin with S ≤ b_i, A38). | `tools/gfp/gfp_common.py:27-41` | Harness only. `blz` keeps `[b_i, b_{i+1})`. | Keep as is. |
| Q15 | **Trigger credit re-reports unchanged instances.** A parallel or repeated trigger re-emits the same SG pair or cycle (A5, A8). | `oracle.cpp:87-91, 178-181` | Kept: a row counts the instances its edge completes, and a parallel edge completes them again. | Keep (R9). |
| Q16 | **Min size 2.** | `features.hpp:75-79`, `include/blz/registry.hpp:26`, `lower.cpp:565` | Kept for histogram blocks: an SG with one intermediate is not a scatter-gather, and a cycle has two or more edges. Count blocks (R7) have no minimum. | Keep for histograms. |
| Q17 | **Windows and `k_max` copied from GFP's setup** (D-8; `dg2_wb` uses GFP's library defaults, F7). | `configs/profiles/*.yaml` | Parameters, not semantics. The clean profiles take their windows from the data (§3.3). | Re-choose on validation. |

The ledger is a starting point. An agent who finds another quirk adds a row, with its effect
measured in E16.

### 3.2 The clean rules (normative)

These rules define every profile of the `clean` family. The clean oracle implements them literally,
with no optimisation, and is the reference for every check in §4.

- **R1. The mining graph has no self-loops.** An edge u→u is not inserted into any mining index and
  is never a hop, a neighbour, a witness or a vertex-statistics value. A self-loop trigger's pattern
  rows are all zero. Its vertex-statistics row is computed like any other trigger's, over u's
  non-loop edges. Its raw attributes stay classifier inputs.
- **R2. One window.** Pattern P with span δ_P sees `W_P(e) = { e' ∈ Vis(e) : t − δ_P < t' ≤ t }` for
  trigger e at time t. This holds for every pattern and every vertex-statistics block.
- **R3. Timestamp visibility.** `Vis(e) = { e' : t' ≤ t }`. Consequences, each a metamorphic test:
  (a) permuting the rows within one timestamp changes no feature row (a row moves with its edge);
  (b) two edges with the same (u, v, t) have the same pattern rows; (c) a replay can emit a
  timestamp's rows once the next timestamp arrives.
- **R4. Non-decreasing order.** Where a pattern orders its hops (tcyc, tchain, ordered sg), the
  order is `≤`, and the trigger is a latest edge of its instance. For tcyc, the node sequence
  `(c_0 = v, …, c_{k−1} = u)` of distinct nodes counts when there are window edges
  `c_{i−1} → c_i` at times `t − δ < t_1 ≤ t_2 ≤ … ≤ t_{k−1} ≤ t`. The earliest-witness rule holds
  with "earliest at or after the previous hop" (§4.2, proved for the strict version and to be
  re-proved for this one).
- **R5. lcyc.** The simple node cycles through the trigger edge, of length 2 to L, whose other
  edges are all in the window, in any time order. Ties count.
- **R6. Scatter-gather, two blocks.** An instance is a pair (s, w), s ≠ w, sized by its set of
  intermediates I(s, w), and reported when |I(s, w)| ≥ 2.
  - **R6a. `sg`, unordered.** `I(s, w) = (N_out(s) ∩ N_in(w)) \ {s, w}` over the window. With R1
    the two endpoints are excluded the same way in both roles, which removes Q1's role-dependent
    exclusion. Both of v1's roles report (§4.5): a trigger x → w reports (s, w) for each source
    s ∈ N_in(x) \ {x, w} (gather), and a trigger s → x reports (s, w) for each sink
    w ∈ N_out(x) \ {s, x} (scatter). The two roles never yield the same pair for one trigger, so
    the block is their sum, as in v1.
  - **R6b. `sg_ord`, ordered.** `I(s, w) = { y ∉ {s, w} : ∃ s→y at t_1, y→w at t_2 in the window,
    t_1 ≤ t_2 }`. A trigger e = (x → w) reports, for each source s with an edge s→x in the window,
    the instance (s, w) (x is always in I, through s→x and the trigger). Only gather edges report:
    under R4 a scatter edge s→y at t whose y→w is also at t has that gather edge as a co-latest
    edge, which reports it. This is V2's `sg_ordered` (D-27, D-28) under R2–R4.
  - Every R6b instance's I is a subset of its R6a I, so `sg_ord` never exceeds what `sg`'s gather
    role reports for the same trigger. Both blocks share their scans, so a backend MAY compute
    them in one pass.
- **R7. Count blocks.** These emit their count as one `uint32` column, 0 included, for every
  trigger (a self-loop trigger gets 0, R1). Each definition is the v1 or v2 one with R1–R3 applied:
  - fan_in = |N_in(v)|, fan_out = |N_out(u)| (distinct neighbours, the trigger's included; §4.3);
  - deg_in, deg_out = the window edges into v, out of u, parallel edges and the trigger included
    (§4.4);
  - fan_total = |N_in(v) ∪ N_out(v)|, fan_oneway_out = |N_out(u) \ N_in(u)| (V2 S4);
  - gs (V2 S5, D-42) for a trigger x → w: A = the distinct sources s ∉ {x, w} with s → x in the
    window and ts ≤ t (ties included, R4; v2 used ts < t); B = fan_out's set for the trigger;
    gs = min(|A|, |B|).
  Bins remain a profile option for anyone who wants a histogram of a count; the clean default emits
  counts.
- **R8. Vertex statistics.** For each of the four sides (u's out- and in-edges, v's out- and
  in-edges, D-57) in `W(e)` and each value
  column (Amount Received, as V2's F1 used, and the age t − ts' in seconds): fan, degree,
  ratio = degree / fan, and the sum, mean, population variance, skew = m3 / m2^1.5 and kurtosis =
  m4 / m2² (not excess) of the values. Outputs are float64.
  - **Order independence.** The values of a side are accumulated in ascending order of
    (value, ts), never in eid order, and in two passes (mean = sum / n, then central moments). Rows
    are then a function of the side's multiset of (value, ts) pairs, so R3's tie-permutation test
    holds bit for bit.
  - **One arithmetic.** The oracle, the CPU backend and the CUDA backend run the same sequence of
    IEEE-754 float64 operations in that order: round to nearest, no FMA contraction, no fast-math
    (CUDA builds this code with `-fmad=false`).
  - **Small sides.** n = 0: every column 0. n = 1: sum = mean = the value, variance = skew =
    kurtosis = 0, fan = degree = 1, ratio = 1. When every value of the side is equal (min = max,
    tested on the values, not on the computed variance, which rounding can leave positive),
    variance = skew = kurtosis = 0. Otherwise the variance is the computed m2, even when rounding
    makes it 0 (distinct values such as 10⁻²⁰⁰ and 2·10⁻²⁰⁰); whenever a computed denominator is
    0 (m2 is 0, or a positive m2 whose m2·√m2 or m2² underflows, D-58), skew and kurtosis are 0,
    never 0/0. No other guard.
- **R9. Credit.** A row holds the instances its trigger edge completes (D-5). Which edges complete
  an instance is part of each pattern's definition: every trigger for count blocks (R7); the
  trigger as a latest edge for tcyc and tchain (R4); the trigger as one of the cycle's edges for
  lcyc (R5); both roles for `sg` (R6a); gather edges only for `sg_ord` (R6b), so a scatter edge
  never reports an ordered instance even when it is a latest edge.
- **R10. Sizes and saturation.** Each histogram block's size is defined per pattern: SG = |I|
  (distinct intermediates); tcyc and lcyc = the number of cycle edges (= nodes); tchain = the
  number of chain edges (V2 S5, D-42: a simple node path ending in the trigger, each hop no later
  than the next under R4, one instance per node sequence, lengths 2 to `k_max`). Counts saturate at
  `UINT32_MAX` with a global saturation counter, as in v1.
- **R11. Work limits are resource guards.** In clean profiles, a per-trigger work limit (V2 D-39)
  is not semantics: reaching it aborts the run with an error and no rows. A pruning change may let
  a run complete that used to abort, but must never change a completed row. A case that hits a
  **work limit** on any side is **inconclusive**: it never counts toward a parity gate, a coverage
  claim or a mutant kill. (A certificate rejection, §4.4 V-cert, is different: it is the expected
  failure that kills a mutant.) Each gate states its case count and meets it with cases that every
  compared implementation completed on the same inputs; for AC-V3-1, fuzz graphs completed by the
  oracle, the CPU backend and the CUDA backend. Legacy profiles keep D-39's behaviour.

### 3.3 Clean profiles

- **`clean_default`.** R1–R11, the v1 windows (fan and degree 24 h, sg and sg_ord 6 h, tcyc 24 h),
  `k_max` 10, v1's histogram bins for sg, sg_ord and tcyc. The regression and hillclimb anchor.
- **`clean_wc`.** The same with W-C windows (V2 D-25: p90 of the fit split's complete attempt spans
  per pattern group, rounded up to the hour, capped at 20 days), computed per IBM dataset from its
  `Patterns.txt` join (V2 §3.5; P15 runs the joins for LI-Small and HI-Medium). A pattern group with
  no complete fit attempt keeps `clean_default`'s window for that group. TransXion has no attempt
  file, so it uses HI-Small's W-C windows, labelled as borrowed. This is the long-window workload of
  the hillclimb and the classifier's first candidate.
- An agent MAY add profiles. Each gets a content-derived id (V2 D-23).

### 3.4 Legacy profiles and what is deleted

- **Subtract first (P12):** GFP batch mode (Q11) and the GFP plumbing in the runtime (Q12) leave the
  default build and the clean path: deleted, or isolated behind legacy tooling where a parity
  helper still needs them, with every caller migrated or retired. The S1 results and E8's verdict
  stand in `results/`. If batch mode is deleted, the frozen DG-2 recipe can no longer be recomputed
  in C++; that is acceptable (DG-2 is closed, the test split is spent).
- **Keep, frozen:** `v1_default`, `v1_parity` and the other non-batch v2 profiles, so AC-V3-0's
  hashes and the GFP parity tooling keep working. They run through the existing generated code and
  the existing oracle. **New fast paths need not support legacy semantics**: a legacy profile MAY
  fall back to the v2 path.
- **Gate G-LEGACY (end of P16):** if keeping the legacy semantics costs complexity in the code that
  the hillclimb touches, propose retiring them (keeping the recorded hashes and golden files) and
  ask the user.

### 3.5 Choices the user should confirm

Q4 (timestamp visibility), Q5 (non-decreasing order) and Q7 (SG order) change what a feature
means, not only how it is computed. On 2026-11-22 the user confirmed Q4 and Q5, and chose to keep
both SG variants for Q7 (R6a and R6b). E16 still measures each change separately on validation.

### 3.6 E16: what the clean semantics change

On HI-Small's fit and validation rows (§6.3), and on TransXion's once P15 loads it: each quirk that
changes values (Q1–Q10), turned on one at a time over the legacy profile that has the affected
block (`v1_default` for the six v1 blocks, `lcyc_f2` for Q6, a vertex-statistics profile with
V2 F1's windows for Q9 and Q10), and then all together in `clean_e16`, which is `clean_default`
plus lcyc and that vertex-statistics block, so every ablated block is present. Its legacy
comparison is `v1_e16`: `v1_default` plus `lcyc_f2`'s lcyc block and the same vertex-statistics
block under legacy semantics, with the same windows. Per block: instances,
active rows and the share of positive rows that are active; then validation F1 with the v2
classifier settings (V2 R0's XGBoost parameters). Harness-only and upstream-only rows (Q12, Q14,
and GFP facts such as A29 and A33 that `blz` never copied) are marked not applicable. Output:
`results/e16_clean.{json,md}`. E16 informs §3.5 only.

---

## 4. Formal verification

### 4.1 What deserves verification

The rule from `/create-formal-verification-skill`: verify a promise the code makes where a break
silently harms users and tests rarely reach the case that breaks it.

| Target | Promise | Why tests are not enough |
|---|---|---|
| T1 | The generated `<pattern>_trigger` of every spec in `patterns/`, as compiled, equals the oracle of its own profile family (clean specs against the clean oracle, legacy specs against the legacy oracle) on every graph within a stated bound, and on sampled larger graphs (translation validation of `blzc`'s output; the unbounded part is T3's). | Fuzz graphs are small, dense and random. Boundary cases (an edge exactly at t − δ, ties, parallel edges, a binding excluded by two filters) are rare in them. |
| T2 | Every optimised path equals the unoptimised one: the sweep lowering (V2 D-29), scan fusion (D-46), packing (D-45), and every hillclimb algorithm (§5.3). | A speedup that drops one row in a million is invisible to a hash on one dataset until it changes a classifier. |
| T3 | The algorithmic lemmas the generated code and the optimisations rely on: the earliest-witness rule (§4.6, R4) and its latest-witness mirror (`pair_latest`), pruning soundness (V2 D-43 or its replacement), the slab lemmas of §5.3, interval-sum and wedge identities. | They must hold at every size; a bound is not enough. |
| T4 | Index primitives' contracts: `range`, `distinct_first` (`prev_same`), `pair_exists`, and the index builder that establishes them. | Every pattern's correctness proof assumes them. |
| T5 | Runtime certificates for results computed by unverified code (e.g. SCC labels, §5.3 H1). | A certificate checked on every run is cheaper than a proof of the producer. |

Not verified, on purpose: CSV parsing and `blz prepare` (asserted invariants), the classifier and
XGBoost glue, the CLI, reports, the Python tools and the GFP harness, CUDA scheduling and OpenMP
drivers (covered by the bit-exact differential checks), and `blzc`'s front end and planner as
programs (their output is validated per spec instead, T1).

### 4.2 Tool evaluation

Each tool was prototyped on the same two targets on 2026-11-22: **T-code**, the real generated
`sg_gather_trigger` (`tests/snapshots/sg_gather.hpp` with the real `include/blz/primitives.hpp`)
against §4.5's gather role, and **T-algo**, §4.6's earliest-witness rule. Each prototype had to
catch planted mutations. The prototypes are throwaway, kept outside the repository (Appendix E).

| | Lean 4 (core) | Dafny | CBMC | Property-based testing |
|---|---|---|---|---|
| Version, install | 4.36.0 via elan; 608 MB download, 3.1 GB on disk, 27 s, no Mathlib | 4.11.0; 65 MB self-contained zip (bundled .NET and Z3), 4 s | CBMC 6.11.0, 73 MB .deb unpacked without dpkg, 212 MB, 4 s; ESBMC 8.7.0 also tried (302 MB, static binary) | RapidCheck at commit `2c3c4365` (untagged; pin the SHA); 2 MB source, builds in 7 s, runs inside Catch2 v3 |
| Reads the real C++ | No | No (C++ backend emits code, it does not read it) | CBMC: not as shipped. It needed 5 mechanical lowerings of the real headers, one on the emitter's `const TCsrView& L = c ? a : b` idiom that every intersection uses. ESBMC parses everything unchanged | Yes: calls the real templated trigger functions over the real `GraphIndex` |
| T-algo | Proved for all sizes (143 lines). Also proved: V2 D-43 pruning soundness against a 3-fact search contract, a hand model of `sg_gather_trigger` equal to §4.5, wedge-counting SG and a sliding-window degree sweep (about 470 lines in all) | Proved both directions over T-CSR-shaped arrays (130 lines); a machine-typed copy also proves no int32/uint32/int64 overflow | Bounded, value-symbolic, fast when "exists" is a nondeterministic choice: 9 hops in 0.9 s, 32 hops in 244 s, ts up to 2·10⁹ | Sampled, not proved: 200,000 random hop lists against exhaustive search |
| T-code | Hand model only; links to the C++ by human comparison and goes stale when `blzc` changes | Hand model only (121 lines), same limit | Every graph with ≤ 5 edges (n = 4, ts < 4, both visibilities, every trigger): pass, 7,528 CPU s at m = 5, about 2 CPU days extrapolated for m = 6. ESBMC ran out of memory under 2 GB at m = 3. **Plain enumeration in compiled C++ of the unmodified code covers the same m = 5 domain (58.7 M graphs, 2.94 G trigger checks) in 133 s on one thread** | Real code against a 30-line naive §4.5 reference, both visibilities; plus a reference-free check of optimised against `blzc --disable` builds |
| Check time | 1.1 s clean build of all proofs | 4–9 s per file | T-code m = 4: 278 CPU s; the SAT solver is the whole cost (symex 0.9 s, solving 291 s) | about 50 µs a case; 10,000 cases in 0.4–0.8 s per property |
| Mutants caught | 13/13 (4 at a brittle proof step, not at the false claim) | 12/12, with line-level errors | 7/7 on T-code at m = 4, 3/3 on T-algo in under 0.2 s; its built-in bounds, overflow and conversion checks caught a planted underflow and an off-by-one read; no false alarms | 12/13, each in a median of 27–65 cases, shrunk to 3–5 edges (18 for a sort-stability bug). Missed a length-10-cycle bug in 200,000 cases until hop and δ ranges were widened together. The existing `fuzz_diff` runs found all 13, but as 156–278-edge seed graphs with no shrinking; its AC1 run missed the timestamp-visibility mutant and the timestamp share missed an eid-tie mutant; the metamorphic suite caught 4 |
| Friction | Proofs follow the model's syntax; toolchain deprecations need small repairs | First draft hit Z3's 30 s timeout until restructured; batch mode timed out on one mutant | C++ front end gaps (no C++17 nested namespaces, no `using` aliases, crashes on conditional references); a wrong `malloc` declaration gave 241 bogus failures; `--unwindset` cannot name loops in member functions; cannot read CUDA or check OpenMP | Generator ranges must be derived from the spec constants (K_MAX, bins) or whole bug classes go unseen; needs `OMP_NUM_THREADS=1` per case |

### 4.3 The chosen stack

The prototypes agree on the shape: no single tool both reads the shipped C++ and proves claims for
every size. The stack therefore has one tool per job, and each layer covers the gap of the one
above it.

| Layer | Tool | Job | Why this tool |
|---|---|---|---|
| L1 proof | **Lean 4, core only, pinned toolchain** | V-proof for T3: every algorithmic lemma, for all sizes, against a Lean model of §3.2's rules. Each lemma states the contracts it assumes of the code (as the prototype's 3-fact search contract did for V2 D-43). | Proved every prototype target plus two new algorithms in about 470 lines; 1.1 s to check; kernel-checked, so no solver timeouts. Dafny did the same algorithm work equally well but hit Z3 timeouts, and a second proof language would double the upkeep. |
| L2 bounded code check | **Exhaustive enumeration in compiled C++** | V-code for T1, T2, T4: the real, unmodified generated code (and every optimised path) against the clean oracle, or optimised against unoptimised, on **every** graph within a stated bound, every trigger, both visibilities where they exist. A second pass of the same harness under ASan and UBSan at a smaller bound gives the memory and overflow checks. | The CBMC prototype proved m ≤ 5 in 7,528 CPU s; enumeration covered the same domain in 133 s on one thread, on code CBMC could not read unmodified. The harness is ordinary C++ that every agent can read and extend. |
| L3 always-on | **RapidCheck** properties in `blz_unit_tests`, plus the existing `fuzz_diff` and metamorphic suites | V-diff beyond L2's bound: larger graphs, long cycles, every pattern through the registry, every `blzc --disable` variant against the full build (reference-free), CPU and CUDA. Fixed seeds in CI, random nightly, shrinking for every failure. | Found 12 of 13 planted bugs in 27–65 cases with 3–5-edge counterexamples; `fuzz_diff` found all 13 but as 156–278-edge graphs. Generator ranges must come from the spec constants (the missed length-10 bug). |
| L4 value lemmas | **CBMC**, plain C, optional | Lemmas over full-width integers that enumeration cannot reach: slab and window arithmetic at int32/int64 extremes, interval endpoints, saturation. | Bounded but value-symbolic: 32 hops with ts up to 2·10⁹ in minutes. Use only where L1's unbounded integers abstract away a machine-width question. |
| L5 runtime certificates | the code itself | V-cert (T5): checks that run on every mine and abort on failure. | Verifies a result from unverified code without verifying the code. |

**The exhaustive layer is the main code-level check.** Its domain is every edge list with up to
m edges over n nodes with timestamps in [0, T), every span up to T and every trigger. The prototype
reached m = 5, n = 4, T = 4 in 133 s on one thread. An agent SHOULD push the bound with symmetry
reduction (canonical node relabelling, compressed timestamps) and threads, but no exhaustive bound
reaches every case: at `k_max` = 10, ten-edge cycles alone give trillions of lists after
relabelling. So each pattern's check has two parts:
- an **exhaustive core** at a bound that stays within a few minutes, stated in the harness, and at
  least the smallest instance of the pattern (4 edges for SG, 2 for a 2-cycle) plus one edge;
- **targeted generators** (L3) for what the core cannot reach: maximum-length cycles and chains
  (`k_max` hops), the last histogram bin and saturation, arithmetic extremes (int32 timestamps
  near the limits, δ near the span), packing widths and chunk boundaries.
Each planted code mutation must be caught by one of the two, and the harness records which.

Not adopted:
- **Dafny.** It proved the algorithm targets as well as Lean did, and a machine-typed copy proved
  no overflow. But it cannot read C++, its C++ backend needs workarounds, and Z3 timeouts needed
  proof restructuring. Lean covers its proving role and L2/L4 its overflow role.
- **CBMC as the main code checker.** It needed five lowerings of the real headers, one on an idiom
  every intersection uses, and was about 60× slower than enumeration over the same domain.
  ESBMC parsed the code unchanged but ran out of memory at m = 3 under 2 GB.
- **Verifying `blzc` itself.** A Lean or Dafny model of the IR and the lowering would take weeks
  (the prototypes' estimate) and would still trust the C++ emitter text. Validating each
  generated output (L2, L3) checks what ships.

An agent MAY replace any layer's tool with evidence, recorded as a decision.

### 4.4 What "verified" means for a speed change

A change to how rows are computed (a new algorithm, lowering, layout or kernel) is verified when
all of these hold, each shown able to fail on a planted break:

1. **V-proof.** The algorithm's identity or soundness lemma is proved for all sizes, against the
   clean reference model, with the assumptions it makes on the code stated as contracts.
2. **V-code.** The real C++ of the new path, as compiled, is checked against the reference path
   on every input within a stated bound, and the contracts of V-proof are checked on the code.
3. **V-cert.** Where the algorithm consumes a result from code that V-proof and V-code do not
   cover (a graph labelling, a partition, a set of marks), either that producer is itself proved
   and checked like the algorithm, or a cheap certificate is checked on every run and failure
   aborts the run. A filter whose producer has neither MUST NOT land: a dropped mark silently
   zeroes a valid row.
4. **V-diff.** Bit-exact rows against the clean oracle on HI-Small, TransXion and 10,000 fuzz graphs
   per backend, including a **sparse-time fuzz setting** with δ well below the time span (so that
   slab and window boundaries are crossed and graphs hold several components), plus the
   metamorphic suite.
5. **V-mut.** A recorded list of planted code mutations, each caught by at least one of the
   above.
6. **Path hit.** Every executable check (L2, L3, V-diff) records evidence that the new path
   actually ran: counters of the triggers, bindings or lists it handled, nonzero in the check's
   runs. A proof (L1) has no runtime; its coverage record names the code each lemma describes, and
   the executable checks carry the path hits. A planner can silently
   decline a path (V2 D-43's reachability pruning is accepted only for strict-order chains, the
   registry's default profile omits `lcyc`, and the `--disable` ablation builds are CPU only), so
   each check names the profile and build that enables the path, and CUDA-only paths are checked
   on a CUDA build. Clean test profiles enable every pattern in `patterns/`.

A legacy-profile fast path (§3.4) needs the same evidence against the legacy oracle.

### 4.5 Project-local verify skills

P14 runs `/create-formal-verification-skill` once per verification area. The intended areas:

- **verify-algorithms**: the proof project of §4.3 (T3), one module per lemma, the mutant runner,
  and a coverage record mapping each lemma to the code it justifies.
- **verify-codegen**: the bounded code checks (T1, T2, T4), one harness per pattern and per
  optimisation, generated from the pattern list, not written by hand per pattern.

Each skill records the files and functions it covers and the commit at which the coverage was last
confirmed. An agent changing covered code reruns the check, and moves the sync commit only after
confirming the verification still describes the code. `/maintain-formal-verification-skill`
brings a skill back in line after changes that skipped it. An agent MAY merge, split or rename the
skills.

---

## 5. Hillclimb

### 5.1 Metric

- **Metric M.** Median wall time of mining every block of a clean profile on a dataset, from the
  loaded `.blzg` in memory to the delivery of every logical row to a **bounded host sink**: rows
  arrive in trigger-range chunks of a fixed size, are decoded to the row contract (one `uint32`
  per count column, the histogram bins, float64 vertex statistics), hashed, and dropped. Baseline
  and candidates use the same sink, contract and hashes. M includes every index, slab, label or
  auxiliary structure the algorithm builds, chunk assembly and decoding, and the certificate
  checks. It excludes CSV parsing, `blz prepare` and file writes, which are reported beside it.
  Lower is better.
- **Samples and noise.** Median of 5 runs on an idle machine; min and max are kept. A change is kept
  only when its median beats the incumbent's by more than the incumbent's min-to-max spread. A
  workload counts as regressed when its median exceeds its baseline's max by more than 5 %.
- **Each run prints** its error count, the triggers mined, the instances emitted, and the row hash
  per block, so a fast run that did less work is visible.

### 5.2 Workloads and target

| Workload | Role |
|---|---|
| HI-Medium, `clean_default`, CPU 12 threads on `local` | **Target.** The stop predicate is measured here. |
| HI-Medium, `clean_wc` | Long-window target (tcyc and sg dominate, F16). |
| HI-Small, both profiles | Development loop; fast to rerun. |
| TransXion, `clean_default` | A different shape: 47,526 accounts for 3.0 M edges (mean degree about 64, merchant hubs). Guards against tuning to the IBM generator. |
| 1650S, HI-Small, both profiles; HI-Medium in trigger-range chunks | Secondary backend and guard. |

The two HI-Medium CPU rows are the **primary** workloads; the others are **guards**. A workload a
backend cannot hold (memory) is recorded as a capacity limit, not a failure. The capacity
exclusions are frozen with the baseline: a candidate that cannot complete a workload the baseline
completed has regressed on it.

**Stop predicate.** All of:
1. M on both primary workloads is at least 4× lower than the clean baseline (P15's measurement of
   P13's implementation, which uses v2's algorithms);
2. every seeded hypothesis H1–H4 was attempted, or ruled out by a recorded measurement, so a lucky
   early win cannot end the run;
3. no guard workload regressed (§5.1);
4. every kept change is verified (§4.4).
The 1650S goal, 3× on HI-Small `clean_default`, is reported but does not gate. These targets come
from §5.3's inferred estimates; if the user names others, use theirs.

### 5.3 Seed hypotheses

Two independent surveys (Opus and Codex, 2026-11-22) ranked candidates from the measured timings
and, for the Opus survey, an operation-counting replica of the generated code. The list below merges
them. Counts are [M3] from that replica; gains are [I].

| # | Hypothesis | Mechanism | Evidence | Verification obligation |
|---|---|---|---|---|
| H1 | **Slab SCC filter** for tcyc and lcyc (CPU and CUDA). | Cut time into slabs of width δ. Graph k holds the edges with ts in [(k−1)δ, (k+1)δ), which contains every window of a trigger in slab k. Label graph k's strongly connected components. Skip a trigger whose endpoints are in different components, and a binding whose node is in a different component from u. | HI-Small: 2,387 / 67,639 / 221,352 of 4.49 M triggers survive at 24 h / 89 h / 20 d (99.95 %, 98.5 %, 95.1 % cut); HI-Medium 15,593 / 411,425 / 2,704,941 of 29.3 M. 0 rows differ from the unpruned DFS. 398 bindings left at 24 h on HI-Small. Inferred tcyc 10–30× on CPU. | V-proof: the slab contains the window (int64 floor division, negative t); a closed walk has one label when labels never increase along an edge; a filter every instance passes leaves the rows unchanged. V-cert: check label monotonicity on every slab edge each run (a split component fails, a merged one is still sound). V-code: slab index and eid ranges at boundaries. lcyc's work limit becomes a resource guard (R11), so a run that aborted unpruned may complete. |
| H2 | **Pair marks for SG.** | In each slab graph, count 2-paths s→y→w per s and keep the pairs with two or more distinct y; a trigger runs only when its edge is a leg of a kept pair, a binding only when its pair is kept. Cheaper fallback with a local proof: a gather trigger x → w runs only when w has at least two distinct in-neighbours other than w in the window, and a binding (s, w) only when s has at least two distinct out-neighbours other than s and w and w at least two distinct in-neighbours other than s and w (necessary for \|I\| ≥ 2). R6's ordered instances are a subset of unordered ones, so marks built without order stay complete. | HI-Small 6 h: 3,053 of 15.5 M bindings survive, 0 instances lost; the fallback filters keep 669 K. Risk: 2-paths grow faster than edges (476 M on HI-Medium against 21 M). | V-proof: an instance implies a mark. The mark builder has no cheap certificate, so under §4.4 V-cert it needs its own completeness proof (V-proof) and V-code check; if that is not done, use only the fallback filters, whose producers are the verified window primitives. |
| H3 | **Count blocks without per-trigger scans** (R7 makes them scalars). | Degree is a difference of two range ranks. Fan is an interval sum: each list entry counts for a contiguous range of triggers, so a difference array and one prefix sum give every count (CUDA); the CPU sweep (V2 D-29) already does this in order. | fan_out windows sum to 4.32 G entries at 24 h on HI-Small; 1650S fused fan and degree kernels 206 ms, inferred 15–30 ms. | V-proof: interval-sum identity with `prev_same`; prefix-sum identity. V-code: the interval endpoints against `count_distinct`. |
| H4 | **Compact output.** | Count blocks send one integer per row; histogram blocks send sparse rows (788 nonzero sg rows and 1,109 tcyc rows of 5.08 M on HI-Small). | A30 D2H is 33 % of T; host widening bounds packed D2H. | V-proof: encode/decode identity. V-code: boundary bins, saturation, row order, chunk offsets. |
| H5 | **Shared hop-bounded reachability labels** for tcyc (Codex's first). | Latest-departure labels to u per (u, time slab), reused across triggers. | Likely subsumed by H1 (398 bindings left at 24 h); try only if H1 leaves heavy residue at 20 d (939 K bindings). | V-proof by induction on hops (the shape of V2 D-43's proof). |
| H6 | **CPU two-phase emission** for count blocks. | Count in list order, write rows in eid order with streaming stores. | Inferred bound: one random row write per trigger. Profile first. | Permutation lemma; V-diff. |
| H7 | **Mining in trigger ranges** for HI-Medium. | Rows for a range of eids at a time, with the index resident. | HI-Medium rows are 8.55 GB at 67 dense columns; R7 brings `clean_default` to 33 columns (4 counts, sg 10, sg_ord 10, tcyc 9), about 4.2 GB. Needed for the 1650S's 4 GB. | Range-split identity (rows depend only on the trigger); V-diff. |

Subtract as you go: if H1 lands, V2 D-43's reachability pruning and V2 D-44's hybrid split may have
no measured job left (at 24 h all 2,876 heavy tcyc triggers fail H1's root test). Measure, and
delete them if so; that also removes their verification burden. H3 retires the CUDA light/heavy
split for count blocks.

Considered and not seeded, with the reason: row sharing for duplicate (u, v, t) triggers (2.0 % of
rows); 2SCENT, temporal Johnson, path bundling, meet-in-the-middle and hop-constrained search (after
H1 nothing is left to prune at 24 h; revisit at 20 d if H1 leaves enough); index compression,
bitsets, merge-path and warp-cooperative intersection (no measured bound after H1–H3). An agent MAY
revive any of them with a measurement.

### 5.4 After the algorithms

Once H1–H3 land, output rows bound every backend (inferred). R7 and H4 are the levers: fewer
columns and fewer bytes per row. A change to the row contract (what `blz mine` hands to the
classifier) is a decision, not a hillclimb step.

### 5.5 Loop

The pstack Hillclimb playbook, with this spec's rules:

1. P16 starts from the harness (§5.1) and the baseline M for every workload in §5.2, both frozen
   once in P15 after range mining is verified, and records a green regression gate (`ctest -L
   fast`, the AC-V3-1 differential checks, the verify skills).
2. One hypothesis per attempt, from §5.3 or a new one grounded in a profile or a counter. Order
   them by measured share of M.
3. Each attempt: implement, run its §4.4 verification, measure M, keep or revert in full. One commit
   per kept change on `main`. Log every attempt in `decision.tsv` (outside the tree): id,
   hypothesis, change, before, after, delta, verification, verdict, note.
4. Independent hypotheses MAY run in parallel worktrees. Timing runs never overlap each other or a
   build.

### 5.6 Stop

Stop when §5.2's predicate holds, or when the remaining ideas are marginal and each costs more than
it can save. Never relax the predicate to meet it. If stuck, report the profile and the ideas left.

---

## 6. Datasets

### 6.1 Loading

- A **dataset descriptor** (one small file per dataset) names the transaction file, its column
  mapping, its timestamp format, and optional node-attribute tables. `blz prepare` reads it instead
  of the hard-coded IBM layout (Q13).
- IBM files (HI-Small, LI-Small, HI-Medium): `YYYY/MM/DD HH:MM`, columns Timestamp, From Bank,
  Account, To Bank, Account, Amount Received, Receiving Currency, Amount Paid, Payment Currency,
  Payment Format, Is Laundering. Nodes are (bank, account) strings (D-2).
- TransXion `tx.csv`: `YYYY-MM-DD HH:MM:SS`, columns Timestamp, From Bank, From Account, To Bank,
  To Account, then the IBM order. `person.csv` and `merchant.csv` attach attributes to accounts by
  (`bank`, `bank_account_number`). Check that every account in `tx.csv` resolves, and report the
  ones that do not.
- The epoch is midnight of the first row's day; timestamps stay int32 seconds and `blz prepare`
  rejects a range that does not fit. The stable sort, dense ids and `.blzg` cache stay as in v1/v2
  unless an agent changes them.
- Invariants per dataset: row and node counts (Table 1 for the IBM sets; F20 records LI-Small's
  6-row difference from Table 1, reported, not chased), self-loops, distinct timestamps and
  positives. The gate uses counts under `blz prepare`'s exact (bank, account) key only: HI-Small's
  are the existing invariants (591,212 self-loops); for each new dataset P15 measures them once,
  checks them by an independent count over the CSV, and pins them. F18/F19's survey figures are
  estimates and are not gate values.

### 6.2 HI-Medium preflight

HI-Medium has 31.9 M rows.

- **Prepare runs in bounded memory.** Today's parser holds every raw column in memory, about
  4.24 GB on HI-Medium before sorting (`src/io/csv.cpp:68`). P15 makes `blz prepare` stream the CSV
  (for example two passes, or chunks with an external merge) under a stated memory cap, and
  measures its peak on HI-Small and LI-Small before running it on HI-Medium. A trigger sample
  cannot preflight this step.
- **Separate budgets.** The preflight estimates, and the run records, peak host RSS for each of:
  index construction (T-CSR, P-CSR including its distinct-pair metadata, not only the
  ≈ 0.47 GB edge-run floor), auxiliary structures (slabs, labels, marks), the row chunks in
  flight, and training; and peak device memory for the graph upload, scratch and output chunks on
  the 1650S. A step whose estimate exceeds what is free is aborted and recorded, never truncated.
- Expected sizes [I]: T-CSR both directions without self-loops ≈ 0.94 GB; `clean_default` rows
  ≈ 4.2 GB in all, so rows are always delivered in chunks (§5.1), never held whole.
- Training runs from a batched `QuantileDMatrix`. If it does not fit on `local`, the preflight
  result goes to the user with the options (fewer negative rows for training only, a rented
  instance).

### 6.3 Splits

- **New datasets:** temporal 60/20/20, with V2 D-17's boundary rule at both cuts (the first eid at
  or after ⌊0.6·m⌋ that starts a new timestamp, and the same at ⌊0.8·m⌋), so no timestamp
  straddles a cut and R3 cannot let a validation row see a test edge of its own timestamp.
- **HI-Small keeps its legacy split** (fit < 3,047,216 ≤ validation < 4,062,676 ≤ test, V2 §3.3);
  its test rows stay spent. If eid 4,062,676 does not start a new timestamp, clean validation on
  HI-Small ends at the first eid of the timestamp that holds it, and those rows are scored nowhere.

### 6.4 Evaluation protocol

The new datasets' test splits are unspent. Per dataset:

1. Select the classifier recipe (feature profile among the clean ones, XGBoost settings) on that
   dataset's validation split only, with V2 D-19's rule, within a budget of 12 fits per dataset.
2. Commit the frozen recipe.
3. Score the test split once. Report F1 ×100, precision, recall and PR-AUC, next to the paper's
   Table 2 numbers where they exist.
4. No change after the test run.

TransXion MAY add its person and merchant attributes as raw classifier columns; mining ignores
them. With and without the attributes is a validation-only comparison inside step 1's budget; only
the frozen winner is scored on test.

---

## 7. Phases

Phases land on `main` in order. Each ends in a check. Within a phase the agent orders the work.

```
P12 subtract ──► P13 clean semantics ──► P14 verification stack ──► P16 hillclimb ──► P17 exit
                       │                                                ▲
                       └──────────────► P15 datasets ───────────────────┘
```

P14 and P15 are independent once P13 lands. P16 needs both.

### P12 — Subtract

- [x] Depends on: nothing.
- [x] Change: take GFP batch mode and `credit: participants` (Q11) and the GFP plumbing in the
      runtime (Q12) out of the default build and the clean code path: delete them, or isolate them
      behind legacy tooling if a GFP parity helper still needs them. Either way, migrate or retire
      each caller in the same change (for example `tools/gfp/vstats_vs_gfp.py` runs
      `blz fuzz --vstats`, and the coverage and DG-2 tools read batch recipes).
- [x] Result: the clean path and the default build have no batch mode and no GFP-only flags; every
      remaining GFP tool still runs or is recorded as retired.
- [x] Unit: `ctest -L fast` passes.
- [x] Live: `bench/ac1_cpu.sh` and, when built, `bench/ac1_cuda.sh` pass; AC-V3-0's hashes hold for
      `v1_default` and `v1_parity`.
- [x] Perf: none (no row computation changes).

**Outcome (2026-11-24): done.** Batch mode is deleted, not isolated (D-55). The vertex statistics
lose the GFP ts offset (D-56).
- The commits:
  - 0cd309d: batch mode, `credit: participants`, the GFP path of `blz coverage`, and their callers;
  - b047934: the ts offset;
  - 2df9a0c: the comments and one test that the deletions left behind;
  - 37a8f34: `bench/s5_report.py` reads E9's frozen eligibility from `results/e9_coverage.json`,
    since nothing writes `features/e9/` any more;
  - the exit checks' results, committed with this outcome.
- Retired, and last runnable at 9bdc1ee: `blz fuzz --batch`, `tools/gfp/fuzz_vs_gfp.py --batch`,
  `blz coverage --eligibility`, `tools/gfp/coverage_gfp.py`, `bench/e9.sh` with `bench/e9/`, the
  probes A34–A37 and A39–A42 with `tests/golden/gfp_batch_probes.json`, the HI-Small mode of
  `tools/gfp/vstats_vs_gfp.py`, and the profiles `batch_b4`, `batch_b32` and `dg2_g1`. Their
  results (S1, S2, E9) stay in `results/`.
- The checks:
  - `ctest -L fast` passes, 128 of 128 tests on `rel` (one CUDA-only test skipped) and 144 of
    144 on `rel-cuda`.
  - `bench/ac1_cpu.sh` and `bench/ac1_cuda.sh` pass (`results/ac1_cpu_p12_run.log`,
    `results/ac1_cuda_p12_run.log`). Every block's fnv1a equals P11's, so AC-V3-0 holds for
    `v1_default` and `v1_parity`.
  - The binaries of 9bdc1ee and of 2df9a0c were run on the same inputs, giving 122 outputs: the
    fuzz-diff summary and profile id of each of the 14 remaining profiles and feature configs on
    cpu and cuda, `blz fuzz`, `blz vstats`, `blz metamorphic`, `blz coverage` of `v1_default`, and
    `blz mine` manifests and rows. 119 outputs are identical. The other 3 are the intended
    changes: the ts columns of `blz fuzz --vstats`, whose means move by exactly 1, and the
    dropped `ts_offset` key in two `blz vstats` manifests.
  - The GFP tools still run (`results/p12_gfp_tools.json`). `probes.py` writes the remaining
    golden files byte for byte. `fuzz_vs_gfp.py` finds no mismatch on 2,000 graphs.
    `vstats_vs_gfp.py` at raw ts gives S2's counts exactly: 196,100 mismatched values, all on
    sides with evicted edges (A45).
- The review: sol-loop on gpt-6.2-sol took 2 rounds (max, then xhigh) and approved. Round 1 found
  that `bench/s5_report.py` still read `features/e9/`, fixed in 37a8f34. Its nit, that
  `fuzz_vs_gfp.py --report ""` no longer falls back to the default report, was declined because
  no caller passes an empty report.

### P13 — Clean semantics

- [x] Depends on: P12.
- [x] Change: the clean oracle (R1–R11, literal), the compiler and runtime support for the clean
      profiles (R1's loop-free index, R2's single window, R3's timestamp-only ranges, R4's order,
      R6's two sg blocks, R7's count blocks, R8's statistics), `clean_default` and `clean_wc`, and the
      new metamorphic relations (tie permutation, duplicate (u, v, t)).
- [x] Result: `blz mine --profile clean_default` runs on HI-Small on every backend.
- [x] Unit: oracle unit tests written from §3.2's rules, one per rule, each with a hand-checked
      graph; the metamorphic suite.
- [x] Live: AC-V3-1's HI-Small and fuzz parts (CPU, CUDA at the default thresholds, the
      sparse-time setting); E16's HI-Small part.
- [x] Perf: none gating; timings here are informal (the harness and baseline are frozen in P15).

**Outcome (2026-11-25): done.** `clean_wc` is `configs/profiles/clean_wc_hi_small.yaml`, with
HI-Small's W-C windows (V2 D-25). LI-Small's and HI-Medium's W-C profiles come with their
`Patterns.txt` joins in P15. `configs/profiles/clean_all.yaml` has every clean block plus R8's
statistics at 86,400 s, and the gates run on it. R8 has four sides (D-57). A computed denominator
of 0 gives skew and kurtosis 0 (D-58).
- The commits:
  - count blocks and the loop-free index: 9d4ca75, ae5c432, a2deb4f, f4e51e4, 216503f (the
    thirteen clean specs), f84b6c2 (`clean_default`, `clean_wc_hi_small`, `clean_all`), 1a8867c;
  - the clean oracle: 5ffb04d, dbf2b64 and 08f25d2 (its tests), 94af16b (D-58), fcb7b2a (the
    reference of clean profiles, with `blz mine --backend oracle --quirks`);
  - fuzz and metamorphic: d0c0e91 (metamorphic on the hybrid; a block at the work limit is
    inconclusive), 50c6b50 (tie permutation, duplicate (u, v, t)), 10021cb (the dense fuzz
    graphs pinned by a hash), 422af7e (the sparse-time setting);
  - the vertex statistics: d132baf, c69a84a, a916a1e, b5b80b5;
  - the review's fixes: 86c7776, 89523d2;
  - the gates and the A30 campaign: df0af58, f3c946b. bec68e8 lets `bench/p11_checks.py` read a
    sweep block's plan by value, so the clean profiles' `clean_<block>` patterns pass. E16:
    bfb6691, 55619aa;
  - the exit checks' results, committed with this outcome.
- The checks:
  - Unit: `tests/unit/test_clean_oracle.cpp` has one test case per rule (R1–R11, with R6a and
    R6b as two), each on a hand-checked graph. It also checks the legacy equivalence: with every quirk
    on, the clean oracle gives the legacy oracle's rows and `vstats_cpu`'s bits on fuzz graphs.
    `ctest -L fast` passes at HEAD: 168 tests on `rel` (one CUDA-only test skipped), 187 on
    `rel-cuda`.
  - AC-V3-0: `bench/ac1_cpu.sh` and `bench/ac1_cuda.sh` pass at HEAD
    (`results/ac1_{cpu,cuda}_p13_run.log`). Every block's fnv1a equals P12's.
  - HI-Small on the A30 (`results/p13_a30.json`): `clean_default`, `clean_wc_hi_small` and
    `clean_all` give the same `output_fnv1a` for every block on the oracle, cpu, cuda and hybrid.
    The vertex statistics are included.
  - Fuzz graphs on the A30 (`results/ac_v3_1_a30.json`, logs in `results/ac_v3_1_a30_logs.tgz`;
    `bench/ac_v3_1.sh`, `clean_all`, work limit 10⁶). The four gates are fuzz-diff of cpu and
    cuda against the oracle, dense and sparse. Every block, the vertex statistics included, has
    0 mismatching rows. Dense: 11,000 graphs; lcyc reached the work limit on 658, so 10,342
    completed (R11). Sparse: 10,000 graphs, none at the limit. cuda ran at its default heavy
    thresholds (0, 4, all).
  - Scope: the user stopped the campaign after these gates on 2026-11-25. The hybrid fuzz-diff
    and the metamorphic suite on every backend would have taken about 54 more hours on the A30.
    They ran only at smoke scale in U2b, with 0 mismatching rows:
    - hybrid fuzz-diff: 200 dense and 300 sparse graphs;
    - metamorphic, dense / sparse: oracle 200 / 1,000, cpu 300 / 1,000, cuda 100 / 500, hybrid
      50 / 200.
  - Run time (informal): fd_cpu_dense took 39 min on 16 vCPUs. fd_cuda_dense took 8.7 h, against
    the 5 h projected from the 1650S's smoke rate (1.65 s per graph); not investigated. Two
    likely parts: each 256-seed batch waits for its slowest oracle graph (one sampled batch's
    oracle phase took about 3 minutes, with the 16 threads about 30 % busy), and lcyc's few-warp
    searches leave most of the A30 idle (inferred).
  - E16's HI-Small part (`results/e16_clean.{json,md}`, `bench/e16.py`):
    - The setup: each quirk alone over `clean_e16` on the oracle, Q7 and Q8 as profile variants,
      and `v1_e16` as the legacy comparison. Each is fit on the fit rows with V2 R0's XGBoost
      settings and scored on validation, which ends at eid 4,062,463 (§6.3). No test row was
      scored.
    - Q2 (vertex statistics that keep self-loops) is the only quirk whose effect exceeds training
      noise. It adds +27.07 F1 points in the main fit and +10.26 to +13.57 in three resampled
      fits, against a clean F1 of 9.89. Reinvestment self-loops carry the signal: 481,056 of the
      569,769 self-loops in the fit and validation rows, none of them laundering.
    - Q4, Q8, Q9 and Q10 gain 5 to 12 points in the main fit but lose 0.06 to 7.62 points in the
      resamples. Q1, Q3, Q5, Q6 and Q7 leave the main fit's F1 unchanged.
    - The legacy check holds: `clean_e16_q8` with `--quirks all` gives `v1_e16`'s output on every
      block, the vertex statistics bit for bit.
- For §3.5 (the user's decision): whether a clean feature should keep the self-loop signal that
  Q2 shows. Nothing was changed.
- The review: sol-loop on gpt-6.2-sol (max, then xhigh) approved in round 3. Then Gemini 4 Neon
  (high) approved in round 1.
  - Round 1 found that a clean mine reaching the work limit could leave rows from earlier blocks.
    Fixed in 86c7776: lcyc runs first, and the sample's eids are written with the first block.
  - Round 1 also found `compiled_k_max` reading the clean patterns. Fixed in 89523d2.
  - Round 1's concern that work-limit stops could hide a mismatch was resolved without a change.
    Every implementation counts the same bindings, so a stop on one side only is itself a mismatch
    and fails the gate.
- Follow-ups, not fixed:
  - `CleanVstats` rejects a non-finite amount even on a self-loop, whose amount R1 keeps out of
    the values.
  - `blz bench --blocks vstats` drops the block without an error.
  - `CleanVstats::rows` trusts that sampled eids are below m; blz's trigger selection ensures it.
  - CUDA vstats issues a 0-byte copy on an empty graph.
  - `bench/a30.sh`'s wait loop runs in the local session. When the session ended mid-campaign,
    the instance kept billing with nothing waiting to stop it. A stand-in watcher downloaded the
    outputs and destroyed the instance. `jl run` has no stop-on-exit for detached runs, so the
    wait needs a way to re-attach to a running run.

### P14 — Verification stack

- [x] Depends on: P13 (the clean reference to verify against).
- [x] Change: pin the tools of §4.3; run `/create-formal-verification-skill` for each area of §4.5;
      re-prove R4's earliest-witness rule and its latest-witness mirror (`pair_latest`,
      V2 `witness: latest`) for the non-strict order; cover T1 for every pattern in `patterns/`, T2
      for the existing optimisations (sweep lowering, scan fusion, packing, reachability pruning:
      note that V2's planner applies pruning only to strict-order chains, so under R4 either extend
      and re-prove it or record it as legacy-only) and T4.
    - L1 (D-63): `verify/algorithms.sh`, Lean 4.36.0 pinned in verify/lean/lean-toolchain. It has
      24 main theorems, among them R4's earliest witness (`clean_tcyc_earliest_witness`) and its
      mirror (`clean_tchain_latest_witness`), each also in the legacy strict form. It also proves
      sg_ord's and gs's order tests, the distinct-first scan, the sweep lowering under both
      visibilities, and pruning for the strict order only (D-60). verify/lean/COVERAGE.md maps
      each theorem to the generated function and snapshot it describes, and each contract to its
      executable check.
    - L2 (D-61): `verify/codegen.sh`. `blz fuzz-diff --bound` checks every canonical graph of a
      bound, packed into disjoint unions, against the oracle. It covers every spec of `patterns/`
      (tests/profiles/l2_legacy_{stream,tsvis}.yaml, both visibilities) and of `patterns/clean/`
      (clean_all, with vstats), on CPU (both schedules, pruning both ways), CUDA (H = 0, 4, all;
      fuse and pack, the fused members at different windows), the hybrid, the `blzc --disable`
      build (rel-noopt) and ASan + UBSan (dev). `blz index-contracts` checks T4 and the `Search`
      contract that the pruning proofs assume.
    - L3 (D-62): RapidCheck pinned at commit 2c3c4365 in cmake/deps.cmake. tests/unit/test_l3.cpp
      has nine properties for what L2's bound cannot reach: K_MAX cycles and chains, R10
      saturation, packing at 2 and 4 bytes, int32 limits and batch position.
    - CBMC (L4) is not adopted in P14 (D-64).
- [x] Result: each verify skill runs with one command and reports pass with its bounds. The skills
      are `.claude/skills/verify-algorithms` (L1) and `.claude/skills/verify-codegen` (L2, L3, T4,
      V-mut), synced at 3a02e81.
- [x] Unit: each check fails on its planted breaks (V-mut) and passes on the real code. E20
      (results/e20_verification.{json,md}, 3a02e81) ran 79 checks and all passed. A check passes
      only when its report shows the comparisons ran: every block compared, the limit reached in
      the limit-1 runs, and the paths it asks for hit.
    - Proof mutants: 37 of 37 failed the build at their recorded declarations (mutant 37 was added
      after the Live run).
    - Code mutants: 13 of 13 were caught by the check `verify/mutants.py` records for each (D-65),
      and a catch counts only as a reported mismatch or a falsified property. Eight were caught by
      L2 or T4, at 1 to 4 edges. Fused members at different windows is one, caught once L2
      staggered the windows, and `keep_eq`, the review's break of the pruning check, is another.
      The other five need L3 and pass L2 at its full bounds (`--escape`): a K_MAX cycle in
      pruning, a count clamp at UINT32_MAX, int32 timestamps, packing at 4 bytes, and the pack
      kernel's tail.
- [x] Live: following each generated skill cold, on a scratch mutation of the artifact it checks (the proof project for the proof skill, the real C++ for the code skill), catches it.
    - verify-codegen, on a sweep.hpp line that counts a neighbour's second copy as new. The quick
      check failed at 2 edges (`0->1@0 0->1@0`, fan_in, got [2], want [1]).
    - verify-algorithms, on `orderTest` answering true for a missing witness. `order_test` failed
      with "Missing cases" exactly at the missing-witness inputs, and the agent named
      clean_sg_ord.hpp's `if (pw_intermediates == kNoTs) continue;` as the reference.
    - Both runs found gaps in the skills, now fixed (28839d3, fe68eeb). The staleness diff
      `SYNC..HEAD` missed uncommitted edits. A green L1 says nothing about an edit to a
      hand-written header. A model-only edit is judged against the snapshot.
- [x] Perf: the full verification run time, recorded (target: short enough to run on every
      relevant change).
    - The quick per-change subset (D-66) took 3 s for `verify/algorithms.sh`, 130 s for
      `ctest --preset rel` and 630 s for `ctest --preset rel-cuda`. About 600 s of the last
      predates P14, so P14's own fast checks add about 30 s.
    - The full run took 108 min at 3a02e81: L2 on four builds 51 min, the code mutants with
      `--escape` 53 min, the rest under 4 min. Foreign load averaged 8 % to 31 % of a core per
      step, with peaks to 63 % (measured).
- Not covered, by design or measured as out of reach (results/e20_verification.md):
  - CUDA under compute-sanitizer;
  - the hybrid's work limit;
  - a single theorem for sg_ord's count `len_I`;
  - the DFS over node sequences and machine width, which are L2's and L3's, not L1's;
  - the `Search` check beyond L2's bound;
  - CPU races, run at 2 threads only, with no thread sanitizer;
  - CUDA on sm_80 (the A30 was not used).
- The review: sol-loop on gpt-6.2-sol (max, then xhigh) approved in round 3. Then Gemini 4 Neon
  (high) approved in round 2. Every finding was in the checks, not in the generated code.
  - sol-loop found checks that could pass on wrong code. L2 compared only sg's sum of its two
    roles, the `search` contract tested one threshold of `keep`, UBSan only warned, a run could
    pass with nothing compared, any exit counted as a mutant catch, and E20 could mix revisions.
    Fixed in 90b9ee7, e79b190 and 6b72ad8.
  - Gemini found that the L3 k_max property failed on correct code in about a quarter of
    `l3_random` runs: its sanity check read the case's random bins. It also found three missing
    path gates. Fixed in 3a02e81.
  - Declined: counting sweep and search work inside the hand-written hot headers. The mutants
    caught at L2, T4 and L3 show those paths run at the checked bounds.

### P15 — Datasets

- [ ] Depends on: P13.
- [ ] Change: dataset descriptors, bounded-memory `blz prepare` (§6.2) for both schemas and
      TransXion's node tables, per-dataset epoch and range checks (Q13), the `Patterns.txt` joins
      for LI-Small and HI-Medium and their W-C windows (§3.3; the join is preflighted on its own,
      since today's `blz` join scans the whole CSV and allocates a per-row inverse index, and it is
      made bounded-memory if it exceeds the cap), the HI-Medium preflight, and
      trigger-range mining with the bounded sink (§5.1, H7).
- [ ] Result: `.blzg` caches for LI-Small, HI-Medium and TransXion; clean rows for each.
- [ ] Unit: prepare invariants per dataset (§6.1); trigger-range mining equals whole-graph mining
      on HI-Small, by row hash, for several chunk sizes.
- [ ] Live: AC-V3-1's dataset parts: oracle = CPU = CUDA on LI-Small and TransXion, oracle = CPU on
      HI-Medium in ranges; E16's TransXion part; E17.
- [ ] Perf: E17 records M and the §6.2 budgets per dataset and backend. **The harness (§5.1, the
      bounded sink included) and the clean baseline M for every §5.2 workload are frozen here,
      once**, after range mining is verified against §4.4, as the hillclimb's starting point.

### P16 — Hillclimb

- [ ] Depends on: P14 and P15.
- [ ] Change: §5's loop until §5.6.
- [ ] Result: `decision.tsv`, one commit per kept change, `results/e19_hillclimb.{json,md}`.
- [ ] Unit / live: §4.4 for every kept change; the regression gate green after each.
- [ ] Perf: M per workload, baseline to final, with spread and limiter.
- [ ] Gate G-LEGACY (§3.4).

### P17 — Exit

- [ ] Depends on: P16.
- [ ] Change: E18 (§6.4) per dataset; an optional A30 campaign (CUDA parity on sm_80 and E19 on the
      A30) with the user's approval; the status of this spec.
- [ ] Unit: `ctest -L fast`; every verify skill green at the final commit.
- [ ] Live: AC-V3-0 … AC-V3-5 checked and recorded.
- [ ] Perf: E19 final table.

---

## 8. Experiments

| ID | What | Output |
|---|---|---|
| E16 | Effect of each quirk change alone, and of `clean_e16` against `v1_e16` (§3.6) | `results/e16_clean.{json,md}` |
| E17 | Datasets: prepare invariants, mining M and peak RSS per backend | `results/e17_datasets_<host>.{json,md}` |
| E18 | Classifier per new dataset: validation selection, frozen recipe, one test score | `results/e18_<dataset>.{json,md}` |
| E19 | Hillclimb: baseline and final M per workload; the decision log summarised | `results/e19_hillclimb_<host>.{json,md}` |
| E20 | Verification: per check, bounds, run time, mutants caught | `results/e20_verification.{json,md}` |

---

## 9. Risks and gates

| ID | Risk | Rule |
|---|---|---|
| R14 | A clean rule makes a feature fire far more (R4 with minute ties, R5 with ties) or far less (R6b), and the cost or the classifier moves a lot. | E16 measures each rule alone; §3.5 lets the user flip it. Preflight on a 1 % sample before a full run. |
| R15 | HI-Medium does not fit in `local`'s RAM or the 1650S's memory. | §6.2 preflight; trigger-range mining (H7); report and ask before renting. |
| R16 | A bounded code check passes because the bound is too small to reach the failing case. | V-mut: every check must catch planted breaks; record the bound and the smallest graph each mutant needs. |
| R17 | A proof proves the hand model, not the code. | Each proof states its contracts; V-code checks the contracts on the real code; the coverage record pins the code it describes. |
| R18 | Fuzz graphs miss slab and window boundaries (dense, short time spans). | The sparse-time fuzz setting of §4.4. |
| R19 | The verification run becomes too slow to run on every change. | E20 tracks it; split the fast subset into `ctest -L fast` and keep the rest for phase exits. |
| R20 | H2's 2-path counts explode on HI-Medium hubs. | Degree-ordered enumeration or a heavy/light split of centres, or the fallback filters; measure before committing. |

---

## Appendix C‴ — decisions added by v3

| ID | Decision |
|---|---|
| D-47 | SPEC v3 is malleable (§0): agents own data structures, module boundaries, optimisation strategies and sequencing, and replace any earlier design with a recorded decision. |
| D-48 | `clean` is the default semantic family (R1–R11). Legacy profiles stay frozen; new fast paths need not support them (§3.4). |
| D-49 | GFP batch mode and `credit: participants` leave the default build and the clean path, deleted or isolated behind legacy tooling with their callers migrated or retired (Q11, P12); S1's results stand. |
| D-50 | Streaming replay (V2 S8) is postponed out of v3. |
| D-51 | The verification stack is Lean 4 (proofs), exhaustive enumeration of the real compiled code (bounded code checks), RapidCheck with `fuzz_diff` and the metamorphic suite (always-on), optional CBMC (machine-width lemmas) and runtime certificates (§4.3). |
| D-52 | A speed change is verified only with V-proof, V-code, V-cert where applicable, V-diff and V-mut (§4.4). |
| D-53 | The new datasets' test splits are scored once each, after a committed per-dataset freeze (§6.4). |
| D-54 | The user confirmed timestamp visibility (Q4, R3) and non-decreasing hop order (Q5, R4) on 2026-11-22, and chose to keep both SG variants for Q7: unordered `sg` (R6a) and ordered `sg_ord` (R6b), as two blocks of the clean profiles. |
| D-55 | P12 deletes GFP batch mode and `credit: participants` instead of isolating them. No GFP parity helper needs them once `fuzz_vs_gfp.py --batch`, `coverage_gfp.py` and E9's G1 path retire. `credit:` stays a required profile key whose only value is `trigger`, so every remaining profile id is unchanged. The frozen DG-2 recipe can no longer be recomputed in C++ (§3.4). |
| D-56 | The vertex statistics lose D-12's ts offset: `VstatsParams::ts_offset`, `blz vstats --ts-offset`, and the +1 of `blz fuzz --vstats`. `tools/gfp/vstats_vs_gfp.py` sends GFP raw ts. GFP's ts-0 artifact (A33) only drops temporal cycles, and the tool compares no temp-cycle column. The tool's HI-Small mode is retired, because F1 was mined at ts + 1. `fuzz_amounts` stays, because its ties, repeats, tiny values and large values are the cases that R8's equal-value and small-side rules need. |
| D-57 | R8's vertex statistics have four sides, GFP's: u's out-edges, u's in-edges, v's out-edges and v's in-edges, 13 columns each (`fan, deg, ratio`, then mean, sum, var, skew and kurt of the age and of the amount; 52 in all, named `vs_<side>_<stat>` with sides `u_out, u_in, v_out, v_in`). R8's first text listed two. V2's F1 and E16's comparison use four, and the side count is not a quirk of §3.1. |
| D-58 | R8's skew and kurtosis are 0 whenever their computed denominator (m2·√m2 or m2²) is 0, which includes a positive m2 that underflows (m2 = 10⁻³²⁰), not only m2 = 0; never 0/0. Q10 keeps GFP's own division, which its variance guard protects. |
| D-59 | On 2026-11-24 the user removed §2.5's limit of one A30 campaign, at the end: a campaign runs whenever a phase needs one, with the user's approval. |
| D-60 | Reachability pruning (V2 D-43) stays legacy-only in P14. The planner applies it only to strict-order chains (clean tcyc has `cpu_reach` false), and its binding check `ld[c] > h` is unsound under R4's ≤: proof mutant 36 ("prune: extended to R4's ≤ greedy") is rejected by the checker. A pruner for R4 is a P16 candidate with its own §4.4 evidence. |
| D-61 | L2's domain is every canonical graph of a bound {m, n, t}: 1..m edges over at most n nodes with ts in [0, t), ts[0] = 0, node ids in first-appearance order (one graph per relabelling orbit), every tie order, the windows 1..t (src/graph/exhaustive.hpp). Graphs are checked in disjoint-union batches through fuzz_diff's one comparison owner (`DiffSource`): every row depends only on its trigger's component, and the stable merge keeps each member's eid order (checked). That fixes each union's right answer, not that a backend runs a member the same way batched as alone: L2 claims the unions and each-alone runs its bounds hold, other standalone launch sizes are L3's, and a mismatch in a union fails the run even when no member fails alone. The bounds are A = m=5,n=5,t=4 (the smallest sg, sg_ord and gs instance plus one edge, cycles to 5) on every backend, B = m=6,n=4,t=3 on the CPU, and m=4,n=4,t=3 for the smoke tests (label fast), the sanitizer run and the CUDA work-limit run. Each graph runs alone at a work limit of 1 to check R11's stop eids. Fused members run again at different windows. CUDA is compared with the oracle, which cost no more than comparing with the CPU (measured). |
| D-62 | L3 is RapidCheck at commit 2c3c4365, run inside Catch2 v3 by test_l3.cpp's own `check()` (not `RC_ENABLE_CATCH`, which builds an old Catch). Each test's fixed seed is a hash of its name (label fast); `l3_random` (label verify) runs fresh seeds with 10× the cases, rerun with `BLZ_L3_SEED` and `BLZ_L3_SCALE`. Shrinking stops after 60 s at the smallest failing case. CUDA is compared with the CPU's rows after CPU = oracle on the same case. Generator ranges come from the compiled K_MAX, the profiles' bins and the windows; negative timestamps are in the domain. |
| D-63 | L1 models a primitive's answer as a function whose contract (`IsEarliest`, `IsLatest`, `Search`) is a hypothesis, and timestamps as unbounded integers (machine width is L3's). `blz index-contracts` checks each contract on the real code. `Search` is checked as `keep(x, h)` at every threshold h from t − δ − 1 to ldJ[x] − 1 against a transcription of the proof's round-by-round `ldJ` after 9 rounds, which is sufficient because the proofs read ld at round 9 only. `verify/algorithms.sh` builds a fresh copy of verify/lean, so every run starts from an empty `.lake`. |
| D-64 | CBMC (L4) is not adopted in P14. No lemma needs machine-width reasoning beyond what L3's int32-limit property reaches; the code mutant that narrows `first_ts_at_least`'s bound to int32 is caught there. |
| D-65 | V-mut for code is `verify/mutants.py`: one-line, text-matched mutants applied in a scratch worktree (never the main tree), built CPU-only or for sm_75 with XGBoost off, and checked cheapest first. The unmutated tree must pass every check up to the deepest recorded catch first. A mutant caught by another check than recorded (MOVED) fails the run, because its record is stale. `--escape` confirms that the L3-only mutants pass L2 at its full bounds. |
| D-66 | The quick per-change verification is `verify/algorithms.sh` plus `ctest --preset rel-cuda` (`ctest --preset rel` without a GPU). The full verification, at phase exits and before a new algorithm, lowering, optimised path or check lands, is `verify/algorithms.sh --mutants`, `verify/codegen.sh`, `l3_random` and `verify/mutants.py --escape`; `bench/e20.py` runs it with a load sampler. |
| D-67 | Dataset descriptors are `configs/datasets/<id>.yaml` with the keys `id`, `transactions`, `schema` (`ibm` or `transxion`), `split` (`legacy_hi_small` or `temporal_60_20_20`), optional `patterns` and `node_tables` (`{file, bank, account}`), and `invariants`. Paths are relative to the repository root, and an unknown key is a located error. `blz prepare --dataset <descriptor> --out <.blzg>` replaces `--csv`. The schemas are a two-row C++ table (header, date separator, seconds), not format strings. Every dataset pins edges, nodes, self_loops, distinct_timestamps, positives, b_fv, b_test and the fit, val and test positives. An `ibm` dataset also pins concatenated_keys (V2 §4.4 R1), and a dataset with node tables pins unresolved_accounts and unmatched_node_rows. `prepare` fails when a value differs, or when a count is not pinned or a pin is not counted. The pins come from `bench/p15_counts.py`, a Python count with its own parsing. It reproduces every HI-Small value of SPEC §3.5 and V2 §3.3, and adds distinct_timestamps = 15,018. `check_hi_small_invariants` and `kMaxTimestamp` are deleted (Q13). |
| D-68 | `blz prepare` reads the CSV twice through a 4 MiB buffer and never maps the whole file. The first pass counts lines, so every column is allocated once at its final size: a vector that grows by doubling can hold up to 3× a column while it reallocates. The second pass parses and hashes. The columns are kept in file order with provisional node ids and dictionary codes. The sort keys (ts − epoch) << 32 \| row take ts's place, and because they are distinct, `std::sort` gives the stable sort. Columns are then permuted one at a time. Stated cap: 44 B per row + 200 B per node + the buffer + about 11 MiB for the binary. Measured peak RSS (getrusage) and time on `local`, with the cap in brackets: HI-Small 303 MiB in 5.3 s [326 MiB] (the v2 builder took 1,395 MiB and 6.0 s), LI-Small 403 MiB in 7.5 s [440 MiB], HI-Medium 1,643 MiB in 37.7 s [1,750 MiB], TransXion 145 MiB in 2.2 s [151 MiB]. |
| D-69 | The epoch is midnight, with no time zone, of the earliest timestamp's day. §6.1's "first row" means the first row of the stream, since a file's first line need not be its earliest. ts is int32 seconds from the epoch, and `prepare` rejects a range above 2³¹ − 1. HI-Small, LI-Small and HI-Medium start on 2022-09-01, so HI-Small's ts are unchanged, and TransXion starts on 2025-01-01. `gt-join` takes the schema and the epoch from the `.blzg`, so its `ts_min` and `ts_max` count from the dataset's epoch. |
| D-70 | `.blzg` version 3 keeps every version 2 section byte for byte (HI-Small: the header's m, n and CSV hash and all 14 sections equal, `bench/p15_blzg_sections.py`). It adds a dataset section at the end: epoch, b_fv, b_test, the descriptor's SHA-256, and the strings id, epoch date, schema and split rule. The schema is stored as well, because `gt-join` reads the CSV again. The reader rejects versions 1 and 2, and a file whose splits differ from its rule's on its timestamps, so a change to a rule can never move a split silently. The file's own SHA-256 changes with the version, so features mined from a version 2 file fail `coverage`'s and E16's `blzg_sha256` checks and must be mined again. |
| D-71 | A `.blzg` owns its `Splits {b_fv, b_test}`. `select_triggers` (fit, val, train, test), `gt-join`, `stats`, `coverage` and `train` read the splits instead of recomputing them from ts. legacy_hi_small: b_test = ⌊0.8 m⌋ and b_fv = min(start(⌊0.6 m⌋), b_test). Every consumer already applied that min, and it gives HI-Small's 3,047,216 and 4,062,676. temporal_60_20_20 puts both cuts at timestamp starts (§6.3). Pinned splits: LI-Small 4,154,859 and 5,539,449, HI-Medium 19,139,453 and 25,519,347, TransXion 1,817,503 and 2,423,336. `--edges k` keeps the whole stream's splits, and a named split that ends past k is an error, not clipped. Synthetic graphs get their splits from `compute_splits`. |
| D-72 | P15 stores none of TransXion's node attributes. It only checks the resolution: 0 of the 47,526 tx accounts resolve to no person or merchant row, and 204 of the 47,730 rows match no tx account. Attributes are a P17 classifier option (§6.4), and `prepare` is cheap to rerun (2.2 s). |
| D-73 | `blz gt-join` reads the CSV through `prepare`'s 4 MiB buffer (D-68) and hashes it in the same pass, checking the SHA-256 against the `.blzg`'s after the pass. It reads the CSV a second time only when a listed line needs rule `fields`. Only the matched rows get eids, through a hash map from data row to eid filled by one scan of `file_row`, so the join itself needs the buffer and O(listed lines + matched rows), not O(m). Before this change it mapped the whole CSV and allocated an m-entry inverse index. Its measured peak RSS (getrusage) was 579 MiB on HI-Small and 781 MiB on LI-Small. A model of about 22 MiB + the CSV's bytes + 21 B per row fits both runs. The 21 B are the view's validation reads of src, dst, ts and file_row, its bitmap, the inverse index and label. On HI-Medium the model gives about 3.5 GiB, over §2.5's cap. Measured after the change: HI-Small 106 MiB in 2.1 s, LI-Small 135 MiB in 3.0 s, HI-Medium 594 MiB in 15.0 s. The model predicted 547 MiB for HI-Medium; the rest is probably the JSON of its 22,743 listed rows (inferred). HI-Small's output equals `results/gt_join.json` apart from git and paths (`bench/p15_gt_join_diff.py`). |
| D-74 | The joins of P15. LI-Small (`results/gt_join_li_small.json`) has 117 attempts and 1,023 listed rows. HI-Medium (`results/gt_join_hi_medium.json`) has 2,756 attempts and 22,743 rows. In both, every listed row matches one CSV row by text, every attempt is complete, and no listed row has label 0. Positives that no listed row maps to: LI-Small 2,542 of 3,565, HI-Medium 12,487 of 35,230 (HI-Small: 1,968 of 5,177). This is recorded, not chased (§2.5). |
| D-75 | `bench/p15_wc.py` computes the W-C windows (§3.3, V2 D-25) from a join and writes `configs/profiles/clean_wc_li_small.yaml` and `clean_wc_hi_medium.yaml`. Each is `clean_wc_hi_small` with its own windows, its own name and a header that names its join. `--check` fails on any difference, and on HI-Small it reproduces `clean_wc_hi_small`'s windows from `results/gt_join.json`. The windows, with the complete fit attempts of each group: LI-Small fan and degree 342,000 s (95 h, 14 attempts), sg and sg_ord 284,400 s (79 h, 3), tcyc 183,600 s (51 h, 1). HI-Medium fan and degree 493,200 s (137 h, 243), sg and sg_ord 514,800 s (143 h, 88), tcyc 489,600 s (136 h, 99). Every group has a complete fit attempt, so none falls back to `clean_default`, and no window reaches the 20-day cap. LI-Small's tcyc window rests on one attempt and its sg windows on three. That is recorded, not chased. GATHER-SCATTER sets no window, as in V2 D-25. |
| D-76 | TransXion has no attempt file, so its `clean_wc` workload uses `configs/profiles/clean_wc_hi_small.yaml` itself (profile id d46576bd06654fdf), labelled borrowed in reports. No TransXion profile file is added. |
| D-77 | Mining runs through one chunk driver (`src/runtime/chunk_driver.*`, H7). `ChunkDriver` builds what a backend mines from (the profile's oracle, or its index, and for cuda and hybrid the device mirror and a miner) and `run` mines each unit (a block, or a fused CUDA unit, SPEC v2 §8 S7) in chunks of `chunk_rows` triggers (a sample's next `chunk_rows` eids), lcyc first (R11), then vstats. Each block writes into its own stream of a `BlockSink` (`src/runtime/block_sink.*`), so a fused unit has two streams open. A stream takes whole rows in eid order and keeps one tally for every sink: FNV-1a over the data bytes continued across chunks (the whole block's `output_fnv1a`), rows, chunks, instances (the sum of the uint32 cells) and saturated emits. `NpySink` writes `<block>.npy.tmp` and renames it at finish, and a stream destroyed unfinished removes it, so a work-limit abort leaves no block file. `HashSink` hashes, counts and drops (§5.1). `MemorySink` keeps rows for tests and for fuzz-diff through the driver. Pinned host buffers and the CUDA miner's device scratch grow once and are reused by later chunks and blocks. `blz mine` is its flags, the driver and `NpySink`. Its manifest adds `chunk_rows`, `memory` (index bytes, row buffer bytes, RSS by kind) and per block `chunks`, `whole`, `instances` and `write_seconds` (the sink's time). `write_vstats` is deleted. cuda and hybrid still take an eid range only. `file_sha256` reads through a 4 MiB buffer: mapping HI-Medium's 1.2 GB `.blzg` to hash it kept it all in RSS. |
| D-78 | The sinks hash with an exact FNV-1a over cells. A uint32 cell below 256 is one step with the prime to the 4th power (its 3 zero high bytes), below 65,536 two steps, eight zero cells one multiply by the prime to the 32nd. A float64 skips its low zero bytes the same way. The value equals byte-wise FNV-1a (unit test on adversarial cells, splits at every position, and HI-Small's 7 clean_default blocks). Measured on those blocks' 670 MB: 0.069 s against 0.69 s byte-wise, while mining them takes about 1.2 s on 12 threads. At byte-wise speed the hash would have been about 30 % of HI-Small's M and a floor on every hillclimb speedup. |
| D-79 | A block with a CPU sweep plan (SPEC v2 §7.1: fan and degree) runs once over the whole selection on cpu and on the hybrid, whose CPU side runs it too, and its rows go to the sink in chunks. A sweep walks every list entry whatever the range, so chunking it would repeat the walk per chunk. `include/blz/sweep.hpp` is unchanged. The buffer is rows × cols × 4 B: 128 MB for a clean count block on HI-Medium. A legacy binned fan block would be 32 times that, but legacy profiles run on HI-Small only. |
| D-80 | `blz measure` is §5.1's M. Every block of the profile and its vstats run through the chunk driver into `HashSink`, `--reps` times (5). The `.blzg` is mapped, its used columns are faulted in, and for cuda and hybrid the CUDA context is created, before the first clock. A rep is timed from the start of index construction (plus the graph upload for cuda and hybrid) to the last chunk hashed. It records its errors, triggers, instances, saturated emits, index and row buffer bytes, RSS, `/proc/loadavg` before and after, and per block its time and row hash. A rep that throws, or whose hashes differ from the first completed rep's, is an error. The result holds the median, min and max, per block too, and `blz bench`'s environment. It exits 2 on any error. |
| D-81 | The harness chunk is 2^20 = 1,048,576 triggers (`kDefaultChunkRows`), chosen by a rule written before the run: 2^20 if its median overhead over one chunk is within one chunk's min-to-max spread on cpu and cuda, else 2^22. `bench/p15_chunk_cost.py` ran HI-Small `clean_default` at one chunk, 2^22, 2^20 and 2^18, two interleaved rounds of 5 reps each (`results/p15_chunk_cost_local.json`; about one core of foreign load). cpu, 12 threads: one chunk 1.551 s [1.501, 1.774], 2^20 1.556 s [1.502, 1.796]. cuda: one chunk 1.285 s [1.263, 1.366], 2^20 1.201 s [1.186, 1.245]. The cuda gain is outside the blocks: one chunk pins 203 MB of rows (sg's 10 cells × 5.08 M), about 105 ms more than 2^20's 42 MB. Inside the blocks 2^20 costs about +5 ms (cpu) and +10 ms (cuda), and 2^18 about +35 ms and +52 ms, so smaller chunks start to cost. At 2^20 a HI-Medium chunk of a 10-cell block is 42 MB on the host, and as much again pinned and on the device. |
| D-82 | HI-Medium (m = 31,898,238, n = 2,077,023) mines within the desktop's memory. Both runs below are `clean_default` on cpu with 12 threads at the 2^20 chunk. `blz mine --triggers range:15000000:17000000` (2,000,000 triggers) takes 9.6 s, 3.9 s of it the index. Its peak RSS is 2,093 MiB: 1,564 anonymous and 528 file-backed. `blz measure --reps 1` over the whole graph (`results/p15_measure_hi_medium_local.json`) gives M = 32.8 s with 0 errors and 0 saturated emits: index 3.2 s, tcyc 15.7 s, sg 9.1 s, sg_ord 2.6 s and about 0.55 s for each sweep block. Its peak RSS is 2,303 MiB, reached during the sweep blocks. File-backed: 526 MiB, the used columns (src, dst, ts) faulted in plus the binary. Anonymous: 1,777 MiB. The index takes 1,420 MiB (1,489 MB, 47 B per edge, the term that grows with m). The row buffers take 122 MiB: the sweep blocks' whole-selection rows, which sg's 40 MiB chunks reuse. The distinct sweep's per-thread `seen` arrays take 95 MiB (12 × n × 4 B). About 140 MiB is left unattributed (thread stacks and the allocator). After the sweeps the anonymous part settles at 1,647 MiB. While `file_sha256` reads the 1.2 GB file (the first 4.5 s), file-backed RSS stays at 529 MiB. |
| D-83 | The chunked L2 checks (H7, §4.4 V-code) were chosen by measured cost. Times are on `local`, 12 threads. The cpu checks (`cpu-chunks`) run on the three L2 profiles. In chunks of 1, 2 and 3 at SMOKE they take 3.2 s on clean_all and 2.4 s on each legacy profile. In chunks of 3 and 1000 at A they take 132, 91 and 100 s, against 16,384-edge unions; chunks of 3 alone took 96 s on clean_all for 522 M puts. The GPU checks (`cuda-chunks`, `cuda-fuse-pack-chunks`, `hybrid-chunks`) run at H = C = 4. In chunks of 1000 at A on the three profiles they take 40 to 108 s each. In chunks of 1, 2 and 3 at CHUNKS_GPU = SMOKE they run on clean_all only, in 2,048-edge unions (cuda 322 s, --fuse --pack 361 s, hybrid 452 s, 12.8 M puts each). A put costs the GPU about 25 µs when 16 cases' drivers overlap, against 60 µs (cuda) and 158 µs (hybrid) for a case alone. At A, chunks of 1 to 3 would be about 0.5 G puts on the GPU, which is hours. The 2,048-edge unions give 59 cases at SMOKE where 16,384-edge unions give 10, so the drivers overlap. The legacy profiles get the GPU's chunks of 1000 at A, and the cpu's chunks of 1 to 3. The driver's code is the same for every profile, and bench/p15_chunks.py has checked HI-Small's legacy and clean profiles through it on every backend (c59e8a6). H = 4 runs both kernels (at SMOKE with --fuse --pack, 17.7 M light and 9,324 heavy triggers), and the direct checks cover H = 0 and all. `--smoke`, ctest's `l2_smoke_chunks_cuda`, `_cuda_fuse_pack` and `_hybrid`, and the mutant rungs run the GPU's chunks of 1 to 3 at CHUNKS_GPU_SMOKE = m=3,n=4,t=3. They use 256-edge unions, which took 11, 14 and 17 s where 2,048-edge unions took 17, 17 and 42 s. They run at H = C = 2: at H = C = 4, 3-edge graphs give no heavy triggers and no CPU-side triggers. ctest's `l2_smoke_chunks_cpu_<profile>` runs the cpu's at SMOKE. A chunked check also needs, in every block and in vstats, a stream of more than one put. It needs a run over a range, and the path its backend takes: sweep streams (cpu, hybrid), fused streams and 1-byte packed rows (--fuse --pack), and a sample (cpu). The chunked checks add 5.5 min to rel's part of verify/codegen.sh and 28 min to rel-cuda's (all 18 passed in 35.7 min). `verify/codegen.sh --only=GLOB` runs the checks whose names match. |
| D-84 | `blz fuzz-diff --chunk-rows K1,K2,…` (`fuzz_diff_chunks`) runs the backend through `ChunkDriver` into a `MemorySink`, as `blz mine` and `blz measure` do. It runs instead of the backend's direct paths, not beside them. It runs once per chunk size and threshold, over each case's selections: every trigger, the range [m/4, m − m/5), and on cpu a 40 % sample. cpu runs the dynamic schedule. The hybrid's CPU side runs the sorted schedule with pruning, as `fuzz_diff_hybrid` does. Each stream's rows and vstats are compared with the reference's at its triggers. Its tally is compared too: rows, ⌈rows / k⌉ puts, FNV-1a against `fnv1a_bytes` over the reference's cells, and instances. A `std::logic_error` from the driver (a stream's guard, `slice`'s precondition) counts as a mismatch of the block whose stream opened last, and its counterexample prints a note. Any other error still stops the run. Under R11, the run over every trigger must stop at the reference's lcyc stop eid and keep no stream. A sub-selection of a case the reference stopped is inconclusive. The cases run in parallel on every backend, 16 at a time, and each case's drivers own their device graph, miner and pinned buffers. Parallel cases ran 2.3× faster on cuda at SMOKE. The report adds `chunk_rows`, `selections` and `path_hits.chunk_driver`. Seeds work too. On 20 dense seeds at a work limit of 10⁶, cuda took 25 s in chunks of 64 at H = 4 and 293 s in chunks of 7 and 64 at the three default thresholds. The direct paths took 68 s. A small chunk probably splits the GPU's lcyc searches into many short launches (inferred, not profiled). `bench/ac_v3_1.sh` passes `CHUNKS` to its fuzz-diff runs. The 10,000-graph and metamorphic runs and E20 were not rerun through the driver. Chunk sizes near a graph's trigger count keep those runs near their direct cost. |
| D-85 | A bug, found by D-83's path hit and fixed in 024be12. From b8bfa99 on, the chunk driver reported a packed D2H width of 4 for every block. `BlockRun::cuda` took the widest of its chunks' widths with max, starting from `CudaBlockTiming`'s default of 4. So `blz mine --pack`'s manifest (`cuda.pack.width`) said 4 bytes whatever was copied. Rows and hashes were not affected. The width now starts at 0. HI-Small `clean_default` on cuda --fuse --pack (triggers 0 to 10⁶) now reports widths 1 and 2, matching its bytes. `cuda-fuse-pack-chunks` requires 1-byte packed rows. |
| D-86 | H7's V-proof is `verify/lean/Blz/Chunks.lean`, in D-63's style. A backend's answer for a part is a function `mine`. Its contract `Local` (each part's cells are its triggers' rows, whatever the part) is a hypothesis, and D-84's runs check it on the code. The module models the driver's loops (`chunksFrom`, `runBlock`, `runChunked`), `slice`, and `BlockStream::put`'s FNV-1a continued across puts. It has 8 main theorems. `chunks_cover`, `mem_chunks` and `length_chunks` cover the chunk enumeration, `slice_eids` covers `slice`, `runChunked_flatten` and `runBlock_flatten` (any part size, so D-79's sweep path too) give chunked mining = whole mining, and `stream_hash` and `runBlock_hash` give the hash. Proof mutants 38 to 48 each fail where recorded. Out of the model: the lost-emit count, the block order and the abort path, the sinks' bytes, and the buffers' reuse. |
| D-87 | L3 adds one chunked property. Its cases are random graphs past the bound, and stars and layered graphs with cells past 255, mined through the driver at a chunk size drawn from 1 to m + 1. cpu runs every profile against the oracle. In a CUDA build, cuda fused and packed runs on clean_all and the hybrid on one profile, both against the CPU. It reaches cases L2 cannot: 2-byte cells in the driver's packed chunks, a hub's heavy triggers, and a chunk that holds every trigger of a graph past the bound. It takes about 7 s at its fixed seed. The driver and its sinks get 8 one-line code mutants: `chunk_end_past_sel`, `chunk_last_dropped`, `whole_offset_rows`, `hash_restarts`, `row_buffer_unzeroed`, `sample_slice_start`, `vstats_slice_start` and `fused_slot_swapped`. The new rungs `l2-smoke-cpu-chunks` (cpu) and `l2-chunks-cuda`, `l2-chunks-cuda-fuse-pack` and `l2-chunks-hybrid` (cuda, at CHUNKS_GPU_SMOKE) sit after each ladder's direct smoke rungs, which never enter the driver. At b5b70af all 8 are caught where recorded. `l2-smoke-cpu-chunks` catches the 7 cpu mutants in 2.7 to 5.5 s, on graphs of 1 or 2 edges. `l2-chunks-cuda-fuse-pack` catches `fused_slot_swapped` in 15.7 s. The 5 mutants recorded at `l3` are still caught there behind the new rungs. The unmutated tree passes every rung up to `l3` on both ladders. The run took 952 s with builds. `whole_offset_rows` passes on clean_all, whose sweep blocks are count blocks of one cell, and l2_legacy_stream's binned fan and degree blocks catch it. |
| D-88 | Sync commits after U3b. `/verify-algorithms` moves from 3a02e81 to b5b70af. Between them, the only change to its covered files before U3b was SPEC_V3.md outside §3.2 (P15's §5 to §7 and decisions), which no model describes. At b5b70af `verify/algorithms.sh --mutants` passes (32 main theorems, 48 mutants) and `blzc_snapshots` passes. `/verify-codegen` stays at 3a02e81. Its record now covers the chunk driver, its sinks and `blz measure`'s hash path. Its chunked checks (D-83), the L3 property and the driver's mutants (D-87) passed at 6edf4ab, but its full run (verify/codegen.sh on every build, `l3_random`, `verify/mutants.py --escape`) has not run since 3a02e81. |
| D-89 | A feature folder belongs to a graph when it was mined from the same CSV: `blz coverage`, `bench/dg2/ladder.py`'s `Features` and `bench/e16.py`'s `Blocks` compare the folder's recorded CSV SHA-256 (`run.csv_sha256` from `blz mine`, the top-level `csv_sha256` of `blz vstats` and tools/gfp) with the `.blzg` header's, not the `.blzg` file's own SHA-256. D-70 keeps every column byte for byte across versions, so the CSV's hash identifies the stream, and the folders mined from a version 2 file stay usable without mining them again. Checked with no mining at HI-Small v3: `Features` accepts features/p7_cpu_default, dg2_R6_G1 (tools/gfp) and p10_sg_ordered_v1w_oracle, whose recorded `.blzg` hash 9ac57fd… is v2's (v3's is 7b2e608…), and `blz coverage` on p7_cpu_default under v1_default runs in 3.2 s and gives results/e9_coverage.json's eligibility and activation for every typology (detection needs `--pred`, which this check did not pass). |

## Appendix E — planning prototypes (not normative)

Built on 2026-11-22 to settle §4.2 and §5.3; throwaway, outside the repository, and they may be
gone. An agent may reuse them or ignore them.

Copied to `~/.cache/blz-fv/prototypes-2026-11-22/` (tool installs stay in `~/.cache/blz-fv/<tool>/`):

- `lean/`: the five Lean modules (earliest witness, pruning soundness, the sg hand model, wedge
  counting, the degree sweep), `mutate.py` (13 mutants) and `check.sh`.
- `dafny/`: `tcyc_greedy.dfy`, `greedy_native.dfy`, `sg_gather.dfy`, `mutants.py`, `run.sh`.
- `cbmc/`: `tcode.cpp`, `index_cf.hpp`, `talgo.c`, `tsweep.cpp`, the lowering script `prep.sh`, and
  **`exh.cpp`, the exhaustive enumeration harness that L2 starts from**, with `diff_index.cpp`.
- `pbt/`: `pbt.cpp` (six RapidCheck properties, including the reference-free `flags` property),
  `mutate.py`, `rerun.sh`, `results/`.
- `algos/`: both speed surveys (`opus.md`, `codex.md`), the operation-counting replica
  `scripts/work.cpp` and the HI-Medium counters `scripts/scc_csv.cpp`, `scripts/bfly_csv.cpp`, with
  their outputs. These produced every [M3] count in §1 and §5.3.

# BlazingAML replication — implementation SPEC v2

Status: draft for review · Date: 2026-11-11 · Scope: HI-Small (other datasets gated, S9), C++20
(OpenMP + CUDA), no FraudGT.

This document plans v2. It is a **delta on `SPEC.md` (v1)**: every v1 rule stays normative unless a
section here replaces it, and section references without a prefix (§4.6, D-8, A28) point into
`SPEC.md`. Where the two disagree, this document wins for v2 work. MUST / SHOULD / MAY carry their
RFC 2119 meaning.

Tags, in addition to v1's [P] (paper), [G] (GFP docs or docstring), [X] (GFP probe) and [D]
(decision):

- **[V1]** a measured v1 result, with its file under `results/`.
- **[W]** an external source with its URL, read on 2026-11-11.
- **[R]** recalled, not checked against a source. An [R] fact MUST be checked before the step that
  depends on it runs, and the check is recorded in the DG-2 ledger (§4.8).

Provenance: planned on 2026-11-11 with a one-round brainstorm with GPT-6.1 Sol (max effort); the
scope tiers, the validation protocol, the sweep lowering (§7.1) and the metamorphic tests (§3.4)
come from that round.

---

## 1. Where v1 ended

These facts drive the plan.

| # | Fact | Source |
|---|---|---|
| F1 | AC1–AC5 hold. The oracle equals GFP stream mode on every HI-Small row; generated CPU and CUDA code equal the oracle bit for bit on HI-Small and 10,000 fuzz graphs, on sm_75 and sm_80. | [V1] `e1_parity.json` |
| F2 | **DG-2 is open and parked.** F1 ×100 for the five Table 2 sets: 0.0, 0.9, 6.1, 6.6, 6.4 against the paper's 11.1, 23.2, 38.4, 42.6, 46.6. Precision 0.77, recall 0.03. | [V1] `e2_table2.md` |
| F3 | The XGB-only gap (0.0 against 11.1) involves no mining at all, so part of DG-2 lies in the input encoding or the classifier, not in the features. | F2 |
| F4 | The v1 DG-2 diagnostics already looked at **test** scores (§17). The test split is therefore exposed: v2 results are disciplined replication comparisons, not a blind evaluation. | §17 |
| F5 | Under the v1 windows (D-8: SG 6 h, the rest 24 h) the cycle and SG features barely fire on HI-Small: GFP finds 1,125 temporal cycles (1,101 of length 2) and 808 SG instances in the parity bins. | [V1] `e0_profile.json`, §16 P1 |
| F6 | The listed laundering attempts span days, not hours. Median span (max ts − min ts): SCATTER-GATHER 89 h, CYCLE 72 h, FAN-IN 85 h, FAN-OUT 77 h, GATHER-SCATTER 151 h, STACK 73 h. Only 0% of SG attempts and 2% of CYCLE attempts fit in 6 h; 2% and 9% fit in 24 h. | `dataset/HI-Small_Patterns.txt`, computed 2026-11-11 |
| F7 | snapml 1.17.2's **library defaults** differ from D-8: `fan_tw` = `degree_tw` = 43,200 (12 h), `scatter-gather_tw` = 432,000 (5 d), `temp-cycle_tw` = 1,728,000 (20 d), `lc-cycle_tw` = 864,000 (10 d), `lc-cycle_len` = 10, `vertex_stats` on with `vertex_stats_tw` = 1,728,000, `vertex_stats_cols` = [3], `vertex_stats_feats` = [0,1,2,3,4,8,9,10], bins 2..30. The defaults enable only `fan` and vertex statistics. | [G] `GraphFeaturePreprocessor().get_params()`, local venv |
| F8 | Altman et al. run GFP with **batch size 128**, SG 6 h, "one day" for the other features, simple cycles up to length 10, and vertex statistics on Amount and Timestamp. Account IDs are not features. The split is temporal 60/20/20. HI-Small minority-class F1: GFP + LightGBM 62.86 ± 0.25, GFP + XGBoost 63.23 ± 0.17. So D-8's windows match this paper. | [W] https://arxiv.org/html/2306.16424 (§4, App. D, Table 2) |
| F9 | IBM's Multi-GNN formatter assigns `EdgeID` = file row index, account ids in **file order** (key = bank string + account string, one shared dictionary, source before destination within a row), and codes by first appearance in file order: one dictionary shared by both currency columns (Receiving Currency coded before Payment Currency within a row) and a separate one for Payment Format. Timestamps = seconds since the first row's midnight minus 10 s. It then sorts rows by timestamp and does **not** renumber `EdgeID`. Output header: `EdgeID,from_id,to_id,Timestamp,Amount Sent,Sent Currency,Amount Received,Received Currency,Payment Format,Is Laundering`. | [W] https://raw.githubusercontent.com/IBM/Multi-GNN/main/format_kaggle_files.py |
| F10 | CPU time is dominated by `fan_out` (10.7 s of 17.2 s at 1 thread). It rescans an overlapping window per trigger. | [V1] `e4_cpu_local.md` |
| F11 | GPU tcyc is dominated by a few thousand heavy triggers (A30: 29 ms for 2,885 triggers). On the A30, transfers cost as much as kernels (57.5 ms D2H + 29 ms upload against 84.4 ms of kernels). | [V1] `e5_gpu_a30.md` |
| F12 | E6 (the XGBoost sensitivity grid) was never run. | `results/` |
| F13 | `HI-Small_Patterns.txt` lists 370 attempts with 3,209 transaction rows, against 5,177 positive rows. At least 1,968 positives belong to no listed attempt. | computed 2026-11-11 |

---

## 2. Goals, scope, acceptance

### 2.1 Goals

- **G6. Close DG-2** with a bounded, pre-registered pipeline-identification ladder (§4). The
  ladder ends with a verdict whatever the numbers are. Matching the paper is not required.
- **G7. Ground-truth coverage.** Explain *where* the features see laundering: per typology, does
  the structure fit the windows, does a feature fire, does the classifier detect it (§5).
- **G8. Temporal expressivity.** Implement the two temporal capabilities the paper claims but never
  evaluates: timestamp visibility (Fig. 5's literal `break if e.t > t`) and cross-operand ordering
  inside an intersection (Fig. 3 option 1), with a new `sg_ordered` feature (§6).
- **G9. Compiler performance.** A sliding-window lowering for fan and degree, and the optimization
  ablation the paper never gives (§7).
- **G10. Contracts.** Versioned semantic profiles, feature manifests, a validation split and
  metamorphic tests, so that v2's many variants cannot silently change v1 outputs (§3).
- **G11. The v1 P6 stretch list**, each item as a gated v2 stretch item with its own acceptance
  test (§8).

### 2.2 Tiers

- **v2 core** = P7–P11 (§12). v2 is done when AC-V2-0 … AC-V2-5 hold.
- **v2 stretch** = S1–S9 (§8). Each starts only after P10 and the local part of P11, when its gate
  is met and the user approves it, so GPU stretch work can still join the A30 campaign (§7.3). A
  stretch item MUST NOT block v2 core.

Every v1 P6 item appears in v2:

| v1 P6 item | v2 home |
|---|---|
| `credit=participants` and `visibility=timestamp`, with F1 impact | timestamp: core (§6.1, E10); participants: S1 |
| Cross-operand ordering, `sg_ordered` | core (§6.2–§6.3) |
| Raw-attribute and vertex-statistics features | DG-2 rungs R3 and R7 (core); C++ vertex statistics: S2 |
| `lc-cycle` | S3 (and DG-2 rung R7 through GFP) |
| `union` / `difference` | S4 |
| Stack, bipartite, gather-scatter | S5 |
| Hybrid CPU/GPU for tcyc | S6 |
| Streaming replay | S8 |
| Other datasets | S9 |

### 2.3 Out of scope for v2

- FraudGT and its Fig. 12 comparison. A BlazingAML-only throughput number is S8.
- LI-Large and HI-Large; Medium datasets only through S9's preflight.
- Multi-GPU; GNN classifiers; a CPU cloud; a live ingestion service.
- Any DG-2 experiment that is not on the pre-registered ladder (§4.4).
- Development builds on the A30 (memory: the instance is paid and destroyed after each run).

### 2.4 Acceptance criteria (v2 core)

| ID | Criterion |
|---|---|
| AC-V2-0 | **No regression.** `ctest -L fast` passes. Mining profile `v1_default` (§3.1) with the oracle, the CPU backend and (when built) the CUDA backend gives, for every block, the `output_fnv1a` recorded in `results/e4_cpu_local.json` (fan_in 10768036772042710147, fan_out 904878896487262018, deg_in 3321375241280117763, deg_out 3341707662053112211, sg 18124683238158106329, tcyc 7737672036140136160). `bench/ac1_cpu.sh` and `bench/ac1_cuda.sh` pass again at the end of P11. |
| AC-V2-1 | **Contracts.** Every `blz mine` output carries a profile id and a manifest (§3.2). The metamorphic tests (§3.4) pass for the oracle and the CPU backend on 2,000 fuzz graphs per supported profile, and the fast suite runs 200 of them, also on CUDA when built. |
| AC-V2-2 | **DG-2 closed.** `results/dg2_ledger.{json,md}` lists every rung and candidate of §4.4, evaluated or marked skipped with a reason. Budget totals are within §4.2. Selection used validation scores only. The frozen recipe was committed before the test run (the freeze commit precedes E8's commit). E8 reports the five Table 2 columns and the Table 3 matrix for `v1_default` and for the frozen recipe, with the verdict of §4.6. §17's DG-2 row is updated to *closed*. |
| AC-V2-3 | **Coverage.** All 3,209 listed rows and all 370 attempts have a join status (§3.5). E9 (§5) reports, per typology and per block, structural eligibility, activation and detection, for `v1_default` and for the frozen recipe, plus the count of unlisted positives. |
| AC-V2-4 | **Temporal semantics.** Under `visibility: timestamp`, the six v1 blocks, and `sg_ordered` under both visibilities, are bit-identical across oracle, CPU and CUDA on HI-Small and on 10,000 fuzz graphs (CUDA at H = 0, 4 and `all`, on the 1650S; sm_80 in the A30 campaign, §7.3). tcyc rows are equal under both visibilities (§6.1). Each new invalid-spec case (§6.4) fails with its code at its `file:line:col`. E10 is published. |
| AC-V2-5 | **Performance.** The sweep lowering (§7.1) is bit-identical to the oracle on HI-Small for every profile that uses it and on 10,000 fuzz graphs. E11 (§7.2) publishes the ablation, with every variant's output hash equal to the fully optimized build's. The A30 campaign (§7.3) publishes E5 for `v1_default` and E12. |

### 2.5 Working rules (v1 rules kept, plus new ones)

- **No tuning on test.** In v2 this means: every choice is made on the validation split (§3.3),
  and test scores are computed once, after the freeze (§4.5).
- **Report the gap, then move on.** DG-2 closes at the end of P8 whatever its verdict. Leads that
  remain are listed, not chased (§4.6).
- **Bit-exact backends; same-machine ratios only** (v1 R6).
- **Shared desktop timing.** Before a timing run, check `ps --sort=-pcpu` for foreign jobs and wait
  for an idle window; never stop the user's jobs. Every timing run logs the top CPU consumers every
  10 s into its results JSON. No Sol review that builds may overlap a timing run.
- **A30.** One campaign at the end of v2 core (§7.3). Stretch GPU work rides along only if it is
  finished before the campaign; a second campaign needs the user's approval.
- **Preflight before anything large.** A step whose time or memory is unknown first runs on a
  sample (§11 `--triggers sample:`), and it is aborted, never silently truncated, when the
  extrapolation exceeds its limit. The abort is recorded.

---

## 3. Contracts (P7)

### 3.1 Semantic profiles

A **profile** names everything that determines feature values. Profiles live in
`configs/profiles/<name>.yaml`:

```yaml
version: 2
profile: v1_default          # identifier, recorded in every output
visibility: stream           # stream | timestamp (§6.1)
credit: trigger              # trigger | participants (S1; rejected until S1 exists)
k_max: 10
blocks:                      # output order = listed order
  fan_in:  {patterns: [fan_in],  window: 86400, bins: [2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 30, 50]}
  fan_out: {patterns: [fan_out], window: 86400, bins: [2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 30, 50]}
  deg_in:  {patterns: [deg_in],  window: 86400, bins: [2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 30, 50]}
  deg_out: {patterns: [deg_out], window: 86400, bins: [2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 30, 50]}
  sg:      {patterns: [sg_gather, sg_scatter], window: 21600, bins: [2, 3, 4, 5, 6, 8, 10, 13, 16, 20]}
  tcyc:    {patterns: [tcyc], window: 86400, bins: [2, 3, 4, 5, 6, 7, 8, 9, 10]}
```

- **Compatibility.** A config file without `version` is a v1 feature config
  (`configs/features_default.yaml`, `features_parity.yaml`). It loads as a profile with
  `visibility: stream`, `credit: trigger`, and, per block, every registered pattern whose `block`
  matches. `features_default.yaml` and `profiles/v1_default.yaml` therefore resolve to the same
  profile and the same profile id.
- **Profile id** = the first 16 hex digits of SHA-256 over the canonical JSON (sorted keys, no
  whitespace) of the resolved profile, with each listed pattern replaced by the SHA-256 of its spec
  file, plus a `block_order` array of the block names in output order (sorted keys would otherwise
  hide a reordering). It identifies semantics, not the build.
- **Frozen profiles.** `v1_default` and `v1_parity` are frozen: AC-V2-0 pins their outputs. A new
  semantic choice is a new profile, never an edit of a frozen one [D-23].
- **Registration** (extends §7.6). A profile that names an unknown pattern, or whose `visibility`
  conflicts with a pattern that pins its visibility (§6.1), fails with **E403**. A profile that a
  backend cannot run (for example `credit: participants` on CUDA) fails with **E404** naming the
  backend. Both report the profile file and key.

As implemented (P7):

- The loader is `src/runtime/profile.{hpp,cpp}`. `blz mine` and `blz bench` take exactly one of
  `--profile` and `--config`; each loads either format. A profile requires `version: 2`,
  `profile` (an identifier), `visibility`, `credit`, `k_max` (≥ 2) and a non-empty `blocks` map,
  each block with `patterns`, `window` and `bins` (strictly increasing). An unknown or duplicate
  key, block or listed pattern is an error at `<file>:<line>:<col>`. Window lower bounds still
  come from each pattern's spec (fan and degree open, sg and tcyc closed).
- The canonical JSON is `{visibility, credit, k_max, blocks: {<block>: {patterns: [<spec
  SHA-256>…], window, bins}}, block_order}`. Patterns keep their listed order. Ids: `v1_default` =
  `features_default.yaml` = **b21580d6e129821a**, `v1_parity` = `features_parity.yaml` =
  **8c8237bf05998d07**. The SHA is taken over the spec's text, so any edit of a library spec,
  even a comment, changes every id that uses it. This is deliberately conservative.
- `PatternInfo` (§7.6 of v1) gains `spec_sha256`, `visibility` and `visibility_loc`. No library
  spec has a `visibility:` key yet, so every pin is empty and the E403 visibility check fires only
  for specs that add one (P10).
- E403 also covers a pattern listed under a block it does not emit into.
- E404, at start-up, against the profile key:
  - `credit: participants` on every backend (until S1);
  - `visibility: timestamp` on the oracle (until P8) and the generated backends (until P10);
  - on the oracle, a block whose pattern list is not the set that §9 of v1 computes for it
    (`sg` = `sg_gather` + `sg_scatter`);
  - on cpu and cuda, a `k_max` other than the compiled `K_MAX` of a listed pattern that declares
    one (today only `tcyc`).
- A v1 config on cpu or cuda keeps the v1 start-up checks (E402, the K_MAX message), so it
  behaves as in v1. **P10 must revisit this:** once `sg_ordered` is registered with a block of its
  own, a v1 config that does not define that block fails E402 on the generated backends.
  Resolved in P10: the v1 check skips registered patterns of known non-v1 blocks (§6.3, "As
  implemented (P10)").

### 3.2 Feature manifest

`features.json` (§8 of v1) gains: `profile`, `profile_id`, the resolved profile, each pattern's
spec SHA-256, the backend and its plan options (§7.2), the trigger selection, and the `.blzg` SHA.
Every results JSON that consumes features records the profile ids it used.

As implemented (P7): `features.json` has `manifest_version: 2` and these keys:

- `profile`: name, file, format, id, `causal`, `row_sees` (what a row may see, §3.3), and the
  resolved profile;
- `patterns`: each listed pattern's block, spec, spec SHA-256 and visibility pin;
- `backend`: name, threads, `disabled_optimizations` (empty until P11), and the schedule or the
  heavy threshold;
- `triggers`, `blzg_sha256`, `csv_sha256`, `peak_rss_mb`, and `output_fnv1a` per block.

The `run` key, which decides whether earlier blocks in the directory are kept, now includes
`blzg_sha256`, `profile_id` and the trigger selection. `blz train` records the features' profile
and profile id and rejects sampled features. `blz bench`, `fuzz-diff` and `metamorphic` record
the profile name and id.

### 3.3 Splits [D-17]

v1's split stays: train = `eid < 4,062,676`, test = the last 1,015,669 edges (§3.4). v2 splits
train into **fit** and **validation** for every selection decision:

- `b_fv` = the first eid ≥ ⌊0.6·m⌋ = 3,047,007 whose timestamp differs from its predecessor's, so
  that no timestamp is split between fit and validation. On HI-Small `b_fv` = **3,047,216**
  (fit ends at ts 480,960, validation starts at ts 481,020 = 2022/09/06 13:37).
- fit = `eid < 3,047,216` (2,299 positives); validation = `3,047,216 ≤ eid < 4,062,676`
  (1,015,460 edges, 1,081 positives); test unchanged (1,797 positives).
- `blz prepare` asserts these counts as new §3.5 invariants.
- The paper's 80/20 boundary splits a timestamp (§3.4); that is kept for comparability with the
  paper.

**Mining stays on the whole stream.** For a causal profile (stream visibility, trigger credit) a
row depends only on earlier edges, so the split needs no special handling. A non-causal profile
(timestamp visibility, GFP batch mode, participants credit) can let a row see later edges. Its
manifest MUST say so, and §4.4 states, per rung, which later edges a row may see.

As implemented (P7): `fit_val_boundary()` (`include/blz/edges.hpp`) and `--triggers fit|val`.
`blz prepare` asserts five new invariants: `b_fv`, the 1,015,460 validation edges, the fit and
validation positives, and R1's 515,088 bank+account keys. It also runs the `file_row` check of
§3.5. All of them hold on HI-Small.

### 3.4 Metamorphic tests

These catch semantic mistakes that backend parity cannot, because all backends would share them.
They run on fuzz graphs (§13.3) for the oracle and the CPU backend, and on CUDA when built.

| Test | Rule | Profiles |
|---|---|---|
| Prefix closure | Mining the first k edges of a stream gives the same rows for those k edges as mining the whole stream. Under timestamp visibility, k is cut at a timestamp boundary. | causal profiles; timestamp visibility with the cut rule |
| Relabeling | Permuting node ids consistently leaves every row unchanged. | all |
| Time shift | Adding a constant to every timestamp leaves every row unchanged. | all |
| Visibility invariance of tcyc | tcyc rows are equal under stream and timestamp visibility (§6.1). | tcyc |
| Order relaxation | For every row, the number of `sg_ordered` instances (the sum over its bins) is ≤ the number of `sg_gather` instances when `sg_gather` is mined alone at the same window and bins. Ordering only removes sources and intermediates. Bins are not compared, because an instance can move to a lower bin. | `sg_ordered` |

As implemented (P7): `src/runtime/metamorphic.{hpp,cpp}` and
`blz metamorphic --seeds a:b [--backend oracle|cpu|cuda] [--profile p] [--bins …] [--heavy …]`.

- Prefix closure uses two cuts per graph, k drawn from [1, m) by a seeded generator. It applies
  to trigger credit only.
- Relabeling applies a seeded permutation of all node ids.
- Time shift adds a seeded constant in [1, 2^30].
- CUDA runs every graph at each heavy threshold (default 0, 4, `all`).
- Visibility invariance runs on the oracle since P8, for profiles with tcyc. It mines tcyc alone under the
  other visibility and compares rows; it joins cpu and CUDA in P10. Order relaxation joins in P10.
- A test runs only where it applies (`meta_test_applies`), and `blz metamorphic` reports `applies`
  for each test. Prefix closure under timestamp visibility moves its cut to a timestamp boundary
  (`tests/profiles/tsvis.yaml`, slow test `metamorphic_tsvis_oracle_2k`).
- Fast suite: 200 graphs on the oracle and cpu, 60 graphs of `v1_parity` on cpu, and 200 on CUDA.
  Slow suite: `metamorphic_<profile>_<backend>_2k` for both frozen profiles.

As implemented (P10): order relaxation mines `sg_gather` alone, with the same backend, at
`sg_ord`'s window and bins, and compares the per-row sums. The sum is sound because the last bin is
open-ended: it counts every instance of size ≥ max(2, b₀). The oracle mines `sg_gather` alone
through `Oracle::mine_sg_gather`, because its sg block is both roles. `registry_profile()` now
includes `sg_ord`, so the fast suite runs order relaxation on every backend. The slow suite adds
`metamorphic_sg_ordered_{v1w,tsvis}_<backend>_2k`.

### 3.5 Ground-truth join

`blz gt-join --graph data/hi_small.blzg --csv dataset/HI-Small_Trans.csv --patterns dataset/HI-Small_Patterns.txt --out results/gt_join.json`

- Each `Patterns.txt` transaction line is matched to CSV rows with **identical text** (both use
  the CSV row format). If none matches, a field-wise match is tried (timestamp, banks and accounts
  as strings, amounts parsed as decimals, currencies, format), and the result records which rule
  matched.
- Identical CSV lines form a duplicate group. If a group has as many matches as listed rows, the
  k-th listed occurrence maps to the k-th occurrence in stream order (`resolved-duplicate`).
  Otherwise the rows are `ambiguous`.
- Row status: `unique`, `resolved-duplicate`, `ambiguous`, `unmatched`. Attempt status:
  `complete` (all rows unique or resolved), `partial`, `unmatched`.
- Per attempt: typology, header text (e.g. "Max 16-degree Fan-Out"), eids, account count, span,
  and split membership (`fit`, `val`, `test`, or `straddles`).
- **Unlisted positives** = positive eids that no attempt row maps to.
- `gt_join.json` is small (3,209 rows) and is committed.

`blz prepare` also starts storing `file_row u32[m]`, each edge's 0-based row in the CSV, in
`.blzg` format version 2 [D-31]. The v1 reader rejects version 2 with a clear message;
`blz prepare` re-creates the cache. Invariants: `file_row` is a permutation of 0..m−1, and the
stable sort is reproduced from it.

As implemented (P7): `src/graph/gt_join.{hpp,cpp}`.

- `blz gt-join` refuses a CSV whose SHA-256 differs from the one the graph was prepared from.
- The field rule is tried only against CSV rows that match no listed line as text.
- An attempt's span and account count cover all its listed rows. Its split is `null` when no
  row maps.
- The file has one line per attempt and per row, so diffs stay readable.
- `file_row` is stored after `format`, so the offsets of the first nine `.blzg` arrays, which
  the Python tools read, do not change. The file grows from 178.6 to 198.9 MB.

**HI-Small (`results/gt_join.json`):**

- All 3,209 listed rows match exactly one CSV row by text: no duplicates, no field-rule matches,
  no ambiguity. All 370 attempts are `complete`.
- Every mapped eid is a positive, and 1,968 of the 5,177 positives are unlisted.
- Attempt splits: fit 104, val 16, test 95, straddles 155. The median attempt spans 74 h and the
  longest 202 h.
- R4's W-C therefore draws on only 26 fan, 9 SG and 19 cycle attempts that lie wholly in fit.

### 3.6 Experiment ledger

`results/dg2_ledger.json` holds one entry per fit and per feature pass:

- rung and candidate id;
- the profile id, or the GFP call recipe for a GFP-produced pass;
- the column list (names, count);
- validation metrics: F1 ×100, precision, recall, PR-AUC (§10.3), confusion counts;
- wall time, peak RSS, the machine, the foreign-load log;
- running totals of fits, feature passes and machine-hours (§4.2);
- for [R] facts, the check that confirmed or refuted them.

`bench/dg2/ledger.py` renders `results/dg2_ledger.md`.

---

As implemented (P7): `bench/dg2/ledger.py`.

- Commands: `init` (the §4.4 ladder with every candidate `pending`), `add <entry.json>`,
  `skip <rung> <candidate> budget|limit`, `render` and `selftest`. The selftest is a fast ctest.
- `add_entry()` is the API for P8's `ladder.py`:
  - it validates an entry's fields (§3.6), including exactly one of `profile_id` and
    `gfp_recipe`;
  - it recomputes the running totals;
  - it refuses a pass beyond the pass cap and any entry after the fit or time cap.
- R8's seven grid points are named X1–X7; P8 binds them to grid values before R8 runs.
- `results/dg2_ledger.{json,md}` is committed as the skeleton.

## 4. DG-2 identification ladder (P8)

### 4.1 Principles

- **Identify, don't optimize.** Each rung tests one source-backed hypothesis about how the paper
  built its pipeline. The ladder is fixed by this section before any rung runs; committing this
  file is the pre-registration [D-18]. A change to the ladder needs a new `D-n` recorded before the
  affected rung runs.
- **Validation only.** Fits train on *fit* rows and score *validation* rows (§3.3). Test rows are
  scored once, in E8 (§4.5).
- **Ordered, not factorial.** One recipe is carried forward from each rung (§4.3). Branches the
  budget skips are listed in the ledger.
- **Cheapest first.** Rungs that need only new fits come before rungs that need feature passes.
- **Python tools** [D-26]. Fits use Python XGBoost 3.4.x (v1 showed that it gives the same
  confusion counts as the C API path, §16 P2) through a `QuantileDMatrix` built from row batches,
  so that peak RSS stays ≤ 3 GB on `local` (v1 R4). The tool is `bench/dg2/ladder.py`. A feature
  pass comes from `blz mine` (any backend for which AC1 holds), from the oracle (for variants that
  only the oracle has), or from GFP itself through `tools/gfp/dg2_features.py`. That script is
  never used for parity, and it is the only place where batch mode and `fit_transform` are
  allowed.
- **Classifier defaults.** Every fit uses `configs/xgb_default.yaml` (D-11) unless its rung changes
  the classifier.

### 4.2 Budget and stop rule [D-20]

Three caps:

- **40 fits** (one XGBoost training plus its validation scoring);
- **8 feature passes** (one full HI-Small mining of one profile or GFP recipe; a pass aborted in
  preflight counts);
- **4 machine-hours** of wall time on `local`, summed over fits and passes.

Reaching the pass cap stops further passes only; fits that need no new pass (such as R8) continue.
Reaching the fit cap or the time cap stops the ladder. The final test run (E8) is outside this
budget. Candidates not run are recorded as `skipped: budget`.

### 4.3 Selection rule [D-19]

- The metric is validation F1 ×100 of the laundering class at threshold 0.5. PR-AUC is reported
  next to it.
- Each rung names a **selection set**: `xgb_only` for R1, `all` for every other rung. Both sets are
  fitted and recorded where the rung says so.
- A candidate replaces the incumbent when its validation F1 on the selection set is **at least 2.0
  points** higher. If several qualify, the highest wins; ties within 0.1 go to the candidate with
  fewer columns, then to the one listed first in §4.4.
- Otherwise the incumbent is kept. A rung never makes things worse by construction, and small,
  noisy gains are not chased (about 1,081 validation positives give an F1 standard error of the
  order of 1 point).

### 4.4 Rungs

The incumbent at the start is **R0 = v1**: stream-order dense ids `src_id`, `dst_id`; no raw
columns; XGBoost defaults; profile `v1_default`. Its validation scores are fitted first (2 fits:
`xgb_only` and `all`).

| Rung | Hypothesis | Candidates | Sets fitted | Selection set | Fits | Passes |
|---|---|---|---|---|---:|---:|
| R1 | Input encoding (F3, F9) | I1, I2, I3 | `xgb_only`, `all` | `xgb_only` | 6 | 0 |
| R2 | Classifier contract | C1 | `xgb_only`, `all` | `all` | 2 | 0 |
| R3 | Raw schema | S1, S2 | `xgb_only`, `all` | `all` | 4 | 0 |
| R4 | Windows (F5–F8) | W-B, W-C | `all` | `all` | 2 | 2 |
| R5 | Visibility (Fig. 5 literal) | V1 | `all` | `all` | 1 | 1 |
| R6 | GFP call semantics (F8, A28, A29) | G1, G2, G3 | `all` | `all` | 3 | 3 |
| R7 | Feature families (F7, F8) | F1, F2 | `all` | `all` | 2 | 2 |
| R8 | XGBoost grid (the v1 E6 grid) | 7 grid points | `all` | `all` | 7 | 0 |
| | **Total with R0** | | | | **29** | **8** |

Every feature pass, GFP passes included, is aborted when it exceeds 60 minutes of wall time or 3 GB
of peak RSS. GFP has no trigger-sample mode, so this time limit is its only preflight. An aborted
pass counts against the budget and its candidate is recorded as `skipped: limit`.

**R1 — input encoding** (fits only). The paper's baseline is "source and destination account IDs"
(Table 2 caption) or "transaction ID, source account and destination account" (§8.4) [P §9.19].

- I1: account ids assigned as in Multi-GNN (F9): key = bank string + account string, one dictionary,
  in file order, sources before destinations. P7 asserts that this key yields the same 515,088
  nodes as D-2.
- I2: I1 + `EdgeID` = `file_row` (F9: Multi-GNN never renumbers it after sorting).
- I3: R0 + `eid` (stream order), the v1 E6 variant.
- A large I2 gain with no I1 gain would mean that file order itself carries label signal. That is
  a property of the dataset files, and E8 reports it as such.

**R2 — classifier contract** (fits only).

- C1: `base_score = 0.5` fixed. [R] XGBoost ≥ 2.0 estimates the intercept from the labels, and
  earlier versions used 0.5 (https://xgboost.readthedocs.io/en/stable/parameter.html, cited by the
  brainstorm, to be checked).
- Older XGBoost defaults for `tree_method` are not tested: `QuantileDMatrix` supports only
  `hist`, and a plain `DMatrix` for a non-hist method does not fit the RAM budget (R12).
- The resolved configuration of every fit (`Booster.save_config()`) is stored in the ledger.

**R3 — raw schema** (fits only). Added raw columns are part of the base: every cumulative set,
`xgb_only` included, gets them.

- S1: the Multi-GNN columns (F9): Timestamp, Amount Sent (= `amount_paid`), Sent Currency, Amount
  Received, Received Currency, Payment Format, with codes assigned exactly as the formatter assigns
  them (F9).
- S2: S1 without Timestamp. Under a temporal split, test timestamps lie outside the training
  range, so a timestamp column may only add noise.

**R4 — windows** (2 passes, `blz mine` on the CPU backend). Bins stay D-9; only δ changes [D-25].

- W-B: snapml 1.17.2 library defaults (F7): fan and degree 43,200; SG 432,000; tcyc 1,728,000.
- W-C: per GFP pattern group, the 90th percentile of the span of the `complete` attempts of the
  matching typologies that lie entirely in the fit split (§3.5), rounded up to a whole hour and
  capped at 1,728,000. fan (both blocks) and degree (both blocks) ↔ FAN-IN ∪ FAN-OUT, sg ↔
  SCATTER-GATHER, tcyc ↔ CYCLE. One window per group keeps W-C runnable in GFP, which has one
  `fan_tw` and one `degree_tw`. Using fit-split ground truth is training-label use, not test use.
- Preflight: each candidate first mines a 1% trigger sample (seed 0). The pass is aborted if the
  extrapolated wall time exceeds 45 minutes at 12 threads, or the extrapolated peak RSS exceeds
  3 GB. tcyc keeps `K_MAX = 10` here; DG-4 (§13) re-checks it before E8 if the window grows.

**R5 — visibility** (1 pass, the oracle with `--visibility timestamp`, §6.1).

- V1: `visibility: timestamp`. A row may see edges with the same timestamp that come later in the
  file. HI-Small has minute resolution and a mean of 338 edges per timestamp (`e0_profile.json`), so
  this differs materially from stream visibility for fan, degree and SG. tcyc is unchanged
  (§6.1).

**R6 — GFP call semantics** (3 passes, GFP itself, at the incumbent windows and bins, all four
v1 pattern groups, `vertex_stats=False`, ts + 1 as in D-12).

- G1: a fresh preprocessor, `transform()` on consecutive batches of 128 edges in stream order over
  the whole stream (Altman's batch size, F8). A batch is cut short at `b_fv` and at 4,062,676, so
  no batch mixes fit, validation and test rows; the same preprocessor continues, so the graph state
  carries over.
- G2: the same with batches of 2,048.
- G3: `fit_transform(fit rows)` then `transform(validation rows)` on snapml 1.17.2, the usual
  scikit-learn call pattern. A29 predicts inflated fit-row counts; the rung measures what that does
  to F1. If adopted, E8 uses `fit_transform(train)` then `transform(test)`.
- What a row may see: in G1 and G2, later edges of its own batch (A28), never edges of a later
  split; in G3, every fit row sees every other fit row.
- First, `tools/gfp/probes.py` gains the batch probes of Appendix A′ (A34–A37), so that G1–G3
  are interpreted against measured behaviour.

**R7 — feature families** (2 passes, GFP itself).

- F1: GFP vertex statistics, stream mode (batch 1, fresh preprocessor, ts + 1), on Amount Received
  and Timestamp, window 86,400 (Altman, F8), default `vertex_stats_feats`. They enter the base like
  raw columns. Altman says only "Amount"; Amount Received is this spec's pre-declared reading.
- F2: the GFP lc-cycle block (simple cycles, length ≤ 10, window 86,400, bins as tcyc), stream
  mode, appended after sg. It does not replace tcyc.
- Which XGBoost column means "Fan" or "Degree" in Table 2 (a histogram block or a vertex-statistic
  scalar) cannot be decided by the `all` score. E8 reports both readings descriptively and does not
  select between them.

**R8 — XGBoost grid** (fits only). The v1 E6 grid, `scale_pos_weight ∈ {1, 5}` × `max_depth ∈
{6, 10}` × `num_boost_round ∈ {100, 500}`, minus the incumbent point, on the carried recipe.

As implemented (P8): `bench/dg2/ladder.py` (a uv script: numpy, xgboost 3.4.x, pyyaml),
`bench/dg2/blzg.py` (a numpy `.blzg` v2 reader) and `tools/gfp/dg2_features.py`.

- **Running.** `uv run --python 3.12 bench/dg2/ladder.py --rung R<k> [--candidate <id>]
  [--facts <json>]` runs a rung's pending candidates in §4.4 order, appends every fit and pass to
  the ledger, and decides the rung (§4.3) once none is pending. Rungs run in order, and R0 cannot
  be rerun.
- **Smoke mode.** `--smoke --ledger <scratch>` exercises the plumbing on every rung: it trains on
  eids [0, 1,000,000), scores [2,000,000, 3,000,000), and uses stand-in feature directories (no
  pass runs). It refuses the real ledger.
- **Fits.**
  - A `QuantileDMatrix` is built from batches of 262,144 rows; the validation matrix uses
    `ref` = the fit matrix.
  - `configs/xgb_default.yaml`, with `nthread` = every core.
  - Each entry stores `Booster.save_config()`, the resolved `base_score`, the ten columns with the
    most gain, and peak RSS as VmHWM (reset through `/proc/self/clear_refs`).
  - The foreign-load log samples, every 10 s, the processes outside the ladder that use ≥ 5% of a
    core.
- **Columns.** In order:
  1. ids: `src_id`, `dst_id` (v1, I3, + `eid`), or `from_id`, `to_id` (I1, I2, + `EdgeID` =
     `file_row`);
  2. raw (S1, S2);
  3. vertex statistics (F1);
  4. for `all`, the six blocks in v1's order, then `lcyc` (F2).
- **F9 codes.**
  - Accounts: key = bank + account, numbered by first appearance in file order (`file_row`),
    the source before the destination.
  - Currencies: one dictionary, Receiving Currency before Payment Currency within a row.
  - Payment Format: its own dictionary.
  - `Timestamp` = ts + 10. The formatter counts from 10 s before midnight of the first row's day.
  - Amounts are rounded as the formatter's `%f` rounds them.
- **Passes.**
  - A pass runs under a monitor that kills its process group past 60 minutes or 3 GB.
  - Peak RSS is the largest VmHWM, sampled every 0.2 s. A child's `ru_maxrss` cannot be used: it
    keeps the ladder's RSS from the fork, and so does `blz`'s own `peak_rss_mb` when the ladder
    launches it.
  - R4's preflight is `blz mine --backend cpu --triggers sample:1:0`. Extrapolated time = index
    build + Σ `extrapolated_mine_seconds`. Extrapolated RSS = the preflight's peak + (m − rows) ×
    the widest block × 4 B, because `blz` holds one block's rows at a time.
  - A preflight is also stopped early, once its elapsed time extrapolates past twice the limit.
  - A pass, preflight included, may also run no longer than what is left of the time cap. A pass
    killed at the cap counts and its candidate is `skipped: budget`.
  - The cap is checked before each fit, and a fit that has started runs to completion: reaching
    the cap stops the ladder (§4.2).
- **R2.** The entry stores the `--facts` check of the [R] fact together with R0's resolved
  `base_score`.
- **R4.** W-C from `results/gt_join.json`:

  | Group | Fit-split `complete` attempts | p90 (s) | Window (s) |
  |---|---:|---:|---:|
  | fan, degree | 26 | 318,180 | 320,400 |
  | sg | 9 | 340,500 | 342,000 |
  | tcyc | 19 | 319,404 | 320,400 |

  The profiles are `configs/profiles/dg2_wb.yaml` (a04862978f42e397) and `dg2_wc.yaml`
  (64747da4a40c0d70). `ladder.py` recomputes W-C before the pass and refuses a mismatch.
- **R5.** `ladder.py` writes `configs/profiles/dg2_v1.yaml`: the incumbent's resolved profile
  with `visibility: timestamp`, mined by the oracle.
  - The graph is cut at 4,062,676 (`blz mine --edges`, §11). The validation/test boundary splits
    timestamp 663,120, which has 213 validation rows and 105 test edges. Under timestamp
    visibility, those validation rows would otherwise see test edges.
  - A frozen V1 would need the same cut for E8's train rows.
- **R6, R7.** `dg2_features.py --recipe G1|G2|G3|F1|F2 --like <incumbent>/features.json`.
  - Windows and bins are taken from `--like`, which may be a `blz` manifest or one written by this
    script. GFP is given `gfp_bins()` (D-32).
  - The manifest records:
    - `gfp_recipe` (id, call pattern, params, snapml version);
    - the rows and what a row may see (`row_sees`, `causal`);
    - the `.blzg` and CSV SHA-256, and `like`;
    - seconds, VmHWM and git.
- **R8.** X1–X7, in nesting order (`scale_pos_weight`, `max_depth`, `num_boost_round`):

  | Point | `scale_pos_weight` | `max_depth` | `num_boost_round` |
  |---|---:|---:|---:|
  | X1 | 1 | 6 | 500 |
  | X2 | 1 | 10 | 100 |
  | X3 | 1 | 10 | 500 |
  | X4 | 5 | 6 | 100 |
  | X5 | 5 | 6 | 500 |
  | X6 | 5 | 10 | 100 |
  | X7 | 5 | 10 | 500 |

  `ledger.py init` writes the grid.
- **Selection.** `ledger.py decide()` applies §4.3 with a tolerance of 10⁻⁹.
  - A candidate whose fits are incomplete (a cap reached part-way through) is not selectable.
  - The incumbent records its source, recipe, scores per set and column counts.
  - A set the winning rung did not fit is recorded as `null`.

### 4.5 Freeze and final test (E8)

1. **Freeze.** The carried recipe after the last rung reached (input encoding, classifier, raw
   schema, profile, GFP recipe, families, XGBoost params) is written to
   `configs/dg2_frozen.yaml` and committed with the ledger.
2. **Pre-test checks.** DG-3 and DG-4 (§13) are resolved and recorded.
3. **Test run.** For `v1_default` (which must reproduce v1's E2/E3 counts exactly) and for the
   frozen recipe:
   - train on the whole train split, score the test split;
   - the five cumulative Table 2 sets and the Table 3 matrix. Base columns (ids, raw, vertex
     statistics) are in every set; the pattern blocks are added in v1's order (§10.1);
   - the R8 grid on the frozen recipe, reported in full as sensitivity, never used to select;
   - per-block gain importances, so that the cumulative columns can be read.
4. No recipe changes after the test run. A defect found later is reported, not fixed in place.

### 4.6 Verdict and closure [D-21]

E8 ends with one verdict, decided by these pre-declared thresholds:

- **AGREE:** every one of the five Table 2 cells is within ±3.0 F1 points of the paper, and Table 3
  precision and recall are within ±0.03 of 0.947 and 0.311 [P §7.2].
- **PARTIAL:** `all` is within ±5.0 points but AGREE fails.
- **GAP:** otherwise. Each cell's distance is reported.

AGREE means the evidence is compatible with the paper; it does not prove that the pipeline is the
paper's. In every case DG-2 is marked **closed** in §17, the remaining leads are listed, and v2 moves
on.

### 4.7 Cost

About 4 machine-hours of ladder time plus about 1 hour for E8, on `local`. No A30 time.

---

## 5. Ground-truth coverage (P9, E9)

For each `complete` attempt (§3.5), each block b, and each profile π in {`v1_default`, frozen}:

- **Structurally eligible**: mining b under π on a graph that contains *only the attempt's edges*
  (eids preserved) gives a non-zero row for at least one of them. This uses the oracle, so it
  applies exactly the semantics of §4 and §6 and needs no per-typology rule. For a block that the
  frozen recipe takes from GFP (R6, R7), eligibility runs that same GFP recipe (call mode, window,
  bins, ts + 1) on the isolated attempt graph instead, and is marked so; 370 tiny graphs cost
  seconds.
- **Activated**: in the full HI-Small mining under π, at least one attempt edge has a non-zero row
  in b.
- **Detected**, split-restricted: `det_val` = the share of the attempt's validation edges predicted
  positive by the validation-selected model, and `det_test` = the share of its test edges predicted
  positive by E8's model. A `straddles` attempt contributes only its edges in each split; fit edges
  are training data and are never scored. An attempt with no edges in a split has `N/A` there, and
  every mean reports its denominator (attempts and edges).

`results/e9_coverage.{json,md}` reports, per typology:

- attempt counts by join status and split;
- span percentiles;
- for each block, the share of attempts structurally eligible and the share activated;
- mean detection;
- the counts of unlisted positives and of positives that no block activates.

E9 is descriptive. It is computed after the freeze and never feeds back into §4.

Cost: 370 tiny oracle runs plus two full minings already done; minutes.

As implemented (P9): `blz coverage` (`src/runtime/coverage.{hpp,cpp}`), `tools/gfp/coverage_gfp.py`,
`bench/e9/predict.py` and `bench/e9/report.py`, run in that order by `bench/e9.sh`.

- **Isolated graph.** The attempt's mapped eids (unique and resolved rows), ascending, so stream
  and ts order are kept, with nodes renumbered by first appearance. Every §4 rule is invariant
  under the renumbering (§3.4 relabeling). The oracle mines it at the profile's windows, bins,
  `k_max` and visibility.
- **GFP blocks.** `blz coverage` refuses GFP-produced features without `--eligibility` from
  `tools/gfp/coverage_gfp.py`. That tool reruns the pass's recipe: the manifest's
  `gfp_recipe.params` verbatim, ts + 1, original eids and node ids. Each edge goes in the batch it
  had in the full stream [D-33]. Every run first checks that the prefix [0, 4,096) (32 whole fit
  batches) gives the pass's rows exactly. `blz coverage` still runs the oracle at the profile
  (`dg2_wc` for the frozen recipe) and reports it as `eligible_oracle`.
- **Activation** reads a full mining under the profile: a `blz` manifest with the same profile id
  and every trigger, or a GFP pass with the same window and bins per block.
- **Detection.** `--pred` is `u8[m]`: 0 and 1 = predicted negative and positive at the threshold
  of `configs/xgb_default.yaml` (score > 0.5), 255 = not scored.
  - Fit rows must be 255.
  - A split is scored in full or not at all.
  - Neither the ladder nor E8 kept a model or its predictions, so `bench/e9/predict.py` refits
    the four models: R0 `all` and the frozen recipe's source (R8/X7) on fit → validation, and
    E8's two `all` models on train → test.
  - It writes nothing unless each refit reproduces its recorded confusion counts exactly [D-34].
- **Conventions.**
  - `any` = any block of the profile.
  - A *listed* positive is an eid that some attempt row maps to.
  - The frozen recipe's F1 vertex statistics are aggregates, not pattern blocks: they get no
    eligibility or activation, but they are in its detection model [D-34].
- **Also reported:**
  - attempts eligible but not activated, expected 0: every pattern's count only grows when edges
    are added;
  - the base rate of non-zero rows per block among all edges and among positives;
  - recall on listed and on unlisted positives.

---

## 6. Temporal semantics extensions (P10)

### 6.1 Timestamp visibility

**Definition.** `Vis_ts(e) = {e' : t' ≤ t}`. Everything else in §4.1–§4.6 of v1 is unchanged: the
window lower bound per block, the self-loop rule, min size 2, the binning.

- **tcyc is invariant.** Every non-trigger cycle edge needs `t_i < t` (§4.6), so edges at
  `t' = t` can never take part. The metamorphic test of §3.4 asserts it.
- **Runtime parameter** [D-22]. Visibility becomes a profile parameter passed in `Params`, like the
  window span. The generated range construction selects the upper bound once per range
  (`eid ≤ e` or `ts ≤ t`); there is no per-element cost. The CUDA bodies are the same text.
- **The spec key.** `visibility:` in a spec becomes optional. If present it **pins** the pattern
  to that visibility, and a profile with a different visibility fails with E403. The library specs
  drop the key. Their snapshots change, and AC-V2-0 shows that the outputs do not.
- E501 no longer covers `visibility: timestamp`.
- **Oracle.** `--visibility timestamp` filters by `t' ≤ t` instead of `id' ≤ id`. P8 needs only
  this oracle part (R5); the compiler and CUDA parts are P10.
- **Probes.** GFP has no timestamp visibility, so there is no GFP parity. The oracle definition
  above is normative.

As implemented (P8): the oracle reads the profile's `visibility` (`--backend oracle --profile p`;
§11). A window's upper end is `t' ≤ t` (a partition point on timestamps) instead of `eid' ≤ eid`.
cpu and cuda reject `visibility: timestamp` with E404 until P10. A unit test checks that timestamp
visibility equals stream visibility once the trigger is moved to the end of its timestamp group
(200 fuzz graphs, against the cpu backend).

As implemented (P10):

- `Params` gains `ts_visibility`, and `blz_core` passes the profile's visibility to every run.
- The generated prologue sets `vis_e` (e, or `kAnyEid`) and `vis_t` (`kNoUpper`, or t) once per
  trigger. Every range and probe takes `min_ts(<own upper bounds>, vis_t)` as its upper time and
  `vis_e` as its eid bound.
- The work estimates use the same bounds. The CUDA bodies are the same text.
- `mine_cpu` and `CudaMiner::mine` with a pattern list take the visibility as a required argument.
  The v1 forms (a `Block`) stay stream.
- E404 no longer covers `visibility: timestamp`, and neither does E501. A spec's `visibility:` key
  only pins: the generated code is the same, and the registry records the pin for E403.
- `blzc --dump-ir` prints `visibility runtime`, or `pinned <v>` for a spec with the key.
- Visibility invariance of tcyc runs on every backend. Its tcyc-only profile is not re-checked: a
  v1 config would fail E402.
- The snapshots were regenerated in their own commit. `v1_default` keeps its id and its six
  pinned hashes on cpu.

### 6.2 Cross-operand ordering in an intersection

v1 rejects a predicate that compares the two scanned edges of one intersection (E203, "P6"). v2
allows exactly one such predicate per intersection stage, as an **order constraint**.

- **Form.** `skip_if: "ea.t OP eb.t"` where `edges: [ea, eb]` are the stage's two scanned edges,
  OP ∈ {`<`, `<=`, `>`, `>=`}, with no arithmetic offset and not inside `or`.
- **Meaning (existential)** [D-27]. Element y is kept when there is at least one pair of witness
  edges (a on ea's operand, b on eb's operand, each within its own bounds) for which the predicate
  is false. Example: `skip_if: "e2.t >= e3.t"` keeps y when some `S→y` edge is strictly earlier
  than some `y→W` edge.
- **Lowering.** With `min`/`max` = the earliest/latest witness timestamp of an operand for y:

  | Predicate | y is kept when |
  |---|---|
  | `skip_if: ea.t >= eb.t` | `min(a) < max(b)` |
  | `skip_if: ea.t > eb.t` | `min(a) <= max(b)` |
  | `skip_if: ea.t <= eb.t` | `max(a) > min(b)` |
  | `skip_if: ea.t < eb.t` | `max(a) >= min(b)` |

- **New primitive.** `PCsrView::pair_latest(x, y, lo_ts, hi_ts, max_eid)` returns the latest
  witness timestamp, or `kNoTs` (two binary searches, like `pair_exists`).
- **Plan.** The scan side is still chosen by window size at runtime (§7.3 of v1). The scan only
  enumerates distinct candidates. Each witness that the lowering needs comes from P-CSR
  (`pair_exists` for an earliest, `pair_latest` for a latest), except an earliest witness on the
  scan side, which distinct-first already gives. The plan comment records which.
- **Scope.** The edges of an ordered intersection are not exported to later stages; referring to
  them is E102.
- **Errors.** A second order constraint in one stage, an order constraint inside `or`, in a
  `break_if`, with `==` / `!=`, with an offset, or between edges that are not the stage's two
  scanned edges is **E205**. Other mixed predicates stay E203.

As implemented (P10):

- `lower.cpp` classifies the parts of a stage's `skip_if` after `split_or`. Before splitting, a
  comparison between the two scanned edges inside `or`, or in a `break_if`, is E205. A part that
  compares two scanned edges goes to the order constraint. Its checks run in this order: `==`/`!=`,
  an offset (an operand that is not a bare `e.t`), an edge that is not one of the stage's two
  (including the same edge on both sides), then a second constraint. All are E205.
- A constraint written in reverse (`eb.t OP ea.t`) is flipped first. The IR's `Order` holds `a`,
  `b`, the normalized `skip`, `keep`, and whether each side needs its latest witness
  (`a_latest`, `b_latest`). `blzc --dump-ir` prints `order skip if … : keep if min(e2) < max(e3)`.
- In the generated loop, the probed side's witness comes from the existence probe itself:
  `pair_latest` or `pair_exists`, according to the side the runtime scans. The scanned side's
  earliest witness is its distinct-first entry. Its latest witness costs one more `pair_latest`
  call, on the scanned side's own P-CSR. Every call takes the visibility bounds (`vis_t`, `vis_e`).
- A unit test checks every operator, in both writing orders, against brute force over random
  witness lists with ties and duplicates.
- A binding intersection with an order constraint exports no edges, so a later use is E102 (case
  `e102_ordered_edge_scope.yaml`). The P6 case `e203_cross_operand.yaml` is replaced by six E205
  cases. A same-edge comparison in a count stage (`e1.t <= e1.t`) is E205 now.

### 6.3 `sg_ordered` [D-28]

Fig. 3 option 1 [P §2.1]: each intermediate forwards after it receives. Under backward windows the
trigger completes such an instance only as a **gather** edge: a scatter edge at time t would need a
later forward hop, which is not visible. So `sg_ordered` has the gather role only.

**Definition.** For a trigger `e = (x→w, t)` with `x ≠ w`, and each source
`s ∈ N_in(x) \ {x, w}` that has an edge `s→x` in the window with timestamp `< t`:

    I_ord(s, w) = { y ≠ w : ∃ (s→y)@a, (y→w)@b in the window with a < b }

Emit `|I_ord|` when it is ≥ 2. The trigger's own intermediate x is always in `I_ord` (its earliest
`s→x` edge precedes the trigger). The exclusions are those of the gather role (§4.5): s can be in
`I_ord` through its own self-loop.

```yaml
# patterns/sg_ordered.yaml — gather role, intermediates forward after they receive
version: 1
pattern: sg_ordered
block: sg_ord
window: {span: delta_sg, lower: closed}
trigger: {edge: e0, src: X, dst: W}
constraints: [{skip_if: "X == W"}]
stages:
  - id: sources
    op: for_all
    src: X.in
    edge: e1
    dst_var: S
    constraints:
      - break_if: "e1.t >= t"          # x received from s strictly before it forwarded
      - skip_if: "S in [X, W]"
  - id: intermediates
    op: intersection
    src: S.out
    dst: W.in
    edges: [e2, e3]
    output: I
    element: Y
    constraints:
      - skip_if: "Y == W"
      - skip_if: "e2.t >= e3.t"        # order constraint (§6.2)
    emit: {size: "len(I)"}
```

- Block `sg_ord` uses the sg bins and `delta_sg`. It is a **new feature**: there is no GFP oracle,
  and its oracle is written from the definition above, separately from the DSL.
- The oracle tests the existential condition by enumerating every witness pair, not through
  min/max, so that the §6.2 lowering itself is tested.
- Profiles `sg_ordered_v1w` (v1 windows) and `sg_ordered_tsvis` (timestamp visibility) are added.
  E10 reports their instance counts and their F1 effect: on validation, and after the freeze on test
  as sensitivity only.

As implemented (P10):

- `patterns/sg_ordered.yaml` is the spec above. Block `sg_ord` is the seventh block (`kAllBlocks`).
  `kV1Blocks` holds the six v1 blocks; GFP probes and v1 feature configs use those.
- A v1 feature config does not define `sg_ord`, so the E402 check of v1 configs skips registered
  patterns of known non-v1 blocks. A profile must still list every block it mines.
- The oracle's `sg_ordered` enumerates every witness pair. For each source s and each y it
  collects the times of `s→y` and of `y→w` in the window, and keeps y when some `a < b`.
- Each profile is v1_default's six blocks plus `sg_ord: {patterns: [sg_ordered], window: 21600,
  bins: sg's}`, 77 columns. They differ only in `visibility`.
- `blz fuzz` (the GFP comparison) runs the six v1 blocks.
- Slow tests: `fuzz_diff_sg_ordered_{v1w,tsvis}_{cpu,cuda}_10k` and the metamorphic tests above.

### 6.4 Compiler changes

| Item | Change |
|---|---|
| §6.4 classification | Adds the order-constraint class (§6.2) before the E203 fallback. |
| E205 | New: invalid order constraint (§6.2). |
| E403, E404 | New: profile registration errors (§3.1). Checked at runtime start, like E402. |
| E501 | No longer covers `visibility: timestamp`. |
| IR | `Stage` gains `order?: {a: edge_var, b: edge_var, keep_if: (min\|max, cmp, min\|max)}`. `blzc --dump-ir` prints it. |
| Snapshots | The library specs lose `visibility:`; every snapshot is regenerated once, in its own commit, before any other P10 change. |
| Invalid-spec suite | At least one case per E205 form and one per E403/E404 registration case, each with its exact `file:line:col`. |

As implemented (P10): the IR's `order` is `{a, b, skip, keep, a_latest, b_latest}`. `keep_if` is
`(a_latest ? max : min, keep, b_latest ? max : min)`. The E205 cases are `e205_second`,
`e205_in_or`, `e205_break_if`, `e205_equality`, `e205_offset` and `e205_same_edge`.

### 6.5 Cost

About 4–6 engineering days. Correctness runs take about an hour on `local` (10,000 fuzz graphs ×
3 backends × 2 visibilities, plus HI-Small).

---

## 7. Compiler performance (P11)

### 7.1 Sweep lowering for sliding-window counts [D-29]

fan and degree count a window over the trigger's own adjacency list. The triggers that share a
list are exactly that list's entries, so a single forward pass over the list can maintain the
window instead of rescanning it per trigger (F10).

**Recognition.** The planner chooses the sweep plan for a pattern when all of these hold:

1. There is no binding stage, and every stage is a root-level `count` over `N.dir`, where N is the
   trigger's destination with `dir = in`, or the trigger's source with `dir = out`, so that the
   list entries of N in that direction are the triggers whose N is that node.
2. The count's element filters reference only the element variable, N, and constants, and its
   scanned edge has no time bound other than the block window and the visibility bound (a
   `break_if: e1.t >= t`, for example, keeps the per-trigger plan).
3. The emit uses only the count output. Trigger-level filters MAY reference any trigger variable;
   they are evaluated per trigger.

**Driver.** Parallel over nodes, dynamic schedule, nodes ordered by list length (longest first).
For node x, positions j = 0, 1, … of its list are visited in order. Both window ends move
monotonically:

- the lower end, by the block's open or closed lower bound at `t_j`;
- the upper end: position j under stream visibility, or the last position with `ts ≤ t_j` under
  timestamp visibility.

The running state is updated as entries enter and leave: an edge count for `multiplicity: edges`,
and for `distinct`, per-neighbour counts in a per-thread dense array plus a distinct counter. An
entry enters the state only if it passes the element filter. Trigger j emits once, when its trigger
filters pass. The dense array is cleared by walking the window of the last trigger, so cost stays
O(list length). With a trigger subset (`--triggers`), the sweep still walks every entry and emits
only the selected triggers.

**Scope.**

- CPU only. CUDA keeps the per-trigger plan, which is already fast for fan and degree (A30: 3–19 ms
  per block).
- Cost is O(m) per block, independent of δ. This matters for R4 if longer windows are adopted.
- The plan is recorded in the generated code's plan comment, and `--disable sweep` (§7.2) restores
  the v1 plan.

As implemented (P11):
- **Recognition** (`sweep_stage()`, `src/ir/plan.cpp`) also requires exactly one stage. Every
  library spec has one, and a second count would need a second running state.
  - The scan's lower bound must be the window bound alone, with no upper bound beyond visibility.
  - Element filters may reference only the element, N and constants.
  - The emit may use only the count output and constants.
  - The plan comment and `blzc --dump-ir` name the sweep: `cpu sweep <stage>`.
- **Driver.** `blz::sweep_count<Distinct>` (`include/blz/sweep.hpp`) gets three lambdas from the
  generated code:
  - `lower(t)`: the window's inclusive lower bound;
  - `keep(N, element)`: the element filters;
  - `emit(e, count, sink)`: the trigger's variables, its filters and the emit.

  Nodes are sorted by a stable counting sort, longest list first, and run in dynamic chunks of
  `kSweepChunk` = 16 nodes. At 12 threads, chunks of 1 and of 64 nodes measured 0.075 s and
  0.068 s on fan_out.
- **Triggers.** A trigger subset is a `BlockOut::selects(e)` test. Range selection now carries its
  end (`BlockOut::last`), and the sweep ignores the schedule and chunk arguments.
- **Manifests.** The manifest records each block's per-pattern CPU plan (`cpu_plans`:
  `sweep` or `per_trigger`). The registry records it as `PatternInfo::cpu_sweep`.
  - `blz mine --triggers sample:…` extrapolates only the per-trigger part of a block from the
    sample. The sweep's time is the same for any subset.
- **CUDA** is unchanged (D-29).
- **Generated code.** With every optimization on, only the four fan and degree specs' generated
  code changed, and `tests/snapshots` was regenerated for them.
- **Tests.**
  - `tests/unit/test_sweep.cpp` checks the generated sweep drivers against each pattern's
    per-trigger function on 300 dense, tie-heavy and fuzz graphs. It covers spans 0, 1, 3, the
    window and 2⁴⁰, both visibilities, every selection kind, and 1 and 3 threads.
  - It also checks `sweep_count` against `count_distinct`/`count_edges` with a closed bound and
    an element filter.
  - `tests/unit/test_dsl_lower.cpp` covers recognition, including its negative cases.

### 7.2 Optimization ablation (E11)

The paper lists optimizations qualitatively and gives no ablation [P §5.4]. `blzc --disable
<opt>[,…]` generates the CPU code with optimizations switched off; the CMake option
`BLZ_DISABLE_OPTS` builds such a variant in its own build directory, with `BLZ_WITH_XGBOOST=OFF`.

| Option | Disabled behaviour |
|---|---|
| `find_start` | Every scan starts at the list head; the window lower bound becomes a per-element skip (the paper's printed loops, B-4). |
| `distinct_first` | Distinct neighbours come from a per-thread hash set cleared per scan. |
| `degree_order` | An intersection always scans its `src` operand and probes `dst`. |
| `count_only` | A set used only through `len()` is materialized into a per-thread vector, then counted. |
| `sweep` | The per-trigger plan of v1 for fan and degree. |

Runtime options already exist: `--schedule dynamic|sorted` and, on GPU, the heavy kernel (H = `all`
disables it).

**E11** (`results/e11_ablation_local.{json,md}`): per block and per variant, at 1 and 12 threads,
1 warm-up and 3 timed runs (median, min, max), profile `v1_default` and the frozen profile if it is
compiled. Every variant's `output_fnv1a` MUST equal the fully optimized build's. The table reports
slowdowns relative to the full build, and the sweep's speedup. Timing hygiene as in §2.5.

As implemented (P11):
- **Parsing.** `ir::parse_disable()` rejects unknown or empty names. `blzc` exits with 2 on them.
- **What each option changes** (`src/codegen/cpu_emitter.cpp`):
  - `find_start`: `TCsrView::head_range`, plus a per-entry `ts` skip.
  - `distinct_first`: a `thread_local` `blz::DistinctSet` per scan site
    (`include/blz/ablation.hpp`, open addressing, cleared in O(size)).
  - With `find_start` off, the `distinct_first` test also checks the previous same-neighbour
    entry's ts against the lower bound (`distinct_first_ts`), because the scan no longer starts at
    the bound.
  - `degree_order`: `scan_x = true`.
  - `count_only`: `thread_local std::vector` sets.
- **CPU only.** A variant's `.cu` is an `#error` stub, and CMake rejects `BLZ_DISABLE_OPTS` with
  `BLZ_ENABLE_CUDA` or `BLZ_WITH_XGBOOST`.
- **Recording.** The registry lists the disabled options (`blz::disabled_optimizations()`), and the
  manifest and bench JSON record them (`disabled_optimizations`). Profile ids do not change, since
  they hash the specs.
- **Snapshot tests.** `blzc_snapshots_disable_<opt>` (`tests/snapshots_disable/<opt>`) pin one
  representative spec per option: sg_gather for the first four, fan_out for `sweep`.
- **E11 tool** (`bench/e11.py`):
  - builds each variant in `build/e11/<variant>`, reusing the dependency sources of an existing
    build;
  - gates each variant with `blz fuzz-diff` on 1,000 graphs;
  - waits for an idle machine;
  - per block, thread count and variant, runs a preflight on a 1 % trigger sample and skips a cell
    estimated over 900 s, recording it as aborted;
  - then runs `blz bench` with the dynamic schedule and `OMP_PROC_BIND=close`.

  It logs the top CPU consumers every 10 s into the JSON. It records whether a variant's generated
  code for the block is the same as the full build's (comments aside), so an unchanged variant is
  told apart from noise.
- **Profile.** E11 runs `v1_default` only: the frozen DG-2 recipe is not compiled (DG-3).

### 7.3 A30 campaign (one session)

One instance session after P11, using v1's runbook (`bench/a30.sh`; D-15: `rel-cuda`,
`CMAKE_CUDA_ARCHITECTURES=80`, no XGBoost). Target ≤ 60 minutes of instance time.

- The fast tests; `blz fuzz-diff --backend cuda` on 10,000 fuzz graphs at H = 0, 4 and `all` for
  `v1_default`, the timestamp-visibility profile and `sg_ordered` (AC-V2-4 on sm_80).
- **E12:** E5 for `v1_default` (a regression check against v1's E5) and for every compiled v2
  profile, with H swept as in v1.
- E11 on the instance CPU if it has more than 12 vCPUs.
- GFP is not timed. A new instance is not the same machine as v1's, so E12 reports no new speedup
  over GFP: only GPU times, their change against v1's E5, and the ratio to the CPU backend measured
  on the same instance.

As implemented (P11):
- **Remote script** (`bench/a30_remote.sh`). It builds `rel-cuda` for sm_80 without XGBoost, then:
  - runs the fast tests;
  - runs `blz fuzz-diff --backend cuda` (H = 0, 4, `all`) on 10,000 graphs for `v1_default`,
    `sg_ordered_v1w` and `sg_ordered_tsvis`, and HI-Small `blz mine` on CUDA under each, into
    `results/p11_a30.json`;
  - runs E12, `blz bench` on CUDA (H = 0, 8, 16, 32, 64, 128, 256, 1024, `all`; 5 runs) and on
    the CPU at `nproc` threads, for every block of `v1_default`, `sg_ordered_v1w`,
    `sg_ordered_tsvis` and the three DG-2 window profiles, into `results/e12_gpu_a30.json`;
  - runs E11 at 1 and `nproc` threads, if the instance has more than 12 vCPUs and the run is at
    most 30 minutes in.
- **Local driver** (`bench/a30.sh`) uploads the working tree and the binary cache and downloads the
  results.
- **Report** (`bench/e12_report.py`) checks:
  - the CUDA hashes against P10's oracle hashes;
  - cuda = cpu for every E12 block;
  - `v1_default`'s rows against v1's E5.

  It tunes H per profile as E5 does.

---

## 8. Stretch items (S1–S9)

Each item starts after P10 and the local part of P11 (§2.2), when its gate is met and the user
approves it. Each one ships with
its own acceptance test and a short addendum to this file. The recommended order is the list order.

### S1 — `credit: participants` and GFP batch compatibility

- **Gate:** in R6 a GFP batch or `fit_transform` recipe beats G0 (stream) by ≥ 2.0 validation
  points, or the user asks.
- **Steps:**
  1. Probes A34–A37 (Appendix A′), extended until every rule below is pinned.
  2. A normative addendum that defines instance identity per pattern, which parallel edges count as
     participants, the recount rule for a growing instance (A5), batch visibility, and the eviction
     rule (`max_no_edges`, `time_window`).
  3. The oracle.
  4. The CPU backend: each instance credits participant rows through per-thread buffers merged
     in eid order, which keeps integer counts deterministic.
- **CUDA:** rejected with E404 until a later decision.
- **AC:** oracle = GFP batch mode on 2,000 fuzz graphs at B ∈ {4, 32}; CPU = oracle on 10,000 fuzz
  graphs and on HI-Small.
- **Trap:** participants credit is retrospective. It MUST NOT rewrite a fit row from validation or
  test evidence; the addendum fixes which later edges a row may see (§3.3).

#### S1 addendum (normative)

Pinned by A34–A42 (Appendix A′). The user pulled S1 forward under DG-3 (§13) on 2026-11-14.

- **Profile.** `visibility: batch`, `credit: participants` and `batch: {size: B, cuts: none |
  splits}` come together; any one without the others is an error at load. The canonical JSON gains
  `batch: {size, cuts}` only for these profiles, so every earlier profile id is unchanged.
  `credit: participants` exists only as GFP batch mode. CUDA rejects it with E404, and so does a
  block other than the six v1 blocks. Each block has the oracle's pattern list (E404 on cpu too).
- **Batches.** A batch is one GFP `transform()` call: B consecutive eids, counted from 0 and again
  from each cut. `cuts: splits` cuts at `b_fv` and at the train/test boundary of the whole stream
  (§3.3), as R6 did; `cuts: none` has no cuts. GFP's `time_window` is the largest block window
  (GFP's −1). A row is returned at the end of its batch.
- **Graph during a call (batch visibility, eviction).** The call's graph G is the edges of earlier
  batches with ts ≥ ts[lo − 1] − time_window (lo = the batch's first eid), plus every edge of the
  batch; the first batch drops nothing (A34, A37). No row ever sees a later batch. `max_no_edges`
  is not modelled: profiles have no such key, and GFP's default (−1) keeps everything.
- **Credit.** Every edge of the batch is a trigger, self-loops included. A trigger's instances
  credit their participants that are in the batch; earlier batches' rows were already returned
  and get nothing (A35). A row gets one count per instance that credits it, at that instance's
  size. Instances are never merged across triggers.
- **Rule 1 (fan, degree).** Trigger e = (u → v, t): W = the non-loop edges of G into v (out of u)
  with t − δ < ts ≤ t, the batch's later eids with ts = t included. Fan's size is W's distinct
  neighbours, degree's |W|. Every edge of W in the batch is a participant, parallel edges included.
  A self-loop trigger computes its W too, but is never in it (A39).
- **Rule 2 (tcyc).** Trigger e = (u → v, t), u ≠ v: its instances are the stream-mode cycles (§4.6
  of v1: one per node sequence, every other edge in t − δ ≤ ts < t, so the batch's later edges never
  take part). An instance's participants are e and every window edge c_{i−1} → c_i that some valid
  increasing timestamp assignment of the sequence uses; a parallel edge that fits no assignment is
  not one (A40). `k_max` caps lengths at run time; GFP has none.
- **Rule 3 (SG).** Trigger e = (a → b, t), δ = SG's window. up = G's edges with ts ≥ t − δ (no
  upper bound: the batch's later edges are in it); win = up's edges with ts ≤ t. Pairs: gather
  (s, w = b, x = b) for each s ∉ {a, b} with s → a in up; scatter (s = a, w, x = a) for each
  w ∉ {a, b} with b → w in up. I = {y ≠ x : s → y and y → w in win}. When |I| ≥ 2 the pair is
  reported, unless a batch edge f with ts > t suppresses it:
  - gather: f = y → w (y ≠ x) with s → y in G, or f = s → y (y ≠ x) with y → w in up;
  - scatter: f = s → y (y ≠ x) with y → w in G, or f = y → w (y ≠ x) with s → y in up.

  A report credits each batch edge of win that is s → y or y → w with y ∈ I, parallel edges
  included. Crediting up instead of win would give the same rows: such a later edge suppresses the
  report (A41, A42).
- **Growing instances (A5).** Fan, degree and tcyc instances are per trigger, so a growing star
  credits its early edges once per later trigger, at each trigger's size. An SG pair that a later
  batch edge extends is suppressed at the earlier triggers, so it is reported by the triggers with
  its latest timestamp only, at its size then. Ties report once per tied trigger (A41).
- **The trap (§3.3).** A row sees its whole batch, and SG rows see later edges of it. With
  `cuts: splits` no batch holds eids on both sides of `b_fv` or of the train/test boundary, so a fit
  row sees only fit edges and a train row only train edges. `blz mine` refuses a batch layout in
  which a batch holds both sides of either boundary (on HI-Small, `cuts: none` at B = 128 does).

As implemented (S1):

- Batch mode is a runtime module, not DSL output [D-36]: `src/runtime/batch.{hpp,cpp}` (the
  layout, the shared driver, the CPU module over the T-CSR/P-CSR index) and
  `src/oracle/batch_oracle.{hpp,cpp}`. The oracle is written from the rules above, with per-node
  eid lists and `std::set`; it shares no index code with the CPU module. Its tcyc keeps every
  feasible arrival and departure time. Batches are independent, so both run them in parallel,
  each into a buffer of its own; the selected rows are copied out. This is §8's "per-thread
  buffers merged in eid order": the buffers are disjoint, and the rows are deterministic.
- `blz mine` runs batch profiles on the oracle and cpu; the manifest records `batch` {size, cuts,
  batches, time_window, module}. `blz fuzz --batch B` writes the batch oracle's rows (batches of B,
  no cuts, time_window = the graph's window); `blz fuzz-diff --profile <batch profile>` diffs the
  CPU module against the batch oracle, with each graph's window for every block and `cuts: splits`
  on the graph's own boundaries. `blz metamorphic`, `blz bench` and `blz coverage` reject batch
  profiles.
- Profiles: `configs/profiles/dg2_g1.yaml` (G1's call: W-C windows, D-9 bins, B = 128 with
  `cuts: splits`, k_max 33) and the acceptance profiles `batch_b4.yaml` and `batch_b32.yaml`
  (`cuts: none`, unit bins 2..33, k_max 33). k_max 33 is above every tcyc length GFP finds on
  HI-Small at the W-C window (the longest is 11, DG-4).
- Probes: `tools/gfp/probes.py` adds A39–A42, 31 fixtures. A search over random graphs with
  gapped timestamps, distinct SG and other windows and random cuts found, for each rejected
  alternative of a rule, a graph on which GFP agrees with the rule and not with the alternative.
  Each graph is shrunk and kept with GFP's rows. `tests/golden/gfp_batch_probes.json` has 49
  cases. The unit tests run the oracle and the CPU module on every plain-`transform()` case
  (A34, A35, A39–A42) and check B = 1 against v1's stream mode with trigger credit (300 fuzz
  graphs, all six blocks).
- **Outcome:** see `results/s1_batch.json`.
  - Oracle = GFP batch mode on 2,000 fuzz graphs at B = 4 and B = 32 (403,557 edges): 0
    mismatched rows in every block.
  - CPU = oracle on 10,000 fuzz graphs (2,053,897 edges) for `batch_b4`, `batch_b32` and
    `dg2_g1`: 0 mismatched rows.
  - HI-Small, `dg2_g1`: CPU = oracle, and both equal the frozen GFP G1 blocks
    (`features/dg2_R6_G1`) byte for byte. The data SHA-256 of each block is the `output_sha256`
    that `configs/dg2_frozen.yaml` records. Time: CPU 9.2 s at 12 threads (RSS 0.6 GB), oracle
    9 min 44 s, GFP's G1 pass 709 s.
  - G1 is therefore compiled: `blz mine --backend cpu --profile configs/profiles/dg2_g1.yaml`
    reproduces the frozen recipe's pattern blocks (DG-3, §13).

### S2 — vertex statistics in C++

- **Gate:** R7 F1 is adopted.
- A runtime module (`src/runtime/vstats.cpp` and its CUDA twin), not DSL output: vertex statistics
  are aggregates, not patterns.
- **Determinism:** per trigger, a fixed accumulation order (window entries in eid order) in double.
  The CPU build uses `-ffp-contract=off`, the CUDA build `--fmad=false`, so CPU and CUDA are bit-exact.
- **AC:** integer statistics (fan, degree) equal GFP exactly. Floating statistics are within a
  relative error of 1e-6 of GFP (absolute 1e-9 near zero) on 2,000 fuzz graphs and HI-Small,
  because GFP's accumulation order is unknown. A larger difference needs a probe and a `D-n`.

#### S2 addendum (normative)

Pinned by A43–A45 (Appendix A′). The user pulled S2 forward under DG-3 (§13) on 2026-11-14.

- **Rows.** One row per trigger e (GFP's stream mode, one `transform()` per edge), 52 float64
  values: 4 sides (the out- and in-edges of e's source, then of its destination) × 13
  statistics: fan, degree, ratio, then avg, sum, var, skew and kurtosis of ts and of
  `amount_received`. The columns are `vs_<side>_<stat>` in GFP's order (`vstats_names()` in
  `tools/gfp/dg2_features.py`).
- **Window (A43).** A side's window is the node's edges in that direction with eid ≤ e and
  ts ≥ ts[e − 1] − W, or every such edge at e = 0. That is GFP's graph at the call, after
  eviction. The bound is closed, and a self-loop counts on both sides of its node. W is GFP's
  `time_window`, the largest pattern window of the call: vertex statistics have no window of their
  own. A `vertex_stats_tw` ≥ W has no effect. A smaller one shrinks the window of the value
  statistics but not of fan and degree; the rule does not model that, so a GFP call matches it only
  with `vertex_stats_tw` ≥ W (F1 has 86,400 for both).
- **Statistics (A44).** fan = the distinct neighbours, degree = the window's size, ratio = degree /
  fan (0 for an empty side). For each value column x (ts plus the run's ts offset, or the amount),
  with n = the degree and m_k = the k-th central moment about the mean:
  - avg = Σx / n and sum = Σx;
  - var = m2 (population variance), skew = m3 / m2^1.5, and kurtosis = m4 / m2² (not excess).

  n = 0 gives zeros. var, skew and kurtosis are 0 when n = 1, when m2 ≤ 0, or when m2 < 10⁻⁵ ·
  avg² (GFP's cancellation guard, which applies to ts too).
- **Accumulation.** Two passes over the window in eid order, in double: Σx, then Σd², Σd³ and Σd⁴
  with d = x − avg. The CPU build compiles the module with `-ffp-contract=off` and the CUDA twin
  with `--fmad=false`. Both run the same `BLZ_HD` code, so they agree bit for bit.
- **GFP's drift (A45, D-37).** GFP keeps running sums per node. It adds an edge's values when the
  edge arrives and subtracts them when the edge is evicted, and the subtraction leaves a residue:
  - a mean that is not Σx / n, and a nonzero sum on an empty side;
  - negative variances, and kurtoses of −10⁵.

  `blz` computes each window afresh and does not reproduce the residue. Every mismatch with GFP is
  on a side that lost an edge to eviction before the row.

As implemented (S2):

- `src/runtime/vstats.{hpp,cpp}` holds the `BLZ_HD` core and the OpenMP driver: one row per
  trigger, for any `--triggers` selection. `src/runtime/vstats_cuda.cu` runs one thread per row
  over an eid range.
- `blz vstats --graph g --window W [--ts-offset o] [--backend cpu|cuda] [--threads n] [--triggers
  ...] --out dir` writes `vstats.npy` (float64 [rows, 52]) and a manifest with the data's
  `output_sha256`; cuda rejects samples. `--window 86400 --ts-offset 1` is F1's rule.
- `blz fuzz --vstats` writes each fuzz graph's amounts and `blz`'s rows at the graph's window, with
  ts offset 1. The amounts come from `fuzz_amounts`: cents, small integers, amounts up to 10⁷, tiny
  values and repeats.
- `tools/gfp/vstats_vs_gfp.py` has two modes:
  - it replays the fuzz graphs into GFP with F1's call;
  - or it compares a `blz vstats` run with F1, rounded to float32 and clipped as F1 was stored.

  Both modes classify each mismatch by eviction.
- Probes: `tools/gfp/probes.py` adds A43–A45 (11 cases in `tests/golden/gfp_vstats_probes.json`).
  `tests/unit/test_vstats.cpp` checks the CPU module:
  - against those cases;
  - against a naive reference, bit for bit, at 1 thread, at every thread, and on a range;
  - against CUDA, bit for bit, on 300 fuzz graphs.
- **Outcome** (`results/s2_vstats.json`):
  - Fuzz (2,000 graphs, 403,557 rows, 20,984,964 values):
    - Fan and degree equal GFP everywhere.
    - 196,100 values (0.93 %) in 49,495 rows are outside the tolerance, all on sides that lost an
      edge to eviction. Sides without an evicted edge have 0 mismatches.
    - GFP reports 23,792 negative variances.
  - HI-Small against F1 (5,078,345 rows, 264,073,940 values):
    - Fan and degree equal.
    - 3,216,785 values (1.2 %) in 1,036,757 rows (20 %) are outside the tolerance, all after
      eviction; 942,547 of them by more than 100 % relative.
    - F1 holds 93,585 negative variances.
  - CPU = CUDA byte for byte on HI-Small (`output_sha256` 5487fce7…):
    - CPU: 2.5 s at 12 threads (index 0.4 s), plus 8.2 s to write and hash the 2.1 GB; RSS 0.7 GB.
    - CUDA: 16.1 s on the GTX 1650 SUPER, whose float64 rate is low.
  - The vertex statistics are therefore compiled with D-37's deviation: `blz vstats` reproduces F1's
    rule, not F1's bytes.

### S3 — lc-cycle

- Probes first: GFP lc-cycle's window rule, its bound, whether the trigger must be on the cycle, and
  ties (O-2).
- A `patterns/lcyc.yaml` that has tcyc's shape without the ordering constraints. Every hop is an
  `exists` or a `for_all` over window edges, with distinct nodes.
- Unordered enumeration can explode. A per-trigger work limit aborts the run with the trigger's eid;
  it never truncates silently.
- **AC:** AC1 (oracle = CPU = CUDA) at length ≤ 10 and window 86,400. Oracle = GFP on HI-Small and on
  2,000 fuzz graphs, except tie cases reproduced as probes and accepted by a `D-n`.

#### S3 addendum (normative)

Pinned by A46–A49 (Appendix A′). The user pulled S3 forward under DG-3 (§13) on 2026-11-14.

- **Instances (A47, A48).** A trigger e = (u → v, t) with u ≠ v counts the simple cycles through
  its own edge. A self-loop trigger emits nothing.
  - A cycle is a node sequence v = c_0, c_1, …, c_h, u of distinct nodes, with a window edge
    c_i → c_{i+1} for each hop and c_h → u.
  - The window edges come in any time order. Parallel edges collapse: one instance per node
    sequence.
  - The size is h + 2, from 2 to L inclusive, with L = `K_MAX` = 10 (GFP's `lc-cycle_len`).
  - GFP rejects L = 1.
- **Window (A46).** The other edges have earlier eids and t − W ≤ ts < t, a closed lower bound.
  - W is the block's own window: GFP's `lc-cycle_tw`, not another pattern's window or the
    eviction window.
  - The rule never uses an edge with ts = t, so it gives the same rows under either visibility.
- **Ties (A49, D-38).** `blz` leaves out every edge tied with the trigger (ts = t). GFP uses a tie
  edge x → y iff rank(y) < rank(v).
  - rank is the order in which GFP's graph first saw the node, over the whole stream, src before
    dst within an edge. It is kept after eviction.
  - Equivalently, the trigger must be the cycle's largest edge by (ts, rank(dst)).
  - This resolves O-2 (SPEC Appendix A). (1,1,2) has no tie and is found. In (1,2,2), the tie edge
    closes at u, which GFP saw after v, so it is not found. With u seen first, it is (A49).
- **Work limit.** Unordered enumeration can explode, so a `work_limit: true` pattern counts its
  bindings per trigger.
  - A binding is a `for_all` binding past its stage's filters. The oracle counts each extension of
    the path to a new node, which is the same number.
  - Every backend and schedule stops on the same triggers: those whose count exceeds the limit.
  - The run then fails with `WorkLimitExceeded`, which names the smallest such eid, whatever the
    thread count, schedule or kernel. No row is written, and none is truncated.
  - The default limit is 10⁸ bindings (`--work-limit`).

As implemented (S3):

- **DSL and code generation.**
  - The DSL has a top-level `work_limit: true` key (parsed, lowered, printed in the IR dump).
  - The generated code calls `out.over(p.work_limit)` once per binding.
  - The OpenMP driver records the smallest aborted eid and skips any trigger whose eid is larger
    than an abort already seen.
  - The CUDA light kernel counts in its `RowSink`. The heavy kernel counts in one shared-memory
    counter per warp. Both kernels `atomicMin` the eid and skip larger eids; they read it with a
    relaxed atomic load.
  - The counter costs each warp 8 B of the heavy block's 48 KiB of shared memory. A CUDA block
    therefore takes 1 to 767 bins, one fewer than the 768 of SPEC §7.5.
- **`patterns/lcyc.yaml`.**
  - It has tcyc's shape without the ordering constraints. Every hop and every close has
    `break_if: "x.t >= t"`, which keeps ties out.
  - Distinct nodes come from `skip_if: "C{k} in [U, V, C{1..k-1}]"`, and `close_k` is an `exists`
    on U.
  - `Block::lcyc` has a closed lower bound.
  - `registry_profile()` leaves lcyc out. It runs only from a profile that lists it:
    `configs/profiles/lcyc_f2.yaml` (F2's window 86,400, bins 2..10, k_max 10).
- **Oracle.** `Oracle::lcyc` is a literal DFS over each node's set of window successors.
- **CLI.**
  - `blz mine --work-limit`: the manifest records `work_limit` on the lcyc block.
  - `blz fuzz-diff --work-limit`: see D-39.
  - `blz fuzz --lcyc [--work-limit]`: the oracle's rows. Every row of a limited graph is
    UINT32_MAX.
  - `tools/gfp/lcyc_vs_gfp.py`: fuzz and HI-Small modes. It checks every mismatching row against
    both rules: ties left out for `blz`, A49's rank rule for GFP.
- **Tests.**
  - `tests/unit/test_lcyc.cpp`: the 16 probe cases on the oracle, the CPU (both schedules) and
    CUDA (H = 0, 2 and all), and the work-limit error.
  - The lcyc parity and invariance tests in `test_cpu_backend.cpp` and `test_cuda_backend.cpp`:
    40 fuzz graphs at a limit of 20,000, with rows or the stop eid compared across ranges, threads,
    schedules and thresholds.
  - Slow: `fuzz_diff_lcyc_f2_{cpu,cuda,cuda_light}_10k`.
- **Outcome** (`results/s3_lcyc.json`):
  - Probes: A46–A49 hold on all 16 cases (13 `rule`, 3 `tie`), and GFP rejects L = 1.
  - AC1 on fuzz graphs (10,000 graphs, 2,053,897 edges):
    - CPU, both schedules, limit 10⁶: 0 mismatching rows. 537 graphs are limited, with equal stop
      eids.
    - CUDA at H = 0 and 4, limit 10⁶: 0 mismatching rows, with the same 537 limited graphs.
    - CUDA at H = all, limit 10⁵ (D-39): 0 mismatching rows. 1,727 graphs are limited, with equal stop
      eids.
  - AC1 on HI-Small (window 86,400, length ≤ 10):
    - oracle = CPU = CUDA byte for byte (`lcyc.npy` sha256 63e31aee…), 0 saturated;
    - 2.9 s on the oracle, 1.3 s on the CPU (12 threads) and 1.5 s on the GTX 1650 SUPER.
    - No trigger reaches the default limit of 10⁸.
  - GFP on HI-Small: F2's lcyc block equals `blz`'s on all 5,078,345 rows, 1,135 of which have a
    cycle. No usable tie edge occurs, so S3 compiles F2 byte for byte.
  - GFP on 2,000 fuzz graphs, with the 96 limited graphs skipped:
    - 74,723 of the 369,426 rows differ (202,380 rows have a GFP cycle);
    - every one is a tie row: GFP's row is A49's rank rule, and `blz`'s is the rule without ties;
    - 0 mismatches are unexplained.
    - The generator makes ties common on purpose (SPEC §13.3: T ∈ [2, 60]).
  - GFP replayed the 1,904 graphs in 7,534 s on 4 processes; one graph alone took 59 s.

### S4 — `union` and `difference`

Each op ships with one library spec that uses it, so that no op exists without a feature:

- `difference`: `fan_oneway_out` = `|N_out(U) \ N_in(U) \ {U}|`, counterparties paid with no money
  coming back within the window.
- `union`: `fan_total` = `|N_in(V) ∪ N_out(V) \ {V}|`, all counterparties of the trigger's
  destination.
- Lowering reuses the intersection machinery: `difference` keeps scanned elements whose probe
  fails, and `union` = `|A| + |B \ A|`.
- `differentiate` stays reserved (E401); it is what `skip_if` already does (§6.9).
- **AC:** E401 is lifted for the two ops; AC1 on 10,000 fuzz graphs; invalid-spec cases.

### S5 — new typologies

- **Gate:** E9 shows typology T structurally ineligible for every existing block under the frozen
  profile, while a draft spec for T would make most of T's complete attempts eligible.
- **Gather-scatter first.** Draft semantics, to be pinned in the addendum: a trigger `x→w` out of
  hub x completes an instance with `A` = distinct sources that sent to x before t and
  `B` = distinct sinks x paid within the window, both excluding x and w as in §4.5. Size =
  `min(|A|, |B|)`, emitted when ≥ 2. This needs `min()` in `iexpr`.
- **Bipartite and stack** are deferred to v3 unless E9 shows them both frequent among undetected
  positives and expressible within 16 binding levels.
- **Speed:** a Fig. 9 analogue (stack speed, normalized to 1 thread) only if a stack spec exists.
- **AC:** AC1 for every new spec, and its E9 eligibility reported.

### S6 — tcyc hybrid CPU/GPU and reachability pruning

- **Hybrid:** heavy triggers (work ≥ H_cpu) on the CPU backend while the GPU runs the light ones,
  concurrently, with a deterministic merge (disjoint rows).
- **Pruning:** a planner option for chain patterns. Per trigger, a backward search from U computes
  each node's latest departure time towards U inside the window, as the oracle already does (§9 of
  v1), and hops that cannot reach U in time are cut. The search is bounded by `K_MAX − 1` hops.
- **AC:** bit-exact. E13 (1650S, and the A30 if it rides the campaign) reports hybrid against
  GPU-only and CPU-only, with T = max(CPU, GPU) + merge + every transfer.

### S7 — output packing and multi-pattern fusion

- **Packing:** per block, D2H as u8 or u16 with a saturation flag. Any overflow re-copies the block
  as u32, so the result is always lossless.
- **Fusion:** patterns whose first scan is the same operand (fan_in + deg_in, fan_out + deg_out,
  sg_gather + sg_scatter) are fused into one kernel body that writes several blocks.
- **AC:** bit-exact; E12 delta reported (F11: D2H equals kernel time on the A30).

### S8 — streaming replay and end-to-end throughput

- **Incremental index:** per-node append-only adjacency in chunks, plus a pair map for P-CSR runs.
- The generated trigger bodies become templates over the graph view, so the same text runs on the
  static CSR and the incremental index.
- `blz replay --batch B` ingests the stream in micro-batches, mines each batch's rows against the
  index, and optionally runs XGBoost inference with a model trained on the fit split.
- **Reported:** per-batch latency p50/p99, edges per second for mining alone and end to end, peak
  RSS. A BlazingAML-only analogue of Fig. 12 (paper: about 340K tx/s on HI-Small at 128 threads).
- **AC:** replay rows = offline rows (hash) for B ∈ {128, 4096, 65536}, and B = 1 on the first
  100,000 edges.

### S9 — other datasets

- **Gate:** the user downloads the data (Kaggle "IBM Transactions for Anti Money Laundering").
- **LI-Small** first: invariants from Table 1 (705,907 nodes, 6,924,055 edges) plus the counts it
  measures. E2 (with the frozen recipe) and E4 rows.
- **Medium:** a RAM preflight first. The index is about 1 GB, each block's `.npy` is written one at a
  time, and a `QuantileDMatrix` is about 2.2 GB. The preflight decides between `local` and a
  memory-larger rented instance; the latter needs the user's approval.
- **Large** stays out.

---

## 9. Experiments

v1's E0–E5 stand. E6 becomes R8 (validation) and E8's sensitivity table (test).

| ID | What | Output |
|---|---|---|
| E7 | DG-2 ladder (validation only) | `results/dg2_ledger.{json,md}` |
| E8 | Final Table 2/3, `v1_default` vs frozen recipe; R8 grid as test sensitivity; the verdict (§4.6) | `results/e8_table2_v2.{json,md}` |
| E9 | Ground-truth coverage (§5) | `results/e9_coverage.{json,md}` |
| E10 | New semantics: instance counts and F1 effect of `sg_ordered` and timestamp visibility | `results/e10_semantics.{json,md}` |
| E11 | Compiler ablation and sweep lowering, CPU | `results/e11_ablation_<host>.{json,md}` |
| E12 | GPU: E5 for `v1_default` and the v2 profiles | `results/e12_gpu_<host>.{json,md}` |
| E13 | (S6) tcyc hybrid | `results/e13_hybrid_<host>.{json,md}` |
| E14 | (S8) replay latency and throughput | `results/e14_replay_<host>.{json,md}` |
| E15 | (S9) other datasets | `results/e15_<dataset>.{json,md}` |

---

## 10. Testing additions

- **Unit:**
  - the profile loader and profile id (v1 config ≡ `v1_default`);
  - `file_row` and `.blzg` v2;
  - the fit/val boundary;
  - `pair_latest` against a naive scan;
  - order-constraint lowering against brute-force witness pairs, on random lists with ties and
    duplicates;
  - the sweep driver against the per-trigger plan;
  - E403, E404.
- **Golden:** Appendix A′ probes enter `tests/golden/gfp_probes.json` only if they describe stream
  mode. Batch probes are kept in a separate `tests/golden/gfp_batch_probes.json`, used by S1,
  vertex-statistics probes in `tests/golden/gfp_vstats_probes.json`, used by S2, and lc-cycle probes
  in `tests/golden/gfp_lcyc_probes.json`, used by S3.
- **Fuzz:** the generator is unchanged. `blz fuzz-diff` gains `--profile`, so AC1 runs per profile.
- **Metamorphic:** §3.4, in the fast suite (200 graphs) and the slow suite (2,000).
- **Compiler:** the invalid-spec suite grows by the E205 cases; snapshots of `sg_ordered` and of
  each `--disable` variant of one representative spec.
- **Slow integration:** HI-Small, every v2 core profile, oracle = CPU = CUDA.
- **S1:** `tests/unit/test_batch.cpp` (the batch probes on the oracle and CPU, B = 1 against v1,
  CPU = oracle across layouts, trigger selections and thread counts); slow:
  `fuzz_diff_<batch profile>_cpu_10k` for `batch_b4`, `batch_b32` and `dg2_g1`.
- **S2:** `tests/unit/test_vstats.cpp`: the GFP probes, the naive reference bit for bit (1 thread,
  every thread, a range), and CPU = CUDA on 300 fuzz graphs. GFP parity runs through
  `tools/gfp/vstats_vs_gfp.py`, which needs snapml, not through ctest.
- **S3:**
  - `tests/unit/test_lcyc.cpp`: the lc-cycle probes and the work-limit error, on the oracle, the CPU
    and CUDA.
  - lcyc's fuzz parity with the work limit, in `test_cpu_backend.cpp` and `test_cuda_backend.cpp`.
  - Slow: `fuzz_diff_lcyc_f2_{cpu,cuda,cuda_light}_10k`.
  - GFP parity runs through `tools/gfp/lcyc_vs_gfp.py`.

---

## 11. CLI additions

```
blz prepare   ...                           # also writes file_row; .blzg v2; asserts §3.3 counts
blz gt-join   --graph g --csv c --patterns p --out results/gt_join.json
blz mine      ... --profile configs/profiles/<p>.yaml          # replaces --config; --config still loads v1 configs
              [--triggers all|train|test|fit|val|range:a:b|sample:<pct>:<seed>]
                                                                # --backend oracle honours the profile's visibility
blz fuzz-diff ... --profile <p>
blz coverage  --graph g --gt results/gt_join.json --profile <p> --features <dir> [--pred <npy>] --out ...
blzc          ... [--disable find_start,distinct_first,degree_order,count_only,sweep]
bench/dg2/ladder.py  --rung R<k> [--candidate <id>]             # appends to results/dg2_ledger.json
tools/gfp/dg2_features.py --recipe G1|G2|G3|F1|F2 --out features/<run>/
```

---

As implemented (P7):

- `--triggers sample:<pct>:<seed>` selects each eid independently, with probability pct/100,
  from a hash of (seed, eid). The selection does not depend on m.
- Output row r belongs to eid `triggers.npy[r]` (u32[k, 1], written next to the blocks). The
  manifest records `extrapolated_mine_seconds` per block.
- Sampling runs on the oracle and cpu only. On cuda, `blz mine` rejects a sample, because the
  generated bucket launchers take an eid range; the preflight of §2.5 and §4.4 uses cpu.
- `blz fuzz-diff --profile` defaults the bins to each block's own.

As implemented (P8): `blz mine --edges k` mines on the stream prefix [0, k): later edges do not
exist, under either visibility. `--triggers` must then be `all`, `range:a:b` or `sample:`, taken
on the prefix, because the named splits are defined on the whole stream. The manifest records
`edges`. Under stream visibility, the rows of a prefix run equal the full run's first k rows
(v1_default on cpu, k = 4,062,676, all six blocks).

As implemented (S1): `blz fuzz --batch B`, `blz fuzz-diff --profile <batch profile>` and
`tools/gfp/fuzz_vs_gfp.py --batch B` (§8 S1).

As implemented (S2): `blz vstats`, `blz fuzz --vstats` and `tools/gfp/vstats_vs_gfp.py` (§8 S2).

As implemented (S3): `blz mine --work-limit`, `blz fuzz-diff --work-limit`, `blz fuzz --lcyc` and
`tools/gfp/lcyc_vs_gfp.py` (§8 S3).

As implemented (P9): `blz coverage --graph g [--gt results/gt_join.json] --profile p --features
<dir> [--pred <u8[m] npy>] [--eligibility <coverage_gfp.json>] --out <json>`. It checks the
gt-join, the profile (E403, and E404 against the oracle), the features and the predictions against
the graph, and writes one line per attempt (§5).

## 12. Milestones

v1's numbering continues. Each phase lists its exit criterion and a rough cost (engineering days,
then machine time).

### P7 — Contracts

- Profiles, profile id, manifest, E403/E404.
- `.blzg` v2 with `file_row`.
- The fit/val split and its invariants.
- Metamorphic tests.
- `blz gt-join`.
- `--triggers sample:`.
- The ledger skeleton.

**Exit:** AC-V2-0 and AC-V2-1; `results/gt_join.json` committed. About 3 days; under 1 h of machine
time.

**Outcome (2026-11-11).** Rules this phase pinned down are in §3.1–§3.6 and §11 ("As
implemented"). `bench/p7_contracts.py` runs the exit checks into `results/p7_contracts.json` on
`local`, in about 5 minutes.

- **AC-V2-0 holds.** `ctest -L fast` passes on `rel`, `rel-cuda` and `dev` (sanitizers).
  `blz mine --profile configs/profiles/v1_default.yaml` gives the six pinned `output_fnv1a` on
  the oracle (79 s), cpu (5.8 s) and CUDA on the 1650S (5.3 s, including start-up).
- **AC-V2-1 holds.**
  - Every `features.json` of those runs carries the full manifest with profile id
    b21580d6e129821a.
  - The metamorphic tests find 0 changed rows on 2,000 fuzz graphs (403,557 edges) for each
    frozen profile on the oracle, cpu and CUDA (CUDA at H = 0, 4 and `all`).
  - `blz fuzz-diff --profile` finds 0 mismatches on 2,000 graphs per profile, for cpu and CUDA.
- **Sampling.** `--triggers sample:1:7` (50,395 rows) gives the full runs' rows on the oracle
  and cpu.
  - On the oracle, the extrapolated times are within 4% of the full run for the two blocks that
    take over 30 s.
  - On cpu, the extrapolations overestimate by 1.1× (fan_out) to 3× (fan_in). Fixed costs
    dominate runs of a few milliseconds, so the estimate errs on the safe side of the preflight
    limit.
- **`.blzg` v2.** `blz prepare` re-created `data/hi_small.blzg` in 5.6 s, and every v1 and v2
  invariant holds.
- **`results/gt_join.json`** is committed (§3.5).
- **The ledger skeleton** is `results/dg2_ledger.{json,md}` (§3.6).

### P8 — DG-2 ladder

- The oracle's `--visibility timestamp`.
- The Appendix A′ batch probes.
- `bench/dg2/ladder.py` and `tools/gfp/dg2_features.py`.
- R0–R8 under §4.2.
- The freeze, DG-3, DG-4, E8, the verdict.

**Exit:** AC-V2-2. About 3–5 days; about 5 machine-hours on `local`.

**Outcome (2026-11-12): done. R0–R8, the freeze, DG-3, DG-4 and E8 ran. The verdict is GAP,
because the frozen recipe overshoots the paper (E8 below). DG-2 is closed (SPEC §17, AC-V2-2).**
The tooling notes are in §3.4, §4.4, §6.1 and §11 ("As
implemented (P8)"), Appendix A′ (A34–A38) and D-32. The ladder is
`results/dg2_ledger.{json,md}`. Its budget totals: 28 of 40 fits, 8 of 8 passes, 0.80 of 4
machine-hours; the ladder never stopped.

Validation F1 ×100 on each rung's selection set (§4.3):

| Rung | Incumbent | Candidates | Carried |
|---|---:|---|---|
| R0 | — | v1: `xgb_only` 0.00, `all` 4.82 (P 0.69, R 0.025) | v1 |
| R1 (`xgb_only`) | 0.00 | I1 0.74; I2 1.83; I3 0.18 | v1 |
| R2 | 4.82 | C1 5.51 | v1 |
| R3 | 4.82 | S1 10.32; **S2 13.14** | S2 |
| R4 | 13.14 | W-B 14.98; **W-C 18.54** | W-C |
| R5 | 18.54 | V1 16.29 | W-C |
| R6 | 18.54 | **G1 25.08**; G2 24.42; G3 `skipped: limit` | G1 |
| R7 | 25.08 | **F1 38.22**; F2 24.96 | F1 |
| R8 | 38.22 | X1 38.41; X2 22.61; X3 30.44; X4 31.37; X5 43.73; X6 51.29; **X7 53.02** | X7 |

- **Carried recipe.**
  - Ids: `src_id`, `dst_id` (R0).
  - Raw columns: S2, the Multi-GNN columns without Timestamp.
  - Pattern blocks: GFP G1 (fresh preprocessor, `transform()` on batches of 128, cut at the split
    boundaries) at W-C windows and D-9 bins.
  - F1 vertex statistics (52 columns).
  - XGBoost with `scale_pos_weight` 5, `max_depth` 10 and 500 rounds; `base_score` estimated.
  - 126 columns in all.
  - Validation: P 0.866, R 0.382, TP 413, FP 64, FN 668; PR-AUC 0.494.
  - The most gain goes to Payment Format, then `vs_src_out_fan`, `vs_dst_out_fan`, `fan_out[2,3)`
    and `tcyc[2,3)`.
- **The recipe is not causal.** In G1 a row sees the later edges of its own batch of 128, and
  instances credit every participant in the batch (A28, A35). No batch mixes splits. The pattern
  blocks and the vertex statistics are GFP-produced, so DG-3 applies before E8 (§13).
- **R2's [R] fact.** XGBoost 3.4.1 resolves `base_score` to 0.000754 on the fit rows, the positive
  rate. The documentation confirms estimation from the labels, which the 2.0.0 release notes
  announce and which replaced 1.6's default of 0.5. The 1.7 parameter page already describes the
  estimation (`bench/dg2/r2_facts.json`).
- **G3 was aborted** at 3,269 MB after 9 s, so it is `skipped: limit`. Its single
  `fit_transform` call on 3,047,216 rows returns a float64 array of about 1.7 GB on top of GFP's
  graph. About 0.37 GB of the peak was `dg2_features.py`'s own copies of the input, so a leaner
  tool would have landed near 2.9–3.0 GB. There is no pass budget left to retry it; this is
  reported as is.
- **Pass costs on `local`, idle.**
  - W-B and W-C on cpu: 8–9 s each.
  - V1 on the oracle: 149 s.
  - G1: 711 s. G2: 364 s.
  - F1: 115 s. F2: 134 s.
  - Fits: 6–227 s, peak RSS ≤ 2.1 GB.
- **Leads not tested** (the pass cap is spent):
  - GFP's own bin rule on D-9 bins (A38, D-32);
  - G3 with a leaner tool;
  - `credit: participants` in `blz` (S1).
- **Freeze (2026-11-12, with the user's go-ahead).** The carried recipe is in
  `configs/dg2_frozen.yaml`. It records each feature block's `output_sha256`, and it was
  committed before any test row was scored.
- **DG-3.** The frozen recipe is identified with GFP-produced features (the G1 pattern blocks and
  the F1 vertex statistics) and is not yet compiled. After E8, the user decides whether to pull
  S1, S2 or S3 forward (§13).
- **DG-4.** E0's method at the frozen tcyc window of 320,400 (`parity.py --block tcyc --bins
  2..33`, stream mode; 95 s, peak RSS 1.6 GB; `results/dg4_tcyc.json`). 3,187 of 5,078,345
  triggers close a cycle, and 5 of them (9.8 × 10⁻⁷ of all triggers) have a longest cycle longer
  than `K_MAX = 10`, all of length 11. By length, the cycle instances are: 2: 3,109; 3: 190;
  4: 54; 5: 22; 6: 6; 7: 3; 8: 9; 10: 4; 11: 5; none from 12 to 33+. At D-8's 86,400 the longest
  was 5 (E0). The share is below DG-1's 0.1 %. The frozen tcyc block is G1's, and GFP counts every
  length, so `K_MAX` cuts nothing in E8. The share applies only when the recipe is compiled
  (DG-3).

**E8 (2026-11-12).**
- **The run.** `bench/dg2/e8.py` ran once, at the clean commit abd37a6, in 34.5 min, and wrote
  `results/e8_table2_v2.{json,md}`.
  - Train: 4,062,676 rows, 3,380 positives.
  - Test: 1,015,669 rows, 1,797 positives.
  - Before the run, `--dry-run` (fit → validation) reproduced the ledger's R0, R7/F1 and R8 X1–X7
    confusion counts exactly.

Table 2 (F1 ×100):

| | XGB Only | Fan | Fan+Degree | Fan+Degree+Cycle | Fan+Degree+Cycle+SG |
|---|---:|---:|---:|---:|---:|
| Paper | 11.1 | 23.2 | 38.4 | 42.6 | 46.6 |
| `v1_default` (`blz train` = Python) | 0.0 | 0.9 | 6.1 | 6.6 | 6.4 |
| Frozen recipe | 58.0 | 60.6 | 63.5 | 63.9 | 63.1 |
| Frozen minus paper | +46.9 | +37.4 | +25.1 | +21.3 | +16.5 |
| Frozen columns | 59 | 83 | 107 | 116 | 126 |

Table 3 (`all`), with the paper's counts in brackets:

| | Predicted laundering | Predicted non-laundering |
|---|---:|---:|
| Actual laundering | TP 993 (558) | FN 804 (1,239) |
| Actual non-laundering | FP 360 (31) | TN 1,013,512 (1,013,841) |

- **Table 3 metrics.**
  - Precision: 0.734 against the paper's 0.947 (−0.213).
  - Recall: 0.553 against 0.311 (+0.242).
  - PR-AUC: 0.616.
- **Verdict (D-21): GAP.** Every cell is above the paper, `all` by 16.5 points. Precision is
  lower and recall higher than the paper's. `v1_default` would also be GAP, from below: `all`
  is −40.2.
- **R11.** `v1_default` reproduces E2's confusion counts on all five sets, through both
  `blz train` and Python.
- **The Table 2 columns do not compare cell by cell.** Every set holds the base columns: ids,
  S2 raw columns and 52 vertex statistics (§4.5 step 3). So XGB Only is already 58.0, and it is
  not the paper's raw-attribute XGBoost (11.1).
  - Share of total gain in `all`: vertex statistics 0.61, raw columns 0.23, ids 0.05, all six
    pattern blocks together 0.10. The vertex statistics carry 0.60–0.68 in every set.
  - The pattern blocks add 5.9 points from XGB Only to Fan+Degree+Cycle; SG then costs 0.9.
- **Validation to test.** `all` goes from 53.0 to 63.1. Precision falls from 0.866 to 0.734 and
  recall rises from 0.382 to 0.553.
  - The positive rate grows from 0.075 % (fit) and 0.106 % (validation) to 0.177 % (test).
  - The final model also trains on the validation period.
- **Sensitivity: the R8 grid on test, `all`, never used to select.** The frozen point (63.1) is
  the highest of the eight. With `scale_pos_weight` 5 the points give 49.3–63.1. With 1 they give
  5.7–22.5, against 22.6–38.4 on validation.
  - The weight-1 points lose precision on test, with 5,165 to 41,965 FP.
  - The worst point is (1, 6, 500): FP 41,965 and recall 0.716.
- **Not explained, not tested** (§4.5 step 4: no recipe change after E8). The overshoot says the
  frozen recipe is not the paper's pipeline. Candidates:
  - G1's look-ahead within a batch of 128 (A28, A35). No batch mixes splits, and labels are never
    features, but a row's pattern counts see later edges of its batch.
  - Vertex statistics stronger than whatever the paper fed XGBoost.
  - The shift in positive rate between the splits.
  - Leads still open: A38's bin rule on D-9 bins, G3 with a leaner tool, `credit: participants`
    in `blz` (S1).
- **DG-3, after E8.** The user decides whether to pull S1, S2 or S3 forward. The verdict stands
  either way.
  - **Resolved on 2026-11-14:** the user pulled S1–S3 forward, and all three are done. Every
    GFP-produced feature of the frozen recipe now has a compiled counterpart:
    - G1's pattern blocks byte for byte (S1);
    - F1's vertex statistics by F1's rule, with D-37's deviation (S2);
    - F2's lc-cycle block byte for byte on HI-Small (S3, D-38).
  - Nothing was refit or rescored: E8's verdict stands (§4.5 step 4).

### P9 — Coverage

`blz coverage` and E9.

**Exit:** AC-V2-3. About 1–2 days; minutes of machine time.

**Outcome (2026-11-13): done. AC-V2-3 holds.**
- All 3,209 listed rows and all 370 attempts have a join status (§3.5; every attempt is
  `complete`).
- `bench/e9.sh` writes `results/e9_coverage.{json,md}`: per typology and per block, eligibility,
  activation and detection for `v1_default` and the frozen recipe, plus the 1,968 unlisted
  positives.
- The tooling notes are in §5 and §11 ("As implemented (P9)"), D-33 and D-34.

- **Cost on `local`.** The four refits take 9 min (peak RSS 2.4 GB). GFP eligibility takes 6 s.
  Each `blz coverage` run takes about 1 s.
- **The refits reproduce their originals exactly** (TP / FP / FN):

  | Model | Reference | TP / FP / FN |
  |---|---|---|
  | `v1_default`, fit → val | R0 | 27 / 12 / 1,054 |
  | frozen, fit → val | R8/X7 | 413 / 64 / 668 |
  | `v1_default`, train → test | E8 | 60 / 18 / 1,737 |
  | frozen, train → test | E8 | 993 / 360 / 804 |

- **Eligibility** (any block, % of the 370 attempts): `v1_default` 42, frozen 60.
  - The windows make the difference for sg and tcyc. sg on SCATTER-GATHER rises from 2 % to 70 %;
    tcyc on CYCLE from 9 % to 78 %.
  - BIPARTITE and RANDOM are eligible for no block under either profile.
  - STACK is eligible only through tcyc (14 %, 33 %).
  - GATHER-SCATTER is 88 % / 92 % eligible, through fan and degree.
  - S5's gate (§8), first half: met for BIPARTITE and RANDOM, not for gather-scatter.
- **Activation** is near universal and says little: 98–100 % of attempts (any block). 80 %
  (`v1_default`) and 84 % (frozen) of all edges already have a non-zero row in some block, almost
  all of them from fan and degree.
  - sg and tcyc are selective. Under `v1_default`, each fires on 0.02 % of all edges.
  - Under the frozen recipe, sg fires on 1.5 % of edges and tcyc on 0.06 %, against 9.5 % and
    6.0 % of positives.
  - No attempt is eligible without being activated, for any block.
- **GFP against blz eligibility** at the frozen windows. 3 of 2,220 (attempt, block) pairs
  differ, all on tcyc.
  - They are the three "Max 11 hops" CYCLE attempts. Their 11-edge cycle is longer than
    `K_MAX = 10`, so the oracle cannot count it; GFP counts every length (DG-4).
  - Row counts are higher by one on 10 more attempts. Each has edges that shared a G1 batch,
    where every participant is credited (A28).
- **Detection** (mean of the per-attempt shares):
  - `v1_default`: 0.07 on validation, 0.08 on test.
  - Frozen: 0.63 on validation, 0.70 on test.
  - Highest: FAN-OUT and SCATTER-GATHER (0.76–0.84). Lowest: BIPARTITE (0.38, 0.48).
- **Unlisted positives are almost never detected.**
  - `v1_default` detects none: 0 of 389 on validation, 0 of 458 on test.
  - The frozen recipe detects 1 of 389 and 5 of 458.
  - 988 of the frozen recipe's 993 test TPs are listed positives. On listed positives its recall
    is 0.60 (validation) and 0.74 (test).
  - Unlisted positives are 25 % of the test positives.
- **Positives that no block activates:** 874 under `v1_default` (513 listed, 361 unlisted) and
  395 under the frozen recipe (228, 167).
- **Not explained, not tested:** why the unlisted positives are invisible. Reported, not chased (§2.5).

### P10 — Temporal semantics

- Runtime visibility in the generated code.
- Order constraints, `pair_latest`, E205.
- `sg_ordered` and its oracle.
- E10.

**Exit:** AC-V2-4 on the 1650S (the sm_80 part completes in P11's campaign). About 4–6 days; about
1 h of machine time.

**Outcome (2026-11-13): done. AC-V2-4 holds on the 1650S; its sm_80 part is P11's (§7.3).**
- The commits:
  - 32b4f14: runtime visibility, snapshots regenerated;
  - 4c4259e: order constraints, `pair_latest`, E205, `sg_ordered`, its oracle;
  - f63e42d: the exit checks and the E10 tool.
- Before the long runs, a gpt-6.1-sol review (xhigh, 1 round) of the whole P10 diff approved it
  with no findings.
- The tooling notes are in §3.1, §3.4 and §6.1–§6.4 ("As implemented (P10)").
- **AC-V2-4** (`bench/p10_semantics.py` → `results/p10_semantics.json`; 13 min on `local`):
  - `blz fuzz-diff` on 10,000 graphs per profile, `sg_ordered_v1w` and `sg_ordered_tsvis`: 0 rows
    differ on cpu (both schedules) or CUDA (H = 0, 4, `all`).
  - The metamorphic tests on 2,000 graphs per profile, including order relaxation: 0 rows differ
    on the oracle, cpu or CUDA.
  - HI-Small: oracle = cpu = CUDA for every block of `v1_default`, `sg_ordered_v1w` and
    `sg_ordered_tsvis`.
  - The six v1 blocks of `v1_default` and `sg_ordered_v1w` keep the AC-V2-0 hashes.
  - tcyc under timestamp visibility keeps its hash, and its `tcyc.npy` is byte-identical to
    `v1_default`'s.
  - The invalid-spec suite passes with exact positions.
  - On HI-Small the oracle takes 80–91 s per profile; cpu and CUDA take about 6 s. `sg_ord` alone
    takes 5.3 s on the oracle, 0.17 s on cpu and 0.07 s on CUDA.
- **E10** (`bench/e10.py` → `results/e10_semantics.{json,md}`, at f63e42d):
  - **Instances.** On HI-Small, `sg_ord` has 212 instances (195 rows) against sg's 808 (both
    roles). That is 26 % of sg: 157 in the fit split, 31 in validation, 24 in test.
  - Timestamp visibility adds 0.1–0.7 % instances to fan, degree and sg, and none to tcyc.
  - **F1 on validation** (fit → val, set `all`, F1 ×100):

    | Classifier | `v1_default` | `sg_ordered_v1w` | `sg_ordered_tsvis` |
    |---|---:|---:|---:|
    | R0 | 4.82 | 4.82 | 5.51 |
    | R3/S2 | 13.14 | 13.14 | 12.53 |

  - `sg_ord` has a gain share of 0.0000 in every fit, so `sg_ordered_v1w`'s confusion counts are
    exactly `v1_default`'s. At v1's sg window (21,600 s) the block is too sparse to matter.
  - Timestamp visibility moves F1 by less than a point on validation.
  - **Test sensitivity** (after the freeze; never used to select): R0 6.40 / 6.40 / 6.79, R3/S2
    21.67 / 21.67 / 19.96.
  - `v1_default` reproduces the ledger's R0 and R3/S2 counts and E8's test counts exactly.
- **Not run:** `sg_ordered` at wider windows (e.g. W-C's 342,000 s for sg). E10 did not
  pre-register it. Reported, not chased (§2.5).

### P11 — Compiler performance and the A30 campaign

- The sweep lowering, `--disable`, E11.
- The A30 campaign: AC-V2-4 on sm_80, E12.
- `bench/ac1_cpu.sh` and `bench/ac1_cuda.sh` re-run.

**Exit:** AC-V2-5. **v2 core is done.** About 4–6 days; 1–2 h of `local` timing in idle windows, plus
≤ 60 min of A30 time.

**Outcome (2026-11-14): done. AC-V2-5 holds, and AC-V2-0 holds again at the end of P11. v2 core
(P7–P11) is done.**
- The commits:
  - d9307a3: the sweep and `blzc --disable`;
  - 7cbae02: E11, the exit checks, the A30 scripts and the E12 report;
  - 4739bc4: `bench/a30.sh`'s `jl run` wait ends on any state other than `running` (AGENTS.md);
  - 81e7839: the campaign's results;
  - 94ed217: the campaign's status and D-35;
  - the local exit checks' results, committed with this outcome.
- Before the long runs, a gpt-6.1-sol review (xhigh) took 2 rounds and approved. It found two bugs
  in the bench scripts, both fixed:
  - E11 left every process named `blz` out of the foreign load;
  - E12 measured only `sg_ord` for `sg_ordered_v1w`.
- The tooling notes are in §7.1–§7.3 ("As implemented (P11)").
- **The A30 campaign** (24 min, $0.18; at 7cbae02, whose dirty flag is only the untracked
  AGENTS.md and CLAUDE.md; `results/p11_a30.json`, `results/e12_gpu_a30.{json,md}`):
  - **AC-V2-4 on sm_80.** On 10,000 fuzz graphs per profile, at H = 0, 4 and `all`, 0 rows differ
    for `v1_default`, `sg_ordered_v1w` or `sg_ordered_tsvis`. Their HI-Small CUDA hashes equal
    P10's oracle hashes.
  - **E12.** For every block of the six compiled profiles, the CUDA rows equal the CPU rows.
    `v1_default`'s kernels are within 2 % of v1's E5, with H = 64 tuned again. Tuned H per profile:
    32 (`sg_ordered_tsvis`), 64 (`v1_default`, `sg_ordered_v1w`), 128 (`dg2_v1`, `dg2_wc`) and
    256 (`dg2_wb`). The DG-2 windows move the optimum.
  - **GPU vs CPU on the instance** (CUDA at the tuned H, against the CPU backend at 16 threads):
    - 3.5–4.6× per profile;
    - sg 9–13×, tcyc 2.8–5.9×;
    - fan and degree 0.6–2.4× only, because the CPU sweep takes about 30 ms per block.
- **E11** (`results/e11_ablation_a30.{json,md}`, D-35):
  - Every variant passed its fuzz gate on 1,000 graphs, and its hashes equal the full build's.
    No cell was noisy, and none was skipped.
  - **The sweep**, against `--disable sweep`, at 1 and 16 threads:
    - fan_out 32.8× and 24.3×;
    - deg_out 2.4× and 2.1×;
    - fan_in 1.7× and 1.8×;
    - deg_in 1.5× and 1.6×.
  - **The other options, at 1 thread:**
    - `degree_order` 2.63× on sg;
    - `find_start` 1.19× and `distinct_first` 1.15× on tcyc, and 1.16× on sg for `distinct_first`;
    - `count_only` at most 1.02×.
  - With the sweep on, `find_start` and `distinct_first` do not reach fan and degree. Each
    variant switches off one option, so their effect on the per-trigger fan plan is not
    measured. Reported, not chased (§2.5).
- **The local exit checks** (2026-11-14, on `local` at 94ed217; `rel` and a rebuilt `rel-cuda`
  for sm_75). The user deferred them on 2026-11-13 and approved them on 2026-11-14.
  - `ctest --preset rel-cuda`: all 115 fast tests pass.
  - `bench/p11_checks.py` → `results/p11_checks.json` (log `results/p11_checks_run.log`, 15 min).
    All 26 checks pass:
    - **AC-V2-5, HI-Small.** For each of the 8 profiles that use the sweep (the 7 in
      `configs/profiles/` and `tests/profiles/tsvis.yaml`), fan_in, fan_out, deg_in and deg_out
      hash the same on the oracle and on cpu. The cpu manifest names the sweep plan for each.
    - **AC-V2-5, fuzz.** `blz fuzz-diff` on cpu (both schedules), 10,000 graphs: every
      registered block under stream visibility, and `sg_ordered_tsvis`. 0 seeds differ.
    - **AC-V2-1.** The metamorphic tests on cpu, 2,000 graphs for each frozen profile, pass.
    - **AC-V2-0.** `v1_default` on the oracle, cpu and CUDA gives the pinned hashes for all six
      blocks. The cpu build disables nothing.
  - `bench/ac1_cpu.sh` (34 s) and `bench/ac1_cuda.sh` (1 min 48 s) → `results/e1_parity.json`
    (logs `results/ac1_{cpu,cuda}_p11_run.log`). On HI-Small, 0 rows differ from the oracle for
    any of the six blocks in any run:
    - cpu: the default config at 12 threads with the dynamic schedule, and the parity config at
      1 thread with the sorted schedule;
    - CUDA: the default config at H = 32, and the parity config at H = 0 and at `all`.

    `blz fuzz-diff` on 10,000 graphs finds 0 seeds that differ on cpu (both schedules) or on
    CUDA (H = 0, 4 and `all`).

### Stretch

S1–S9 in the order of §8, each after its gate and the user's approval.

Status: the user pulled S1–S3 forward under DG-3 on 2026-11-14. S1 is done (§8 S1 addendum). S2
is done with D-37's deviation (§8 S2 addendum). S3 is done with D-38's tie deviation and D-39's work
limit (§8 S3 addendum). DG-3 is resolved (P8). S4–S9 wait for their gates and the user.

---

## 13. Risks and decision gates

| ID | Risk / gate | Mitigation / rule |
|---|---|---|
| R7 | The test split is exposed (F4). | Validation-only selection, a committed freeze before E8, every rung recorded, no post-test changes (§4.5). |
| R8 | Longer windows (R4, S3) make tcyc, SG or lc-cycle explode. | Preflight on a 1% sample with abort limits (§2.5, §4.4); the sweep (§7.1) makes fan and degree window-independent; S3's work limit. |
| R9 | GFP batch semantics depend on batch boundaries and eviction, so S1 parity may be impossible to pin. | Probes before any implementation (A34–A37); S1 is gated. |
| R10 | `Patterns.txt` rows may not join uniquely. | Explicit statuses (§3.5); E9 uses `complete` attempts only and reports the rest. |
| R11 | The ladder's Python fits diverge from `blz train`. | E8 runs `v1_default` through both `blz train` and the Python tool; both must reproduce v1's E2/E3 counts. If they differ, E8 reports both, takes the `v1_default` row from `blz train`, and records the difference against the frozen row's tool. |
| R12 | RAM on `local` (about 3 GB free) with wider ladder matrices. | `QuantileDMatrix` from batches; RSS recorded per fit; a fit that would exceed 3 GB runs with fewer threads or is recorded as `skipped: memory`. |
| R13 | Shared-desktop timing noise (v1 E4: fan_out 1.21–1.70 s at 12 threads). | §2.5 timing hygiene; the ablation reports min and max, not only the median. |
| DG-3 | The frozen recipe depends on features only GFP produces (R6 batch semantics, R7 vertex statistics or lc-cycle). | Before E8, record it as "identified with GFP-produced features; not yet compiled". After E8, ask the user whether to pull S1, S2 or S3 forward. The verdict stands either way. **Resolved on 2026-11-14:** S1–S3 are done (P8). |
| DG-4 | The frozen tcyc window is longer than D-8's, so `K_MAX = 10` may cut cycles that GFP counts. | Report only. Before E8, measure the GFP tcyc length histogram at the frozen tcyc window (E0's method, under the 60-minute pass limit) and report the share of triggers with cycles longer than `K_MAX` next to E8. The frozen recipe keeps `K_MAX = 10`; DG-1's raise rule is not applied after the freeze. |

---

## Appendix A′ — GFP probes added by v2

Run in P8 before R6 (snapml 1.17.2, a fresh preprocessor per case); A39–A42 were added by S1,
A43–A45 by S2 and A46–A49 by S3.
`tools/gfp/probes.py` asserts every answer below. A34–A37 and A39–A42 describe batch mode, so they
are kept here rather than in Appendix A (stream mode, batch size 1), and their cases and GFP's full
rows are in `tests/golden/gfp_batch_probes.json` (49 cases). A38 is a stream-mode fact about GFP's
bins and is asserted only.

| # | Question | Answer (P8) |
|---|---|---|
| A34 | Batch boundary: does an edge in batch k see edges of batch k+1? (Expected no; A28 shows that it sees later edges of its own batch.) | **No.** A batch's rows see every edge of earlier batches, which are in the graph, and never an edge of a later batch. They do not depend on how the earlier edges were batched: the graph state is the edge set. |
| A35 | Credit across batches: does an instance completed in batch k credit participant edges from earlier batches, whose rows were already returned? | **No.** Only the participants in batch k are credited. Earlier participants' rows were already returned and get nothing. An SG instance (s, w) is credited once per batch edge, at its size within the batch. |
| A36 | Re-sent edges in batch mode, and `fit_transform` on 1.17.2 versus fresh `transform()`, per pattern group (extends A29). | `fit_transform(X)` gives exactly what sending X twice to a fresh preprocessor gives. Fan and SG counts double. Degree instance sizes double, because re-sent edges count as parallel edges. tcyc changes unevenly. `fit(X)` followed by `transform()` on chunks of X differs from one `transform(X)`. |
| A37 | Eviction: with `time_window` / `max_no_edges` set, which edges does a batch still see? | A `time_window` below the largest pattern window has no effect (GFP keeps that window). With `max_no_edges = 2`, a stream edge sees only the 2 most recent earlier edges, but a batch always sees all of its own edges. |
| A38 | (Found while checking R6's tooling.) Which bin does GFP use for an instance of size S? | **The first bin i with S ≤ b_i, or the last bin.** This is not the bin with b_i ≤ S < b_{i+1} that GFP's docstring states and SPEC §4.2 uses. The two rules agree only for unit-width bins (`[2..33]` and `[1..8]`), which covers every v1 parity run and probe, so v1 could not see the difference. With D-9's fan bins, size 6 lands in [7, 9) instead of [5, 7). `gfp_bins()` (`tools/gfp/gfp_common.py`) passes b′_i = b_{i+1} − 1 and keeps the last value. GFP then bins exactly as §4.2 does, given b_0 = 2 (min size 2) or unit-width bins [D-32]. With that translation, G1's code run at batch size 1 gives `blz`'s rows exactly (100,000 edges, v1_default). |

S1's probes (A39–A42) each list the rejected alternatives their fixtures separate
(`S1_RULES` in `tools/gfp/probes.py`); the answers are the rules of the S1 addendum (§8).

| # | Question | Answer (S1) |
|---|---|---|
| A39 | Fan and degree in a batch: which edges form a trigger's instance, and which are credited? | **Rule 1.** The non-loop edges into v (out of u) with t − δ < ts ≤ t in the graph up to the batch's end, including later eids with ts = t; every one of them in the batch is credited, once per trigger. Rejected: only eids ≤ e, later timestamps of the batch, a closed lower bound, self-loops in W, self-loop triggers that do not search, trigger-only credit. |
| A40 | tcyc in a batch: which edges are participants? | **Rule 2.** The trigger and every window edge between consecutive cycle nodes that some increasing timestamp assignment uses, once per node sequence. Rejected: an open lower bound, trigger-only credit, every window edge between consecutive nodes, one instance per edge path. |
| A41 | SG in a batch: which triggers report a pair, and over which views? | **Rule 3.** Pairs come from up (ts ≥ t − δ, the batch's later edges included); I from win (ts ≤ t); self-loop triggers search; tied last edges each report; the excluded node never suppresses. Rejected: pairs from win, pairs from G's lower bound, ties that suppress, x as a suppressor, no self-loop searchers. |
| A42 | SG suppression and eviction: which later batch edges cancel an earlier trigger's report? | **Rule 3.** A later y → w (gather) or s → y (scatter) whose other edge is in G, and a later s → y (gather) or y → w (scatter) whose other edge is in up; G evicts at the largest pattern window. Rejected: no suppression, each other view (up, G, from the batch's first ts, every edge, none) on each side, eviction at SG's or the other patterns' window. |

S2's probes (A43–A45) check GFP's rows against the S2 rule and against each rejected alternative
(`S2_FIXTURES` in `tools/gfp/probes.py`, F1's call); the answers are the rules of the S2 addendum
(§8). GFP's rows are in `tests/golden/gfp_vstats_probes.json` (11 cases).

| # | Question | Answer (S2) |
|---|---|---|
| A43 | Vertex statistics in stream mode: which of a node's edges does row e's side aggregate, and what do `vertex_stats_tw` and the pattern windows change? | **GFP's graph at the call.** The node's edges with eid ≤ e and ts ≥ ts[e − 1] − time_window (closed; every edge at e = 0), with self-loops on both sides; time_window is the largest pattern window. A `vertex_stats_tw` ≥ that has no effect. A smaller one shrinks the window of the value statistics, by the same rule, but not of fan and degree. Rejected: the window [t − W, t] at the trigger, an open bound, `vertex_stats_tw` or the companion pattern's window as W, self-loops left out. |
| A44 | The formulas: ratio, variance, skew, kurtosis, small samples. | ratio = degree / fan (0 when fan = 0); population variance; skew = m3 / m2^1.5; kurtosis = m4 / m2² (not excess); n ≤ 1 gives 0. A variance below 10⁻⁵ · mean² is reported as 0 with its skew and kurtosis, for ts too: ts 1,000,001 and 1,000,006 give 0. Rejected: fan / degree, the sample variance, excess kurtosis, no guard. |
| A45 | After an eviction, are GFP's statistics those of the window? | **No.** GFP subtracts evicted values from running sums and keeps the rounding residue. In the fixture, a side that holds the single amount 4.0 reports a mean of 3.9999999999999933, a variance of −0.0162 and a kurtosis of −1.98 × 10⁵, with or without self-loops. Fan and degree are exact. Accepted as D-37. |

S3's probes (A46–A49) check GFP's rows against the S3 rule, with GFP's tie rank, and against each
rejected alternative (`S3_FIXTURES` in `tools/gfp/probes.py`, `lc_model` in
`tools/gfp/lcyc_vs_gfp.py`; lc-cycle only, `lc-cycle_len` 10, bins 2..10). GFP's rows and `blz`'s
are in `tests/golden/gfp_lcyc_probes.json`: 16 cases, 13 `rule` cases where they are equal and
3 `tie` cases where they differ (D-38).

| # | Question | Answer (S3) |
|---|---|---|
| A46 | lc-cycle's window: which edges, and which window? | Edges with earlier eids and t − W ≤ ts ≤ t, where W is `lc-cycle_tw`. The window is the pattern's own and is anchored at the trigger's t. Rejected: an open lower bound; a start at ts[e − 1] − W (A43's eviction rule); another pattern's window, i.e. GFP's time_window (fan at 100 with lc-cycle at 4). |
| A47 | The length bound. | 2 ≤ size ≤ `lc-cycle_len`, inclusive: a 10-cycle counts at L = 10, an 11-cycle does not, and 2-cycles count. GFP rejects `lc-cycle_len` = 1. Rejected: L − 1, L + 1, no 2-cycles. |
| A48 | Must the trigger be on the cycle, and what is one instance? | The simple cycles through the trigger edge, one per node sequence, with edges in any time order. Rejected: increasing ts along the path (tcyc's order), every cycle through u, edge sequences (parallel edges multiply), walks that repeat a node. |
| A49 | Ties (O-2): may another cycle edge have ts = t? | **Only by GFP's node order.** A tie edge x → y is usable iff rank(y) < rank(v). rank is the order in which GFP first saw the node over the whole stream, src before dst, and it is kept after eviction. This explains O-2: (1,1,2) is found, (1,2,2) is not, and (1,2,2) is found once u was seen before v. Rejected: no tie edges (`blz`, D-38), every tie edge, the src key rank(x) > rank(u), dst before src, node ids as ranks, ranks over the window's edges only. |

## Appendix B′ — deviations added by v2

| # | Paper | This spec | Reason |
|---|---|---|---|
| B-11 | Fig. 3 option 1 states a per-intermediate order, with no trigger role [P §2.1]. | `sg_ordered` = gather role only, existential witness pairs (§6.3). | Under backward windows only a gather edge can complete an ordered instance. |
| B-12 | Fig. 5's loops read `break if e.t > t` (timestamp visibility). | Offered as profile parameter `visibility: timestamp`; stream stays the default. | GFP parity for the default (B-9); the paper's literal reading is now available. |

## Appendix C′ — decisions added by v2

| ID | Decision |
|---|---|
| D-17 | Fit/validation boundary `b_fv` = first eid ≥ ⌊0.6·m⌋ that starts a new timestamp (3,047,216 on HI-Small); test unchanged. |
| D-18 | The DG-2 ladder of §4.4 is pre-registered by committing this file; changes need a `D-n` before the affected rung runs. |
| D-19 | Selection: validation F1 ×100 at threshold 0.5 on the rung's selection set; adopt at ≥ 2.0 points over the incumbent; ties within 0.1 go to fewer columns. |
| D-20 | Ladder budget: ≤ 40 fits, ≤ 8 feature passes, ≤ 4 machine-hours on `local`; E8 outside it. |
| D-21 | Verdict thresholds: AGREE (every Table 2 cell within ±3.0, Table 3 precision and recall within ±0.03), PARTIAL (`all` within ±5.0), GAP. DG-2 closes in every case. |
| D-22 | Visibility is a runtime profile parameter; a spec's `visibility:` key pins it (E403 on conflict). |
| D-23 | Semantic profiles with content-derived ids; `v1_default` and `v1_parity` are frozen. |
| D-24 | GFP-produced features (R6, R7) are allowed only inside DG-2 and are labeled as not compiled (DG-3). |
| D-25 | Window recipes for R4: W-B = snapml 1.17.2 library defaults (F7); W-C = p90 of fit-split attempt spans per GFP pattern group (fan and degree ↔ FAN-IN ∪ FAN-OUT, sg ↔ SCATTER-GATHER, tcyc ↔ CYCLE), rounded up to the hour, capped at 20 days. |
| D-26 | DG-2 fits and E8 use Python XGBoost 3.4.x with a batched `QuantileDMatrix`; `blz train` remains the C++ path and must reproduce `v1_default`'s counts. |
| D-27 | Order constraints are existential over witness pairs and lowered through earliest/latest witnesses (§6.2). |
| D-28 | `sg_ordered` has the gather role only (§6.3). |
| D-29 | The sweep lowering is CPU-only; CUDA keeps the per-trigger plan. |
| D-30 | One A30 campaign for v2 core; stretch GPU work joins it only if finished before it, otherwise a new campaign needs the user's approval. |
| D-31 | `.blzg` format version 2 adds `file_row`; the version 1 reader rejects it and `blz prepare` rebuilds the cache. |
| D-32 | GFP-produced passes (R6, R7) give GFP `gfp_bins(bins)`, so that their blocks are binned as §4.2 bins `blz`'s (A38). R6 and R7 then change only the call semantics or the family, not the binning. GFP's own rule applied to D-9's bins is a separate hypothesis. It is not tested, because the pass cap is fully allocated; E8 lists it as an open lead. F1 runs temp-cycle (window 86,400, one bin `[2]`) alongside, because GFP rejects `vertex_stats` with no pattern enabled. That column is dropped. The vertex statistics are the same with degree or fan as the companion (30,000 edges). F1 stores float32. GFP's float64 values beyond the float32 range are clipped to ±3.4028235·10³⁸ and counted in the manifest. These come from the kurtosis of near-constant inputs: 3 values in the first 3,000,000 rows, in `vs_dst_in_amount_received_kurtosis`. GFP emits each input column's statistics in input-column order, whatever the order of `vertex_stats_cols`; `dg2_features.py` names the columns in that order and checks it on one edge before every F1 pass. XGBoost works in float32 and rejects `inf`; NaN is kept and is missing to XGBoost. |
| D-33 | E9's GFP eligibility of a batch recipe (G1) sends each attempt edge in the batch it had in the full stream (G1's layout: 128 eids from each cut, cuts at `b_fv` and 4,062,676), in eid order, to a fresh preprocessor. The alternative, fresh batches of 128 over the isolated stream, would put nearly every attempt in a single batch, where every edge sees and credits every other edge. That is not what the full pass did. With the full-stream layout, the isolated and full runs differ only in which edges exist. |
| D-34 | E9's detection uses refits of the recorded models (the ladder and E8 kept neither models nor predictions). A refit is used only when its confusion counts equal the ledger's (validation) or E8's (test) exactly. Nothing is selected and nothing feeds back (§4.5 step 4). Vertex statistics are not a pattern block and get no eligibility or activation. |
| D-35 | E11 is the A30 run of the P11 campaign (`results/e11_ablation_a30.{json,md}`: the instance's EPYC 7542 CPU, at 1 and 16 threads), not a run on `local` at 1 and 12 threads as §7.2 states. It is complete: no cell skipped or noisy, and every hash check passed. The user accepted it on 2026-11-13, which saves 30–60 min of idle-window timing on `local`. |
| D-36 | GFP batch mode (S1) is a hand-written runtime module with a literal oracle, not DSL output. Its SG report rule (suppression by later batch edges, role-dependent views, eviction at the largest window) is GFP's call semantics, not a pattern, and the DSL has no construct for "the rest of the call". CUDA rejects it (E404). `dg2_g1` sets k_max 33 so that tcyc counts every cycle GFP counts on HI-Small (the longest has 11 edges); GFP has no cap. |
| D-37 | S2's vertex statistics are each window's exact statistics: two passes in eid order (§8 S2). They do not reproduce the residue of GFP's running sums after an eviction (A45). The mismatches all fall on sides that lost an edge to eviction: 0.93 % of values on 2,000 fuzz graphs and 1.2 % on HI-Small (20 % of rows; F1 holds 93,585 negative variances). They are accepted as GFP's numerical defect. Reproducing it would need GFP's per-node add and subtract sequence over the whole stream, which a per-trigger module does not have, and it would copy the error. Fan and degree equal GFP everywhere. A `vertex_stats_tw` below the largest pattern window (A43) is not modelled; F1 does not use one. |
| D-38 | S3's lcyc leaves out every edge tied with the trigger (ts = t), under both visibilities. GFP decides a tie by its node order (A49): a tie edge x → y counts iff GFP first saw y before v. That order is the history of the call, not the window's edges, and the DSL has no node order: its node comparisons are ==, != and `in`, and a mixed predicate is E203. Matching it would need a per-node insertion rank carried through the index and a new predicate. The tie rows are accepted instead. On the 1,904 fuzz graphs under the limit, 74,723 of 369,426 rows (20 %) differ, every one reproduced by A49's rule (SPEC §13.3 makes ties common). On HI-Small none differ: F2 equals `blz` on every row. This resolves O-2. |
| D-39 | The work limit (§8 S3) counts bindings per trigger. It stops the run with the smallest eid over the limit and never truncates a row. AC1's fuzz runs use a limit of 10⁶ on the CPU and on CUDA at H = 0 and 4. 537 of the 10,000 graphs reach it; for those, the stop eid is compared on every backend, schedule and threshold, not the rows. CUDA's all-light pass (H = all) uses 10⁵, which 1,727 graphs reach (0 mismatching rows). There one GPU thread walks each trigger's bindings serially, so at 10⁶ that pass alone would take about 3.75 h on the GTX 1650 SUPER. The light kernel runs the same generated `<pattern>_trigger` as the CPU driver, which is checked at 10⁶. GFP's fuzz parity skips the graphs limited at 10⁶ (96 of 2,000). HI-Small runs at the default of 10⁸, which no trigger reaches. |

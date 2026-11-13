# E9: ground-truth coverage (SPEC v2 §5)

Git `4bcb83d258b4` (dirty). Attempts: `results/gt_join.json` (§3.5). Descriptive only: computed after the freeze, never fed back (§5).

- **Structurally eligible**: the block gives a non-zero row to at least one of the attempt's edges when the graph holds only those edges (eids kept).
- **Activated**: the block gives a non-zero row to at least one attempt edge in the full HI-Small mining.
- **Detected**: the share of the attempt's validation (test) edges that the validation-selected (E8's) model predicts positive. Fit edges are training data and are never scored.

| Profile | Pattern blocks | Eligibility | Models |
|---|---|---|---|
| v1_default | `blz mine` (cpu), profile `v1_default` b21580d6e129821a | the oracle, profile `v1_default` | val: ledger R0 all, TP 27 FP 12 FN 1054 (reproduces); test: E8 v1_default all, TP 60 FP 18 FN 1737 (reproduces) |
| frozen recipe | GFP G1 (`features/dg2_R6_G1`), windows and bins of `dg2_wc` 64747da4a40c0d70 | GFP G1, each edge in its full-stream batch (D-33) | val: ledger R8/X7 all, TP 413 FP 64 FN 668 (reproduces); test: E8 frozen_recipe all, TP 993 FP 360 FN 804 (reproduces) |

The frozen recipe's F1 vertex statistics are aggregates, not pattern blocks: they have no eligibility or activation here, but they are in its detection model.

## Attempts

| Typology | Attempts | Complete | fit | val | test | straddles | Edges | Span p10 (h) | p50 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BIPARTITE | 49 | 49 | 22 | 2 | 17 | 8 | 263 | 0 | 24 | 43 | 47 |
| CYCLE | 54 | 54 | 19 | 0 | 10 | 25 | 287 | 28 | 72 | 90 | 96 |
| FAN-IN | 40 | 40 | 9 | 1 | 12 | 18 | 318 | 11 | 85 | 95 | 96 |
| FAN-OUT | 48 | 48 | 17 | 3 | 10 | 18 | 342 | 0 | 77 | 95 | 96 |
| GATHER-SCATTER | 51 | 51 | 8 | 0 | 16 | 27 | 716 | 7 | 151 | 184 | 202 |
| RANDOM | 41 | 41 | 12 | 6 | 12 | 11 | 191 | 0 | 46 | 87 | 92 |
| SCATTER-GATHER | 44 | 44 | 9 | 1 | 11 | 23 | 626 | 45 | 89 | 95 | 96 |
| STACK | 43 | 43 | 8 | 3 | 7 | 25 | 466 | 10 | 73 | 101 | 114 |
| **ALL** | 370 | 370 | 104 | 16 | 95 | 155 | 3209 | 0 | 75 | 110 | 202 |

## Eligible / activated, v1_default (% of complete attempts)

| Typology | fan_in | fan_out | deg_in | deg_out | sg | tcyc | any |
|---|---:|---:|---:|---:|---:|---:|---:|
| BIPARTITE | 0 / 73 | 0 / 63 | 0 / 90 | 0 / 82 | 0 / 0 | 0 / 24 | 0 / 98 |
| CYCLE | 0 / 83 | 0 / 76 | 0 / 94 | 0 / 89 | 0 / 0 | 9 / 24 | 9 / 98 |
| FAN-IN | 80 / 90 | 0 / 82 | 80 / 95 | 0 / 88 | 0 / 0 | 0 / 25 | 80 / 100 |
| FAN-OUT | 0 / 88 | 69 / 81 | 0 / 90 | 69 / 92 | 0 / 0 | 0 / 23 | 69 / 98 |
| GATHER-SCATTER | 69 / 96 | 65 / 88 | 69 / 100 | 65 / 98 | 0 / 2 | 18 / 31 | 88 / 100 |
| RANDOM | 0 / 80 | 0 / 68 | 0 / 93 | 0 / 85 | 0 / 0 | 0 / 24 | 0 / 93 |
| SCATTER-GATHER | 73 / 93 | 73 / 84 | 73 / 100 | 73 / 98 | 2 / 2 | 5 / 32 | 77 / 100 |
| STACK | 0 / 91 | 0 / 77 | 0 / 98 | 0 / 91 | 0 / 2 | 14 / 28 | 14 / 100 |
| **ALL** | 27 / 87 | 26 / 78 | 27 / 95 | 26 / 90 | 0 / 1 | 6 / 26 | 42 / 98 |

Eligible but not activated: none (each pattern's count only grows when edges are added, so none is expected).

## Eligible / activated, frozen recipe (% of complete attempts)

| Typology | fan_in | fan_out | deg_in | deg_out | sg | tcyc | any |
|---|---:|---:|---:|---:|---:|---:|---:|
| BIPARTITE | 0 / 90 | 0 / 78 | 0 / 98 | 0 / 94 | 0 / 8 | 0 / 39 | 0 / 100 |
| CYCLE | 0 / 98 | 0 / 87 | 0 / 100 | 0 / 94 | 0 / 7 | 78 / 78 | 78 / 100 |
| FAN-IN | 90 / 95 | 0 / 90 | 90 / 100 | 0 / 95 | 0 / 18 | 0 / 25 | 90 / 100 |
| FAN-OUT | 0 / 96 | 81 / 92 | 0 / 100 | 81 / 100 | 0 / 17 | 0 / 29 | 81 / 100 |
| GATHER-SCATTER | 80 / 98 | 76 / 96 | 80 / 100 | 76 / 100 | 0 / 31 | 25 / 33 | 92 / 100 |
| RANDOM | 0 / 93 | 0 / 78 | 0 / 100 | 0 / 95 | 0 / 12 | 0 / 37 | 0 / 100 |
| SCATTER-GATHER | 75 / 100 | 75 / 93 | 75 / 100 | 75 / 100 | 70 / 73 | 30 / 34 | 100 / 100 |
| STACK | 0 / 100 | 0 / 86 | 0 / 100 | 0 / 93 | 0 / 7 | 33 / 35 | 33 / 100 |
| **ALL** | 30 / 96 | 30 / 88 | 30 / 100 | 30 / 96 | 8 / 21 | 22 / 40 | 60 / 100 |

Eligible but not activated: none (each pattern's count only grows when edges are added, so none is expected).
GFP eligibility against blz's (the oracle, profile `dg2_wc`, stream mode): 3 of 2220 (attempt, block) pairs differ. Oracle, ALL: fan_in 30, fan_out 30, deg_in 30, deg_out 30, sg 8, tcyc 21, any 59 %.

## Detection

Mean of the per-attempt shares, over the complete attempts with edges in the split; n = attempts / edges. The pooled share (predicted edges / edges) is in the JSON.

| Typology | v1_default val | v1_default test | frozen val | frozen test | n val | n test |
|---|---:|---:|---:|---:|---:|---:|
| BIPARTITE | 0.00 | 0.05 | 0.38 | 0.48 | 10 / 50 | 21 / 74 |
| CYCLE | 0.00 | 0.04 | 0.54 | 0.55 | 24 / 65 | 25 / 109 |
| FAN-IN | 0.00 | 0.07 | 0.67 | 0.74 | 19 / 64 | 20 / 137 |
| FAN-OUT | 0.07 | 0.08 | 0.76 | 0.84 | 21 / 65 | 21 / 140 |
| GATHER-SCATTER | 0.13 | 0.07 | 0.65 | 0.78 | 26 / 99 | 39 / 401 |
| RANDOM | 0.12 | 0.12 | 0.52 | 0.74 | 17 / 45 | 17 / 86 |
| SCATTER-GATHER | 0.08 | 0.13 | 0.81 | 0.82 | 23 / 147 | 23 / 252 |
| STACK | 0.09 | 0.08 | 0.55 | 0.64 | 28 / 157 | 26 / 140 |
| **ALL** | 0.07 | 0.08 | 0.63 | 0.70 | 168 / 692 | 192 / 1339 |

## Positives

| | fit | val | test | all |
|---|---:|---:|---:|---:|
| listed (in an attempt) | 1178 | 692 | 1339 | 3209 |
| unlisted | 1121 | 389 | 458 | 1968 |
| listed, no block activates (v1_default) | 204 | 86 | 223 | 513 |
| unlisted, no block activates (v1_default) | 217 | 58 | 86 | 361 |
| listed, no block activates (frozen recipe) | 109 | 48 | 71 | 228 |
| unlisted, no block activates (frozen recipe) | 85 | 39 | 43 | 167 |

Recall on listed and unlisted positives (predicted / positives):

| | val listed | val unlisted | test listed | test unlisted |
|---|---:|---:|---:|---:|
| v1_default | 27 / 692 (4 %) | 0 / 389 (0 %) | 60 / 1339 (4 %) | 0 / 458 (0 %) |
| frozen recipe | 412 / 692 (60 %) | 1 / 389 (0 %) | 988 / 1339 (74 %) | 5 / 458 (1 %) |

## Base rates: non-zero rows (% of edges, all splits)

| Block | v1_default, all edges | v1_default, positives | frozen, all edges | frozen, positives |
|---|---:|---:|---:|---:|
| fan_in | 33.98 | 54.2 | 51.72 | 70.3 |
| fan_out | 54.66 | 47.9 | 62.20 | 59.5 |
| deg_in | 70.17 | 63.4 | 81.75 | 77.9 |
| deg_out | 74.02 | 53.8 | 80.52 | 64.9 |
| sg | 0.02 | 0.1 | 1.49 | 9.5 |
| tcyc | 0.02 | 3.0 | 0.06 | 6.0 |
| any | 80.00 | 83.1 | 84.29 | 92.4 |

# E10: new temporal semantics (SPEC v2 §6.3, §9)

git f63e42d, local (archlinux, AMD Ryzen 5 3600 6-Core Processor, 12 threads, 15910 MB), 2026-11-13T06:00:23+00:00.

Profiles: `v1_default`; `sg_ordered_v1w` = v1_default + `sg_ord` (stream visibility); `sg_ordered_tsvis` = the same under timestamp visibility. Features from the cpu backend; every profile's rows are bit-identical across oracle, cpu and CUDA (`results/p10_semantics.json`).

## Instance counts (HI-Small, all rows)

Instances = the sum of a block's bin columns; rows = triggers with at least one instance.

| Block | v1_default instances | v1_default rows | sg_ordered_v1w instances | sg_ordered_v1w rows | sg_ordered_tsvis instances | sg_ordered_tsvis rows |
|---|---:|---:|---:|---:|---:|---:|
| fan_in | 1,725,512 | 1,725,512 | 1,725,512 | 1,725,512 | 1,727,391 | 1,727,391 |
| fan_out | 2,775,585 | 2,775,585 | 2,775,585 | 2,775,585 | 2,776,879 | 2,776,879 |
| deg_in | 3,563,722 | 3,563,722 | 3,563,722 | 3,563,722 | 3,589,432 | 3,589,432 |
| deg_out | 3,759,204 | 3,759,204 | 3,759,204 | 3,759,204 | 3,774,387 | 3,774,387 |
| tcyc | 1,125 | 1,109 | 1,125 | 1,109 | 1,125 | 1,109 |
| sg | 808 | 788 | 808 | 788 | 812 | 792 |
| sg_ord | — | — | 212 | 195 | 214 | 197 |

By split, `sg_ord` vs `sg` (instances):

| Profile | Split | sg | sg_ord | sg_ord / sg |
|---|---|---:|---:|---:|
| sg_ordered_v1w | fit | 570 | 157 | 0.2754 |
| sg_ordered_v1w | val | 109 | 31 | 0.2844 |
| sg_ordered_v1w | test | 129 | 24 | 0.1860 |
| sg_ordered_v1w | all | 808 | 212 | 0.2624 |
| sg_ordered_tsvis | fit | 572 | 158 | 0.2762 |
| sg_ordered_tsvis | val | 109 | 31 | 0.2844 |
| sg_ordered_tsvis | test | 131 | 25 | 0.1908 |
| sg_ordered_tsvis | all | 812 | 214 | 0.2635 |

## Validation (fit → val), set `all`

| Classifier | Profile | Columns | F1 ×100 | Precision | Recall | PR-AUC | TP | FP | FN | ΔF1 vs v1_default |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 | v1_default | 69 | 4.82 | 0.692 | 0.025 | 0.047 | 27 | 12 | 1054 |  |
| R0 | sg_ordered_v1w | 79 | 4.82 | 0.692 | 0.025 | 0.047 | 27 | 12 | 1054 | +0.00 |
| R0 | sg_ordered_tsvis | 79 | 5.51 | 0.689 | 0.029 | 0.048 | 31 | 14 | 1050 | +0.68 |
| R3/S2 | v1_default | 74 | 13.14 | 0.846 | 0.071 | 0.216 | 77 | 14 | 1004 |  |
| R3/S2 | sg_ordered_v1w | 84 | 13.14 | 0.846 | 0.071 | 0.216 | 77 | 14 | 1004 | +0.00 |
| R3/S2 | sg_ordered_tsvis | 84 | 12.53 | 0.869 | 0.068 | 0.229 | 73 | 11 | 1008 | -0.61 |

Gain share of `sg` and `sg_ord`:

- R0, v1_default: sg 0.0003, sg_ord 0.0000
- R0, sg_ordered_v1w: sg 0.0003, sg_ord 0.0000
- R0, sg_ordered_tsvis: sg 0.0001, sg_ord 0.0000
- R3/S2, v1_default: sg 0.0000, sg_ord 0.0000
- R3/S2, sg_ordered_v1w: sg 0.0000, sg_ord 0.0000
- R3/S2, sg_ordered_tsvis: sg 0.0000, sg_ord 0.0000

## Test (train → test), set `all`: sensitivity only, after the freeze, never used to select

| Classifier | Profile | Columns | F1 ×100 | Precision | Recall | PR-AUC | TP | FP | FN | ΔF1 vs v1_default |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 | v1_default | 69 | 6.40 | 0.769 | 0.033 | 0.073 | 60 | 18 | 1737 |  |
| R0 | sg_ordered_v1w | 79 | 6.40 | 0.769 | 0.033 | 0.073 | 60 | 18 | 1737 | +0.00 |
| R0 | sg_ordered_tsvis | 79 | 6.79 | 0.736 | 0.036 | 0.073 | 64 | 23 | 1733 | +0.39 |
| R3/S2 | v1_default | 74 | 21.67 | 0.690 | 0.129 | 0.207 | 231 | 104 | 1566 |  |
| R3/S2 | sg_ordered_v1w | 84 | 21.67 | 0.690 | 0.129 | 0.207 | 231 | 104 | 1566 | +0.00 |
| R3/S2 | sg_ordered_tsvis | 84 | 19.96 | 0.530 | 0.123 | 0.153 | 221 | 196 | 1576 | -1.71 |

Gain share of `sg` and `sg_ord`:

- R0, v1_default: sg 0.0001, sg_ord 0.0000
- R0, sg_ordered_v1w: sg 0.0001, sg_ord 0.0000
- R0, sg_ordered_tsvis: sg 0.0001, sg_ord 0.0000
- R3/S2, v1_default: sg 0.0000, sg_ord 0.0000
- R3/S2, sg_ordered_v1w: sg 0.0000, sg_ord 0.0000
- R3/S2, sg_ordered_tsvis: sg 0.0000, sg_ord 0.0000

## Reproduction

- validation R0 v1_default vs ledger R0/v1: reproduces
- validation R3/S2 v1_default vs ledger R3/S2: reproduces
- test R0 v1_default vs E8 v1_default all: reproduces

## Notes

- Classifiers: the DG-2 ladder's R0 recipe (v1 ids, XGBoost scale_pos_weight 1, max_depth 6, 100 rounds, base_score estimated) and R3/S2 (the same plus the Multi-GNN columns without Timestamp), both at v1 windows. The frozen DG-2 recipe uses GFP-produced blocks, so it has no compiled sg_ord to add.
- sg_ordered_tsvis on validation is mined on the stream prefix [0, 4,062,676) (blz mine --edges), so the 213 validation rows at timestamp 663,120 do not see its 105 test edges (SPEC v2 §4.4, R5). On test, its train rows come from that prefix mining and its test rows from the full mining.
- Test scores are sensitivity only: DG-2 is closed and frozen (E8), and nothing here selects a recipe.
- At v1's sg window (21,600 s) sg_ord is sparse: 157 instances in the fit split. Its largest gain share in any fit here is 0.0000, so it cannot move F1 at these windows. Wider windows (R4's W-C) were not pre-registered for E10 and are not run.

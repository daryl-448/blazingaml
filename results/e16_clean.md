# E16: what the clean semantics change (SPEC v3 §3.6), HI-Small

git bfb6691 (dirty), local (archlinux, AMD Ryzen 5 3600 6-Core Processor, 12 threads, 15910 MB), 2026-11-24T19:32:12+00:00. Every configuration mined on the oracle (`blz mine --backend oracle`), all 5,078,345 rows; nothing here changes a recipe, profile or default (E16 informs §3.5 only).

- Rows: fit [0, 3,047,216) (2,299 positives); validation [3,047,216, 4,062,463) (1,080 positives). Eid 4,062,676 does not start a timestamp: ts 663,120 holds eids [4,062,463, 4,062,781), so clean validation ends at 4,062,463 (SPEC v3 §6.3). Its 213 rows below 4,062,676 (1 positive) are scored nowhere; every fit here, legacy ones included, uses the same range.
- Classifier: V2 R0's XGBoost settings (`configs/xgb_default.yaml`: library defaults, 100 rounds, threshold 0.5), fit on the fit rows. Every fit has the same base, the DG-2 ladder's carried ids and raw columns: v1 `src_id`, `dst_id` and S2 (Amount Sent, Sent Currency, Amount Received, Received Currency, Payment Format, with Multi-GNN's codes). Then the vertex statistics (52 columns, clipped to the float32 range) and the pattern blocks in v1's order, sg_ord and lcyc last.
- Fit plumbing: v1_default (features/p7_cpu_default) + R3/S2, validation [3,047,216, 4,062,676) gives {'tp': 77, 'fp': 14, 'fn': 1004, 'tn': 1014365}, the ledger {'tp': 77, 'fp': 14, 'fn': 1004, 'tn': 1014365}: reproduces.

## The v1 check

clean_e16_q8 under `--quirks all` (every oracle quirk, plus Q8's histograms) reproduces v1_e16 on every block: the six v1 blocks and lcyc by output hash, the vertex statistics bit for bit against `blz vstats --window 86400`, and sg_ord against V2's sg_ordered (sg_ordered_v1w, oracle). v1_e16's six blocks carry v1_default's AC-V2-0 hashes.

## Quirks

Rows changed = rows of the configuration that differ from clean_e16's (vstats bit for bit; Q8's histograms against clean's counts binned the same way). ΔF1 = validation F1 ×100 minus clean_e16's, with a 95% paired block-bootstrap interval.

| Quirk | How | Blocks with changed rows (fit / validation rows changed) | F1 ×100 | ΔF1 [95%] | ΔF1, training resamples |
|---|---|---|---:|---|---|
| Q1 | oracle quirk | sg 81 / 0, sg_ord 6 / 0 | 9.89 | +0.00 [+0.00, +0.00] | +0.00, +1.00, +0.00 |
| Q2 | oracle quirk | vstats 1,432,535 / 83,787 | 36.95 | +27.07 [+23.51, +30.60] | +12.81, +13.57, +10.26 |
| Q3 | oracle quirk | lcyc 1 / 0, sg 2 / 1, tcyc 1 / 0 | 9.89 | +0.00 [+0.00, +0.00] | +0.00, +0.00, -0.43 |
| Q4 | oracle quirk | deg_in 68,584 / 27,360, deg_out 229,662 / 95,249, fan_in 13,560 / 5,128, fan_out 114,103 / 39,308, sg 2 / 0, sg_ord 1 / 0, vstats 288,914 / 117,361 | 21.92 | +12.04 [+9.17, +14.95] | -4.55, -0.06, -4.51 |
| Q5 | oracle quirk | sg_ord 3 / 0 | 9.89 | +0.00 [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| Q6 | oracle quirk | none | 9.89 | +0.00 [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| Q7 | profile variant | sg_ord removed (its other blocks: none) | 9.89 | +0.00 [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| Q8 | profile variant | fan and degree become histograms (rows changed counted after binning clean's counts: none) | 18.38 | +8.49 [+5.61, +11.39] | -3.49, -3.02, -3.69 |
| Q9 | oracle quirk | vstats 171,008 / 98,150 | 14.94 | +5.06 [+2.51, +7.60] | -2.68, -0.51, -2.96 |
| Q10 | oracle quirk | vstats 2,675,867 / 1,014,434 | 18.48 | +8.60 [+6.27, +11.03] | -7.62, -4.55, -0.57 |
| Q11 | not applicable | GFP batch mode and credit: participants: deleted in P12 (D-55), no value to compare | — | — | — |
| Q12 | not applicable | GFP harness plumbing in the runtime (ts_offset, fuzz knobs): harness only, removed in P12 | — | — | — |
| Q13 | not applicable | input range fixed to HI-Small: an input check, the same rows on HI-Small | — | — | — |
| Q14 | not applicable | GFP's bin rule (A38): harness only, blz keeps [b_i, b_(i+1)) in both | — | — | — |
| Q15 | kept | trigger credit re-reports unchanged instances: clean keeps it (R9), nothing to switch | — | — | — |
| Q16 | kept | min size 2 for histograms: clean keeps it (R10); count blocks have none, which is Q8 | — | — | — |
| Q17 | not applicable | windows and k_max copied from GFP's setup: parameters, re-chosen on validation later; clean_e16 and v1_e16 share v1's windows | — | — | — |
| A29 | not applicable | GFP's fit_transform double-counts re-sent edges (snapml >= 1.14.1): blz never copied it | — | — | — |
| A33 | not applicable | GFP's ts-0 self-loop artifact in tcyc (D-12's ts + 1): harness only, blz never copied it | — | — | — |
| A45 | not applicable | GFP's running-sum drift after eviction in vertex statistics: v2 computes each window exactly (D-37) | — | — | — |

## Validation fits

| Configuration | Columns | F1 ×100 | Precision | Recall | PR-AUC | TP | FP | FN | ΔF1 vs clean [95%] | F1, training resamples |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| clean | 101 | 9.89 | 0.218 | 0.064 | 0.060 | 69 | 247 | 1011 |  | 23.86, 17.68, 20.06 |
| Q1 | 101 | 9.89 | 0.218 | 0.064 | 0.060 | 69 | 247 | 1011 | +0.00 [+0.00, +0.00] | 23.86, 18.69, 20.06 |
| Q2 | 101 | 36.95 | 0.775 | 0.243 | 0.324 | 262 | 76 | 818 | +27.07 [+23.51, +30.60] | 36.67, 31.25, 30.32 |
| Q3 | 101 | 9.89 | 0.218 | 0.064 | 0.060 | 69 | 247 | 1011 | +0.00 [+0.00, +0.00] | 23.86, 17.68, 19.63 |
| Q4 | 101 | 21.92 | 0.739 | 0.129 | 0.221 | 139 | 49 | 941 | +12.04 [+9.17, +14.95] | 19.31, 17.63, 15.55 |
| Q5 | 101 | 9.89 | 0.218 | 0.064 | 0.060 | 69 | 247 | 1011 | +0.00 [+0.00, +0.00] | 23.86, 17.68, 20.06 |
| Q6 | 101 | 9.89 | 0.218 | 0.064 | 0.060 | 69 | 247 | 1011 | +0.00 [+0.00, +0.00] | 23.86, 17.68, 20.06 |
| Q7 | 91 | 9.89 | 0.218 | 0.064 | 0.060 | 69 | 247 | 1011 | +0.00 [+0.00, +0.00] | 23.86, 17.68, 20.06 |
| Q8 | 145 | 18.38 | 0.398 | 0.119 | 0.117 | 129 | 195 | 951 | +8.49 [+5.61, +11.39] | 20.37, 14.67, 16.36 |
| Q9 | 101 | 14.94 | 0.333 | 0.096 | 0.102 | 104 | 208 | 976 | +5.06 [+2.51, +7.60] | 21.18, 17.17, 17.10 |
| Q10 | 101 | 18.48 | 0.210 | 0.165 | 0.084 | 178 | 668 | 902 | +8.60 [+6.27, +11.03] | 16.24, 13.13, 19.48 |
| all | 101 | 3.84 | 0.020 | 0.554 | 0.015 | 598 | 29455 | 482 | -6.04 [-8.02, -4.09] | 4.79, 7.75, 20.53 |
| v1_e16 | 135 | 14.32 | 0.089 | 0.366 | 0.051 | 395 | 4042 | 685 | +4.43 [+2.20, +6.77] | 5.91, 7.64, 10.31 |

## Blocks: clean_e16 against v1_e16

Instances, active rows and the share of positive rows that are active, fit / validation. Count blocks: one instance per trigger with a count of at least 1; the `≥ 2` share is what a min-size-2 histogram marks active.

| Block | clean kind | clean instances | clean active | clean pos. active | v1_e16 instances | v1_e16 active | v1_e16 pos. active | rows changed (v1_e16 vs clean) |
|---|---|---|---|---|---|---|---|---|
| fan_in | count | 2,499,332 / 993,362 | 2,499,332 / 993,362 | 0.998 / 0.999 (≥ 2: 0.558 / 0.575) | 957,690 / 339,809 | 957,690 / 339,809 | 0.555 / 0.575 | 6,464 / 2,316 |
| fan_out | count | 2,499,332 / 993,362 | 2,499,332 / 993,362 | 0.998 / 0.999 (≥ 2: 0.464 / 0.542) | 1,491,831 / 663,475 | 1,491,831 / 663,475 | 0.464 / 0.542 | 1,804 / 356 |
| deg_in | count | 2,499,332 / 993,362 | 2,499,332 / 993,362 | 0.998 / 0.999 (≥ 2: 0.658 / 0.679) | 1,929,300 / 833,506 | 1,929,300 / 833,506 | 0.654 / 0.677 | 56,731 / 21,997 |
| deg_out | count | 2,499,332 / 993,362 | 2,499,332 / 993,362 | 0.998 / 0.999 (≥ 2: 0.513 / 0.598) | 2,045,383 / 897,447 | 2,045,383 / 897,447 | 0.513 / 0.598 | 41,018 / 14,995 |
| tcyc | histogram | 610 / 253 | 599 / 251 | 0.035 / 0.033 | 611 / 253 | 600 / 251 | 0.035 / 0.033 | 1 / 0 |
| sg | histogram | 483 / 108 | 467 / 108 | 0.001 / 0.000 | 570 / 109 | 550 / 109 | 0.001 / 0.000 | 87 / 1 |
| sg_ord | histogram | 151 / 31 | 138 / 31 | 0.000 / 0.000 | — | — | — | — |
| lcyc | histogram | 637 / 262 | 618 / 254 | 0.036 / 0.036 | 638 / 262 | 619 / 254 | 0.036 / 0.036 | 1 / 0 |
| vstats | vstats | — / — | 2,680,775 / 1,014,434 | 1.000 / 1.000 | — / — | 3,047,216 / 1,015,247 | 1.000 / 1.000 | 3,043,961 / 1,015,247 |

## Per-block effects

Each block a configuration changes, against clean_e16, as fit / validation: instances, active rows, the share of positive rows that are active, and rows changed. A count block shows its rows with a count of at least 2 (every non-loop trigger has a count of at least 1), which is also what a histogram marks; Q4's count blocks show the sum of the counts as instances.

| Configuration | Block | Instances, clean → quirk | Active rows, clean → quirk | Pos. active, clean → quirk | Rows changed (positives) |
|---|---|---|---|---|---|
| Q1 | sg | 483 → 568 / 108 → 108 | 467 → 548 / 108 → 108 | 0.001 → 0.001 / 0.000 → 0.000 | 81 (0) / 0 (0) |
| Q1 | sg_ord | 151 → 161 / 31 → 31 | 138 → 144 / 31 → 31 | 0.000 → 0.000 / 0.000 → 0.000 | 6 (0) / 0 (0) |
| Q2 | vstats | — | 2,680,775 → 3,047,216 / 1,014,434 → 1,015,247 | 1.000 → 1.000 / 1.000 → 1.000 | 1,432,535 (1,194) / 83,787 (500) |
| Q3 | lcyc | 637 → 638 / 262 → 262 | 618 → 619 / 254 → 254 | 0.036 → 0.036 / 0.036 → 0.036 | 1 (0) / 0 (0) |
| Q3 | sg | 483 → 485 / 108 → 109 | 467 → 469 / 108 → 109 | 0.001 → 0.001 / 0.000 → 0.000 | 2 (0) / 1 (0) |
| Q3 | tcyc | 610 → 611 / 253 → 253 | 599 → 600 / 251 → 251 | 0.035 → 0.035 / 0.033 → 0.033 | 1 (0) / 0 (0) |
| Q4 | deg_in | 8,799,063 → 8,723,769 / 3,865,939 → 3,835,741 | 1,944,976 → 1,929,300 / 838,673 → 833,506 | 0.658 → 0.654 / 0.679 → 0.677 | 68,584 (55) / 27,360 (15) |
| Q4 | deg_out | 2,228,461,801 → 2,227,351,070 / 1,071,767,514 → 1,071,376,198 | 2,055,232 → 2,045,383 / 899,965 → 897,447 | 0.513 → 0.513 / 0.598 → 0.598 | 229,662 (264) / 95,249 (99) |
| Q4 | fan_in | 4,351,359 → 4,331,741 / 1,592,828 → 1,585,107 | 958,882 → 957,690 / 340,119 → 339,809 | 0.558 → 0.555 / 0.575 → 0.575 | 13,560 (46) / 5,128 (13) |
| Q4 | fan_out | 977,466,931 → 977,093,869 / 444,144,722 → 444,053,348 | 1,492,715 → 1,491,831 / 663,664 → 663,475 | 0.464 → 0.464 / 0.542 → 0.542 | 114,103 (167) / 39,308 (54) |
| Q4 | sg | 483 → 481 / 108 → 108 | 467 → 465 / 108 → 108 | 0.001 → 0.001 / 0.000 → 0.000 | 2 (0) / 0 (0) |
| Q4 | sg_ord | 151 → 150 / 31 → 31 | 138 → 137 / 31 → 31 | 0.000 → 0.000 / 0.000 → 0.000 | 1 (0) / 0 (0) |
| Q4 | vstats | — | 2,680,775 → 2,676,142 / 1,014,434 → 1,013,767 | 1.000 → 1.000 / 1.000 → 1.000 | 288,914 (313) / 117,361 (115) |
| Q5 | sg_ord | 151 → 148 / 31 → 31 | 138 → 135 / 31 → 31 | 0.000 → 0.000 / 0.000 → 0.000 | 3 (0) / 0 (0) |
| Q9 | vstats | — | 2,680,775 → 2,680,776 / 1,014,434 → 1,014,436 | 1.000 → 1.000 / 1.000 → 1.000 | 171,008 (224) / 98,150 (133) |
| Q10 | vstats | — | 2,680,775 → 2,680,775 / 1,014,434 → 1,014,434 | 1.000 → 1.000 / 1.000 → 1.000 | 2,675,867 (2,299) / 1,014,434 (1,080) |
| all | deg_in | 8,799,063 → 8,723,769 / 3,865,939 → 3,835,741 | 1,944,976 → 1,929,300 / 838,673 → 833,506 | 0.658 → 0.654 / 0.679 → 0.677 | 68,584 (55) / 27,360 (15) |
| all | deg_out | 2,228,461,801 → 2,227,351,070 / 1,071,767,514 → 1,071,376,198 | 2,055,232 → 2,045,383 / 899,965 → 897,447 | 0.513 → 0.513 / 0.598 → 0.598 | 229,662 (264) / 95,249 (99) |
| all | fan_in | 4,351,359 → 4,331,741 / 1,592,828 → 1,585,107 | 958,882 → 957,690 / 340,119 → 339,809 | 0.558 → 0.555 / 0.575 → 0.575 | 13,560 (46) / 5,128 (13) |
| all | fan_out | 977,466,931 → 977,093,869 / 444,144,722 → 444,053,348 | 1,492,715 → 1,491,831 / 663,664 → 663,475 | 0.464 → 0.464 / 0.542 → 0.542 | 114,103 (167) / 39,308 (54) |
| all | lcyc | 637 → 638 / 262 → 262 | 618 → 619 / 254 → 254 | 0.036 → 0.036 / 0.036 → 0.036 | 1 (0) / 0 (0) |
| all | sg | 483 → 570 / 108 → 109 | 467 → 550 / 108 → 109 | 0.001 → 0.001 / 0.000 → 0.000 | 87 (0) / 1 (0) |
| all | sg_ord | 151 → 157 / 31 → 31 | 138 → 140 / 31 → 31 | 0.000 → 0.000 / 0.000 → 0.000 | 10 (0) / 0 (0) |
| all | tcyc | 610 → 611 / 253 → 253 | 599 → 600 / 251 → 251 | 0.035 → 0.035 / 0.033 → 0.033 | 1 (0) / 0 (0) |
| all | vstats | — | 2,680,775 → 3,047,216 / 1,014,434 → 1,015,247 | 1.000 → 1.000 / 1.000 → 1.000 | 3,043,961 (2,299) / 1,015,247 (1,080) |
| all+Q8 | deg_in | 1,944,976 → 1,929,300 / 838,673 → 833,506 | 1,944,976 → 1,929,300 / 838,673 → 833,506 | 0.658 → 0.654 / 0.679 → 0.677 | 56,731 (40) / 21,997 (11) |
| all+Q8 | deg_out | 2,055,232 → 2,045,383 / 899,965 → 897,447 | 2,055,232 → 2,045,383 / 899,965 → 897,447 | 0.513 → 0.513 / 0.598 → 0.598 | 41,018 (1) / 14,995 (0) |
| all+Q8 | fan_in | 958,882 → 957,690 / 340,119 → 339,809 | 958,882 → 957,690 / 340,119 → 339,809 | 0.558 → 0.555 / 0.575 → 0.575 | 6,464 (31) / 2,316 (9) |
| all+Q8 | fan_out | 1,492,715 → 1,491,831 / 663,664 → 663,475 | 1,492,715 → 1,491,831 / 663,664 → 663,475 | 0.464 → 0.464 / 0.542 → 0.542 | 1,804 (1) / 356 (0) |
| all+Q8 | lcyc | 637 → 638 / 262 → 262 | 618 → 619 / 254 → 254 | 0.036 → 0.036 / 0.036 → 0.036 | 1 (0) / 0 (0) |
| all+Q8 | sg | 483 → 570 / 108 → 109 | 467 → 550 / 108 → 109 | 0.001 → 0.001 / 0.000 → 0.000 | 87 (0) / 1 (0) |
| all+Q8 | sg_ord | 151 → 157 / 31 → 31 | 138 → 140 / 31 → 31 | 0.000 → 0.000 / 0.000 → 0.000 | 10 (0) / 0 (0) |
| all+Q8 | tcyc | 610 → 611 / 253 → 253 | 599 → 600 / 251 → 251 | 0.035 → 0.035 / 0.033 → 0.033 | 1 (0) / 0 (0) |
| all+Q8 | vstats | — | 2,680,775 → 3,047,216 / 1,014,434 → 1,015,247 | 1.000 → 1.000 / 1.000 → 1.000 | 3,043,961 (2,299) / 1,015,247 (1,080) |
| v1_e16 | deg_in | 1,944,976 → 1,929,300 / 838,673 → 833,506 | 1,944,976 → 1,929,300 / 838,673 → 833,506 | 0.658 → 0.654 / 0.679 → 0.677 | 56,731 (40) / 21,997 (11) |
| v1_e16 | deg_out | 2,055,232 → 2,045,383 / 899,965 → 897,447 | 2,055,232 → 2,045,383 / 899,965 → 897,447 | 0.513 → 0.513 / 0.598 → 0.598 | 41,018 (1) / 14,995 (0) |
| v1_e16 | fan_in | 958,882 → 957,690 / 340,119 → 339,809 | 958,882 → 957,690 / 340,119 → 339,809 | 0.558 → 0.555 / 0.575 → 0.575 | 6,464 (31) / 2,316 (9) |
| v1_e16 | fan_out | 1,492,715 → 1,491,831 / 663,664 → 663,475 | 1,492,715 → 1,491,831 / 663,664 → 663,475 | 0.464 → 0.464 / 0.542 → 0.542 | 1,804 (1) / 356 (0) |
| v1_e16 | lcyc | 637 → 638 / 262 → 262 | 618 → 619 / 254 → 254 | 0.036 → 0.036 / 0.036 → 0.036 | 1 (0) / 0 (0) |
| v1_e16 | sg | 483 → 570 / 108 → 109 | 467 → 550 / 108 → 109 | 0.001 → 0.001 / 0.000 → 0.000 | 87 (0) / 1 (0) |
| v1_e16 | tcyc | 610 → 611 / 253 → 253 | 599 → 600 / 251 → 251 | 0.035 → 0.035 / 0.033 → 0.033 | 1 (0) / 0 (0) |
| v1_e16 | vstats | — | 2,680,775 → 3,047,216 / 1,014,434 → 1,015,247 | 1.000 → 1.000 / 1.000 → 1.000 | 3,043,961 (2,299) / 1,015,247 (1,080) |

Every (block, configuration) pair, with the same counts, is in `results/e16_clean.json` (`blocks`).

## Runs

| Configuration | Profile | Quirks | Wall | Peak RSS | Load before |
|---|---|---|---:|---:|---|
| clean | clean_e16 | none | 174 s | 420 MB | 0.82, 0.77, 6.60 |
| Q1 | clean_e16 | Q1 | 175 s | 420 MB | 6.68, 7.85, 8.04 |
| Q2 | clean_e16 | Q2 | 175 s | 420 MB | 10.17, 9.29, 8.60 |
| Q3 | clean_e16 | Q3 | 177 s | 419 MB | 9.45, 9.85, 8.97 |
| Q4 | clean_e16 | Q4 | 176 s | 420 MB | 11.89, 10.77, 9.48 |
| Q5 | clean_e16 | Q5 | 178 s | 419 MB | 12.65, 11.62, 10.03 |
| Q6 | clean_e16 | Q6 | 178 s | 420 MB | 11.42, 11.63, 10.32 |
| Q7 | clean_e16_q7 | none | 172 s | 419 MB | 12.12, 11.88, 10.65 |
| Q8 | clean_e16_q8 | none | 178 s | 458 MB | 11.19, 11.73, 10.81 |
| Q9 | clean_e16 | Q9 | 175 s | 422 MB | 12.21, 11.93, 11.05 |
| Q10 | clean_e16 | Q10 | 149 s | 426 MB | 11.88, 11.82, 11.15 |
| all | clean_e16 | all | 146 s | 426 MB | 11.19, 11.55, 11.15 |
| all+Q8 | clean_e16_q8 | all | 147 s | 458 MB | 10.03, 6.80, 7.73 |
| v1_e16 | v1_e16 | legacy path | 95 s | 437 MB | 11.16, 5.47, 7.44 |

## Notes

- Training noise dominates single fits. F1 at threshold 0.5 under R0's settings moves a lot when only the training sample changes: clean_e16's four draws (the unweighted fit, then 3 refits on block-bootstrapped fit rows, 100 chunks, the same weights for every configuration) give F1 9.89, 23.86, 17.68, 20.06. The [95%] column covers only the validation sample's noise. Here a quirk has an effect only when its paired ΔF1 keeps one sign over all four draws.
- Only Q2 has one. Its ΔF1 over the four draws is +27.07, +12.81, +13.57, +10.26 and its ΔPR-AUC +0.264, +0.192, +0.132, +0.100. The fit and validation rows hold 569,769 self-loops, 481,056 of them Reinvestment, and 6 laundering ones (0 Reinvestment). Every Reinvestment edge is a self-loop. Q2 puts each account's reinvestment history into its vertex statistics, and R1 takes it out: R1 keeps self-loops as classifier rows, but no feature carries an account's self-loops into its other rows. Q2's effect is the vstats part only: fan and degree drop self-loops under both semantics, self-loop triggers have zero pattern rows under both, and batch mode (A41) is deleted.
- Q5, Q6, Q7 leave every fit identical to clean's in all four draws, and Q1 and Q3 move F1 by at most 1.00 in one draw. The blocks they change (sg, sg_ord, tcyc, lcyc) hold at most 0.09% of the gain in any fit, and HI-Small barely has a cycle or a scatter-gather with two hops in one minute: Q5 changes 3 sg_ord rows and no tcyc row, and Q6 changes no lcyc row. sg_ord has 138 active fit rows, 1 of them positive, and none of the 13 unweighted fits splits on sg or sg_ord, so Q7 (dropping the block) changes nothing.
- Q4 (stream visibility) changes fan, degree and vstats rows, but its ΔF1 takes both signs (+12.04, -4.55, -0.06, -4.51). Within a timestamp, file order carries the label: 1,814 of the 3,379 fit and validation positives that share their minute sit in the first tenth of its rows (mean relative rank 0.300 on fit and 0.228 on validation, against 0.500 for negatives). Under stream visibility a row's windows end at its file position, so its features can encode that rank; R3 removes this.
- Q10 (absolute timestamps) puts 85.2% of validation rows above the fit maximum of u_out's time mean (85.2% for v1_e16; clean's ages: 0.0%), so the trees extrapolate on time. Its ΔF1 takes both signs (+8.60, -7.62, -4.55, -0.57); its PR-AUC is lower in 3 of 4 draws.
- All the oracle quirks together score below clean in 3 of 4 draws on F1 and 4 of 4 on PR-AUC, and v1_e16 in 3 and 4, even though Q2 alone raises both every time. The quirks do not add up, and which of them cancels Q2's gain was not tested. The all-quirks unweighted fit predicts 29,455 false positives.
- Q8 (histograms instead of counts) is information-preserving here (clean's counts, binned, equal Q8's rows exactly), and its ΔF1 takes both signs (+8.49, -3.49, -3.02, -3.69). So does Q9's (+5.06, -2.68, -0.51, -2.96).
- R0's settings make a weak classifier on these columns, and the absolute levels are not comparable to E8: DG-2's tuned recipe reached F1 53.02 on GFP features. Only the paired differences are read here.
- No test label is read. Labels are sliced to the fit and validation rows, and the evidence above uses those rows only. Each run mines every row, like every other HI-Small run, but no test row is scored.
- Count blocks: instances = rows with a count of at least 1 (one instance per trigger, R7). The ≥ 2 share is what v1's histogram marks active.
- Timings are informal (P13). No foreign process used more than 20% of a core before any run. The load average logged before a run still includes the previous E16 run.

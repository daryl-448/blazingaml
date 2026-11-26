# E16: what the clean semantics change (SPEC v3 §3.6), TransXion

git af5dc36, local (archlinux, AMD Ryzen 5 3600 6-Core Processor, 12 threads, 15910 MB), 2026-11-26T16:20:19+00:00. Every configuration mined on the oracle (`blz mine --backend oracle`), rows [0, 2,423,336) of 3,029,170; nothing here changes a recipe, profile or default (E16 informs §3.5 only).

- Rows: fit [0, 1,817,503) (2,960 positives); validation [1,817,503, 2,423,336) (779 positives). The .blzg's fit and validation rows, as `blz prepare` stored them with V2 D-17's boundary rule at both cuts (SPEC v3 §6.3). Every run mines [0, 2,423,336) (`--triggers train`); the test rows [2,423,336, 3,029,170) are neither mined nor read.
- Classifier: V2 R0's XGBoost settings (`configs/xgb_default.yaml`: library defaults, 100 rounds, threshold 0.5), fit on the fit rows. Every fit has the same base, the DG-2 ladder's carried ids and raw columns: v1 `src_id`, `dst_id` and S2 (Amount Sent, Sent Currency, Amount Received, Received Currency, Payment Format, with Multi-GNN's codes). Then the vertex statistics (52 columns, clipped to the float32 range) and the pattern blocks in v1's order, sg_ord and lcyc last.

## The v1 check

clean_e16_q8 under `--quirks all` (every oracle quirk, plus Q8's histograms) reproduces v1_e16 on every block of these rows: deg_in, deg_out, fan_in, fan_out, lcyc, sg, tcyc against v1_e16 by output hash, the others row by row (the vertex statistics bit for bit against `blz vstats --window 86400`), and sg_ord against V2's sg_ordered (sg_ordered_v1w) mined on the same rows on the oracle. AC-V2-0's hashes are HI-Small's and are not checked.

## Quirks

Rows changed = rows of the configuration that differ from clean_e16's (vstats bit for bit; Q8's histograms against clean's counts binned the same way). ΔF1 = validation F1 ×100 minus clean_e16's, with a 95% paired block-bootstrap interval.

| Quirk | How | Blocks with changed rows (fit / validation rows changed) | F1 ×100 | ΔF1 [95%] | ΔF1, training resamples |
|---|---|---|---:|---|---|
| Q1 | oracle quirk | none | 28.93 | +0.00 [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| Q2 | oracle quirk | vstats 48 / 13 | 29.70 | +0.78 [-2.15, +3.80] | +0.37, -6.23, -1.65 |
| Q3 | oracle quirk | none | 28.93 | +0.00 [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| Q4 | oracle quirk | deg_in 696 / 194, deg_out 453 / 141, fan_in 245 / 57, fan_out 12 / 3, lcyc 327 / 93, tcyc 327 / 93, vstats 1,143 / 336 | 30.15 | +1.23 [-1.73, +4.19] | +1.40, -3.56, -5.27 |
| Q5 | oracle quirk | tcyc 660 / 200 | 32.16 | +3.24 [+0.16, +6.46] | -3.29, -0.05, -0.87 |
| Q6 | oracle quirk | lcyc 660 / 200 | 32.16 | +3.24 [+0.16, +6.46] | -3.29, -0.05, -0.87 |
| Q7 | profile variant | sg_ord removed (its other blocks: none) | 28.93 | +0.00 [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| Q8 | profile variant | fan and degree become histograms (rows changed counted after binning clean's counts: none) | 36.04 | +7.12 [+4.21, +9.98] | +2.73, +1.05, -2.23 |
| Q9 | oracle quirk | vstats 4,640 / 1,028 | 24.18 | -4.75 [-7.33, -2.17] | +0.13, +1.18, -0.39 |
| Q10 | oracle quirk | vstats 1,817,499 / 605,823 | 14.82 | -14.11 [-17.26, -10.80] | -20.19, -17.19, -28.62 |
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
| clean | 101 | 28.93 | 0.268 | 0.315 | 0.159 | 245 | 670 | 534 |  | 31.10, 35.26, 37.91 |
| Q1 | 101 | 28.93 | 0.268 | 0.315 | 0.159 | 245 | 670 | 534 | +0.00 [+0.00, +0.00] | 31.10, 35.26, 37.91 |
| Q2 | 101 | 29.70 | 0.300 | 0.294 | 0.190 | 229 | 534 | 550 | +0.78 [-2.15, +3.80] | 31.47, 29.02, 36.27 |
| Q3 | 101 | 28.93 | 0.268 | 0.315 | 0.159 | 245 | 670 | 534 | +0.00 [+0.00, +0.00] | 31.10, 35.26, 37.91 |
| Q4 | 101 | 30.15 | 0.279 | 0.329 | 0.140 | 256 | 663 | 523 | +1.23 [-1.73, +4.19] | 32.51, 31.70, 32.65 |
| Q5 | 101 | 32.16 | 0.281 | 0.376 | 0.142 | 293 | 750 | 486 | +3.24 [+0.16, +6.46] | 27.81, 35.21, 37.04 |
| Q6 | 101 | 32.16 | 0.281 | 0.376 | 0.142 | 293 | 750 | 486 | +3.24 [+0.16, +6.46] | 27.81, 35.21, 37.04 |
| Q7 | 91 | 28.93 | 0.268 | 0.315 | 0.159 | 245 | 670 | 534 | +0.00 [+0.00, +0.00] | 31.10, 35.26, 37.91 |
| Q8 | 145 | 36.04 | 0.308 | 0.435 | 0.204 | 339 | 763 | 440 | +7.12 [+4.21, +9.98] | 33.83, 36.31, 35.69 |
| Q9 | 101 | 24.18 | 0.214 | 0.279 | 0.131 | 217 | 799 | 562 | -4.75 [-7.33, -2.17] | 31.23, 36.44, 37.52 |
| Q10 | 101 | 14.82 | 0.107 | 0.240 | 0.087 | 187 | 1558 | 592 | -14.11 [-17.26, -10.80] | 10.91, 18.07, 9.29 |
| all | 101 | 8.23 | 0.049 | 0.259 | 0.029 | 202 | 3929 | 577 | -20.70 [-23.57, -17.86] | 22.35, 19.68, 6.31 |
| v1_e16 | 135 | 12.57 | 0.082 | 0.268 | 0.048 | 209 | 2337 | 570 | -16.35 [-19.37, -13.33] | 5.01, 25.45, 4.95 |

## Blocks: clean_e16 against v1_e16

Instances, active rows and the share of positive rows that are active, fit / validation. Count blocks: one instance per trigger with a count of at least 1; the `≥ 2` share is what a min-size-2 histogram marks active.

| Block | clean kind | clean instances | clean active | clean pos. active | v1_e16 instances | v1_e16 active | v1_e16 pos. active | rows changed (v1_e16 vs clean) |
|---|---|---|---|---|---|---|---|---|
| fan_in | count | 1,817,490 / 605,823 | 1,817,490 / 605,823 | 0.998 / 0.994 (≥ 2: 0.229 / 0.334) | 1,442,603 / 482,295 | 1,442,603 / 482,295 | 0.229 / 0.334 | 33 / 18 |
| fan_out | count | 1,817,490 / 605,823 | 1,817,490 / 605,823 | 0.998 / 0.994 (≥ 2: 0.388 / 0.257) | 1,065,733 / 371,056 | 1,065,733 / 371,056 | 0.388 / 0.257 | 12 / 3 |
| deg_in | count | 1,817,490 / 605,823 | 1,817,490 / 605,823 | 0.998 / 0.994 (≥ 2: 0.670 / 0.762) | 1,446,027 / 483,089 | 1,446,027 / 483,089 | 0.632 / 0.727 | 442 / 139 |
| deg_out | count | 1,817,490 / 605,823 | 1,817,490 / 605,823 | 0.998 / 0.994 (≥ 2: 0.753 / 0.733) | 1,089,124 / 376,141 | 1,089,124 / 376,141 | 0.716 / 0.693 | 452 / 141 |
| tcyc | histogram | 1,616 / 545 | 1,606 / 535 | 0.179 / 0.229 | 956 / 345 | 946 / 335 | 0.034 / 0.053 | 660 / 200 |
| sg | histogram | 210 / 48 | 210 / 48 | 0.000 / 0.005 | 210 / 48 | 210 / 48 | 0.000 / 0.005 | 0 / 0 |
| sg_ord | histogram | 54 / 12 | 54 / 12 | 0.000 / 0.001 | — | — | — | — |
| lcyc | histogram | 1,710 / 584 | 1,686 / 569 | 0.185 / 0.240 | 1,050 / 384 | 1,027 / 369 | 0.041 / 0.064 | 660 / 200 |
| vstats | vstats | — / — | 1,817,499 / 605,823 | 0.999 / 0.994 | — / — | 1,817,503 / 605,833 | 1.000 / 1.000 | 1,817,503 / 605,833 |

## Per-block effects

Each block a configuration changes, against clean_e16, as fit / validation: instances, active rows, the share of positive rows that are active, and rows changed. A count block shows its rows with a count of at least 2 (every non-loop trigger has a count of at least 1), which is also what a histogram marks; Q4's count blocks show the sum of the counts as instances.

| Configuration | Block | Instances, clean → quirk | Active rows, clean → quirk | Pos. active, clean → quirk | Rows changed (positives) |
|---|---|---|---|---|---|
| Q2 | vstats | — | 1,817,499 → 1,817,503 / 605,823 → 605,833 | 0.999 → 1.000 / 0.994 → 1.000 | 48 (11) / 13 (5) |
| Q4 | deg_in | 34,321,632 → 34,320,936 / 7,756,689 → 7,756,495 | 1,446,233 → 1,446,027 / 483,146 → 483,089 | 0.670 → 0.632 / 0.762 → 0.727 | 696 (223) / 194 (67) |
| Q4 | deg_out | 3,659,530 → 3,659,077 / 1,263,160 → 1,263,019 | 1,089,343 → 1,089,124 / 376,206 → 376,141 | 0.753 → 0.716 / 0.733 → 0.693 | 453 (223) / 141 (67) |
| Q4 | fan_in | 33,476,585 → 33,476,340 / 7,626,676 → 7,626,619 | 1,442,603 → 1,442,603 / 482,296 → 482,295 | 0.229 → 0.229 / 0.334 → 0.334 | 245 (0) / 57 (0) |
| Q4 | fan_out | 3,546,399 → 3,546,387 / 1,238,087 → 1,238,084 | 1,065,739 → 1,065,733 / 371,056 → 371,056 | 0.388 → 0.388 / 0.257 → 0.257 | 12 (0) / 3 (0) |
| Q4 | lcyc | 1,710 → 1,383 / 584 → 491 | 1,686 → 1,360 / 569 → 476 | 0.185 → 0.111 / 0.240 → 0.153 | 327 (220) / 93 (68) |
| Q4 | tcyc | 1,616 → 1,289 / 545 → 452 | 1,606 → 1,279 / 535 → 442 | 0.179 → 0.104 / 0.229 → 0.141 | 327 (220) / 93 (68) |
| Q4 | vstats | — | 1,817,499 → 1,817,499 / 605,823 → 605,823 | 0.999 → 0.999 / 0.994 → 0.994 | 1,143 (443) / 336 (135) |
| Q5 | tcyc | 1,616 → 956 / 545 → 345 | 1,606 → 946 / 535 → 335 | 0.179 → 0.034 / 0.229 → 0.053 | 660 (428) / 200 (137) |
| Q6 | lcyc | 1,710 → 1,050 / 584 → 384 | 1,686 → 1,027 / 569 → 369 | 0.185 → 0.041 / 0.240 → 0.064 | 660 (428) / 200 (137) |
| Q9 | vstats | — | 1,817,499 → 1,817,499 / 605,823 → 605,823 | 0.999 → 0.999 / 0.994 → 0.994 | 4,640 (3) / 1,028 (2) |
| Q10 | vstats | — | 1,817,499 → 1,817,499 / 605,823 → 605,823 | 0.999 → 0.999 / 0.994 → 0.994 | 1,817,499 (2,958) / 605,823 (774) |
| all | deg_in | 34,321,632 → 34,320,936 / 7,756,689 → 7,756,495 | 1,446,233 → 1,446,027 / 483,146 → 483,089 | 0.670 → 0.632 / 0.762 → 0.727 | 696 (223) / 194 (67) |
| all | deg_out | 3,659,530 → 3,659,077 / 1,263,160 → 1,263,019 | 1,089,343 → 1,089,124 / 376,206 → 376,141 | 0.753 → 0.716 / 0.733 → 0.693 | 453 (223) / 141 (67) |
| all | fan_in | 33,476,585 → 33,476,340 / 7,626,676 → 7,626,619 | 1,442,603 → 1,442,603 / 482,296 → 482,295 | 0.229 → 0.229 / 0.334 → 0.334 | 245 (0) / 57 (0) |
| all | fan_out | 3,546,399 → 3,546,387 / 1,238,087 → 1,238,084 | 1,065,739 → 1,065,733 / 371,056 → 371,056 | 0.388 → 0.388 / 0.257 → 0.257 | 12 (0) / 3 (0) |
| all | lcyc | 1,710 → 1,050 / 584 → 384 | 1,686 → 1,027 / 569 → 369 | 0.185 → 0.041 / 0.240 → 0.064 | 660 (428) / 200 (137) |
| all | tcyc | 1,616 → 956 / 545 → 345 | 1,606 → 946 / 535 → 335 | 0.179 → 0.034 / 0.229 → 0.053 | 660 (428) / 200 (137) |
| all | vstats | — | 1,817,499 → 1,817,503 / 605,823 → 605,833 | 0.999 → 1.000 / 0.994 → 1.000 | 1,817,503 (2,960) / 605,833 (779) |
| all+Q8 | deg_in | 1,446,233 → 1,446,027 / 483,146 → 483,089 | 1,446,233 → 1,446,027 / 483,146 → 483,089 | 0.670 → 0.632 / 0.762 → 0.727 | 442 (212) / 139 (61) |
| all+Q8 | deg_out | 1,089,343 → 1,089,124 / 376,206 → 376,141 | 1,089,343 → 1,089,124 / 376,206 → 376,141 | 0.753 → 0.716 / 0.733 → 0.693 | 452 (223) / 141 (67) |
| all+Q8 | fan_in | 1,442,603 → 1,442,603 / 482,296 → 482,295 | 1,442,603 → 1,442,603 / 482,296 → 482,295 | 0.229 → 0.229 / 0.334 → 0.334 | 33 (0) / 18 (0) |
| all+Q8 | fan_out | 1,065,739 → 1,065,733 / 371,056 → 371,056 | 1,065,739 → 1,065,733 / 371,056 → 371,056 | 0.388 → 0.388 / 0.257 → 0.257 | 12 (0) / 3 (0) |
| all+Q8 | lcyc | 1,710 → 1,050 / 584 → 384 | 1,686 → 1,027 / 569 → 369 | 0.185 → 0.041 / 0.240 → 0.064 | 660 (428) / 200 (137) |
| all+Q8 | tcyc | 1,616 → 956 / 545 → 345 | 1,606 → 946 / 535 → 335 | 0.179 → 0.034 / 0.229 → 0.053 | 660 (428) / 200 (137) |
| all+Q8 | vstats | — | 1,817,499 → 1,817,503 / 605,823 → 605,833 | 0.999 → 1.000 / 0.994 → 1.000 | 1,817,503 (2,960) / 605,833 (779) |
| v1_e16 | deg_in | 1,446,233 → 1,446,027 / 483,146 → 483,089 | 1,446,233 → 1,446,027 / 483,146 → 483,089 | 0.670 → 0.632 / 0.762 → 0.727 | 442 (212) / 139 (61) |
| v1_e16 | deg_out | 1,089,343 → 1,089,124 / 376,206 → 376,141 | 1,089,343 → 1,089,124 / 376,206 → 376,141 | 0.753 → 0.716 / 0.733 → 0.693 | 452 (223) / 141 (67) |
| v1_e16 | fan_in | 1,442,603 → 1,442,603 / 482,296 → 482,295 | 1,442,603 → 1,442,603 / 482,296 → 482,295 | 0.229 → 0.229 / 0.334 → 0.334 | 33 (0) / 18 (0) |
| v1_e16 | fan_out | 1,065,739 → 1,065,733 / 371,056 → 371,056 | 1,065,739 → 1,065,733 / 371,056 → 371,056 | 0.388 → 0.388 / 0.257 → 0.257 | 12 (0) / 3 (0) |
| v1_e16 | lcyc | 1,710 → 1,050 / 584 → 384 | 1,686 → 1,027 / 569 → 369 | 0.185 → 0.041 / 0.240 → 0.064 | 660 (428) / 200 (137) |
| v1_e16 | tcyc | 1,616 → 956 / 545 → 345 | 1,606 → 946 / 535 → 335 | 0.179 → 0.034 / 0.229 → 0.053 | 660 (428) / 200 (137) |
| v1_e16 | vstats | — | 1,817,499 → 1,817,503 / 605,823 → 605,833 | 0.999 → 1.000 / 0.994 → 1.000 | 1,817,503 (2,960) / 605,833 (779) |

Every (block, configuration) pair, with the same counts, is in `results/e16_clean_transxion.json` (`blocks`).

## Runs

| Configuration | Profile | Quirks | Wall | Peak RSS | Load before |
|---|---|---|---:|---:|---|
| clean | clean_e16 | none | 4 s | 576 MB | 6.32, 10.67, 11.07 |
| Q1 | clean_e16 | Q1 | 4 s | 576 MB | 6.86, 10.71, 11.08 |
| Q2 | clean_e16 | Q2 | 4 s | 576 MB | 6.86, 10.71, 11.08 |
| Q3 | clean_e16 | Q3 | 4 s | 576 MB | 6.39, 10.55, 11.03 |
| Q4 | clean_e16 | Q4 | 4 s | 576 MB | 7.00, 10.61, 11.04 |
| Q5 | clean_e16 | Q5 | 4 s | 576 MB | 6.52, 10.45, 10.99 |
| Q6 | clean_e16 | Q6 | 4 s | 576 MB | 6.52, 10.45, 10.99 |
| Q7 | clean_e16_q7 | none | 4 s | 576 MB | 6.07, 10.29, 10.94 |
| Q8 | clean_e16_q8 | none | 4 s | 584 MB | 6.55, 10.32, 10.94 |
| Q9 | clean_e16 | Q9 | 4 s | 577 MB | 6.10, 10.16, 10.89 |
| Q10 | clean_e16 | Q10 | 4 s | 576 MB | 6.10, 10.16, 10.89 |
| all | clean_e16 | all | 4 s | 576 MB | 5.70, 10.01, 10.83 |
| all+Q8 | clean_e16_q8 | all | 4 s | 584 MB | 7.24, 10.26, 10.91 |
| v1_e16 | v1_e16 | legacy path | 6 s | 142 MB | 7.62, 10.29, 10.92 |
| sg_ordered_v1w | sg_ordered_v1w | none | 2 s | 142 MB | 7.09, 10.13, 10.86 |

## Notes

- clean_e16's 4 draws (the unweighted fit, then 3 refits on block-bootstrapped fit rows, 100 chunks, the same weights for every configuration) give F1 28.93, 31.10, 35.26, 37.91 and PR-AUC 0.159, 0.146, 0.200, 0.199. The [95%] column covers only the validation sample's noise.
- Paired ΔF1 keeps one sign over all draws for Q10 (-14.11, -20.19, -17.19, -28.62; ΔPR-AUC -0.072, -0.092, -0.131, -0.163); all (-20.70, -8.75, -15.58, -31.60; ΔPR-AUC -0.130, -0.047, -0.102, -0.169); v1_e16 (-16.35, -26.09, -9.81, -32.97; ΔPR-AUC -0.110, -0.125, -0.069, -0.178). Q1, Q3, Q7 give fits identical to clean's in every draw.
- Configurations that change no block row on the fit or validation rows: Q1, Q3, Q7, Q8 (Q7 drops sg_ord and counts as unchanged when its other blocks are).
- The fit and validation rows hold 23 self-loops, 11 of them laundering.
- 564 of the 1,406 fit and validation positives that share their timestamp sit in the first tenth of its rows (mean relative rank 0.578 on fit and 0.577 on validation; negatives 0.500 and 0.500).
- u_out's time mean: 99.7% of validation rows lie above its fit maximum under Q10 (absolute timestamps), 99.7% under v1_e16 and 0.0% under clean's ages.
- No test label is read. Every run mines [0, 2,423,336), labels are read for those rows only, and only validation rows are scored.
- R0's settings (`configs/xgb_default.yaml`), not a tuned recipe: only the paired differences are read.
- Count blocks: instances = rows with a count of at least 1 (one instance per trigger, R7). The ≥ 2 share is what v1's histogram marks active.
- Timings are informal (P13); the load before each run is logged with it.

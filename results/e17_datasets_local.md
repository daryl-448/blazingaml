# E17: datasets, M and memory per backend (SPEC v3 §8)

`build/rel-cuda/blz measure --reps 5 --triggers all`, cpu and hybrid at 12 threads, default options, chunk 2^20 (D-81); commits 7e419b5. M is §5.1's: median of the reps, min and max kept. Estimates are bench/p15_preflight.py's models.

## Prepare invariants

| Dataset | Descriptor hash in .blzg | Recounted (no test label) | Gated by prepare only | Prepare peak | Prepare time |
|---|---|---|---|---:|---:|
| HI-Small | equal | 8 of 8 equal | concatenated_keys, positives, test_positives | 303 MiB | 5.3 s |
| TransXion | equal | 8 of 8 equal | positives, test_positives, unmatched_node_rows, unresolved_accounts | 145 MiB | 2.2 s |
| LI-Small | equal | 8 of 8 equal | concatenated_keys, positives, test_positives | 403 MiB | 7.5 s |
| HI-Medium | equal | 8 of 8 equal | concatenated_keys, positives, test_positives | 1643 MiB | 37.7 s |

Prepare peaks and times are D-68's, recorded and not rerun.

## M per cell

| Dataset | Profile | Backend | M median | min | max | spread | Errors | Peak RSS | Index / rows / rest | Device peak (graph, scratch+out) | Foreign load mean / max | Idle at start |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
| HI-Small | clean_default | cpu | 1.498 s | 1.484 | 1.563 | 5.2 % | 0 | 422 MiB | 223 / 40 / 65 | — | 42 / 60 % | yes (4 % busy) |
| HI-Small | clean_default | cuda | 1.136 s | 1.130 | 1.200 | 6.2 % | 0 | 550 MiB | 223 / 40 / 115 | 398 MiB (281, 53) | 31 / 50 % | yes (2 % busy) |
| HI-Small | clean_default | hybrid | 1.356 s | 1.333 | 1.365 | 2.4 % | 0 | 588 MiB | 223 / 40 / 153 | 442 MiB (281, 97) | 23 / 32 % | yes (1 % busy) |
| HI-Small | clean_wc_hi_small | cpu | 3.709 s | 3.664 | 3.755 | 2.4 % | 0 | 423 MiB | 223 / 40 / 65 | — | 28 / 46 % | yes (2 % busy) |
| HI-Small | clean_wc_hi_small | cuda | 2.087 s | 2.073 | 2.093 | 0.9 % | 0 | 552 MiB | 223 / 40 / 117 | 398 MiB (281, 53) | 17 / 25 % | yes (2 % busy) |
| HI-Small | clean_wc_hi_small | hybrid | 2.344 s | 2.339 | 2.366 | 1.1 % | 0 | 607 MiB | 223 / 40 / 172 | 442 MiB (281, 97) | 20 / 26 % | yes (2 % busy) |
| TransXion | clean_default | cpu | 0.450 s | 0.445 | 0.469 | 5.2 % | 0 | 289 MiB | 167 / 40 / 26 | — | 25 / 25 % | yes (2 % busy) |
| TransXion | clean_default | cuda | 0.494 s | 0.481 | 0.503 | 4.4 % | 0 | 451 MiB | 167 / 40 / 111 | 318 MiB (202, 52) | 14 / 15 % | yes (3 % busy) |
| TransXion | clean_default | hybrid | 0.627 s | 0.620 | 0.638 | 2.9 % | 0 | 466 MiB | 167 / 40 / 125 | 342 MiB (202, 76) | 20 / 27 % | yes (2 % busy) |
| TransXion | clean_wc_hi_small | cpu | 0.945 s | 0.932 | 0.956 | 2.6 % | 0 | 289 MiB | 167 / 40 / 26 | — | 25 / 35 % | yes (1 % busy) |
| TransXion | clean_wc_hi_small | cuda | 0.835 s | 0.834 | 0.866 | 3.8 % | 0 | 452 MiB | 167 / 40 / 112 | 318 MiB (202, 52) | 21 / 22 % | yes (3 % busy) |
| TransXion | clean_wc_hi_small | hybrid | 1.056 s | 1.040 | 1.104 | 6.1 % | 0 | 503 MiB | 167 / 40 / 162 | 342 MiB (202, 76) | 28 / 31 % | yes (2 % busy) |
| LI-Small | clean_default | cpu | 2.498 s | 2.470 | 2.532 | 2.5 % | 0 | 560 MiB | 304 / 40 / 90 | — | 27 / 36 % | yes (1 % busy) |
| LI-Small | clean_default | cuda | 1.738 s | 1.733 | 1.748 | 0.9 % | 0 | 666 MiB | 304 / 40 / 119 | 500 MiB (384, 52) | 23 / 37 % | yes (2 % busy) |
| LI-Small | clean_default | hybrid | 2.107 s | 2.073 | 2.119 | 2.2 % | 0 | 717 MiB | 304 / 40 / 170 | 564 MiB (384, 116) | 22 / 34 % | yes (2 % busy) |
| LI-Small | clean_wc_li_small | cpu | 6.017 s | 5.989 | 6.051 | 1.0 % | 0 | 560 MiB | 304 / 40 / 90 | — | 27 / 50 % | yes (2 % busy) |
| LI-Small | clean_wc_li_small | cuda | 3.009 s | 2.991 | 3.020 | 1.0 % | 0 | 667 MiB | 304 / 40 / 120 | 500 MiB (384, 52) | 18 / 25 % | yes (2 % busy) |
| LI-Small | clean_wc_li_small | hybrid | 3.661 s | 3.645 | 3.687 | 1.2 % | 0 | 743 MiB | 304 / 40 / 196 | 564 MiB (384, 116) | 21 / 30 % | yes (2 % busy) |
| HI-Medium | clean_default | cpu | 35.023 s | 34.251 | 35.240 | 2.8 % | 0 | 2297 MiB | 1420 / 122 / 234 | — | 26 / 98 % | yes (2 % busy) |
| HI-Medium | clean_default | cuda | 14.338 s | 14.237 | 14.610 | 2.6 % | 0 | 2199 MiB | 1420 / 40 / 134 | 1902 MiB (1785, 107) | 21 / 73 % | yes (2 % busy) |
| HI-Medium | clean_default | hybrid | 23.379 s | 22.988 | 23.613 | 2.7 % | 0 | 2451 MiB | 1420 / 122 / 304 | 2284 MiB (1785, 489) | 23 / 60 % | yes (2 % busy) |
| HI-Medium | clean_wc_hi_medium | cpu | 124.759 s | 123.783 | 124.881 | 0.9 % | 0 | 2305 MiB | 1420 / 122 / 235 | — | 24 / 78 % | yes (2 % busy) |
| HI-Medium | clean_wc_hi_medium | cuda | 48.973 s | 48.897 | 49.134 | 0.5 % | 0 | 2207 MiB | 1420 / 40 / 142 | 1902 MiB (1785, 107) | 12 / 119 % | yes (2 % busy) |
| HI-Medium | clean_wc_hi_medium | hybrid | 78.093 s | 77.484 | 78.632 | 1.5 % | 0 | 2496 MiB | 1420 / 122 / 368 | 2284 MiB (1785, 489) | 5 / 46 % | yes (0 % busy) |

Row hashes per block agree across backends: hi_small/clean_default yes, hi_small/clean_wc_hi_small yes, transxion/clean_default yes, transxion/clean_wc_hi_small yes, li_small/clean_default yes, li_small/clean_wc_li_small yes, hi_medium/clean_default yes, hi_medium/clean_wc_hi_medium yes.

## §6.2 budgets: estimate against measured

| Dataset | Profile | Backend | Host estimate (index / rows / rest / peak) | Host measured peak | Device estimate | Device measured |
|---|---|---|---|---:|---:|---:|
| HI-Small | clean_default | cpu | 223 / 40 / 83 / 426 MiB | 422 | — | — |
| HI-Small | clean_default | cuda | 223 / 40 / 96 / 535 MiB | 550 | 398 | 398 MiB |
| HI-Small | clean_default | hybrid | 223 / 40 / 120 / 560 MiB | 588 | 442 | 442 MiB |
| HI-Small | clean_wc_hi_small | cpu | 223 / 40 / 83 / 426 MiB | 423 | — | — |
| HI-Small | clean_wc_hi_small | cuda | 223 / 40 / 96 / 535 MiB | 552 | 398 | 398 MiB |
| HI-Small | clean_wc_hi_small | hybrid | 223 / 40 / 120 / 560 MiB | 607 | 442 | 442 MiB |
| TransXion | clean_default | cpu | 167 / 40 / 31 / 295 MiB | 289 | — | — |
| TransXion | clean_default | cuda | 167 / 40 / 44 / 404 MiB | 451 | 319 | 318 MiB |
| TransXion | clean_default | hybrid | 167 / 40 / 68 / 429 MiB | 466 | 363 | 342 MiB |
| TransXion | clean_wc_hi_small | cpu | 167 / 40 / 31 / 295 MiB | 289 | — | — |
| TransXion | clean_wc_hi_small | cuda | 167 / 40 / 44 / 404 MiB | 452 | 319 | 318 MiB |
| TransXion | clean_wc_hi_small | hybrid | 167 / 40 / 68 / 429 MiB | 503 | 363 | 342 MiB |
| LI-Small | clean_default | cpu | 304 / 40 / 104 / 549 MiB | 560 | — | — |
| LI-Small | clean_default | cuda | 304 / 40 / 117 / 659 MiB | 666 | 501 | 500 MiB |
| LI-Small | clean_default | hybrid | 304 / 40 / 141 / 684 MiB | 717 | 545 | 564 MiB |
| LI-Small | clean_wc_li_small | cpu | 304 / 40 / 104 / 549 MiB | 560 | — | — |
| LI-Small | clean_wc_li_small | cuda | 304 / 40 / 117 / 659 MiB | 667 | 501 | 500 MiB |
| LI-Small | clean_wc_li_small | hybrid | 304 / 40 / 141 / 684 MiB | 743 | 545 | 564 MiB |
| HI-Medium | clean_default | cpu | 1420 / 122 / 255 / 2184 MiB | 2297 | — | — |
| HI-Medium | clean_default | cuda | 1420 / 40 / 268 / 2212 MiB | 2199 | 1902 | 1902 MiB |
| HI-Medium | clean_default | hybrid | 1420 / 122 / 292 / 2318 MiB | 2451 | 1946 | 2284 MiB |
| HI-Medium | clean_wc_hi_medium | cpu | 1420 / 122 / 255 / 2184 MiB | 2305 | — | — |
| HI-Medium | clean_wc_hi_medium | cuda | 1420 / 40 / 268 / 2212 MiB | 2207 | 1902 | 1902 MiB |
| HI-Medium | clean_wc_hi_medium | hybrid | 1420 / 122 / 292 / 2318 MiB | 2496 | 1946 | 2284 MiB |

## Training (estimate, never run here)

| Dataset | Recipe | Columns | Fit rows | Validation rows | Peak RSS | Fit time |
|---|---|---:|---:|---:|---:|---:|
| HI-Small | clean_default | 40 | 3,047,216 | 1,015,460 | 1,058 MiB | 19 s |
| HI-Small | clean_all | 61 | 3,047,216 | 1,015,460 | 1,256 MiB | 29 s |
| TransXion | clean_default | 40 | 1,817,503 | 605,833 | 873 MiB | 11 s |
| TransXion | clean_all | 61 | 1,817,503 | 605,833 | 991 MiB | 17 s |
| LI-Small | clean_default | 40 | 4,154,859 | 1,384,590 | 1,224 MiB | 26 s |
| LI-Small | clean_all | 61 | 4,154,859 | 1,384,590 | 1,493 MiB | 40 s |
| HI-Medium | clean_default | 40 | 19,139,453 | 6,379,894 | 3,460 MiB | 120 s |
| HI-Medium | clean_all | 61 | 19,139,453 | 6,379,894 | 4,700 MiB | 183 s |

Inferred from the preflight's HI-Small fits (bytes per cell, seconds per fit cell).

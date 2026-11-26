# Clean baseline M, frozen (SPEC v3 §5.2, P15)

Harness commit 7e419b5, chunk 1,048,576 triggers, `blz measure --reps 5`, 12 threads on cpu. Frozen 2026-11-27T01:16:44+0800 from `results/e17_datasets_local.json`; never overwritten.

| Workload | Role | M median | min | max | Triggers | Peak RSS |
|---|---|---:|---:|---:|---:|---:|
| hi_medium/clean_default/cpu | target (primary): the stop predicate is measured here | 35.023 s | 34.251 | 35.240 | 31,898,238 | 2297 MiB |
| hi_medium/clean_wc_hi_medium/cpu | long-window target (primary) | 124.759 s | 123.783 | 124.881 | 31,898,238 | 2305 MiB |
| hi_small/clean_default/cpu | guard: development loop | 1.498 s | 1.484 | 1.563 | 5,078,345 | 422 MiB |
| hi_small/clean_wc_hi_small/cpu | guard: development loop | 3.709 s | 3.664 | 3.755 | 5,078,345 | 423 MiB |
| transxion/clean_default/cpu | guard: a different shape | 0.450 s | 0.445 | 0.469 | 3,029,170 | 289 MiB |
| hi_small/clean_default/cuda | guard: 1650S | 1.136 s | 1.130 | 1.200 | 5,078,345 | 550 MiB |
| hi_small/clean_wc_hi_small/cuda | guard: 1650S | 2.087 s | 2.073 | 2.093 | 5,078,345 | 552 MiB |
| hi_medium/clean_default/cuda | guard: 1650S, HI-Medium in trigger-range chunks | 14.338 s | 14.237 | 14.610 | 31,898,238 | 2199 MiB |
| hi_medium/clean_wc_hi_medium/cuda | guard: 1650S, HI-Medium in trigger-range chunks | 48.973 s | 48.897 | 49.134 | 31,898,238 | 2207 MiB |

M = median of the reps (§5.1). A change is kept only when its median beats the incumbent's by more than the incumbent's min-to-max spread; a workload regresses when its median exceeds this baseline's max by more than 5 %; a candidate that cannot complete a workload this baseline completed has regressed on it.

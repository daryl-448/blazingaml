# E4: CPU mining time (a30)

- CPU: AMD EPYC 7542 32-Core Processor; OMP_PROC_BIND=close, OMP_PLACES=threads
- Build: Release, `-O3 -DNDEBUG -march=native`, 12.3.0
- Git: a716c853eb76 (dirty); config configs/features_default.yaml
- Index build (shared by all blocks, not in the times below): 0.45 s, 268 MB; peak RSS 852 MB
- Each cell: median of 5 timed runs after 1 warm-up, over every trigger; the output is identical in every run (SPEC §8)

## Mining time (s), schedule `dynamic`

| block | 1 thr | 2 thr | 4 thr | 8 thr | 12 thr | 16 thr | 1 → max |
|---|---:|---:|---:|---:|---:|---:|---:|
| fan_in | 0.659 | 0.329 | 0.167 | 0.097 | 0.071 | 0.057 | 11.6× |
| fan_out | 13.4 | 6.692 | 3.344 | 1.685 | 1.128 | 0.851 | 15.8× |
| deg_in | 0.606 | 0.308 | 0.158 | 0.096 | 0.069 | 0.055 | 11.0× |
| deg_out | 0.889 | 0.450 | 0.228 | 0.127 | 0.090 | 0.074 | 12.0× |
| sg | 2.753 | 1.379 | 0.700 | 0.378 | 0.264 | 0.207 | 13.3× |
| tcyc | 4.002 | 2.000 | 1.029 | 0.517 | 0.343 | 0.260 | 15.4× |
| **all** | 22.3 | 11.2 | 5.626 | 2.901 | 1.965 | 1.504 | 14.8× |

## Mining time (s), schedule `sorted`

| block | 1 thr | 2 thr | 4 thr | 8 thr | 12 thr | 16 thr | 1 → max |
|---|---:|---:|---:|---:|---:|---:|---:|
| fan_in | 1.328 | 0.706 | 0.379 | 0.213 | 0.154 | 0.125 | 10.6× |
| fan_out | 13.2 | 6.667 | 3.442 | 1.800 | 1.215 | 0.919 | 14.4× |
| deg_in | 1.400 | 0.706 | 0.384 | 0.214 | 0.154 | 0.125 | 11.2× |
| deg_out | 1.636 | 0.872 | 0.470 | 0.257 | 0.182 | 0.146 | 11.2× |
| sg | 3.118 | 1.612 | 0.863 | 0.501 | 0.364 | 0.296 | 10.5× |
| tcyc | 3.324 | 1.723 | 0.908 | 0.487 | 0.342 | 0.268 | 12.4× |
| **all** | 24.0 | 12.3 | 6.445 | 3.473 | 2.412 | 1.878 | 12.8× |

`sorted` includes its schedule cost (work estimates plus a sort of every trigger); loop-only medians are in the JSON (`loop_median`).

## GFP (snapml 1.17.2), transform() only, fresh instance per run

| pattern | threads | batch | median (s) | runs |
|---|---:|---:|---:|---:|
| fan | 16 | 2048 | 77.9 | 3 |
| fan | 16 | 65536 | 84.1 | 3 |
| degree | 16 | 2048 | 43.8 | 3 |
| degree | 16 | 65536 | 58.9 | 3 |
| scatter-gather | 16 | 2048 | 36.1 | 3 |
| scatter-gather | 16 | 65536 | 32.1 | 3 |
| temp-cycle | 16 | 2048 | 23.7 | 3 |
| temp-cycle | 16 | 65536 | 23.7 | 3 |

## Speedup over GFP best (`dynamic` schedule)

T_GFP_best = the fastest batch size at GFP's highest thread count. speedup = T_GFP_best / T_BlazingAML.

| GFP pattern | blocks | T_GFP_best (s) | 1 thr | 2 thr | 4 thr | 8 thr | 12 thr | 16 thr |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fan | fan_in, fan_out | 77.9 (batch 2048) | 6× | 11× | 22× | 44× | 65× | 86× |
| degree | deg_in, deg_out | 43.8 (batch 2048) | 29× | 58× | 114× | 196× | 275× | 339× |
| scatter-gather | sg | 32.1 (batch 65536) | 12× | 23× | 46× | 85× | 122× | 155× |
| temp-cycle | tcyc | 23.7 (batch 65536) | 6× | 12× | 23× | 46× | 69× | 91× |

Caveat (SPEC §11.2): GFP batch mode credits every edge of an instance and lets edges see later edges of the same batch (SPEC Appendix A, A28): the timings compare equal work, not identical outputs (SPEC §11.2).

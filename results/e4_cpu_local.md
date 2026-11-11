# E4: CPU mining time (local)

- CPU: AMD Ryzen 5 3600 6-Core Processor; OMP_PROC_BIND=close, OMP_PLACES=threads
- Build: Release, `-O3 -DNDEBUG -march=native`, 16.2.1 20260810
- Git: 458669bc70b6 (dirty); config configs/features_default.yaml
- Index build (shared by all blocks, not in the times below): 0.34 s, 268 MB; peak RSS 837 MB
- Each cell: median of 5 timed runs after 1 warm-up, over every trigger; the output is identical in every run (SPEC §8)

## Mining time (s), schedule `dynamic`

| block | 1 thr | 2 thr | 4 thr | 6 thr | 8 thr | 12 thr | 1 → max |
|---|---:|---:|---:|---:|---:|---:|---:|
| fan_in | 0.447 | 0.317 | 0.169 | 0.131 | 0.109 | 0.085 | 5.3× |
| fan_out | 10.7 | 6.989 | 3.533 | 2.421 | 2.049 | 1.697 | 6.3× |
| deg_in | 0.478 | 0.314 | 0.168 | 0.120 | 0.107 | 0.100 | 4.8× |
| deg_out | 0.694 | 0.508 | 0.263 | 0.179 | 0.146 | 0.108 | 6.4× |
| sg | 1.981 | 1.353 | 0.701 | 0.485 | 0.395 | 0.290 | 6.8× |
| tcyc | 2.901 | 2.021 | 1.063 | 0.744 | 0.562 | 0.394 | 7.4× |
| **all** | 17.2 | 11.5 | 5.897 | 4.079 | 3.367 | 2.674 | 6.4× |

## Mining time (s), schedule `sorted`

| block | 1 thr | 2 thr | 4 thr | 6 thr | 8 thr | 12 thr | 1 → max |
|---|---:|---:|---:|---:|---:|---:|---:|
| fan_in | 0.935 | 0.718 | 0.413 | 0.312 | 0.266 | 0.213 | 4.4× |
| fan_out | 11.7 | 7.594 | 3.849 | 2.733 | 2.050 | 1.671 | 7.0× |
| deg_in | 0.961 | 0.750 | 0.433 | 0.323 | 0.278 | 0.268 | 3.6× |
| deg_out | 1.236 | 0.941 | 0.532 | 0.395 | 0.314 | 0.330 | 3.7× |
| sg | 2.295 | 1.624 | 0.887 | 0.647 | 0.546 | 0.422 | 5.4× |
| tcyc | 2.502 | 1.706 | 0.910 | 0.647 | 0.518 | 0.383 | 6.5× |
| **all** | 19.7 | 13.3 | 7.024 | 5.056 | 3.972 | 3.287 | 6.0× |

`sorted` includes its schedule cost (work estimates plus a sort of every trigger); loop-only medians are in the JSON (`loop_median`).

## GFP (snapml 1.17.2), transform() only, fresh instance per run

| pattern | threads | batch | median (s) | runs |
|---|---:|---:|---:|---:|
| fan | 12 | 2048 | 141.1 | 3 |
| fan | 12 | 65536 | 98.9 | 3 |
| fan | 6 | 65536 | 135.7 | 1 |
| fan | 4 | 65536 | 178.8 | 1 |
| fan | 2 | 65536 | 302.8 | 1 |
| fan | 1 | 65536 | 642.7 | 1 |
| degree | 12 | 2048 | 85.0 | 3 |
| degree | 12 | 65536 | 296.2 | 3 |
| degree | 6 | 65536 | 341.7 | 1 |
| degree | 4 | 65536 | 120.2 | 1 |
| degree | 2 | 65536 | 192.8 | 1 |
| degree | 1 | 65536 | 305.7 | 1 |
| scatter-gather | 12 | 2048 | 59.6 | 3 |
| scatter-gather | 12 | 65536 | 31.7 | 3 |
| scatter-gather | 6 | 65536 | 33.4 | 1 |
| scatter-gather | 4 | 65536 | 36.6 | 1 |
| scatter-gather | 2 | 65536 | 57.0 | 1 |
| scatter-gather | 1 | 65536 | 106.3 | 1 |
| temp-cycle | 12 | 2048 | 32.6 | 3 |
| temp-cycle | 12 | 65536 | 18.0 | 3 |
| temp-cycle | 6 | 65536 | 18.4 | 1 |
| temp-cycle | 4 | 65536 | 19.6 | 1 |
| temp-cycle | 2 | 65536 | 22.8 | 1 |
| temp-cycle | 1 | 65536 | 31.3 | 1 |

## Speedup over GFP best (`dynamic` schedule)

T_GFP_best = the fastest batch size at GFP's highest thread count. speedup = T_GFP_best / T_BlazingAML.

| GFP pattern | blocks | T_GFP_best (s) | 1 thr | 2 thr | 4 thr | 6 thr | 8 thr | 12 thr |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fan | fan_in, fan_out | 98.9 (batch 65536) | 9× | 14× | 27× | 39× | 46× | 56× |
| degree | deg_in, deg_out | 85.0 (batch 2048) | 73× | 103× | 197× | 285× | 336× | 409× |
| scatter-gather | sg | 31.7 (batch 65536) | 16× | 23× | 45× | 65× | 80× | 109× |
| temp-cycle | tcyc | 18.0 (batch 65536) | 6× | 9× | 17× | 24× | 32× | 46× |

Caveat (SPEC §11.2): GFP batch mode credits every edge of an instance and lets edges see later edges of the same batch (SPEC Appendix A, A28): the timings compare equal work, not identical outputs (SPEC §11.2).

## Measurement notes

- `local` is a shared desktop. At 12 threads, fan_out measured 1.21–1.70 s across separate runs on the same build. This table keeps the 1.70 s value from the full sweep, the conservative figure. The cores held about 3.8 GHz under load.
- GFP degree is bimodal on an idle machine: each traversal takes about 85 s or about 300 s (see `seconds` per point). The degree speedup depends on which mode the `T_GFP_best` runs hit.
- GFP temp-cycle: the grid's temp-cycle points overlapped an unrelated workload (two single-threaded jobs on 2 of 12 hardware threads). One idle-machine run before the grid took 15.7 s at 12 threads with batch 65536, against 18.0 s here, which gives a 12-thread speedup of about 40× rather than 46×. A re-run was also disturbed, so it is not used.

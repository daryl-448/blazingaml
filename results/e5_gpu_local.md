# E5: GPU mining time (local)

- GPU: NVIDIA GeForce GTX 1650 SUPER (sm_75, 20 SMs, 3728 MB); driver 13040, runtime 13040
- Host CPU: AMD Ryzen 5 3600 6-Core Processor
- Build: Release, `-O3 -DNDEBUG -march=native`, 16.2.1 20260810; CUDA: nvcc 13.4.92, sm 75,80, -O3 -DNDEBUG
- Git: a716c853eb76 (dirty); config configs/features_default.yaml
- Index build on the host (shared by all blocks, not in the times below): 0.40 s; device copy 329 MB, uploaded from pageable memory in 33.9 ms (median of 5)
- Each cell: median of 5 timed runs after 1 warm-up, over every trigger. Every run at every threshold produced the same rows (SPEC §8).

Output check: the rows of every block equal the CPU backend's (FNV-1a of the whole block, `e4_cpu_local.json`), which equal the oracle's (AC1).

## Kernel time (ms) by light/heavy threshold H

Kernel time = output zeroing + bucketing (work estimates, partition) + light kernel + heavy kernel, from cudaEvents. A trigger is heavy when its work estimate (the first scan's window) is ≥ H; H = 0 puts every trigger on the heavy kernel (warp per trigger), H = all every trigger on the light kernel (thread per trigger). Bold: the block's fastest H.

| block | H=0 | H=8 | H=16 | H=32 | H=64 | H=128 | H=256 | H=1024 | H=all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fan_in | 40.0 | 60.7 | 59.3 | 59.0 | 58.4 | 58.2 | 58.4 | 58.1 | **33.3** |
| fan_out | 169.3 | 154.7 | 133.8 | 120.5 | 119.3 | 119.9 | 119.8 | **110.9** | 203.5 |
| deg_in | 38.3 | 55.6 | 54.2 | 53.8 | 53.6 | 54.4 | 55.0 | 52.6 | **29.5** |
| deg_out | 71.3 | 72.1 | 66.4 | 61.7 | 60.8 | 62.0 | 62.4 | **56.0** | 113.9 |
| sg | 132.8 | **109.7** | 112.7 | 114.5 | 122.2 | 142.8 | 161.0 | 166.8 | 161.3 |
| tcyc | 236.9 | 226.8 | 221.9 | 216.7 | 214.9 | **213.7** | 215.8 | 214.4 | 245.5 |
| **all** | 688.8 | 679.6 | 648.2 | **626.1** | 629.3 | 651.1 | 672.4 | 658.8 | 787.0 |

Tuned threshold for this GPU (smallest total): **H = 32**. The SPEC default is 32.

## Breakdown at H = 32

| block | patterns | light triggers | heavy triggers | bucket (ms) | light (ms) | heavy (ms) | kernels (ms) | D2H (ms) | T = upload + kernels + D2H (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fan_in | fan_in | 5,065,801 | 12,544 | 23.3 | 33.7 | 0.1 | 59.0 | 18.3 | 111.3 |
| fan_out | fan_out | 4,623,183 | 455,162 | 14.3 | 34.0 | 70.7 | 120.5 | 18.4 | 172.8 |
| deg_in | deg_in | 5,065,801 | 12,544 | 23.3 | 28.7 | 0.1 | 53.8 | 18.5 | 106.2 |
| deg_out | deg_out | 4,623,183 | 455,162 | 14.7 | 28.3 | 16.9 | 61.7 | 18.5 | 114.1 |
| sg | sg_gather, sg_scatter | 10,075,552 | 81,138 | 22.7 | 57.8 | 32.7 | 114.5 | 15.4 | 163.9 |
| tcyc | tcyc | 5,074,856 | 3,489 | 15.6 | 51.3 | 148.8 | 216.7 | 13.9 | 264.5 |
| **all** | | | | | | | 626.1 | 103.0 | 763.1 (one upload) |

Trigger counts are summed over a block's patterns (sg runs sg_gather and sg_scatter over the same triggers). Bucketing includes reading the heavy count back to the host.

## Speedup over GFP best (same machine)

T_GFP_best = GFP's fastest batch size at its highest thread count, on this host's CPU (`e4_cpu_local.json`). T_BlazingAML = median graph upload + the group's kernels + its D2H at H = 32 (SPEC §11.2). speedup = T_GFP_best / T_BlazingAML.

| GFP pattern | blocks | T_GFP_best (s) | T_BlazingAML GPU (s) | speedup | CPU backend, 12 thr (s) | GPU vs CPU |
|---|---|---:|---:|---:|---:|---:|
| fan | fan_in, fan_out | 98.9 (batch 65536, 12 thr) | 0.250 | 395× | 1.782 | 7.1× |
| degree | deg_in, deg_out | 85.0 (batch 2048, 12 thr) | 0.186 | 456× | 0.208 | 1.1× |
| scatter-gather | sg | 31.7 (batch 65536, 12 thr) | 0.164 | 193× | 0.290 | 1.8× |
| temp-cycle | tcyc | 18.0 (batch 65536, 12 thr) | 0.265 | 68× | 0.394 | 1.5× |

Caveat (SPEC §11.2): GFP batch mode credits every edge of an instance and lets edges see later edges of the same batch (SPEC Appendix A, A28): the timings compare equal work, not identical outputs (SPEC §11.2).

## Measurement notes

- The 1650S run was taken with only the desktop and a browser on the machine (no other process above 10% CPU, no other GPU compute process), logged before and after the run; the load average was still decaying from a build that had just finished.
- H = 32 (626.1 ms) and H = 64 (629.3 ms) differ by 0.5%, within run-to-run noise: the tuned value is the SPEC default, by a tie. Per block the best thresholds differ (fan_in and deg_in: every trigger light, because their bucketing costs about 23 ms and almost no trigger is heavy; fan_out and deg_out: 1024; sg: 8; tcyc: 128); one threshold per GPU is what §7.5 asks for.
- tcyc's time is dominated by a few thousand heavy triggers (about 149 ms for 3,489 triggers at H = 32): a warp splits only the first hop, and the deeper search below each hop runs on one lane. The P6 hybrid (heavy tcyc triggers on the CPU) targets this.
- T_GFP_best comes from E4 (results/e4_cpu_local.json). Its temp-cycle value (18.0 s) overlapped an unrelated CPU workload; the one idle-machine run took 15.7 s, which would give about 60× instead of 68×.

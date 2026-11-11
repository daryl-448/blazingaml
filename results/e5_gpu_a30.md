# E5: GPU mining time (a30)

- GPU: NVIDIA A30 (sm_80, 56 SMs, 24163 MB); driver 13000, runtime 12060
- Host CPU: AMD EPYC 7542 32-Core Processor
- Build: Release, `-O3 -DNDEBUG -march=native`, 12.3.0; CUDA: nvcc 12.6.77, sm 80, -O3 -DNDEBUG
- Git: a716c853eb76 (dirty); config configs/features_default.yaml
- Index build on the host (shared by all blocks, not in the times below): 0.43 s; device copy 329 MB, uploaded from pageable memory in 29.1 ms (median of 5)
- Each cell: median of 5 timed runs after 1 warm-up, over every trigger. Every run at every threshold produced the same rows (SPEC §8).

Output check: the rows of every block equal the CPU backend's (FNV-1a of the whole block, `e4_cpu_a30.json`), which equal the oracle's (AC1).

## Kernel time (ms) by light/heavy threshold H

Kernel time = output zeroing + bucketing (work estimates, partition) + light kernel + heavy kernel, from cudaEvents. A trigger is heavy when its work estimate (the first scan's window) is ≥ H; H = 0 puts every trigger on the heavy kernel (warp per trigger), H = all every trigger on the light kernel (thread per trigger). Bold: the block's fastest H.

| block | H=0 | H=8 | H=16 | H=32 | H=64 | H=128 | H=256 | H=1024 | H=all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fan_in | 10.8 | 4.0 | 3.5 | 3.4 | 3.4 | 3.4 | 3.4 | 3.4 | **2.3** |
| fan_out | 50.4 | 30.8 | 21.8 | 19.2 | 19.1 | **19.1** | 19.1 | 19.2 | 78.4 |
| deg_in | 10.5 | 3.7 | 3.2 | 3.2 | 3.1 | 3.1 | 3.1 | 3.1 | **2.0** |
| deg_out | 22.5 | 11.6 | 8.9 | 8.1 | **8.1** | 8.1 | 8.1 | 8.3 | 44.5 |
| sg | 35.1 | **12.3** | 12.6 | 12.9 | 13.1 | 16.4 | 18.7 | 20.4 | 36.0 |
| tcyc | 77.5 | 56.9 | 48.9 | 37.9 | **37.5** | 37.5 | 37.7 | 38.7 | 91.7 |
| **all** | 206.8 | 119.4 | 98.8 | 84.7 | **84.4** | 87.7 | 90.2 | 93.2 | 255.0 |

Tuned threshold for this GPU (smallest total): **H = 64**. The SPEC default is 32.

## Breakdown at H = 64

| block | patterns | light triggers | heavy triggers | bucket (ms) | light (ms) | heavy (ms) | kernels (ms) | D2H (ms) | T = upload + kernels + D2H (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fan_in | fan_in | 5,075,824 | 2,521 | 1.1 | 2.0 | 0.0 | 3.4 | 10.1 | 42.7 |
| fan_out | fan_out | 4,629,431 | 448,914 | 1.0 | 2.0 | 15.8 | 19.1 | 10.3 | 58.5 |
| deg_in | deg_in | 5,075,824 | 2,521 | 1.1 | 1.7 | 0.0 | 3.1 | 10.2 | 42.5 |
| deg_out | deg_out | 4,629,431 | 448,914 | 1.0 | 1.6 | 5.1 | 8.1 | 10.4 | 47.7 |
| sg | sg_gather, sg_scatter | 10,082,609 | 74,081 | 1.5 | 7.5 | 3.9 | 13.1 | 8.4 | 50.6 |
| tcyc | tcyc | 5,075,460 | 2,885 | 0.8 | 7.4 | 29.0 | 37.5 | 8.0 | 74.5 |
| **all** | | | | | | | 84.4 | 57.5 | 171.0 (one upload) |

Trigger counts are summed over a block's patterns (sg runs sg_gather and sg_scatter over the same triggers). Bucketing includes reading the heavy count back to the host.

## Speedup over GFP best (same machine)

T_GFP_best = GFP's fastest batch size at its highest thread count, on this host's CPU (`e4_cpu_a30.json`). T_BlazingAML = median graph upload + the group's kernels + its D2H at H = 64 (SPEC §11.2). speedup = T_GFP_best / T_BlazingAML.

| GFP pattern | blocks | T_GFP_best (s) | T_BlazingAML GPU (s) | speedup | CPU backend, 16 thr (s) | GPU vs CPU |
|---|---|---:|---:|---:|---:|---:|
| fan | fan_in, fan_out | 77.9 (batch 2048, 16 thr) | 0.072 | 1081× | 0.907 | 12.6× |
| degree | deg_in, deg_out | 43.8 (batch 2048, 16 thr) | 0.061 | 717× | 0.129 | 2.1× |
| scatter-gather | sg | 32.1 (batch 65536, 16 thr) | 0.051 | 633× | 0.207 | 4.1× |
| temp-cycle | tcyc | 23.7 (batch 65536, 16 thr) | 0.075 | 317× | 0.260 | 3.5× |

Caveat (SPEC §11.2): GFP batch mode credits every edge of an instance and lets edges see later edges of the same batch (SPEC Appendix A, A28): the timings compare equal work, not identical outputs (SPEC §11.2).

## Measurement notes

- JarvisLabs A30 container (IN2): 16 vCPUs of an AMD EPYC 7542, Ubuntu 22.04, driver 580.126.20; nvcc 12.6 came with the image, g++-12 was installed by bench/a30_remote.sh (results/a30_env.txt, results/a30_run.log). It is a shared cloud host: other tenants may load its CPU, which affects the host-side numbers (E4, GFP) more than the cudaEvent kernel times.
- H = 64 (84.4 ms) and H = 32 (84.7 ms) differ by 0.4%, a tie. Per block: fan_in and deg_in fastest with every trigger light, fan_out 128, deg_out and tcyc 64, sg 8.
- At this GPU's speed the transfers cost as much as the kernels: 57.5 ms of D2H (about 10 ms per block of 5.08M rows) plus a 29 ms graph upload, against 84.4 ms of kernels, all inside T_BlazingAML.
- tcyc is still dominated by its heavy kernel (29 ms for 2,885 triggers at H = 64), as on the 1650S; the P6 hybrid targets this.
- T_GFP_best is GFP at 16 threads on the same host (D-16: nproc threads only, both batch sizes, 3 runs each).

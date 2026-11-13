# E12: GPU mining time by profile, and GPU vs CPU (a30)

- GPU: NVIDIA A30 (sm_80, 56 SMs, 24163 MB); driver 13000, runtime 12060
- Host CPU: AMD EPYC 7542 32-Core Processor
- Build: Release, `-O3 -DNDEBUG -march=native`, 12.3.0; CUDA: nvcc 12.6.77, sm 80, -O3 -DNDEBUG
- Git: 7cbae022da79 (dirty)
- Each cell: median of 5 timed runs after 1 warm-up, over every trigger of HI-Small. Every run at every H produced the same rows (SPEC §8).

## AC-V2-4 on sm_80

| profile | fuzz graphs | H | mismatching rows | seconds | HI-Small: CUDA hashes = oracle |
|---|---:|---|---:|---:|---|
| sg_ordered_tsvis | 10,000 | 0, 4, all | 0 | 84 | yes |
| sg_ordered_v1w | 10,000 | 0, 4, all | 0 | 83 | yes |
| v1_default | 10,000 | 0, 4, all | 0 | 78 | yes |

The oracle hashes are P10's, mined on `local` (`results/p10_semantics.json`).

## dg2_v1: kernel time (ms) by H

| block | H=0 | H=8 | H=16 | H=32 | H=64 | H=128 | H=256 | H=1024 | H=all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fan_in | 12.9 | 10.0 | 5.7 | 4.5 | 4.4 | 4.4 | 4.4 | 4.4 | **2.9** |
| fan_out | 109.5 | 97.5 | 78.4 | 55.1 | 46.1 | 44.6 | **44.6** | 44.7 | 222.3 |
| deg_in | 11.8 | 9.4 | 5.2 | 4.1 | 4.0 | 4.0 | 4.0 | 4.0 | **2.4** |
| deg_out | 33.3 | 30.4 | 25.5 | 20.3 | 17.6 | **17.2** | 17.2 | 17.3 | 125.3 |
| sg | 100.5 | 74.9 | 62.7 | **61.1** | 64.1 | 66.0 | 73.0 | 115.2 | 198.2 |
| tcyc | 328.8 | 308.4 | 296.3 | 268.0 | 194.5 | **173.2** | 179.6 | 177.0 | 293.9 |
| **all** | 596.9 | 530.5 | 473.7 | 413.0 | 330.8 | **309.4** | 322.8 | 362.6 | 845.0 |

Tuned H (smallest total): **128**. Graph upload 33.0 ms.

GPU vs CPU (16 threads, dynamic schedule) on this instance. GPU = kernels + D2H at H = 128; CPU = the block's mining time (`blz bench --backend cpu`).

| block | CPU plan | GPU | CPU | CPU / GPU | rows equal |
|---|---|---:|---:|---:|---|
| fan_in | sweep | 15.6 ms | 34.6 ms | 2.2× | yes |
| fan_out | sweep | 57.1 ms | 34.3 ms | 0.6× | yes |
| deg_in | sweep | 14.9 ms | 33.4 ms | 2.2× | yes |
| deg_out | sweep | 28.8 ms | 31.3 ms | 1.1× | yes |
| sg | per_trigger | 75.7 ms | 1.01 s | 13.3× | yes |
| tcyc | per_trigger | 181.7 ms | 742.3 ms | 4.1× | yes |
| **all** | | 406.8 ms (with upload) | 1.88 s | 4.6× | |

## dg2_wb: kernel time (ms) by H

| block | H=0 | H=8 | H=16 | H=32 | H=64 | H=128 | H=256 | H=1024 | H=all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fan_in | 11.2 | 3.5 | 3.3 | 3.3 | 3.3 | 3.3 | 3.3 | 3.3 | **2.2** |
| fan_out | 30.7 | 14.8 | 11.7 | 11.4 | 11.4 | 11.4 | 11.4 | **11.2** | 42.2 |
| deg_in | 10.5 | 3.3 | 3.1 | 3.1 | 3.1 | 3.1 | 3.1 | 3.1 | **1.9** |
| deg_out | 14.4 | 7.4 | 6.2 | 6.2 | **6.2** | 6.2 | 6.2 | 6.5 | 24.0 |
| sg | 91.1 | 70.5 | 59.1 | **57.3** | 59.9 | 62.3 | 68.1 | 113.8 | 186.3 |
| tcyc | 1021.9 | 998.6 | 981.3 | 955.6 | 899.3 | 597.0 | 575.1 | **564.5** | 845.5 |
| **all** | 1179.8 | 1098.0 | 1064.8 | 1036.9 | 983.1 | 683.3 | **667.1** | 702.4 | 1102.2 |

Tuned H (smallest total): **256**. Graph upload 32.9 ms.

GPU vs CPU (16 threads, dynamic schedule) on this instance. GPU = kernels + D2H at H = 256; CPU = the block's mining time (`blz bench --backend cpu`).

| block | CPU plan | GPU | CPU | CPU / GPU | rows equal |
|---|---|---:|---:|---:|---|
| fan_in | sweep | 14.0 ms | 28.3 ms | 2.0× | yes |
| fan_out | sweep | 23.3 ms | 33.2 ms | 1.4× | yes |
| deg_in | sweep | 13.9 ms | 31.9 ms | 2.3× | yes |
| deg_out | sweep | 17.5 ms | 31.0 ms | 1.8× | yes |
| sg | per_trigger | 77.7 ms | 927.7 ms | 11.9× | yes |
| tcyc | per_trigger | 583.6 ms | 1.61 s | 2.8× | yes |
| **all** | | 762.9 ms (with upload) | 2.66 s | 3.5× | |

## dg2_wc: kernel time (ms) by H

| block | H=0 | H=8 | H=16 | H=32 | H=64 | H=128 | H=256 | H=1024 | H=all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fan_in | 11.0 | 8.5 | 4.7 | 3.7 | 3.7 | 3.7 | 3.7 | 3.7 | **2.5** |
| fan_out | 105.5 | 93.5 | 75.1 | 53.0 | 44.7 | **43.3** | 43.3 | 43.4 | 221.7 |
| deg_in | 10.4 | 7.9 | 4.3 | 3.3 | 3.3 | 3.3 | 3.3 | 3.3 | **2.1** |
| deg_out | 30.9 | 27.8 | 23.3 | 18.7 | 16.4 | **16.1** | 16.1 | 16.1 | 125.5 |
| sg | 86.0 | 63.6 | 53.2 | **51.9** | 54.5 | 56.2 | 62.3 | 96.6 | 167.7 |
| tcyc | 314.3 | 292.0 | 281.2 | 256.9 | 183.8 | 167.9 | 169.2 | **167.8** | 286.8 |
| **all** | 558.2 | 493.4 | 441.8 | 387.6 | 306.4 | **290.4** | 297.9 | 330.9 | 806.3 |

Tuned H (smallest total): **128**. Graph upload 33.1 ms.

GPU vs CPU (16 threads, dynamic schedule) on this instance. GPU = kernels + D2H at H = 128; CPU = the block's mining time (`blz bench --backend cpu`).

| block | CPU plan | GPU | CPU | CPU / GPU | rows equal |
|---|---|---:|---:|---:|---|
| fan_in | sweep | 14.6 ms | 34.2 ms | 2.3× | yes |
| fan_out | sweep | 56.0 ms | 33.7 ms | 0.6× | yes |
| deg_in | sweep | 14.1 ms | 33.0 ms | 2.3× | yes |
| deg_out | sweep | 26.5 ms | 30.7 ms | 1.2× | yes |
| sg | per_trigger | 66.0 ms | 871.4 ms | 13.2× | yes |
| tcyc | per_trigger | 176.6 ms | 726.8 ms | 4.1× | yes |
| **all** | | 386.9 ms (with upload) | 1.73 s | 4.5× | |

## sg_ordered_tsvis: kernel time (ms) by H

| block | H=0 | H=8 | H=16 | H=32 | H=64 | H=128 | H=256 | H=1024 | H=all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fan_in | 12.0 | 4.8 | 4.2 | 4.1 | 4.1 | 4.1 | 4.1 | 4.1 | **2.6** |
| fan_out | 53.5 | 33.3 | 23.4 | 20.4 | 20.3 | 20.3 | 20.3 | **20.3** | 79.2 |
| deg_in | 11.7 | 4.5 | 3.9 | 3.8 | 3.8 | 3.8 | 3.8 | 3.8 | **2.2** |
| deg_out | 19.6 | 13.4 | 10.1 | 9.2 | 9.2 | **9.2** | 9.2 | 9.3 | 44.8 |
| sg | 40.8 | **14.7** | 14.9 | 15.1 | 15.6 | 19.7 | 22.6 | 24.7 | 42.9 |
| sg_ord | 21.4 | **6.7** | 6.9 | 7.1 | 7.4 | 11.1 | 13.4 | 13.3 | 12.2 |
| tcyc | 84.4 | 62.7 | 53.7 | 41.3 | 40.9 | **40.7** | 41.5 | 42.2 | 96.4 |
| **all** | 243.4 | 140.1 | 117.0 | **101.1** | 101.3 | 108.9 | 114.8 | 117.7 | 280.4 |

Tuned H (smallest total): **32**. Graph upload 33.0 ms.

GPU vs CPU (16 threads, dynamic schedule) on this instance. GPU = kernels + D2H at H = 32; CPU = the block's mining time (`blz bench --backend cpu`).

| block | CPU plan | GPU | CPU | CPU / GPU | rows equal |
|---|---|---:|---:|---:|---|
| fan_in | sweep | 15.1 ms | 31.4 ms | 2.1× | yes |
| fan_out | sweep | 30.4 ms | 34.1 ms | 1.1× | yes |
| deg_in | sweep | 14.8 ms | 33.9 ms | 2.3× | yes |
| deg_out | sweep | 20.1 ms | 32.4 ms | 1.6× | yes |
| sg | per_trigger | 24.8 ms | 246.9 ms | 10.0× | yes |
| sg_ord | per_trigger | 15.9 ms | 124.7 ms | 7.8× | yes |
| tcyc | per_trigger | 49.9 ms | 284.9 ms | 5.7× | yes |
| **all** | | 204.1 ms (with upload) | 788.4 ms | 3.9× | |

## sg_ordered_v1w: kernel time (ms) by H

| block | H=0 | H=8 | H=16 | H=32 | H=64 | H=128 | H=256 | H=1024 | H=all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fan_in | 10.8 | 4.0 | 3.5 | 3.4 | 3.4 | 3.4 | 3.4 | 3.4 | **2.3** |
| fan_out | 50.7 | 30.9 | 21.9 | 19.3 | **19.2** | 19.2 | 19.2 | 19.3 | 79.0 |
| deg_in | 10.5 | 3.7 | 3.2 | 3.2 | 3.1 | 3.1 | 3.1 | 3.1 | **2.0** |
| deg_out | 17.4 | 11.7 | 8.9 | 8.1 | **8.1** | 8.1 | 8.1 | 8.3 | 44.7 |
| sg | 35.8 | **12.5** | 12.8 | 13.1 | 13.3 | 16.8 | 19.2 | 21.0 | 37.5 |
| sg_ord | 19.6 | **6.0** | 6.3 | 6.5 | 6.8 | 10.1 | 12.2 | 12.2 | 11.1 |
| tcyc | 77.8 | 57.3 | 49.1 | 38.2 | 37.6 | **37.5** | 37.7 | 39.1 | 92.3 |
| **all** | 222.7 | 126.2 | 105.6 | 91.7 | **91.6** | 98.3 | 102.9 | 106.3 | 268.8 |

Tuned H (smallest total): **64**. Graph upload 33.0 ms.

GPU vs CPU (16 threads, dynamic schedule) on this instance. GPU = kernels + D2H at H = 64; CPU = the block's mining time (`blz bench --backend cpu`).

| block | CPU plan | GPU | CPU | CPU / GPU | rows equal |
|---|---|---:|---:|---:|---|
| fan_in | sweep | 14.4 ms | 30.7 ms | 2.1× | yes |
| fan_out | sweep | 29.7 ms | 33.8 ms | 1.1× | yes |
| deg_in | sweep | 14.1 ms | 33.2 ms | 2.4× | yes |
| deg_out | sweep | 19.8 ms | 32.1 ms | 1.6× | yes |
| sg | per_trigger | 22.9 ms | 216.1 ms | 9.4× | yes |
| sg_ord | per_trigger | 15.7 ms | 114.8 ms | 7.3× | yes |
| tcyc | per_trigger | 46.3 ms | 273.6 ms | 5.9× | yes |
| **all** | | 196.0 ms (with upload) | 734.3 ms | 3.7× | |

## v1_default: kernel time (ms) by H

| block | H=0 | H=8 | H=16 | H=32 | H=64 | H=128 | H=256 | H=1024 | H=all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fan_in | 12.2 | 4.0 | 3.5 | 3.4 | 3.4 | 3.4 | 3.4 | 3.4 | **2.3** |
| fan_out | 50.6 | 30.9 | 21.9 | 19.3 | **19.2** | 19.2 | 19.2 | 19.3 | 79.1 |
| deg_in | 10.5 | 3.7 | 3.2 | 3.2 | 3.1 | 3.1 | 3.1 | 3.1 | **2.0** |
| deg_out | 17.4 | 11.7 | 8.9 | 8.1 | **8.1** | 8.1 | 8.1 | 8.3 | 44.7 |
| sg | 35.8 | **12.5** | 12.8 | 13.1 | 13.3 | 16.8 | 19.2 | 21.0 | 37.1 |
| tcyc | 77.8 | 57.3 | 49.0 | 38.1 | **37.4** | 37.8 | 37.7 | 39.1 | 91.9 |
| **all** | 204.5 | 120.1 | 99.2 | 85.1 | **84.6** | 88.4 | 90.8 | 94.2 | 257.0 |

Tuned H (smallest total): **64**. Graph upload 31.1 ms.

GPU vs CPU (16 threads, dynamic schedule) on this instance. GPU = kernels + D2H at H = 64; CPU = the block's mining time (`blz bench --backend cpu`).

| block | CPU plan | GPU | CPU | CPU / GPU | rows equal |
|---|---|---:|---:|---:|---|
| fan_in | sweep | 14.7 ms | 30.7 ms | 2.1× | yes |
| fan_out | sweep | 29.8 ms | 34.1 ms | 1.1× | yes |
| deg_in | sweep | 14.3 ms | 33.2 ms | 2.3× | yes |
| deg_out | sweep | 18.7 ms | 31.7 ms | 1.7× | yes |
| sg | per_trigger | 22.4 ms | 216.8 ms | 9.7× | yes |
| tcyc | per_trigger | 46.1 ms | 273.8 ms | 5.9× | yes |
| **all** | | 176.9 ms (with upload) | 620.3 ms | 3.5× | |

## v1_default against v1's E5 (`e5_gpu_a30.json`)

Kernel time (ms) at each run's tuned H (v2: 64, v1: 64); a different instance of the same GPU model.

| block | v1 | v2 | v2 / v1 | rows equal |
|---|---:|---:|---:|---|
| fan_in | 3.4 | 3.4 | 1.00 | yes |
| fan_out | 19.1 | 19.2 | 1.00 | yes |
| deg_in | 3.1 | 3.1 | 1.00 | yes |
| deg_out | 8.1 | 8.1 | 1.00 | yes |
| sg | 13.1 | 13.3 | 1.02 | yes |
| tcyc | 37.5 | 37.4 | 1.00 | yes |

## Checks

50 of 50 pass.

# E13: tcyc hybrid CPU/GPU and reachability pruning (local)

- GPU: NVIDIA GeForce GTX 1650 SUPER (compute capability 7.5, 20 SMs); CUDA build: nvcc 13.4.92, sm 75,80, -O3 -DNDEBUG
- CPU: AMD Ryzen 5 3600 6-Core Processor, 12 OpenMP threads, schedule sorted; build Release, `-O3 -DNDEBUG -march=native`
- Git: 0bfe5438dcc7 (dirty)
- Each cell: median of 5 timed runs after 1 warm-up, every trigger of HI-Small, block tcyc (K_MAX = 10).
- GPU-only T = graph upload + kernels + D2H at the GPU light/heavy threshold H (E5's form). CPU-only T = the block's mining time. Hybrid T = graph upload + the call's wall clock: the GPU finds the CPU's triggers (work estimate >= the CPU threshold H_cpu) and copies them to the host, then the CPU mines them while the GPU mines the rest, then the CPU rows are merged. Every transfer is inside T.
- Pruning is the CPU's reachability pruning (D-43); the GPU never prunes.

## v1_default (window 86,400 s, 24 h)

GPU-only and CPU-only:

| | H=32 | H=128 | H=1024 |
|---|---:|---:|---:|
| GPU-only T | 260 ms | **259 ms** | 262 ms |
| GPU heavy triggers | 3,489 | 2,876 | 2,808 |

CPU-only T: 379 ms without pruning, 341 ms with it (1.1×).

Hybrid T by CPU threshold (rows) and GPU threshold, without / with pruning:

| H_cpu | CPU triggers | H=32 off | H=32 on | H=128 off | H=128 on | H=1024 off | H=1024 on |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 931,677 | 336 ms | 150 ms | 340 ms | 158 ms | 341 ms | 152 ms |
| 8 | 308,858 | 273 ms | 120 ms | 278 ms | 121 ms | 278 ms | **120 ms** |
| 16 | 42,191 | 233 ms | 128 ms | 236 ms | 127 ms | 242 ms | 127 ms |
| 32 | 3,489 | 227 ms | 130 ms | 227 ms | 129 ms | 230 ms | 130 ms |
| 64 | 2,885 | 226 ms | 130 ms | 227 ms | 130 ms | 227 ms | 130 ms |
| 128 | 2,876 | 228 ms | 130 ms | 226 ms | 129 ms | 227 ms | 130 ms |
| 256 | 2,871 | 230 ms | 131 ms | 224 ms | 130 ms | 230 ms | 130 ms |
| all | 0 | 261 ms | 261 ms | 261 ms | 262 ms | 263 ms | 263 ms |

Best hybrid (pruning on): H = 1024, H_cpu = 8: T = 120 ms (T_formula 120 ms): upload 33 ms, split 15 ms (on the GPU: 15 ms), CPU side 33 ms on 308,858 triggers, GPU side 82 ms (kernels 53 ms, D2H 14 ms), merge 5 ms.
Without pruning the best hybrid is 224 ms (H = 128, H_cpu = 256).

Rows: every run equal; hash = the recorded one (7737672036140136160).

## dg2_wc (window 320,400 s, 89 h, the frozen recipe's W-C)

GPU-only and CPU-only:

| | H=32 | H=128 | H=1024 |
|---|---:|---:|---:|
| GPU-only T | 721 ms | 675 ms | **669 ms** |
| GPU heavy triggers | 113,178 | 2,889 | 2,852 |

CPU-only T: 910 ms without pruning, 574 ms with it (1.6×).

Hybrid T by CPU threshold (rows) and GPU threshold, without / with pruning:

| H_cpu | CPU triggers | H=32 off | H=32 on | H=128 off | H=128 on | H=1024 off | H=1024 on |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1,741,457 | 934 ms | 336 ms | 927 ms | 339 ms | 918 ms | 328 ms |
| 8 | 1,139,892 | 880 ms | 235 ms | 876 ms | 234 ms | 873 ms | 232 ms |
| 16 | 488,083 | 856 ms | 144 ms | 795 ms | 141 ms | 784 ms | **140 ms** |
| 32 | 113,178 | 728 ms | 168 ms | 706 ms | 166 ms | 708 ms | 166 ms |
| 64 | 9,442 | 677 ms | 191 ms | 678 ms | 185 ms | 669 ms | 185 ms |
| 128 | 2,889 | 671 ms | 195 ms | 689 ms | 188 ms | 661 ms | 188 ms |
| 256 | 2,871 | 701 ms | 194 ms | 664 ms | 188 ms | 653 ms | 188 ms |
| all | 0 | 719 ms | 718 ms | 671 ms | 672 ms | 671 ms | 671 ms |

Best hybrid (pruning on): H = 1024, H_cpu = 16: T = 140 ms (T_formula 140 ms): upload 32 ms, split 16 ms (on the GPU: 16 ms), CPU side 70 ms on 488,083 triggers, GPU side 100 ms (kernels 71 ms, D2H 14 ms), merge 7 ms.
Without pruning the best hybrid is 653 ms (H = 1024, H_cpu = 256).

Rows: every run equal; hash = the recorded one (15263752808203538370).

## dg2_wb (window 1,728,000 s, 20 days)

GPU-only and CPU-only:

| | H=32 | H=128 | H=1024 |
|---|---:|---:|---:|
| GPU-only T | 1.82 s | 1.72 s | **1.65 s** |
| GPU heavy triggers | 370,060 | 9,016 | 2,852 |

CPU-only T: 2.04 s without pruning, 1.14 s with it (1.8×).

Hybrid T by CPU threshold (rows) and GPU threshold, without / with pruning:

| H_cpu | CPU triggers | H=32 off | H=32 on | H=128 off | H=128 on | H=1024 off | H=1024 on |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1,998,981 | 2.05 s | 644 ms | 2.05 s | 646 ms | 2.06 s | 642 ms |
| 8 | 1,438,230 | 2.01 s | 504 ms | 1.99 s | 506 ms | 2.05 s | 503 ms |
| 16 | 884,758 | 1.91 s | 341 ms | 1.96 s | 343 ms | 1.96 s | 341 ms |
| 32 | 370,060 | 1.82 s | **198 ms** | 1.85 s | 199 ms | 1.83 s | 199 ms |
| 64 | 88,567 | 1.67 s | 264 ms | 1.72 s | 255 ms | 1.70 s | 255 ms |
| 128 | 9,016 | 1.62 s | 316 ms | 1.60 s | 301 ms | 1.63 s | 302 ms |
| 256 | 2,903 | 1.58 s | 328 ms | 1.60 s | 323 ms | 1.64 s | 316 ms |
| all | 0 | 1.82 s | 1.82 s | 1.71 s | 1.72 s | 1.64 s | 1.63 s |

Best hybrid (pruning on): H = 32, H_cpu = 32: T = 198 ms (T_formula 198 ms): upload 32 ms, split 16 ms (on the GPU: 16 ms), CPU side 114 ms on 370,060 triggers, GPU side 161 ms (kernels 130 ms, D2H 14 ms), merge 5 ms.
Without pruning the best hybrid is 1.58 s (H = 32, H_cpu = 256).

Rows: every run equal; hash = the recorded one (11834804438481637798).

## Summary

| profile | GPU-only (best H) | CPU-only, pruned | hybrid, pruned (H, H_cpu) | vs GPU-only | vs CPU-only | hybrid, not pruned |
|---|---:|---:|---:|---:|---:|---:|
| v1_default | 259 ms (128) | 341 ms | 120 ms (1024, 8) | 2.2× | 2.8× | 224 ms |
| dg2_wc | 669 ms (1024) | 574 ms | 140 ms (1024, 16) | 4.8× | 4.1× | 653 ms |
| dg2_wb | 1.65 s (1024) | 1.14 s | 198 ms (32, 32) | 8.3× | 5.7× | 1.58 s |

Default CPU threshold of `blz mine --backend hybrid`: **32** (the smallest summed median T over the three profiles with pruning on at H = 32: 4: 1.13 s, 8: 859 ms, 16: 612 ms, 32: 496 ms, 64: 585 ms, 128: 641 ms, 256: 653 ms, all: 2.80 s).

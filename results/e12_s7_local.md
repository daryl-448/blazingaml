# E12 delta for S7: packed D2H and multi-pattern fusion (local)

- GPU: NVIDIA GeForce GTX 1650 SUPER (compute capability 7.5, 20 SMs); CUDA build: nvcc 13.4.92, sm 75,80, -O3 -DNDEBUG
- CPU: AMD Ryzen 5 3600 6-Core Processor (the host widening of packed rows runs on its OpenMP team, `OMP_WAIT_POLICY=passive`)
- Git: 0d52172ad4af (dirty)
- Each cell: median of 5 timed runs after 1 warm-up, every trigger of HI-Small, H swept over 0, 8, 16, 32, 64, 128, 256, 1024, all.
- Variants: **base** (S6's backend), **pack** (`--pack`: each block crosses PCIe as uint8 or uint16 when its largest cell fits, and the host widens it back), **fuse** (`--fuse`: each compiled fusion's blocks in one kernel), **fuse + pack**.
- D2H is the copy (with packing: the pipelined copy of the narrow rows and their widening, host clock); pack is the pack kernels and the read-back of the block's largest cell.
- T = graph upload (once) + kernels + pack + D2H over the profile's blocks, at the variant's tuned H (smallest summed kernel time, as E12).

## v1_default

Hashes: every variant's equal: **yes**; equal to the reference (P10 oracle): **yes**.

| variant | tuned H | kernels (ms) | pack (ms) | D2H (ms) | D2H bytes | T (ms) | vs base |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 64 | 619.7 | 0.0 | 100.9 | 4 B/cell | 755.0 | +0.0 % |
| pack | 64 | 617.1 | 9.8 | 72.4 | 340 MB | 731.4 | +3.1 % |
| fuse | 64 | 535.5 | 0.0 | 103.3 | 4 B/cell | 672.3 | +10.9 % |
| fuse + pack | 32 | 535.2 | 9.8 | 72.5 | 340 MB | 649.8 | +13.9 % |

Packed widths (bytes per cell): deg_in 1, deg_out 1, fan_in 1, fan_out 1, sg 1, tcyc 1.

| fusion | base: kernels / D2H (ms) | fuse: kernels / D2H | fuse + pack: kernels / pack + D2H | kernels saved by fusion |
|---|---:|---:|---:|---:|
| fan_in+deg_in | 112.3 / 36.2 | 86.4 / 37.0 | 86.8 / 29.4 | +23.1 % |
| fan_out+deg_out | 179.5 / 36.2 | 120.1 / 37.0 | 120.4 / 29.4 | +33.1 % |

## sg_ordered_v1w

Hashes: every variant's equal: **yes**; equal to the reference (P10 oracle): **yes**.

| variant | tuned H | kernels (ms) | pack (ms) | D2H (ms) | D2H bytes | T (ms) | vs base |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 32 | 676.8 | 0.0 | 118.7 | 4 B/cell | 828.1 | +0.0 % |
| pack | 32 | 676.3 | 11.3 | 81.7 | 391 MB | 801.7 | +3.2 % |
| fuse | 32 | 594.6 | 0.0 | 117.9 | 4 B/cell | 745.0 | +10.0 % |
| fuse + pack | 32 | 586.3 | 11.2 | 81.8 | 391 MB | 711.6 | +14.1 % |

Packed widths (bytes per cell): deg_in 1, deg_out 1, fan_in 1, fan_out 1, sg 1, sg_ord 1, tcyc 1.

| fusion | base: kernels / D2H (ms) | fuse: kernels / D2H | fuse + pack: kernels / pack + D2H | kernels saved by fusion |
|---|---:|---:|---:|---:|
| fan_in+deg_in | 112.1 / 37.0 | 86.7 / 36.7 | 86.2 / 29.0 | +22.6 % |
| fan_out+deg_out | 182.2 / 37.0 | 119.5 / 36.5 | 119.6 / 28.9 | +34.4 % |

## sg_ordered_tsvis

Hashes: every variant's equal: **yes**; equal to the reference (P10 oracle): **yes**.

| variant | tuned H | kernels (ms) | pack (ms) | D2H (ms) | D2H bytes | T (ms) | vs base |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 32 | 787.3 | 0.0 | 118.7 | 4 B/cell | 938.3 | +0.0 % |
| pack | 32 | 787.1 | 11.2 | 80.8 | 391 MB | 911.4 | +2.9 % |
| fuse | 32 | 667.1 | 0.0 | 118.1 | 4 B/cell | 817.5 | +12.9 % |
| fuse + pack | 32 | 673.0 | 11.3 | 83.4 | 391 MB | 810.7 | +13.6 % |

Packed widths (bytes per cell): deg_in 1, deg_out 1, fan_in 1, fan_out 1, sg 1, sg_ord 1, tcyc 1.

| fusion | base: kernels / D2H (ms) | fuse: kernels / D2H | fuse + pack: kernels / pack + D2H | kernels saved by fusion |
|---|---:|---:|---:|---:|
| fan_in+deg_in | 151.0 / 37.0 | 105.8 / 36.7 | 107.9 / 29.2 | +30.0 % |
| fan_out+deg_out | 219.9 / 37.0 | 139.4 / 36.7 | 139.6 / 29.3 | +36.6 % |

## dg2_v1

Hashes: every variant's equal: **yes**; equal to the reference (E12 A30): **yes**.

| variant | tuned H | kernels (ms) | pack (ms) | D2H (ms) | D2H bytes | T (ms) | vs base |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 128 | 1687.7 | 0.0 | 103.3 | 4 B/cell | 1831.2 | +0.0 % |
| pack | 128 | 1673.9 | 9.8 | 70.7 | 340 MB | 1786.5 | +2.4 % |
| fuse | 128 | 1521.3 | 0.0 | 103.2 | 4 B/cell | 1657.1 | +9.5 % |
| fuse + pack | 128 | 1527.2 | 9.8 | 71.0 | 340 MB | 1640.8 | +10.4 % |

Packed widths (bytes per cell): deg_in 1, deg_out 1, fan_in 1, fan_out 1, sg 1, tcyc 1.

| fusion | base: kernels / D2H (ms) | fuse: kernels / D2H | fuse + pack: kernels / pack + D2H | kernels saved by fusion |
|---|---:|---:|---:|---:|
| fan_in+deg_in | 164.5 / 37.0 | 119.4 / 36.9 | 119.4 / 28.7 | +27.4 % |
| fan_out+deg_out | 403.0 / 37.0 | 296.3 / 36.9 | 299.0 / 29.0 | +26.5 % |

## dg2_wc

Hashes: every variant's equal: **yes**; equal to the reference (E12 A30): **yes**.

| variant | tuned H | kernels (ms) | pack (ms) | D2H (ms) | D2H bytes | T (ms) | vs base |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 128 | 1499.4 | 0.0 | 103.3 | 4 B/cell | 1635.4 | +0.0 % |
| pack | 128 | 1501.5 | 9.8 | 70.6 | 340 MB | 1614.2 | +1.3 % |
| fuse | 128 | 1427.4 | 0.0 | 103.3 | 4 B/cell | 1563.2 | +4.4 % |
| fuse + pack | 128 | 1407.5 | 9.8 | 71.8 | 340 MB | 1521.4 | +7.0 % |

Packed widths (bytes per cell): deg_in 1, deg_out 1, fan_in 1, fan_out 1, sg 1, tcyc 1.

| fusion | base: kernels / D2H (ms) | fuse: kernels / D2H | fuse + pack: kernels / pack + D2H | kernels saved by fusion |
|---|---:|---:|---:|---:|
| fan_in+deg_in | 119.2 / 37.0 | 100.2 / 37.0 | 97.1 / 29.1 | +15.9 % |
| fan_out+deg_out | 348.8 / 37.0 | 278.5 / 37.0 | 271.1 / 29.1 | +20.1 % |

## dg2_wb

Hashes: every variant's equal: **yes**; equal to the reference (E12 A30): **yes**.

| variant | tuned H | kernels (ms) | pack (ms) | D2H (ms) | D2H bytes | T (ms) | vs base |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 128 | 2379.2 | 0.0 | 103.3 | 4 B/cell | 2514.8 | +0.0 % |
| pack | 128 | 2381.1 | 9.8 | 70.2 | 340 MB | 2493.3 | +0.9 % |
| fuse | 256 | 2253.3 | 0.0 | 103.3 | 4 B/cell | 2388.9 | +5.0 % |
| fuse + pack | 256 | 2296.7 | 9.8 | 71.5 | 340 MB | 2410.6 | +4.1 % |

Packed widths (bytes per cell): deg_in 1, deg_out 1, fan_in 1, fan_out 1, sg 1, tcyc 1.

| fusion | base: kernels / D2H (ms) | fuse: kernels / D2H | fuse + pack: kernels / pack + D2H | kernels saved by fusion |
|---|---:|---:|---:|---:|
| fan_in+deg_in | 107.4 / 37.0 | 81.6 / 37.0 | 83.0 / 28.9 | +24.0 % |
| fan_out+deg_out | 140.7 / 37.0 | 95.2 / 37.0 | 97.0 / 29.1 | +32.4 % |

## libgomp's default wait policy

The same `--pack` run of v1_default at H = 32 without `OMP_WAIT_POLICY` (10 timed runs): D2H per run.

| block | D2H min / median / max (ms) | widening median (ms) |
|---|---:|---:|
| deg_in | 12.2 / 12.2 / 18.0 | 11.8 |
| deg_out | 12.2 / 12.2 / 12.5 | 11.8 |
| fan_in | 12.2 / 12.2 / 48.1 | 11.8 |
| fan_out | 12.2 / 12.2 / 22.2 | 11.8 |
| sg | 10.3 / 14.9 / 40.3 | 14.5 |
| tcyc | 9.2 / 9.3 / 47.5 | 8.9 |

## Checks

- v1_default: measured: **pass**
- v1_default: every variant ran: **pass**
- v1_default: every variant's hashes equal: **pass**
- v1_default: hashes = reference: **pass**
- sg_ordered_v1w: measured: **pass**
- sg_ordered_v1w: every variant ran: **pass**
- sg_ordered_v1w: every variant's hashes equal: **pass**
- sg_ordered_v1w: hashes = reference: **pass**
- sg_ordered_tsvis: measured: **pass**
- sg_ordered_tsvis: every variant ran: **pass**
- sg_ordered_tsvis: every variant's hashes equal: **pass**
- sg_ordered_tsvis: hashes = reference: **pass**
- dg2_v1: measured: **pass**
- dg2_v1: every variant ran: **pass**
- dg2_v1: every variant's hashes equal: **pass**
- dg2_v1: hashes = reference: **pass**
- dg2_wc: measured: **pass**
- dg2_wc: every variant ran: **pass**
- dg2_wc: every variant's hashes equal: **pass**
- dg2_wc: hashes = reference: **pass**
- dg2_wb: measured: **pass**
- dg2_wb: every variant ran: **pass**
- dg2_wb: every variant's hashes equal: **pass**
- dg2_wb: hashes = reference: **pass**

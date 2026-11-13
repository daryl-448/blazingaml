# E11: optimization ablation, CPU backend (a30)

- CPU: AMD EPYC 7542 32-Core Processor; OMP_PROC_BIND=close, OMP_PLACES=threads; dynamic schedule
- Build: Release, `-O3 -DNDEBUG -march=native`, 12.3.0; one build per variant in build/e11/<variant>, without XGBoost
- Git: 7cbae022da79 (dirty); profile v1_default (b21580d6e129821a); graph data/hi_small.blzg
- Each cell: median [min–max] of 3 timed runs after 1 warm-up, over every trigger (SPEC v2 §7.2). The index build is not timed.
- A variant is the full build with one optimization switched off (`blzc --disable`, SPEC v2 §7.2): find_start = scans start at the list head; distinct_first = a per-thread hash set; degree_order = intersections always scan their src operand; count_only = len()-only sets are materialized; sweep = fan and degree keep v1's per-trigger plan.
- Not run: the frozen DG-2 recipe (configs/dg2_frozen.yaml): its pattern blocks are GFP-produced (G1, DG-3), so no compiled profile exists

Output check: every timed variant's output_fnv1a equals the full build's, block by block. Fuzz gate (`blz fuzz-diff --seeds 0:1000`, every block against the oracle): mismatching rows full 0, find_start 0, distinct_first 0, degree_order 0, count_only 0, sweep 0.

## 1 thread: seconds, and slowdown against the full build

| block | full | find_start | distinct_first | degree_order | count_only | sweep |
|---|---:|---:|---:|---:|---:|---:|
| fan_in | 0.389 [0.388–0.392] | 0.388 [0.386–0.395] 1.00× | 0.390 [0.389–0.393] 1.00× | 0.395 [0.391–0.396] 1.01× = | 0.394 [0.393–0.398] 1.01× | 0.666 [0.664–0.666] 1.71× |
| fan_out | 0.397 [0.395–0.401] | 0.398 [0.398–0.402] 1.00× | 0.398 [0.397–0.399] 1.00× | 0.399 [0.399–0.402] 1.00× = | 0.397 [0.396–0.399] 1.00× | 13.1 [13.0–13.1] 32.85× |
| deg_in | 0.431 [0.430–0.432] | 0.432 [0.430–0.432] 1.00× | 0.429 [0.429–0.429] 1.00× | 0.432 [0.431–0.433] 1.00× = | 0.428 [0.427–0.429] 0.99× | 0.645 [0.642–0.647] 1.50× |
| deg_out | 0.386 [0.386–0.386] | 0.385 [0.381–0.385] 1.00× | 0.386 [0.385–0.387] 1.00× | 0.386 [0.386–0.387] 1.00× = | 0.384 [0.383–0.385] 0.99× | 0.920 [0.899–0.921] 2.38× |
| sg | 2.835 [2.833–2.846] | 2.871 [2.861–2.939] 1.01× | 3.282 [3.218–3.312] 1.16× | 7.444 [7.444–7.461] 2.63× | 2.895 [2.894–2.900] 1.02× | 2.846 [2.843–2.864] 1.00× = |
| tcyc | 4.000 [3.942–4.010] | 4.740 [4.648–4.827] 1.19× | 4.594 [4.549–4.714] 1.15× | 3.916 [3.912–3.919] 0.98× = | 4.000 [3.987–4.006] 1.00× | 3.957 [3.951–4.012] 0.99× = |

## 16 threads: seconds, and slowdown against the full build

| block | full | find_start | distinct_first | degree_order | count_only | sweep |
|---|---:|---:|---:|---:|---:|---:|
| fan_in | 0.031 [0.031–0.032] | 0.032 [0.031–0.033] 1.04× | 0.031 [0.031–0.032] 1.00× | 0.031 [0.031–0.032] 0.99× = | 0.031 [0.031–0.033] 1.00× | 0.054 [0.054–0.055] 1.75× |
| fan_out | 0.034 [0.034–0.040] | 0.038 [0.034–0.038] 1.12× | 0.035 [0.034–0.041] 1.02× | 0.034 [0.034–0.035] 1.01× = | 0.034 [0.033–0.035] 0.99× | 0.826 [0.825–0.826] 24.26× |
| deg_in | 0.033 [0.033–0.034] | 0.033 [0.033–0.034] 1.00× | 0.035 [0.034–0.035] 1.05× | 0.033 [0.033–0.034] 0.99× = | 0.034 [0.033–0.035] 1.00× | 0.053 [0.053–0.053] 1.58× |
| deg_out | 0.032 [0.032–0.034] | 0.032 [0.031–0.033] 0.98× | 0.032 [0.032–0.033] 0.98× | 0.032 [0.032–0.033] 0.98× = | 0.032 [0.032–0.034] 0.98× | 0.069 [0.068–0.069] 2.12× |
| sg | 0.215 [0.215–0.215] | 0.218 [0.218–0.218] 1.01× | 0.240 [0.239–0.240] 1.12× | 0.502 [0.501–0.503] 2.34× | 0.219 [0.218–0.219] 1.02× | 0.213 [0.213–0.213] 0.99× = |
| tcyc | 0.273 [0.273–0.274] | 0.313 [0.313–0.313] 1.14× | 0.302 [0.298–0.304] 1.11× | 0.269 [0.269–0.270] 0.99× = | 0.270 [0.270–0.272] 0.99× | 0.271 [0.270–0.272] 0.99× = |

`=`: the variant's generated code for this block equals the full build's (comments aside), so the difference is noise.

## The sweep lowering (SPEC v2 §7.1)

Speedup = the per-trigger plan (`--disable sweep`) / the sweep (the full build).

| block | 1 thr: per-trigger → sweep | 16 thr: per-trigger → sweep |
|---|---:|---:|
| fan_in | 0.666 → 0.389 s: 1.7× | 0.054 → 0.031 s: 1.8× |
| fan_out | 13.1 → 0.397 s: 32.8× | 0.826 → 0.034 s: 24.3× |
| deg_in | 0.645 → 0.431 s: 1.5× | 0.053 → 0.033 s: 1.6× |
| deg_out | 0.920 → 0.386 s: 2.4× | 0.069 → 0.032 s: 2.1× |

## Measurement notes

- Idle check before timing: True; the top CPU consumers were logged every 10 s (51 samples, `load_log`).
- Cells with other processes at ≥ 50 % CPU at some sample during their runs: none.

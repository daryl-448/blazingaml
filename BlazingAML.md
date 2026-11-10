# BlazingAML: paper digest for replication and new ideas

**Paper:** *BlazingAML: High-Throughput Anti-Money Laundering (AML) via Multi-Stage Graph Mining*.
Haojie Ye, Arjun Laxman, Yichao Yuan, Krisztian Flautner, Nishil Talati (Univ. of Michigan, UIUC).
arXiv:2604.12241v1 [cs.DC], 14 Apr 2026, 13 pages. The paper links to no code or artifact.

**How to read this file**

- Text outside `> [Note]` blocks is the paper's content. Numbers, code, YAML and formulas are copied exactly. Prose is condensed.
- `> [Note]` blocks hold this digest's own checks, inferences and replication hints. They are not claims made by the paper.
- §9 lists places where the paper contradicts itself. §10 lists what the paper leaves out. Read both before replicating.
- `[n]` is a reference number from the paper's bibliography. The key ones are in §11.

---

## 1. Summary

- **Problem.** Laundering patterns are *fuzzy*:
  - **Structural fuzziness:** the shape is fixed, but the number of intermediate accounts varies.
  - **Temporal fuzziness:** only a local or partial edge order holds, plus a time window. There is no strict global order.

  Mining every exact variant blows up combinatorially. A 3-intermediate scatter-gather already has $6! = 720$ temporal orderings, and the count grows as $O(n!)$, where $n$ is the number of participating edges.
- **Pipeline.** The pipeline is the hybrid one from GFP [4], unchanged. Graph pattern mining produces per-edge pattern-count features, which feed a gradient-boosted classifier (XGBoost [6]). All of the paper's contribution is in the mining front end:
  1. **A multi-stage pattern specification ("Temporal Segment Composition").** A pattern is a chain of stages. Each stage applies an operation (`for_all` neighbor expansion, intersection, union, difference, `differentiate`) to operands (nodes, edges, earlier stage outputs) under constraints (`skip_if`, `break_if`, time window, ordering). Node *sets* per stage capture structural fuzziness. Per-stage constraints capture temporal fuzziness.
  2. **A domain-specific compiler.** It turns declarative specs (e.g. YAML) into nested-loop C++/OpenMP and CUDA code, applying graph-specific optimizations.
- **Headline results** on the IBM synthetic AML data [1]:
  - The mined features are "identical in value" to GFP's, so F1 matches GFP.
  - Mining speedup over GFP running 64 threads is **210× on CPU and 333× on GPU**, reported as "average speedups". See §9 for problems with these numbers.
  - Throughput is **4.9×** that of FraudGT [19], a graph transformer. FraudGT has the higher F1 on 4 of the 6 datasets.

---

## 2. Problem framing (§§2–4 of the paper)

### 2.1 Fuzziness (Fig. 3, shown on scatter-gather)

- **Structural.** The same scatter-gather topology appears with "#Intr. nodes: 2" and with "#Intr. nodes: 4". The number of placement accounts "may range from just a few to dozens".
- **Temporal.** Fig. 3 draws a 3-intermediate scatter-gather. Scatter edges (source → intermediate) are labeled $t_1, t_3, t_5$ and gather edges (intermediate → sink) are labeled $t_2, t_4, t_6$. It shows two options:
  1. Partial order: $t_2 > t_1;\ t_4 > t_3;\ t_6 > t_5$. Each intermediate forwards after it receives. Nothing orders edges across different intermediates.
  2. Time window: $\max(t_i) - \min(t_i) < \delta$.

  The text adds: "only local or partial ordering constraints (e.g., $t_i < t_{i+1}$) holding".
- **Why exact mining is infeasible (§4).**
  - Frameworks with a strict edge order must enumerate $6! = 720$ temporal-constraint combinations for a "3-size" scatter-gather. In general this is $O(n!)$, with $n$ = number of participating edges.
  - Each exact variant is a separate algorithm that rescans the same graph regions.
  - The paper also claims the maintenance burden grows quadratically with the number of variants.
- **Prior work and why it falls short:**
  - **Temporal motif miners** [21, 23, 35] impose a strict global edge order within a window. In classical temporal motifs [23], each edge depends only on the immediately previous edge and timestamps increase monotonically. The paper says this is just a special case of its template.
  - **Cypher** [22] has variable-length paths. It has no syntax for complex topological variation within one query, and no support for temporal fuzziness (partial orders, overlapping windows).
  - **Anti-edge / anti-vertex constructs** (Peregrine [15], [16]) express absence constraints on one or a few manually listed connections. The paper calls this a different problem space.
- **A pattern's presence is not proof of fraud.** For example, fan-in and fan-out are common in legitimate activity. So pattern counts are used as *classifier features*, not as alerts.
- **Patterns drawn in Fig. 2** (taken from [1]): fan-out, fan-in, scatter-gather, gather-scatter, cycle, bipartite, stack.
- **Money-laundering stages:** placement → layering → integration. Detection focuses on **layering**.

### 2.2 Requirements for a pattern language (§4)

- **Expressivity beyond fixed templates.** For example, a lower bound on the number of placement edges ("at least N") rather than one fixed size.
- **Flexible temporal semantics.**
  - "Integration occurs after placement" within a horizon $\delta$, while events inside each phase are interchangeable.
  - Cycle edges may arrive out of cycle order. For example, account A credits B earlier as camouflage, before the illicit funds reach A.
  - A practical language must therefore:
    - (i) represent steps as logical time advances, with optional per-step partial orders;
    - (ii) allow cross-step constraints such as causality, min/max fanout and node inequality;
    - (iii) accept out-of-order evidence ("anticipatory edges"), as long as eventual-consistency constraints are met.
- **Separation of concerns.**
  - Analysts state *what* is suspicious.
  - The compiler decides *how*: neighbor ordering, early exits on temporal violations, degree-aware intersections, and workload balancing across threads, warps and cores.
  - It targets CUDA for GPUs and OpenMP for CPUs.
- **Temporal Segment Composition** is the name of the abstraction. A pattern is a sequence of logical time steps, with interchangeable operations inside a step and explicit constraints across steps. The paper says it "admits static checks, enables aggressive pruning".

---

## 3. System pipeline (Fig. 1)

1. Transactions (Src, Dst, Curr, Amt, Type).
2. Streaming graph representation.
3. Pattern mining and matching.
4. Transactions augmented with graph-mining features. For each edge, the feature value is "the number of instances of each pattern the transaction participates in".
5. A downstream classifier (gradient boosting, XGBoost [6]) predicts licit or illicit for each transaction.

Details:

- **Graph.** A node is a bank account. An edge is a transaction. Attributes include amount, currency, account details and timestamp.
- **Streaming.**
  - "Each incoming transaction edge updates a time-windowed feature representation".
  - A new edge "automatically increments counts for all pattern instances it participates in".
  - New transactions trigger "localized pattern updates rather than requiring full graph recomputation".
- **Classifier.** Better classifiers, such as GNNs, are explicitly left as future work.

> [Note] The IBM data is a directed **multigraph**: one account pair can have many transactions. Nodes must be keyed by the (bank, account) pair; see §12.

---

## 4. The multi-stage framework (§5)

- **Stage.** A logical phase of money flow, such as layering, scattering or reconvergence. It is not one observed transaction. A stage is produced by applying an operation to one or more earlier stages.
- **Operations.** Neighbor expansion (`for_all`), intersection, union, difference, and `differentiate`. `differentiate` means "filtering with conditional logic such as eliminating self-connections".
- **Operands.** Nodes, edges, their attributes (source, destination, timestamp), and outputs of earlier stages.
- **What each stage declares.** An input set, an output set, the operation, node constraints (e.g. account type, currency type) and temporal constraints (time window or ordering). Inputs may reference any earlier stage's outputs "in any time order".
- **Symbolic node.** "either a single graph node or a node set within each stage."

### 4.1 Scatter-gather (Fig. 4a)

Graph: N0 → {intermediates, including N1} → N2. The trigger event is the edge **N0 → N1**.

| Step | Operation |
|---|---|
| Stage 0 → 1 | trigger transaction N0 → N1 |
| Stage 1 → 2 | expand out-neighbors of N1, called N2 (candidate gather account) |
| Stage 2 → 3 | intermediate node set $= \mathrm{N2.in}() \cap \mathrm{N0.out}()$ |

To make intermediates forward only after they receive (option 1 in Fig. 3), the paper adds "an ordering constraint between edges spanning Stage 2 to Stage 1 and Stage 1 to Stage 0".

### 4.2 Four-node cycle (Fig. 4b)

Graph: N0 → N1 → N2 → N3 → N0. The trigger is **N0 → N1**.

| Step | Operation |
|---|---|
| Stage 0 → 1 | trigger N0 → N1 |
| Stage 1 → 2 | expand to N2 (Fig. 4b shows N1 `out()`) |
| Stage 2 → 3 | N3 set $= \mathrm{N2.out}() \cap \mathrm{N0.in}()$, which closes the loop |

The Fig. 4 caption notes the shared skeleton: "the same stage-based structure for the two patterns, with different in- and out-neighbor sets at stages 0 and 2".

**Claimed advantages:**

- It mirrors how investigators reason about money flow.
- Stages are modular, reusable building blocks for a pattern library.
- Work inside a stage parallelizes across interchangeable candidate nodes.
- One set of primitives covers scatter-gather, cycles and chains.

---

## 5. The compiler (§6)

### 5.1 Inputs

- **Primitives.** `for_all` (iterate over all edges or neighbors of a node), `intersection`, `differentiate`, and `operands`.
- **Format.** YAML is one example. The paper says other representations are supported.
- **Each stage has five components:**
  1. Input operands (e.g. `N0.out_neigh`, `N1.in_neigh`).
  2. Operation types (`for_all` or `intersection`).
  3. Optional temporal constraints (e.g. `Find_Start(t-δ)` for time-window boundaries).
  4. Filters that skip nodes or terminate early based on timestamp thresholds.
  5. Output variables passed to later stages.
- **Checks and mapping.** The compiler parses the spec, checks operand dependencies between stages for logical consistency, and maps operations onto code templates designed for power-law graphs.

### 5.2 How a spec is compiled

- **One loop cell per stage.** Each stage becomes one *cell* of a nested loop:
  - an outer loop over the stage's input set;
  - inner logic that runs the set operation, using fresh sets or outputs already materialized by earlier stages.
- **Assembly.** Cells are assembled in stage order.
- **Dependency tracking.** The compiler tracks data dependencies. It decides whether a stage's results feed forward, or whether an earlier stage must be revisited to supply operands for a later set operation.
- **Exact versus fuzzy operations.** A stage's set operation can be:
  - *exact*: enumerate all neighbors, which matches traditional temporal motifs [23];
  - *any expression* over earlier stage sets: union, intersection, difference, or general set differentiation.

**Algorithm 1** (verbatim; italics dropped):

```
Algorithm 1 Generic compiled pseudo-code for one stage of a
user-specified AML pattern (form_nested_loop).
 1: // Stage configuration
 2: // cfg.op ∈ {for_all, union, intersect, differentiate}
 3: // cfg.src: input node set (graph-based or prior stage variable)
 4: // cfg.dst_var: output node-set variable for this stage
 5: // cfg.constraints: {skip_if, break_if (incl. time window)}
 6: edge_start ← Find_Starting_Edge(t, cfg.src)
 7: for each e in IterateEdges(cfg.src, edge_start, cfg.op) do
 8:     // iterate according to op
 9:     if e satisfies cfg.constraints.break_if then
10:         // including time-window overflow
11:         break
12:     if e satisfies cfg.constraints.skip_if then
13:         continue
14:     n ← ExtractNode(e)
15:     // output of each stage is nodes, not edges
16:     cfg.dst_var.add(n)
17:     // surviving nodes feed the next stage of the nested loop
18: if is_final_stage then
19:     Postprocess(cfg.dst_var)
20:     // final stage result assembly
```

### 5.3 Fig. 5 worked examples (verbatim)

The figure writes subscripts (e₀, N₂). They are flattened here to e0, N2. The figure draws each code box separately and does not show how the boxes nest. The labels `# box k` are added here.

**Node roles in Fig. 5 differ from Fig. 4.** The trigger `e: N0→N1 @ t` is the edge that has just arrived, at time `t`:

- **Scatter-gather (Fig. 5a drawing).** N2 → {N0, ·, ·} → N1. The trigger is the *last* (gather) hop. N0 is an intermediate, N1 is the gather sink, and N2 is the scatter source.
- **4-cycle (Fig. 5a drawing).** N0 → N1 (trigger), N1 → N3, N3 → N2, N2 → N0.

#### Scatter-gather spec (Fig. 5a)

```yaml
pattern: scatter_gather
time_window: delta
stages:
  - id: scatter_phase
    op: for_all
    src: N0.in_neigh
    dst_var: N2
    constraints:
      - skip_if: N2 == N1
      - break_if: e0.t > t
  - id: gather_phase
    op: intersection
    src: N2.out_neigh
    dst: N1.in_neigh
    constraints:
      - break_if: e1.t > t
      - break_if: e2.t > t
    output: sg_tx
```

#### Scatter-gather generated code (Fig. 5b)

Stage diagram:

- Stage 0 is `N0.in`. A ∀ step with filter `N2 != N1` leads to Stage 1.
- Stage 1 is `N2.out` ∩ `N1.in`, giving Stage 2.
- A ∀ step with filter `len(Place. Acct) > 2` leads to Stage 3, "Edge Feat. Update".

```
# box 1
e0.start = find_Start(t – δ, N0.in_neigh)
for e0 in N0.in_neigh: (for_all)
  N2 = e0.src
  if ( e0.t > t ) break
  if N2 == N1 continue
# box 2
e1.start = find_Start(t – δ, N2.out_neigh)
e2.start = find_Start(t – δ, N1.in_neigh)
for e1 in N2.out_neigh: (intersect)
  if ( e1.t > t) break
  for e2 in N1.in_neigh:
    if ( e2.t > t) break
    if e1.dst == e2.src:
      sg_tx.add( (e1, e2) )
# box 3
if len(sg_tx) > 2:
  Update sg_tx features
```

#### 4-cycle spec (Fig. 5a)

```yaml
pattern: 4_cycle
time_window: delta
stages:
  - id: first_leg
    op: for_all
    src: N0.out_neigh
    dst_var: N2
    constraints:
      - skip_if: N2 == N0
      - break_if: e0.t > t
  - id: second_leg
    op: intersection
    src: N2.out_neigh
    dst: N0.in_neigh
    dst_var: N3
    constraints:
      - skip_if: N3 == N1
      - break_if: e1.t > t
      - break_if: e2.t > t
    output: cycle_tx
```

#### 4-cycle generated code (Fig. 5c)

Stage diagram:

- Stage 0 is **`N1.out`**. A ∀ step with filter `N2 != N0` leads to Stage 1.
- Stage 1 is `N2.out` ∩ `N0.in`, with filter `N3 != N1`, giving Stage 2.
- A ∀ step with filter `len(Place. Acct) > 0` leads to Stage 3, "Edge Feat. Update".

```
# box 1
e0.start = find_Start(t – δ, N1.out_neigh)
for e0 in N0.out_neigh: (for_all)
  N2 = e0.dst
  if ( e0.t > t ) break
  if N2 == N0 continue
# box 2
e1.start = find_Start(t – δ, N2.out_neigh)
e2.start = find_Start(t – δ, N0.in_neigh)
for e1 in N2.out_neigh: (intersect)
  N3 = e1.dst
  if ( e1.t > t) break
  if N3 == N1 continue
  for e2 in N0.in_neigh:
    if ( e2.t > t) break
    if N3 == N1 continue
    if e1.dst == e2.src:
      cycle_tx.add((e,e0,e1,e2,e3))
# box 3
if len(cycle_tx) > 2:
  Update cycle_tx features
```

> [Note] **Inferred execution semantics, for replication.**
> - Adjacency lists must be **sorted by timestamp**. That is the only way `break if e.t > t` is correct, and the paper mentions "sorted neighbor lists and binary search".
> - `find_Start(t − δ, L)` is then a binary search for the first edge in `L` at the lower window boundary $t-\delta$. **Whether an edge exactly at $t-\delta$ is included is not specified.** Fig. 3's strict $\max(t_i) - \min(t_i) < \delta$ implies it is excluded, giving the window $(t-\delta,\ t]$.
> - **The printed loops never use the computed `e*.start`.** Each loop is written `for e in <list>`, which starts at the beginning of the list. Read literally, the code enforces only the upper bound (`e.t > t`) and has no lower bound. Starting each scan at `e*.start` is an inferred repair; the printed code does not show it.
> - With that repair, every stage scans only edges whose timestamps fall between $t-\delta$ and $t$; the lower bound is inclusive or exclusive, as discussed above. Here $t$ is the trigger edge's timestamp, so the window looks **backwards** from the newly arrived edge. This fits the streaming model.
> - Neither generated code enforces an order *within* the window. There is no $e_1.t < e_2.t$ check, i.e. no requirement that scatter comes before gather. Only the window is checked.
>
> [Note] **The consistent reading of the 4-cycle is `first_leg: src: N1.out_neigh`.** That gives N0 → N1 → N2 → N3 → N0, with N2 ≠ N0 and N3 ≠ N1. It matches Fig. 4b, the Stage 0 label `N1.out` in Fig. 5c, and the `find_Start(…, N1.out_neigh)` call. The literal `N0.out_neigh` makes N3 close the 3-cycle N0 → N2 → N3 → N0. That cycle includes the trigger edge only when N2 = N1. See §9.
>
> **This repair fixes connectivity but does not guarantee four distinct nodes.** The printed checks are only N2 ≠ N0 and N3 ≠ N1. A self-loop therefore lets a degenerate walk through, for example A→B→B→C→A with N2 = N1 = B. Self-loops are common in the data: `HI-Small_Trans.csv` has 591,212 rows where the source equals the destination. The paper does not say whether self-loops are filtered or whether nodes must be distinct.

### 5.4 Optimizations the compiler applies

The paper describes these only qualitatively. It gives no ablation.

- **Listed in the introduction:** power-law-aware memory-access pattern generation; degree-based workload balancing; pipelined CPU-GPU execution based on pattern structure.
- **Decided per stage.** Using neighborhood size, the expected size of intermediate sets and how selective the time window is, the compiler decides:
  - how each stage is evaluated;
  - which set operations to reorder;
  - whether the stage runs on CPU, GPU, or a hybrid pipeline.
- **General techniques:**
  - GPU load balancing;
  - memory-coalesced neighbor iteration;
  - ordering set operations by estimated cost;
  - avoiding materialization through on-demand propagation;
  - set intersection with degree-based ordering, choosing the smaller neighborhood at runtime;
  - overlapping CPU-GPU execution that pipelines data transfer with mining across temporal windows.
- **Scatter-gather on GPU:**
  - CUDA threads are assigned to different candidate nodes;
  - memory access is coalesced and neighbor lists sit in shared memory;
  - temporal constraints become conditional branches that exit early.
- **Cycle:**
  - memory locality is prioritized for sequential neighbor traversal;
  - the second-leg intersection uses sorted neighbor lists and binary search, exploiting sparse connectivity.
- **Hybrid CPU/GPU for scatter-gather, fan and cycle patterns:**
  - "shallow traversals" go to the GPU, for massive parallelism and to mitigate load imbalance;
  - "a small number of deep traversals" go to a CPU post-processing stage, which has a higher clock and a latency-optimized memory hierarchy;
  - the generated code moves data between them transparently.
- **Fan.** The neighborhood-search data structure is "further optimized in CUDA".
- **Plan sensitivity.** Small changes in operand selection or stage order can lead the compiler to entirely different execution plans.

---

## 6. Experimental setup (§7, §8.1)

**Datasets.** Table 1, IBM synthetic AML data [1]:

| Dataset | Vertices | Edges |
|---|---:|---:|
| LI-Small | 705,907 | 6,924,055 |
| LI-Medium | 2,032,095 | 31,251,483 |
| LI-Large | 2,070,980 | 176,066,557 |
| HI-Small | 515,088 | 5,078,345 |
| HI-Medium | 2,077,023 | 31,898,238 |
| HI-Large | 2,116,168 | 179,702,229 |

HI means a higher share of illicit transactions; LI means a lower share.

**Protocol.**

- **Split.** Train on the first 80% of transactions by timestamp. Test on the last 20%.
- **Features.**
  - The baseline is "XGB Only": source and destination account IDs. §8.4 says instead: transaction ID, source account and destination account.
  - Then pattern counts per edge are added cumulatively: Fan, then Degree, then Cycle, then Scatter-Gather (SG). They are concatenated and passed to XGBoost.
- **Metric.** F1, chosen because of class imbalance.

**Baselines.**

- **GFP [4] (IBM).** Mines patterns, adds them as edge features, and uses a gradient-boosting classifier. This is the same pipeline shape. It is run at 64 threads and serves as the 1.0× reference.
- **FraudGT [19].** A graph-transformer detector.

**Hardware.**

- CPU: 2× Intel Xeon Platinum 8380 (40 physical cores and 80 SMT threads each), 8 memory channels, 1 TB of memory.
- GPU: up to 4× NVIDIA A40 (48 GB GDDR6). The results report a single A40.
- BlazingAML's CPU runs use OpenMP with 1–256 threads.

**Scalability data.** Trovares synthetic temporal-triangle datasets [30], from 10K to 100M edges.

---

## 7. Results (§8)

### 7.1 F1 scores (Table 2; Fig. 11 plots the same data)

| Dataset | XGB Only | Fan | Fan+Degree | Fan+Degree+Cycle | Fan+Degree+Cycle+SG |
|---|---:|---:|---:|---:|---:|
| LI-Small | 10.1 | 8.1 | 10.3 | 12.7 | 18.4 |
| HI-Small | 11.1 | 23.2 | 38.4 | 42.6 | 46.6 |
| LI-Med | 3.1 | 4.0 | 11.0 | 14.0 | 21.4 |
| HI-Med | 10.4 | 29.3 | 47.1 | 50.4 | 51.1 |
| LI-Large | 5.4 | 9.1 | 16.0 | 17.7 | 17.8 |
| HI-Large | 20.2 | 39.1 | 55.7 | 57.5 | 58.1 |

**The paper's claims about this table:**

- Adding features "correlates with consistent performance improvements across all datasets".
- The full feature set gives the best score on every dataset.
- HI datasets far outperform LI datasets (HI-Large 58.1 vs LI-Large 17.8).
- There is a "clear performance hierarchy (Scatter-Gather > Cycle > Degree > Fan)".

> [Note] **Marginal F1 gain from each feature group.** Features are added cumulatively in a fixed order:
>
> | Dataset | +Fan | +Degree | +Cycle | +SG |
> |---|---:|---:|---:|---:|
> | LI-Small | −2.0 | +2.2 | +2.4 | +5.7 |
> | HI-Small | +12.1 | +15.2 | +4.2 | +4.0 |
> | LI-Med | +0.9 | +7.0 | +3.0 | +7.4 |
> | HI-Med | +18.9 | +17.8 | +3.3 | +0.7 |
> | LI-Large | +3.7 | +6.9 | +1.7 | +0.1 |
> | HI-Large | +18.9 | +16.6 | +1.8 | +0.6 |
>
> This table does not support the paper's "SG > Cycle > Degree > Fan" ranking. Fan and Degree give the largest gains on the HI datasets. Fan *lowers* F1 on LI-Small. There is no ablation that varies the order in which features are added.

### 7.2 Confusion matrix (Table 3: HI-Small, all features, XGBoost)

| | Predicted laundering | Predicted non-laundering |
|---|---:|---:|
| Actual laundering | 558 | 1239 |
| Actual non-laundering | 31 | 1013841 |

> [Note] **Checks against the confusion matrix.**
> - The four cells sum to 1,015,669, which is exactly $0.2 \times 5{,}078{,}345$. This confirms the 20% test split.
> - With laundering as the positive class: precision $=558/589\approx 0.947$ and recall $=558/1797\approx 0.311$.
> - So $F_1 = \frac{2\,TP}{2\,TP+FP+FN} = \frac{1116}{2386} \approx 0.468$, i.e. 46.8 on the paper's 0–100 scale. Table 2 reports 46.6 and Table 4 reports 46.4.
> - The model is high-precision and low-recall.

### 7.3 Mining speed (Figs. 6–10)

Each value is the label printed on the figure's last bar group. In Figs. 6–8 and 10 that group is the geometric mean (GM) across datasets. In Fig. 9 the group's x-axis label is "HI-Large", and it is probably not a GM; see §9, item 13. Speedups are relative to the baseline column.

| Fig. | Pattern | Baseline (= 1.0) | 1 t | 2 t | 4 t | 8 t | 16 t | 32 t | 64 t | 128 t | 256 t | GPU |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | Scatter-Gather | GFP, 64 threads | 7.0 | 13.6 | 27.3 | 54.0 | 103 | 184 | 219 | 220 | 201 | **333** |
| 7 | Cycle | GFP, 64 threads | 10.9 | 22.8 | 42.1 | 68.4 | 86 | 113 | 127 | 122 | 122 | 156 |
| 8 | Fan-in + Fan-out | GFP, 64 threads | 0.6 | 1.2 | 2.3 | 4.4 | 7.5 | 10.9 | 11.4 | 8.2 | 6.4 | 26.7 |
| 9 | Stack (last group, labeled "HI-Large") | BlazingAML, 1 thread | 1.0 | 1.8 | 3.6 | 7.1 | 12.9 | 22.9 | 25.6 | 25.8 | 24.2 | 33.5 |
| 10 | SG, Trovares 10K–100M | GFP, 64 threads | 1.2 | 2.3 | 4.4 | 8.2 | 14.4 | 23.3 | 27.5 | 24.2 | 21.3 | 43.7 |

Here "t" means threads. Figs. 6–8 are plotted on a log scale.

**What the paper's text says about each pattern** (§9 compares this with the figures):

- **Scatter-gather.**
  - A single thread is already "comparable" to GFP running 64 threads.
  - Scaling is near-linear up to 64 threads, "reaching 219–220× ... depending on dataset".
  - "At 128 and 256 threads ... peak improvements of 333× ... on the largest datasets".
- **Cycle.**
  - A single thread is 10.9× faster than GFP.
  - Scaling is linear up to 64 threads (127×) and then plateaus.
  - The GPU gives "159×".
- **Fan.**
  - GFP at 64 threads is comparable to BlazingAML with fewer than 8 threads.
  - Speed improves "up to 32 threads (11.4×)" and then drops ("8.2× at 128 threads").
- **Stack.**
  - No GFP comparison; it is normalized to BlazingAML's own single-thread time.
  - Speed scales "up to 64 threads (25.8×)". The GPU gives 33.5×.
- **Trovares.** The paper says all of the following:
  - 21.8× at 32 threads on 10K edges.
  - 40.8× on 1M and 27.8× on 10M.
  - On average, 27.5× at 64 threads.
  - 19.5× and 27.5× at 64 threads on 10M and 100M respectively.
  - On the GPU, 24.4× on 100M.
  - Speed scales nearly linearly from 1 to 64 threads.
- **General claim.** "high-core-count CPUs deliver competitive speedups" compared with the GPU.

> [Note] **Thread counts above 160 oversubscribe the CPU.** The server has 2 × 80 = 160 hardware threads, so the 256-thread runs exceed it. That likely explains the drop at 256 threads.
>
> [Note] **Per-dataset speedups measured from the bars.** The paper prints numbers only for the last group. These values were read from bar heights in pixels, calibrated against the printed labels, and are good to roughly ±10%.
>
> | Fig. / pattern | Bar | LI-S | HI-S | LI-M | HI-M | LI-L | HI-L |
> |---|---|---:|---:|---:|---:|---:|---:|
> | 6 / SG | 64 t | 196 | 203 | 173 | 162 | 308 | 308 |
> | 6 / SG | 128 t | 196 | 209 | 173 | 162 | 318 | 318 |
> | 6 / SG | GPU | 298 | 361 | 262 | 230 | 483 | 453 |
> | 7 / Cycle | 64 t | 104 | 97 | 122 | 127 | 135 | 135 |
> | 7 / Cycle | GPU | 194 | 237 | 70 | 77 | 201 | 188 |
> | 8 / Fan | 64 t | 4.8 | 5.6 | 2.0 | 2.0 | 126 | 121 |
> | 8 / Fan | GPU | 22 | 30 | 7.9 | 7.9 | 85 | 76 |
> | 9 / Stack | 64 t | 24 | 23 | 29 | 30 | 26 | (last group: 25.6) |
> | 9 / Stack | GPU | 34 | 37 | 39 | 41 | 36 | (last group: 33.5) |
>
> For Fig. 10 (SG on Trovares; 32 t / 64 t / GPU): 10K ≈ 21 / 17 / 43; 100K ≈ 31 / 40 / 67; 1M ≈ 33 / 40 / 62; 10M ≈ 20 / 26 / 33; 100M ≈ 13 / 19 / 24.
>
> Two things are hidden by the GM summaries:
> - **The GPU is slower than 64 CPU threads in some cases:** Cycle on the Medium datasets, and Fan on the Large datasets.
> - **Fan speedup varies enormously by dataset size.** It is about 2–6× on Small and Medium but about 120× on Large.

### 7.4 Comparison with FraudGT (§8.5)

F1 scores (Table 4):

| Method | LI-Small | HI-Small | LI-Medium | HI-Medium | LI-Large | HI-Large |
|---|---:|---:|---:|---:|---:|---:|
| FraudGT | 28.6 | 69.6 | 24.0 | 62.3 | 11.0 | 54.3 |
| BlazingAML | 18.4 | 46.4 | 21.4 | 51.1 | 17.8 | 58.1 |

The paper explains the F1 gap by the different ML models: "feature extension + XGB" versus a transformer. BlazingAML's features equal GFP's.

**Throughput (Fig. 12).**

- The y-axis is "Kilo Transaction per seconds". BlazingAML runs at 128 threads.
- The headline: "4.9× higher throughput" on average. The caption says "edges per second".
- The paper attributes this to the efficiency of feature extraction plus gradient boosting, compared with "the quadratic complexity inherent in Transformer architectures".

> [Note] **Approximate throughput read off Fig. 12** (thousands of transactions per second; the paper prints no numbers):
>
> | | LI-Small | HI-Small | LI-Med | HI-Med | LI-Large | HI-Large |
> |---|---:|---:|---:|---:|---:|---:|
> | BlazingAML | ~295 | ~340 | ~120 | ~130 | ~115 | ~115 |
> | FraudGT | ~45 | ~45 | ~33 | ~33 | ~25 | ~22 |
>
> The ratios run from about 3.6× to 7.6×, consistent with the reported 4.9× average.

---

## 8. Stated limitations and future work

- The classifier is deliberately simple (gradient boosting). GNNs and other classifiers are named as future work.
- The paper notes that purely pattern-based detection is insufficient because laundering strategies keep changing. This is why mining is used as feature generation.

> [Note] **Claimed in the design but not separately demonstrated or ablated in the evaluation:**
> - per-step partial orders;
> - anticipatory (out-of-order) edges;
> - cross-step min/max-fanout constraints;
> - `union`, `difference` and `differentiate` stages;
> - multi-GPU execution (the results report a single A40);
> - hybrid and pipelined CPU-GPU execution. §6 says it is generated for the scatter-gather, fan and cycle patterns, but the results do not say which bars, if any, use it;
> - streaming or incremental mining. The 80/20 split applies to the classifier. How mining itself was run during evaluation (whole graph at once, or edge by edge) is not stated.
>
> Only Fan, Degree, Cycle and SG features feed the classifier. Gather-scatter, bipartite and stack are drawn in Fig. 2 but are not used as features, and stack is benchmarked for speed only. Any of these gaps could seed new work.

---

## 9. Places where the paper contradicts itself

Treat the figures and tables as primary evidence. Where the prose disagrees with them, the prose is the less reliable source.

**Pattern specification**

1. **4-cycle first-leg source.**
   - The YAML (`src: N0.out_neigh`), the loop (`for e0 in N0.out_neigh`) and the §6 prose ("traverse from N0.out_neigh ... skip self-loops (N2 == N0)") all say `N0.out_neigh`.
   - The Fig. 5c Stage 0 label `N1.out`, the `find_Start(t – δ, N1.out_neigh)` call, and Fig. 4b all say `N1.out`.
   - Only the `N1.out` version yields a 4-cycle that contains the trigger edge.
2. **4-cycle tuple.** `cycle_tx.add((e,e0,e1,e2,e3))` lists 5 edges for a 4-edge cycle. `e3` is never defined. `e` is presumably the trigger edge.
3. **4-cycle threshold.** The diagram filter says `len(Place. Acct) > 0`, but the code says `if len(cycle_tx) > 2`.
4. **Duplicated check.** The 4-cycle code repeats `if N3 == N1 continue` inside the inner `e2` loop. It is redundant.
5. **Scatter-gather node roles differ between figures.**
   - Fig. 4a and the §5 prose: the trigger is the first, scatter hop. N0 is the source and N2 the sink. Intermediates $=\mathrm{N2.in}\cap\mathrm{N0.out}$.
   - Fig. 5 and the §6 prose: the trigger is the last, gather hop. N2 is the source, N1 the sink, and N0 an intermediate. Intermediates $=\mathrm{N2.out}\cap\mathrm{N1.in}$.
   - Fig. 5's version is the one that works with the backward window ending at the trigger time $t$ (see the notes in §5.3).
6. **Edges or nodes?**
   - Algorithm 1 says "output of each stage is nodes, not edges".
   - Fig. 5 adds edge tuples instead: `sg_tx.add((e1, e2))`.
   - The filter is labeled `len(Place. Acct)` (number of accounts) but tests `len(sg_tx)` (number of edge pairs). On a multigraph these differ: two transactions through the same intermediate count as two pairs but one account.
7. **Start-of-window call.**
   - Algorithm 1 calls `Find_Starting_Edge(t, cfg.src)`.
   - Fig. 5 calls `find_Start(t – δ, ·)`.
   - The five-component list writes `Find_Start(t-δ)`.
8. **Operation names.**
   - Algorithm 1 has `{for_all, union, intersect, differentiate}`.
   - The prose has union, intersection, difference, differentiate.
   - The stage-component list allows only "`for_all` or `intersection`".
   - Algorithm 1 has a single `cfg.src`, but an intersection needs two operands. The YAML uses `src` + `dst`, plus `dst_var` for the output.

**Results**

Items 9–14 use the per-dataset bar measurements in the second note of §7.3.

9. **The 333× figure.**
   - The abstract and introduction give 333× as the GPU speedup. In Fig. 6 it is the scatter-gather GPU GM label.
   - §8.2 instead says 333× is reached "At 128 and 256 threads ... on the largest datasets". On LI-Large and HI-Large the bars measure about 318× at 128 threads and about 289× at 256 threads. So the claim roughly holds for 128 threads only. The GM labels are 220× (128 threads) and 201× (256 threads).
   - 333× applies to scatter-gather only: the GPU GM is 156× for cycle and 26.7× for fan. Yet the introduction calls 210× / 333× "average speedups".
10. **The 210× CPU figure is never derived.**
    - Fig. 6's GM labels are 219× (64 threads) and 220× (128 threads).
    - §8.2 says "219–220× ... depending on dataset". But the per-dataset 64-thread bars measure about 162–308×. The numbers 219 and 220 are GM labels at two thread counts, not a per-dataset range.
11. **Cycle on GPU.** The text says 159×. The GM label is 156×, and no single dataset's GPU bar is near 159× (they measure about 70–240×).
12. **Fan.** The text says "up to 32 threads (11.4×)".
    - 11.4× is the GM label at **64** threads; the 32-thread GM is 10.9×.
    - No dataset's 32-thread bar is near 11.4×. They measure about 2–6× on Small and Medium and about 110× on Large.
13. **Stack (Fig. 9).**
    - The text says "up to 64 threads (25.8×)". In the printed last group, 25.8× is the **128**-thread label and the 64-thread label is 25.6×. LI-Large's 64-thread bar (≈26×) is within reading error of 25.8×, so this may be a near-miss rather than a mix-up.
    - **The last bar group is probably not a GM.** Fig. 9 has only 6 bar groups: five datasets plus a last group set off by a dashed line, with x-axis label "HI-Large". Figs. 6–8 instead have six datasets plus a separate GM group.
    - The last group's GPU value (33.5) is below the GM of the other five groups' GPU bars (≈37), so it cannot be their GM. It is most likely HI-Large's own data, placed where the other figures put the GM.
14. **Trovares (Fig. 10).**
    - §8.3 says the 64-thread runs reach "19.5× and 27.5× on Trovares-10M and 100M". The bars show the reverse: about 26× on 10M and about 19× on 100M. The two values look swapped.
    - The other Trovares numbers in the text match the bars: 21.8× (10K, 32 threads); 40.8× (1M) and 27.8× (10M), both with no thread count stated; and 24.4× (GPU, 100M).
    - "Five orders of magnitude" is wrong: 10K → 100M is four orders of magnitude across five datasets.
15. **HI-Small F1** has three values: 46.6 (Table 2), 46.4 (Table 4), and 46.8 computed from Table 3.
16. **FraudGT throughput.**
    - §8.5 says the gap is "particularly pronounced on larger datasets, where BlazingAML processes between 300-400 thousand transactions per second compared to FraudGT's 50-100 thousand".
    - Fig. 12 shows about 300–340K only on the **Small** datasets, and about 115–130K on Medium and Large. FraudGT stays at or below about 50K everywhere.
17. **Accuracy claim versus FraudGT.** The contributions list says "4.9× higher throughput compared to the state-of-the-art with the same level of accuracy". But F1 is well below FraudGT's on 4 of 6 datasets (e.g. 46.4 vs 69.6). "Same accuracy" is only true relative to GFP.
18. **Table 2 prose.**
    - "Consistent improvements across all datasets" is contradicted by LI-Small, where Fan takes F1 from 10.1 to 8.1.
    - The "SG > Cycle > Degree > Fan" ranking is unsupported (see the note in §7.1).
19. **Baseline features.** Table 2's caption says source and destination account IDs. §8.4 says transaction ID, source account and destination account.
20. **Bolding in Table 4.** It bolds FraudGT only on LI-Medium and HI-Medium, even though FraudGT also wins on LI-Small and HI-Small.
21. **Typos that could confuse a reader.**
    - The Fig. 5b and 5c captions say "Complication example"; it should be *Compilation*.
    - "furtner" should be *further*.
    - "safely and audibly" probably means *auditably*.
    - The legend says "BLazingAML GPU".
    - "The average speedup of BlazingAML GFP" is missing a word.

---

## 10. What the paper does not specify (replication gaps)

**Mining**

- $\delta$, the time window, for each pattern. The paper gives no value at all.
- The largest cycle length counted (only a 4-cycle is shown).
- Exact definitions of the **Fan** and **Degree** features. Degree is not a pattern in Fig. 2.
- Exact definitions of the gather-scatter, bipartite and stack specs.
- **Feature semantics.** The features are said to equal GFP [4]'s, so GFP is the ground truth: its windows, which patterns it counts, and any bin or size limits.
- How multi-edges, self-loops and timestamp ties are handled.
- Whether every edge in `sg_tx` / `cycle_tx` gets its count incremented, including the trigger edge.

**Classifier**

- XGBoost hyperparameters and class weighting.
- Whether raw attributes are used (amount, currency, payment format). The baseline mentions only IDs.
- Which class F1 is computed on. It is very likely the laundering class.

**Timing**

- What "end-to-end throughput" covers: graph construction, I/O, feature write-back, XGBoost?
- How streaming is simulated, if it is at all.
- FraudGT's configuration and hardware, and whether its throughput includes training.
- GFP's version and settings, apart from 64 threads.
- How the 210× CPU number was computed.

**Implementation**

- The compiler, its intermediate representation, and the generated code are not released or linked.
- How the multi-GPU setup is used.
- The Trovares dataset generation parameters.

> [Note] **External pointers, not from the paper; verify before relying on them.** GFP ships in IBM's Snap ML library as `snapml.GraphFeaturePreprocessor`. The IBM AML datasets [1] are public; they are distributed via Kaggle as "IBM Transactions for Anti Money Laundering (AML)".

---

## 11. Key references (the paper's numbering)

- [1] Altman et al., *Realistic synthetic financial transactions for anti-money laundering models*. NeurIPS 36 (2024). The IBM AML / "IT-AML" datasets.
- [4] Blanuša et al., *Graph Feature Preprocessor: Real-time Extraction of Subgraph-based Features from Transaction Graphs*. arXiv:2402.08593 (2024). GFP: the main baseline and source of the features.
- [6] Chen & Guestrin, *XGBoost*. KDD 2016.
- [15] Jamshidi et al., *Peregrine*. EuroSys 2020. [16] Jamshidi et al., *Anti-vertex for neighborhood constraints in subgraph queries*. GRADES-NDA 2022.
- [19] Lin et al., *FraudGT: a simple, effective, and efficient graph transformer for financial fraud detection*. ICAIF 2024.
- [21] Mackey et al., *A chronological edge-driven approach to temporal subgraph isomorphism*. IEEE BigData 2018.
- [22] Neo4j, *Cypher*.
- [23] Paranjape, Benson & Leskovec, *Motifs in temporal networks*. WSDM 2017.
- [28] Talati et al., *Mint: An accelerator for mining temporal motifs*. MICRO 2022.
- [30] Trovares, *Temporal Triangles xGT Datasets* (2024).
- [35] Yuan et al., *Everest: GPU-Accelerated System For Mining Temporal Motifs*. arXiv:2310.02800 (2023).

---

## 12. Local dataset in this repo (not from the paper)

**`dataset/HI-Small_Trans.csv`**

- Header: `Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering`
- 5,078,345 rows and 515,088 distinct **(bank, account)** nodes. Both match Table 1's HI-Small exactly.
- 5,177 rows have `Is Laundering = 1`.
- 591,212 rows are self-loops, where the source (bank, account) equals the destination.
- Timestamps run from `2022/09/01 00:00` to `2022/09/18 16:18`, at minute resolution.
- **The file is not sorted by time.**
- **Reproducing the paper's split:**
  - Stable-sort by `Timestamp`, keeping file order for ties.
  - The test set is the last $5{,}078{,}345 - \lfloor 0.8 \cdot 5{,}078{,}345 \rfloor = 1{,}015{,}669$ rows.
  - That test set holds **1,797** positives, which equals $558 + 1239$ from Table 3.
  - The boundary timestamp `2022/09/08 16:12` appears on both sides of the split, so the way ties are broken matters.
  - Splitting in file order instead gives 1,251 positives.

**`dataset/HI-Small_accounts.csv`**

- Header: `Bank Name,Bank ID,Account Number,Entity ID,Entity Name`
- 518,581 accounts. Only 515,088 of them appear in transactions.

**`dataset/HI-Small_Patterns.txt`**

- Lists ground-truth laundering attempts, each starting with a `BEGIN LAUNDERING ATTEMPT - <TYPE>` line.
- Counts by type: BIPARTITE 49, CYCLE 54, FAN-IN 40, FAN-OUT 48, GATHER-SCATTER 51, RANDOM 41, SCATTER-GATHER 44, STACK 43.

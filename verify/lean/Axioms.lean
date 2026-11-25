import Blz

/-! Each main theorem's axioms. verify/algorithms.sh runs this file after the build and fails if
any line names `sorryAx`. The expected axioms are `propext`, `Classical.choice` and `Quot.sound`
(or a subset). -/

-- EarliestWitness: tcyc (R4 and Q5)
#print axioms Blz.earliest_witness
#print axioms Blz.clean_tcyc_earliest_witness
#print axioms Blz.legacy_tcyc_earliest_witness
-- LatestWitness: tchain (R4 and Q5)
#print axioms Blz.latest_witness
#print axioms Blz.clean_tchain_latest_witness
#print axioms Blz.legacy_tchain_latest_witness
-- OrderTest: sg_ord (R6b) and legacy sg_ordered
#print axioms Blz.order_test
#print axioms Blz.clean_sg_ord_order_test
#print axioms Blz.legacy_sg_ord_order_test
-- Scan: the hop answers and count_distinct
#print axioms Blz.countDistinct_eq
#print axioms Blz.scanBinding_isEarliest
#print axioms Blz.tchain_hop_isLatest
#print axioms Blz.scanBinding_eq_earliestIn
#print axioms Blz.tchainAnswer_eq_latestIn
-- Gs: R7
#print axioms Blz.gsRow_eq
#print axioms Blz.clean_gs_row
#print axioms Blz.legacy_gs_row
-- Sweep: the sweep lowering
#print axioms Blz.Sweep.sweep_correct
#print axioms Blz.Sweep.clear_zero
-- Prune: reachability pruning, legacy only
#print axioms Blz.root_kept
#print axioms Blz.bindings_kept
#print axioms Blz.greedy_bindings_kept
#print axioms Blz.ldJ_search
#print axioms Blz.one_round_cuts

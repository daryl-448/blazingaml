"""L1's proof mutants (SPEC v3 §4.3, AC-V3-3): each one changes one line of a model of the code, of
a contract or of a claim, and must make the build fail. Run by `verify/algorithms.sh --mutants`:

    python3 -I mutants.py BUILT_PROJECT_DIR

BUILT_PROJECT_DIR is a copy of verify/lean whose `lake build` has passed; each mutant builds in a
fresh copy of it (with its .lake, so only the mutated module and what imports it rebuild).

Each entry is (name, module, old, new, where, claim); `old` must occur exactly once in the module.
`where` lists the declarations that fail to elaborate, in source order, as Module.decl; the run
checks it, so the record cannot go stale. `claim` is the one among them whose statement the mutant
makes false, so that no proof of it can exist. The first failure is then either

  claim  the false claim itself, or
  step   a proof step of a declaration whose statement still holds (a brittle proof), before the
         false claim fails further down.

A mutant that builds is SURVIVED, one that fails elsewhere than `where` is MOVED; either, or a
pattern not found exactly once (BAD), fails the run.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

LAKE = os.environ.get("LAKE") or os.path.join(
    os.environ.get("ELAN_HOME", os.path.expanduser("~/.cache/blz-fv/lean/elan")), "bin", "lake")

R4_GREEDY = "| some x => (greedy o first lo hi (max lo (o.next x)) Ls).map (x :: ·)"
CLEAN_TCYC = "(greedy .le first (t - δ + 1) t (t - δ + 1) hops).isSome"
LEGACY_TCYC = "(greedy .lt first (t - δ) (min (t - 1) vis_t) (t - δ) hops).isSome"
MIRROR = "| some y => (latestGreedy o last lo vis (min (o.prev y) vis) Ls).map (y :: ·)"
CLEAN_TCHAIN = "(latestGreedy .le last (t - δ + 1) t t hops.reverse).isSome"
LEGACY_TCHAIN = "(latestGreedy .lt last (t - δ) vis_t (min (t - 1) vis_t) hops.reverse).isSome"

MUTANTS = [  # (name, module, old, new, where, claim)
    # Hops: the rule itself
    ("spec drops the closing link t_k R t", "Hops",
     "  | [x], t => R x t", "  | [x], _ => True",
     ["Hops.Chain.head_rel", "Hops.Chain.le_end", "Hops.Chain.mono_end", "Hops.chain_snoc"],
     "Hops.Chain.head_rel"),
    ("spec drops the window's lower bound", "Hops",
     "Picks ts hops ∧ (∀ x ∈ ts.head?, lo ≤ x) ∧ Chain o.rel ts t",
     "Picks ts hops ∧ True ∧ Chain o.rel ts t",
     ["EarliestWitness.Witness.mono_lo", "EarliestWitness.Witness.cons",
      "EarliestWitness.Witness.uncons", "EarliestWitness.clean_tcyc_earliest_witness",
      "EarliestWitness.legacy_tcyc_earliest_witness", "LatestWitness.Witness.snoc",
      "LatestWitness.witness_snoc", "LatestWitness.clean_tchain_latest_witness",
      "LatestWitness.legacy_tchain_latest_witness"],
     "EarliestWitness.Witness.uncons"),
    # EarliestWitness: tcyc, R4 and Q5
    ("R4: the greedy advances to x + 1 under ≤", "EarliestWitness",
     R4_GREEDY, R4_GREEDY.replace("(o.next x)", "(x + 1)"),
     ["EarliestWitness.greedy_sound", "EarliestWitness.greedy_complete",
      "EarliestWitness.example@229"],
     "EarliestWitness.greedy_complete"),
    ("R4: a strict spec under the ≤ greedy", "EarliestWitness",
     "Chain (· ≤ ·) ts t) ↔", "Chain (· < ·) ts t) ↔",
     ["EarliestWitness.clean_tcyc_earliest_witness"],
     "EarliestWitness.clean_tcyc_earliest_witness"),
    ("R4: an open upper bound, t_k < t", "EarliestWitness",
     CLEAN_TCYC, CLEAN_TCYC.replace("(t - δ + 1) t (t - δ + 1)", "(t - δ + 1) (t - 1) (t - δ + 1)"),
     ["EarliestWitness.clean_tcyc_earliest_witness"],
     "EarliestWitness.clean_tcyc_earliest_witness"),
    ("R4: a closed lower bound, t - δ ≤ t_1", "EarliestWitness",
     CLEAN_TCYC, CLEAN_TCYC.replace("(t - δ + 1) t (t - δ + 1)", "(t - δ) t (t - δ)"),
     ["EarliestWitness.clean_tcyc_earliest_witness"],
     "EarliestWitness.clean_tcyc_earliest_witness"),
    ("Q5: the greedy stays at x under <", "EarliestWitness",
     R4_GREEDY, R4_GREEDY.replace("(o.next x)", "x"),
     ["EarliestWitness.greedy_sound", "EarliestWitness.greedy_complete"],
     "EarliestWitness.greedy_sound"),
    ("Q5: the trigger's own time included, min(t, vis_t)", "EarliestWitness",
     LEGACY_TCYC, LEGACY_TCYC.replace("(min (t - 1) vis_t)", "(min t vis_t)"),
     ["EarliestWitness.legacy_tcyc_earliest_witness"],
     "EarliestWitness.legacy_tcyc_earliest_witness"),
    ("Q5: an open lower bound, t - δ < t_1", "EarliestWitness",
     LEGACY_TCYC, LEGACY_TCYC.replace("(t - δ) (min (t - 1) vis_t) (t - δ)",
                                      "(t - δ + 1) (min (t - 1) vis_t) (t - δ + 1)"),
     ["EarliestWitness.legacy_tcyc_earliest_witness"],
     "EarliestWitness.legacy_tcyc_earliest_witness"),
    ("contract: a hop's answer need not be the earliest", "EarliestWitness",
     "  | some y => y ∈ L ∧ b ≤ y ∧ y ≤ hi ∧ ∀ x ∈ L, b ≤ x → x ≤ hi → y ≤ x",
     "  | some y => y ∈ L ∧ b ≤ y ∧ y ≤ hi",
     ["EarliestWitness.greedy_sound", "EarliestWitness.greedy_complete",
      "EarliestWitness.earliestIn_spec", "EarliestWitness.IsEarliest.unique"],
     "EarliestWitness.greedy_complete"),
    # LatestWitness: tchain's mirror, R4 and Q5
    ("R4 mirror: the cap drops to y - 1 under ≤", "LatestWitness",
     MIRROR, MIRROR.replace("(o.prev y)", "(y - 1)"),
     ["LatestWitness.latestGreedy_sound", "LatestWitness.latestGreedy_complete",
      "LatestWitness.example@222"],
     "LatestWitness.latestGreedy_complete"),
    ("Q5 mirror: the cap stays at y under <", "LatestWitness",
     MIRROR, MIRROR.replace("(o.prev y)", "y"),
     ["LatestWitness.latestGreedy_sound", "LatestWitness.latestGreedy_complete"],
     "LatestWitness.latestGreedy_sound"),
    ("R4 mirror: hop 1's cap t - 1", "LatestWitness",
     CLEAN_TCHAIN, CLEAN_TCHAIN.replace("(t - δ + 1) t t", "(t - δ + 1) t (t - 1)"),
     ["LatestWitness.clean_tchain_latest_witness"],
     "LatestWitness.clean_tchain_latest_witness"),
    ("Q5 mirror: hop 1's cap min(t, vis_t)", "LatestWitness",
     LEGACY_TCHAIN, LEGACY_TCHAIN.replace("(min (t - 1) vis_t)", "(min t vis_t)"),
     ["LatestWitness.legacy_tchain_latest_witness"],
     "LatestWitness.legacy_tchain_latest_witness"),
    ("R4 mirror: a closed lower bound t - δ", "LatestWitness",
     CLEAN_TCHAIN, CLEAN_TCHAIN.replace("(t - δ + 1) t t", "(t - δ) t t"),
     ["LatestWitness.clean_tchain_latest_witness"],
     "LatestWitness.clean_tchain_latest_witness"),
    ("R4 mirror: the hops walked forward, not from the trigger back", "LatestWitness",
     CLEAN_TCHAIN, CLEAN_TCHAIN.replace("hops.reverse", "hops"),
     ["LatestWitness.clean_tchain_latest_witness"],
     "LatestWitness.clean_tchain_latest_witness"),
    ("contract: a hop's answer need not be the latest", "LatestWitness",
     "  | some y => y ∈ L ∧ lo ≤ y ∧ y ≤ c ∧ ∀ x ∈ L, lo ≤ x → x ≤ c → x ≤ y",
     "  | some y => y ∈ L ∧ lo ≤ y ∧ y ≤ c",
     ["LatestWitness.latestGreedy_sound", "LatestWitness.latestGreedy_complete",
      "LatestWitness.latestIn_spec", "LatestWitness.IsLatest.unique"],
     "LatestWitness.latestGreedy_complete"),
    # OrderTest: sg_ord
    ("sg_ord: the test compares max(e3) R min(e2)", "OrderTest",
     "  | some a, some c => decide (o.rel a c)", "  | some a, some c => decide (o.rel c a)",
     ["OrderTest.order_test", "OrderTest.example@75"],
     "OrderTest.order_test"),
    ("sg_ord: y → w probed for its earliest edge, not its latest", "OrderTest",
     "(hA : IsEarliest A lo hi ea) (hB : IsLatest B lo hi la)",
     "(hA : IsEarliest A lo hi ea) (hB : IsEarliest B lo hi la)",
     ["OrderTest.order_test", "OrderTest.clean_sg_ord_order_test",
      "OrderTest.legacy_sg_ord_order_test"],
     "OrderTest.order_test"),
    ("sg_ord: a missing witness keeps y (the cold run of verify-algorithms)", "OrderTest",
     "  | _, _ => false", "  | _, _ => true",
     ["OrderTest.order_test"],
     "OrderTest.order_test"),
    ("sg_ord R6b: the window's upper bound t - 1", "OrderTest",
     "(hA : IsEarliest A (t - δ + 1) t ea) (hB : IsLatest B (t - δ + 1) t la)",
     "(hA : IsEarliest A (t - δ + 1) (t - 1) ea) (hB : IsLatest B (t - δ + 1) (t - 1) la)",
     ["OrderTest.clean_sg_ord_order_test"],
     "OrderTest.clean_sg_ord_order_test"),
    # Scan: distinct_first and count_distinct
    ("distinct_first: a scan visits every entry", "Scan",
     "if e.nbr ∈ pre then firstsFrom (e.nbr :: pre) es else e :: firstsFrom (e.nbr :: pre) es",
     "if e.nbr ∈ pre then e :: firstsFrom (e.nbr :: pre) es else e :: firstsFrom (e.nbr :: pre) es",
     ["Scan.mem_nbr_firstsFrom", "Scan.mem_firstsFrom", "Scan.firstsFrom_nodup",
      "Scan.firstsFrom_earliest"],
     "Scan.mem_nbr_firstsFrom"),
    ("scan: the earliest binding without the list sorted by ts", "Scan",
     "theorem scanBinding_isEarliest {L : List Entry} (hs : L.Pairwise (fun a c => a.ts ≤ c.ts))",
     "theorem scanBinding_isEarliest {L : List Entry}",
     ["Scan.scanBinding_isEarliest", "Scan.scanBinding_eq_earliestIn"],
     "Scan.scanBinding_isEarliest"),
    ("range: an open lower bound b < ts", "Scan",
     "decide (b ≤ e.ts) && decide (e.ts ≤ hi) && decide (e.eid ≤ me)",
     "decide (b < e.ts) && decide (e.ts ≤ hi) && decide (e.eid ≤ me)",
     ["Scan.scanBinding_isSome", "Scan.scanBinding_isEarliest"],
     "Scan.scanBinding_isSome"),
    # Gs: R7
    ("gs: the gate emits a when b = 0", "Gs",
     "def gsRow (a b : Nat) : Nat := if a = 0 then 0 else if b = 0 then 0 else min a b",
     "def gsRow (a b : Nat) : Nat := if a = 0 then 0 else if b = 0 then a else min a b",
     ["Gs.gsRow_eq"],
     "Gs.gsRow_eq"),
    ("gs R7: A keeps the source w", "Gs",
     "gsRow (countDistinct (fun s => !(s == w))", "gsRow (countDistinct (fun _ => true)",
     ["Gs.clean_gs_row"],
     "Gs.clean_gs_row"),
    ("legacy gs: A's order test ts ≤ t", "Gs",
     "(Lin.filter (inRange (t - δ + 1) (min (t - 1) vis_t) me))",
     "(Lin.filter (inRange (t - δ + 1) (min t vis_t) me))",
     ["Gs.legacy_gs_row"],
     "Gs.legacy_gs_row"),
    # Sweep: the sweep lowering
    ("sweep: evicts ts ≤ lower(t_j), one too many", "Sweep",
     "if _h : s.lo < s.hi ∧ c.ts s.lo < lts then", "if _h : s.lo < s.hi ∧ c.ts s.lo ≤ lts then",
     ["Sweep.advLo_spec"],
     "Sweep.advLo_spec"),
    ("sweep, stream visibility: hi < j leaves the trigger out", "Sweep",
     "decide (k < c.b) && (if c.tsVis then decide (c.ts k ≤ c.ts j) else decide (k ≤ j))",
     "decide (k < c.b) && (if c.tsVis then decide (c.ts k ≤ c.ts j) else decide (k < j))",
     ["Sweep.advHi_spec"],
     "Sweep.advHi_spec"),
    ("sweep, timestamp visibility: ts < t leaves the ties out", "Sweep",
     "decide (k < c.b) && (if c.tsVis then decide (c.ts k ≤ c.ts j) else decide (k ≤ j))",
     "decide (k < c.b) && (if c.tsVis then decide (c.ts k < c.ts j) else decide (k ≤ j))",
     ["Sweep.advHi_spec"],
     "Sweep.advHi_spec"),
    ("sweep, distinct: leave tests seen before the decrement", "Sweep",
     "(if c.distinct then s.count - (if seen' y = 0 then 1 else 0) else s.count - 1)",
     "(if c.distinct then s.count - (if s.seen y = 0 then 1 else 0) else s.count - 1)",
     ["Sweep.CountInv.leave"],
     "Sweep.CountInv.leave"),
    ("sweep, distinct: clearing misses the window's last entry", "Sweep",
     "(List.range' s.lo (s.hi - s.lo)).foldl", "(List.range' s.lo (s.hi - s.lo - 1)).foldl",
     ["Sweep.clear_zero"],
     "Sweep.clear_zero"),
    # Prune: reachability pruning, legacy only
    ("prune: the search relaxes only ts < ld - 1", "Prune",
     "decide (e.ts < ld e.dst)", "decide (e.ts + 1 < ld e.dst)",
     ["Prune.ldJ_search", "Prune.two_rounds_keep"],
     "Prune.ldJ_search"),
    ("prune: the search seeds U below t", "Prune",
     "| 0 => fun x => if x = U then t else bot", "| 0 => fun x => if x = U then t - 1 else bot",
     ["Prune.ldJ_U", "Prune.ldJ_search", "Prune.two_rounds_keep"],
     "Prune.ldJ_U"),
    ("prune: one round fewer than hops", "Prune",
     "(V :: cs).Nodup → (V :: cs).getLast? = some U → cs.length ≤ R →",
     "(V :: cs).Nodup → (V :: cs).getLast? = some U → cs.length ≤ R + 1 →",
     ["Prune.bindings_kept", "Prune.greedy_bindings_kept"],
     "Prune.bindings_kept"),
    ("prune: the root check keep(V, t - δ)", "Prune",
     "(hR : cs.length ≤ R) (hw : Witness .lt lo t (hopLists E (V :: cs)) ts) : lo ≤ ld R V := by",
     "(hR : cs.length ≤ R) (hw : Witness .lt lo t (hopLists E (V :: cs)) ts) : lo < ld R V := by",
     ["Prune.root_kept"],
     "Prune.root_kept"),
    ("prune: extended to R4's ≤ greedy", "Prune",
     "(hg : greedy .lt first lo (t - 1) lo (hopLists E (V :: cs)) = some ts)",
     "(hg : greedy .le first lo (t - 1) lo (hopLists E (V :: cs)) = some ts)",
     ["Prune.greedy_bindings_kept"],
     "Prune.greedy_bindings_kept"),
]

DECL = re.compile(r"^(?:@\[[^\]]*\]\s*)?(?:(?:private|protected|noncomputable)\s+)*"
                  r"(theorem|lemma|def|example|instance|structure|abbrev|inductive)\b\s*([^\s:({\[]*)")
ERROR = re.compile(r"^error: (?:\./)*Blz/(\w+)\.lean:(\d+):\d+:")


def decl_at(lines, line):
    """The declaration that contains 1-based `line`."""
    for i in range(line - 1, -1, -1):
        m = DECL.match(lines[i])
        if m:
            return m.group(2) if m.group(1) != "example" else f"example@{i + 1}"
    return f"line{line}"


def failed_decls(root, out):
    found = []
    for l in out.splitlines():
        m = ERROR.match(l)
        if not m:
            continue
        mod, line = m.group(1), int(m.group(2))
        with open(os.path.join(root, "Blz", mod + ".lean"), encoding="utf-8") as f:
            name = f"{mod}.{decl_at(f.read().split(chr(10)), line)}"
        if name not in found:
            found.append(name)
    return found


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python3 -I mutants.py BUILT_PROJECT_DIR")
    base = os.path.abspath(sys.argv[1])
    bad = 0
    with tempfile.TemporaryDirectory(prefix="blz-l1-mutants-") as tmp:
        for i, (name, mod, old, new, where, claim) in enumerate(MUTANTS, 1):
            root = os.path.join(tmp, str(i))
            shutil.copytree(base, root, symlinks=True)
            path = os.path.join(root, "Blz", mod + ".lean")
            with open(path, encoding="utf-8") as f:
                src = f.read()
            if src.count(old) != 1:
                print(f"BAD      {i:2} {name}: the pattern occurs {src.count(old)} times in {mod}.lean")
                bad += 1
                continue
            with open(path, "w", encoding="utf-8") as f:
                f.write(src.replace(old, new))
            r = subprocess.run([LAKE, "build"], cwd=root, capture_output=True, text=True)
            got = failed_decls(root, r.stdout + r.stderr)
            status = "SURVIVED" if r.returncode == 0 else "KILLED" if got == where else "MOVED"
            bad += status != "KILLED"
            kind = "claim" if got[:1] == [claim] else "step"
            print(f"{status:8} {i:2} {name}")
            print(f"{'':11} fails at {', '.join(got) or '(no error line)'}")
            if status == "KILLED":
                tail = "" if kind == "claim" else f"; the false claim: {claim}"
                print(f"{'':11} first failure: {kind}{tail}")
            else:
                print(f"{'':11} recorded: {', '.join(where)}")
            shutil.rmtree(root)
    print(f"{len(MUTANTS) - bad} of {len(MUTANTS)} mutants killed where recorded")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""SPEC v2 §8 S5: assemble results/s5_typologies.json.

`blz fuzz-diff --profile configs/profiles/typologies.yaml --report results/s5_typologies.json
--name <run>` writes the AC1 runs into the file; this script adds, in place:

- `gate`: the structure of each typology's listed attempts (results/gt_join.json on the .blzg) that
  decides S5's gate, next to E9's eligibility under the frozen recipe (any block);
- `hi_small_ac1`: per profile, the output hashes of `blz mine` on each backend (features/s5/);
- `e9_eligibility`: per profile, the share of complete attempts eligible and activated per block
  (`blz coverage`, features/s5/coverage_<profile>.json), and eligible under the frozen recipe or
  the new blocks;
- the mutation check and the fuzz relations of tests/unit/test_typologies.cpp, as recorded.

Run: uv run --python 3.12 --with numpy bench/s5_report.py
"""
import collections
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench" / "dg2"))
from blzg import Blzg  # noqa: E402

OUT = ROOT / "results" / "s5_typologies.json"
PROFILES = {"typologies": "hi_small", "typologies_wc": "hi_small_wc"}
BACKENDS = ("oracle", "cpu", "cuda")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def structure(g, attempts):
    """Per typology: the share of attempts that are matchings (no two edges share an account), the
    largest weakly connected component in edges, the share with a 2-hop temporal chain a -> u -> v
    (a != v, ts(a -> u) < ts(u -> v), any span), and the share with a hub of >= 2 distinct senders
    and >= 2 distinct receivers."""
    by = collections.defaultdict(list)
    for a in attempts:
        if a["status"] != "complete":
            continue
        eids = [r["eid"] for r in a["rows"]]
        src, dst, ts = g.src[eids].tolist(), g.dst[eids].tolist(), g.ts[eids].tolist()
        parent = {}

        def find(x):
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for s, d in zip(src, dst):
            parent[find(s)] = find(d)
        comp = collections.Counter(find(s) for s in src)
        chain = any(d1 == s2 and s1 != d2 and s1 != s2 and s2 != d2 and t1 < t2
                    for s1, d1, t1 in zip(src, dst, ts) for s2, d2, t2 in zip(src, dst, ts))
        ins, outs = collections.defaultdict(set), collections.defaultdict(set)
        for s, d in zip(src, dst):
            ins[d].add(s)
            outs[s].add(d)
        hub = any(len(ins[x]) >= 2 and len(outs[x]) >= 2 for x in ins)
        by[a["typology"]].append((max(comp.values()) == 1, max(comp.values()), chain, hub))
    pct = lambda xs: round(100 * sum(xs) / len(xs), 1)
    return {t: {"attempts": len(v), "matching_pct": pct([x[0] for x in v]),
                "largest_component_edges_max": max(x[1] for x in v),
                "two_hop_chain_pct": pct([x[2] for x in v]), "gs_hub_pct": pct([x[3] for x in v])}
            for t, v in sorted(by.items())}


def shares(cov):
    out = {}
    for t, v in cov["by_typology"].items():
        c = v["complete"]
        out[t] = {"complete": c,
                  "eligible_pct": {b: round(100 * x / c, 1) for b, x in v["eligible"].items()},
                  "activated_pct": {b: round(100 * x / c, 1) for b, x in v["activated"].items()}}
    return out


def with_frozen(cov, e9_attempts):
    """Per typology: complete attempts eligible for any frozen-recipe block, for any new block, and
    for either (E9's frozen eligibility per attempt, results/e9_coverage.json)."""
    fz = {a["id"]: any(a["frozen"]["eligible"]) for a in e9_attempts if a["status"] == "complete"}
    by = collections.defaultdict(lambda: [0, 0, 0, 0])
    for a in cov["attempts"]:
        if a["status"] != "complete":
            continue
        new = any(a["eligible"])
        for t in (a["typology"], "ALL"):
            r = by[t]
            r[0] += 1
            r[1] += fz[a["id"]]
            r[2] += new
            r[3] += fz[a["id"]] or new
    return {t: {"complete": r[0], "frozen_any_pct": round(100 * r[1] / r[0], 1),
                "new_any_pct": round(100 * r[2] / r[0], 1), "either_pct": round(100 * r[3] / r[0], 1)}
            for t, r in sorted(by.items(), key=lambda kv: (kv[0] == "ALL", kv[0]))}


def main():
    res = json.loads(OUT.read_text())
    gt = json.loads((ROOT / "results" / "gt_join.json").read_text())
    g = Blzg(str(ROOT / gt["inputs"]["graph"]))
    if g.csv_sha256 != gt["inputs"]["csv_sha256"]:
        raise SystemExit("results/gt_join.json is not a gt-join of " + gt["inputs"]["graph"])
    e9_doc = json.loads((ROOT / "results" / "e9_coverage.json").read_text())
    frozen = e9_doc["profiles"]["frozen"]
    e9 = {t: round(100 * v["eligible"]["any"] / v["complete"], 1) for t, v in frozen["by_typology"].items()}
    res["gate"] = {
        "rule": "SPEC v2 §8 S5: E9 shows typology T ineligible for every existing block under the frozen profile, "
                "while a draft spec for T would make most of T's complete attempts eligible",
        "structure": structure(g, gt["attempts"]),
        "frozen_eligible_any_pct": e9,
        "frozen_recipe": frozen["profile"]["file"],
        "decision": "RANDOM passes the gate's first half (0 % eligible under the frozen recipe). STACK is an "
                    "exception admitted with it: 33 % eligible, all through tcyc (13 attempts are 2-cycles, one a "
                    "4-cycle), so 67 % are seen by no block. A chain spec (tchain) makes most of both eligible. "
                    "GATHER-SCATTER fails it (92 % eligible: 84 % through fan and degree, 4 more attempts through "
                    "tcyc only); the user chose on 2026-11-19 to build its draft (gs) as well. BIPARTITE cannot "
                    "pass: every attempt is a matching (D-41)",
    }
    ac1 = {}
    for prof, stem in PROFILES.items():
        runs = {}
        for be in BACKENDS:
            man = json.loads((ROOT / "features" / "s5" / f"{stem}_{be}" / "features.json").read_text())
            runs[be] = {b: {"output_fnv1a": v["output_fnv1a"], "saturated": v["saturated"]}
                        for b, v in man["blocks"].items()}
            runs[be]["profile_id"] = man["profile"]["id"]
        blocks = [b for b in runs["cpu"] if b != "profile_id"]
        same = all(runs[be] == runs["cpu"] for be in BACKENDS)  # profile id, and every block's hash and saturation
        cpu_dir = ROOT / "features" / "s5" / f"{stem}_cpu"
        ac1[prof] = {"backends": runs, "hash_parity": same,
                     "cpu_sha256": {b: sha256(cpu_dir / f"{b}.npy") for b in blocks},
                     "note": "blz mine on each backend (cuda on the GTX 1650 SUPER, default heavy threshold); every "
                             "backend's npy was byte-identical, and only the cpu one is kept (for blz coverage). "
                             "Not a timing run."}
    res["hi_small_ac1"] = ac1
    elig = {}
    for prof, stem in PROFILES.items():
        cov = json.loads((ROOT / "features" / "s5" / f"coverage_{prof}.json").read_text())
        br = cov["base_rates"]
        elig[prof] = {"profile_id": cov["profile"]["id"],
                      "windows": {b: v["window"] for b, v in cov["profile"]["resolved"]["blocks"].items()},
                      "by_typology": shares(cov), "with_frozen": with_frozen(cov, e9_doc["attempts"]),
                      "with_frozen_note": "frozen_any = eligible for a block of E9's frozen recipe ("
                                          + frozen["profile"]["file"] + ", W-C windows) under both profiles here",
                      "nonzero_pct": {b: {"edges": round(100 * v["edges"]["all"] / br["edges"]["all"], 2),
                                          "positives": round(100 * v["positives"]["all"] / br["positives"]["all"], 2)}
                                      for b, v in br["nonzero"].items()}}
    res["e9_eligibility"] = elig
    res["mutation_check"] = {
        "change": "the CPU emitter's for_all ignores witness: latest (it binds the scanned edge's own ts, the earliest witness)",
        "seeds": "0:2000, configs/profiles/typologies.yaml, cpu vs oracle", "mismatched_rows": 244352,
        "mismatched_graphs": 1479, "note": "the emitter was restored and rebuilt after the check"}
    res["fuzz_relations_unit_test"] = {
        "test": "tests/unit/test_typologies.cpp (oracle, seeds 0:300)", "rows": 58893, "gs_nonzero": 26007,
        "gs_below_fan_out": 15480, "tchain_ge_4_edges": 21888,
        "note": "gs <= fan_out holds on every row; the fuzz graphs exercise gs's min() (it is below fan_out on 26 % "
                "of rows) and chains of 4 edges or more (37 % of rows)"}
    OUT.write_text(json.dumps(res, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

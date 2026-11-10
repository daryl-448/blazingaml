# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "xgboost>=3.4,<3.5"]
# ///
# DG-2 diagnostic (SPEC §17), not a headline result: it looks at test-split scores.
# uv run --python 3.12 bench/dg2/global.py data/hi_small.blzg features/oracle_default <variant>
# variants: ids+global, ids+global+win
# DG-2 diagnostic: fan/degree with no time window (all earlier edges), raw sizes instead of bins.
import json, sys
import numpy as np, xgboost as xgb
graph, feats, variant = sys.argv[1], sys.argv[2], sys.argv[3]
raw = np.memmap(graph, dtype=np.uint8, mode="r")
m = int(np.frombuffer(raw[8:16], np.uint64)[0]); n = int(np.frombuffer(raw[16:24], np.uint64)[0])
al = lambda x: (x + 63) // 64 * 64
off = 64
def take(dt, size):
    global off
    a = np.frombuffer(raw[off:off + size * m], dt); off = al(off + size * m); return a
src = take(np.uint32, 4).astype(np.int64); dst = take(np.uint32, 4).astype(np.int64)
ts = take(np.int32, 4); label = take(np.uint8, 1)
eid = np.arange(m); nl = src != dst  # self-loops: excluded, and self-loop triggers emit nothing (A13, A14)

def running(key, first_flag):
    """For each non-loop edge e: sum of first_flag over non-loop edges e' <= e with the same key."""
    out = np.zeros(m, np.float32)
    idx = eid[nl]; k = key[nl]; f = first_flag[nl].astype(np.int64)
    order = np.lexsort((idx, k)); ks = k[order]; cs = np.cumsum(f[order])
    start = np.r_[0, np.flatnonzero(ks[1:] != ks[:-1]) + 1]
    base = np.repeat(np.r_[0, cs][start], np.diff(np.r_[start, len(ks)]))
    out[idx[order]] = cs - base
    return out

pair = src * n + dst
first = np.zeros(m, bool)
_, fi = np.unique(np.where(nl, pair, -1), return_index=True); first[fi] = True; first &= nl
ones = np.ones(m, bool)
g = {"deg_out": running(src, ones), "deg_in": running(dst, ones),
     "fan_out": running(src, first), "fan_in": running(dst, first)}
cols, names = [src.astype(np.float32), dst.astype(np.float32)], ["src_id", "dst_id"]
if "global" in variant:
    for k in ["fan_in", "fan_out", "deg_in", "deg_out"]: cols.append(g[k]); names.append("g_" + k)
if "win" in variant:
    for b in ["fan_in", "fan_out", "deg_in", "deg_out", "tcyc", "sg"]:
        a = np.load(f"{feats}/{b}.npy", mmap_mode="r")
        cols += [a[:, j].astype(np.float32) for j in range(a.shape[1])]; names += [f"{b}[{j}]" for j in range(a.shape[1])]
X = np.stack(cols, axis=1); del cols
ntr = m * 4 // 5
params = {"objective": "binary:logistic", "tree_method": "hist", "max_depth": 6, "eta": 0.3,
          "max_bin": 256, "seed": 0, "nthread": 12}
bst = xgb.train(params, xgb.DMatrix(X[:ntr], label=label[:ntr], nthread=12), 100)
p = bst.predict(xgb.DMatrix(X[ntr:], nthread=12)) > 0.5
y = label[ntr:] == 1
tp, fp, fn = int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum())
gain = bst.get_score(importance_type="gain")
top = sorted(((v, names[int(k[1:])]) for k, v in gain.items()), reverse=True)[:5]
print(json.dumps({"variant": variant, "cols": X.shape[1], "tp": tp, "fp": fp, "fn": fn,
                  "f1": round(200 * tp / (2 * tp + fp + fn), 1), "top_gain": [n for _, n in top]}))

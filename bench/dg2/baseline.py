# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "xgboost>=3.4,<3.5"]
# ///
# DG-2 diagnostic (SPEC §17), not a headline result: it looks at test-split scores.
# uv run --python 3.12 bench/dg2/baseline.py data/hi_small.blzg features/oracle_default <variant>
# variants: ids, ids+eid, ids+raw, ids+raw+all
# DG-2 diagnostic: does adding raw transaction fields to the inputs move F1 toward the paper?
import json, sys
import numpy as np, xgboost as xgb
graph, feats, variant = sys.argv[1], sys.argv[2], sys.argv[3]
raw = np.memmap(graph, dtype=np.uint8, mode="r")
m = int(np.frombuffer(raw[8:16], np.uint64)[0])
al = lambda x: (x + 63) // 64 * 64
off = 64
def take(dt, size):
    global off
    a = np.frombuffer(raw[off:off + size * m], dt); off = al(off + size * m); return a
src = take(np.uint32, 4); dst = take(np.uint32, 4); ts = take(np.int32, 4); label = take(np.uint8, 1)
paid = take(np.float64, 8); recv = take(np.float64, 8)
pay_ccy = take(np.uint8, 1); recv_ccy = take(np.uint8, 1); fmt = take(np.uint8, 1)
f32 = lambda a: np.asarray(a, dtype=np.float32)
cols, names = [f32(src), f32(dst)], ["src_id", "dst_id"]
if "eid" in variant: cols.append(f32(np.arange(m))); names.append("eid")
if "raw" in variant:
    cols += [f32(paid), f32(recv), f32(pay_ccy), f32(recv_ccy), f32(fmt)]
    names += ["amount_paid", "amount_received", "pay_ccy", "recv_ccy", "format"]
if "all" in variant:
    for b in ["fan_in", "fan_out", "deg_in", "deg_out", "tcyc", "sg"]:
        a = np.load(f"{feats}/{b}.npy", mmap_mode="r")
        cols += [f32(a[:, j]) for j in range(a.shape[1])]; names += [f"{b}[{j}]" for j in range(a.shape[1])]
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
                  "f1": round(200 * tp / (2 * tp + fp + fn), 1),
                  "top_gain": [n for _, n in top]}))

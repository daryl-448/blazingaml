# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "xgboost>=3.4,<3.5"]
# ///
"""Independent check of `blz train` (SPEC §10.2, DG-2): the same .blzg + .npy blocks, trained with
Python XGBoost from a plain in-memory DMatrix instead of the C API QuantileDMatrix callback.

uv run --python 3.12 bench/xgb_xcheck.py <graph.blzg> <features dir> <block,block,...|''>
Prints one JSON line with the test-split confusion counts and F1 (x100). Needs ~3 GB of RAM.
"""
import json
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent / "dg2"))
from blzg import Blzg  # noqa: E402

graph, feats = sys.argv[1], sys.argv[2]
blocks = [b for b in sys.argv[3].split(",") if b]
g = Blzg(graph)
if g.dataset()["id"] != "hi_small":  # SPEC v3 §6.4: a new dataset's test split is scored once, after its freeze
    sys.exit(f"{graph}: dataset {g.dataset()['id']}; this DG-2 check scores the test split of HI-Small only")
src, dst, label = g.src, g.dst, g.label

cols = [src.astype(np.float32), dst.astype(np.float32)]
for b in blocks:
    a = np.load(f"{feats}/{b}.npy", mmap_mode="r")
    cols += [a[:, j].astype(np.float32) for j in range(a.shape[1])]
X = np.stack(cols, axis=1)
del cols
ntr = g.splits()["test"][0]  # train = [0, b_test), as `blz train` reads it
params = {"objective": "binary:logistic", "tree_method": "hist", "max_depth": 6, "eta": 0.3,
          "max_bin": 256, "seed": 0, "nthread": 12}
bst = xgb.train(params, xgb.DMatrix(X[:ntr], label=label[:ntr], nthread=12), 100)
p = bst.predict(xgb.DMatrix(X[ntr:], nthread=12)) > 0.5
y = label[ntr:] == 1
tp, fp, fn = int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum())
print(json.dumps({"blocks": blocks, "xgboost": xgb.__version__, "columns": X.shape[1],
                  "tp": tp, "fp": fp, "fn": fn, "f1_x100": 200 * tp / (2 * tp + fp + fn)}))

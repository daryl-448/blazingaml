# E8: final Table 2/3, v1_default and the frozen DG-2 recipe (SPEC v2 §4.5, §4.6)

Train rows [0, 4062676], scored rows [4062676, 5078345]. Frozen recipe: `configs/dg2_frozen.yaml` (R8/X7). Git: `abd37a64eac41f2adb52e6e2a0574e47c5003dc9`.

Every set holds the recipe's base columns (ids, raw columns, vertex statistics; §4.5 step 3), so the frozen recipe's XGB Only cell is not the paper's raw-attribute XGBoost.

## Table 2 (F1 ×100)

| | XGB Only | Fan | Fan+Degree | Fan+Degree+Cycle | Fan+Degree+Cycle+SG |
|---|---:|---:|---:|---:|---:|
| Paper | 11.1 | 23.2 | 38.4 | 42.6 | 46.6 |
| v1_default, `blz train` | 0.0 | 0.9 | 6.1 | 6.6 | 6.4 |
| v1_default, Python | 0.0 | 0.9 | 6.1 | 6.6 | 6.4 |
| **Frozen recipe** | **58.0** | **60.6** | **63.5** | **63.9** | **63.0** |
| Frozen Precision | 0.810 | 0.859 | 0.806 | 0.804 | 0.734 |
| Frozen Recall | 0.452 | 0.468 | 0.524 | 0.531 | 0.553 |
| Frozen PR-AUC | 0.550 | 0.590 | 0.615 | 0.619 | 0.616 |
| Frozen columns | 59 | 83 | 107 | 116 | 126 |

## Table 3 (`all`; paper, then v1_default, then the frozen recipe)

| | Predicted laundering | Predicted non-laundering |
|---|---:|---:|
| Actual laundering | 558 / 60 / **993** | 1239 / 1737 / **804** |
| Actual non-laundering | 31 / 18 / **360** | 1013841 / 1013854 / **1013512** |

Frozen: precision 0.734 (paper 0.947), recall 0.553 (paper 0.311).

## Verdict (§4.6, D-21): **GAP**

F1 minus paper: xgb_only +46.9, fan +37.4, fan_deg +25.1, fan_deg_cyc +21.3, all +16.4; precision -0.213, recall +0.242.

## v1_default reproduction (R11)

- python xgb_only: reproduces E2
- blz train xgb_only: reproduces E2
- python fan: reproduces E2
- blz train fan: reproduces E2
- python fan_deg: reproduces E2
- blz train fan_deg: reproduces E2
- python fan_deg_cyc: reproduces E2
- blz train fan_deg_cyc: reproduces E2
- python all: reproduces E2
- blz train all: reproduces E2

## Sensitivity: the R8 grid on the frozen recipe, `all` (never used to select)

| scale_pos_weight | max_depth | rounds | F1 ×100 | precision | recall | PR-AUC |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6 | 100 | 22.5 | 0.140 | 0.575 | 0.422 |
| 1 | 6 | 500 | 5.7 | 0.030 | 0.716 | 0.403 |
| 1 | 10 | 100 | 17.5 | 0.107 | 0.485 | 0.163 |
| 1 | 10 | 500 | 21.6 | 0.140 | 0.470 | 0.178 |
| 5 | 6 | 100 | 50.7 | 0.432 | 0.614 | 0.590 |
| 5 | 6 | 500 | 60.5 | 0.680 | 0.545 | 0.587 |
| 5 | 10 | 100 | 49.3 | 0.424 | 0.588 | 0.571 |
| 5 | 10 | 500 (frozen) | 63.0 | 0.734 | 0.553 | 0.616 |

## Gain share by block, frozen recipe

| set | ids | raw | vstats | fan_in | fan_out | deg_in | deg_out | tcyc | sg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| xgb_only | 0.06 | 0.25 | 0.68 | - | - | - | - | - | - |
| fan | 0.06 | 0.25 | 0.62 | 0.02 | 0.04 | - | - | - | - |
| fan_deg | 0.05 | 0.24 | 0.61 | 0.02 | 0.04 | 0.02 | 0.02 | - | - |
| fan_deg_cyc | 0.05 | 0.23 | 0.60 | 0.02 | 0.04 | 0.02 | 0.02 | 0.02 | - |
| all | 0.05 | 0.23 | 0.61 | 0.02 | 0.03 | 0.02 | 0.02 | 0.01 | 0.01 |

DG-3: the frozen recipe is identified with GFP-produced features (G1 blocks, F1 vertex statistics); not yet compiled (SPEC v2 §13).

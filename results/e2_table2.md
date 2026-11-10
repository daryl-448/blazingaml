# E2/E3: HI-Small, oracle features, default XGBoost (SPEC §10)

| | XGB Only | Fan | Fan+Degree | Fan+Degree+Cycle | Fan+Degree+Cycle+SG |
|---|---:|---:|---:|---:|---:|
| Paper F1 (Table 2) | 11.1 | 23.2 | 38.4 | 42.6 | 46.6 |
| **This repo F1** | **0.0** | **0.9** | **6.1** | **6.6** | **6.4** |
| Precision | 0.000 | 0.727 | 0.826 | 0.785 | 0.769 |
| Recall | 0.000 | 0.004 | 0.032 | 0.035 | 0.033 |
| PR-AUC | 0.011 | 0.021 | 0.072 | 0.074 | 0.073 |
| Columns | 2 | 26 | 50 | 59 | 69 |

Confusion matrix, `all` (Table 3 layout; paper in parentheses):

| | Predicted laundering | Predicted non-laundering |
|---|---:|---:|
| Actual laundering | 60 (558) | 1737 (1239) |
| Actual non-laundering | 18 (31) | 1013854 (1013841) |

XGBoost 3.4.2, `configs/xgb_default.yaml`, 100 rounds, threshold 0.5, 12 threads. Test positives: 1797. Git: `cf7a2610bfbb966bfd1a5887a7916e358f85368a`.

# AC-V3-1, dataset parts

git 42d880c. Row hash per block of `blz measure --reps 1` (SPEC v3 §5.1's sink) over the same triggers. Oracle wall time in all: 3,143.3 s of a 3,600 s budget (cap 1,081.7 s per item; D-92: HI-Medium clean_default whole only if <= 20 min, clean_wc_hi_medium in 3 ranges; a cap c with the estimates summing to the budget; whole if <= c, else 3 ranges (start, middle, end) sized to c, at most 2,000,000 triggers).

Verdict: **PASS**.

| Dataset | Profile | Selection | Triggers | Backends | Equal | Seconds |
|---|---|---|---:|---|---|---|
| transxion | clean_default | all | 3,029,170 | cpu, cuda, oracle | yes | oracle 1.5, cpu 0.5, cuda 0.5 |
| transxion | clean_wc_hi_small | all | 3,029,170 | cpu, cuda, oracle | yes | oracle 10.2, cpu 0.9, cuda 0.9 |
| transxion | clean_all | all | 3,029,170 | cpu, cuda, oracle | yes | oracle 3.8, cpu 2.0, cuda 2.0 |
| li_small | clean_default | all | 6,924,049 | cpu, cuda, oracle | yes | oracle 171.0, cpu 2.5, cuda 1.8 |
| li_small | clean_wc_li_small | range:0:1772938 | 1,772,938 | cpu, cuda, oracle | yes | oracle 64.5, cpu 1.1, cuda 0.8 |
| li_small | clean_wc_li_small | range:2575555:4348493 | 1,772,938 | cpu, cuda, oracle | yes | oracle 451.6, cpu 1.7, cuda 1.0 |
| li_small | clean_wc_li_small | range:5151111:6924049 | 1,772,938 | cpu, cuda, oracle | yes | oracle 453.0, cpu 2.5, cuda 1.5 |
| li_small | clean_wc_li_small | all | 6,924,049 | cpu, cuda | yes | cpu 6.0, cuda 3.0 |
| li_small | clean_all | all | 6,924,049 | cpu, cuda, oracle | yes | oracle 490.0, cpu 22.2, cuda 15.6 |
| hi_medium | clean_default | range:0:2000000 | 2,000,000 | cpu, oracle | yes | oracle 19.0, cpu 3.7 |
| hi_medium | clean_default | range:14949119:16949119 | 2,000,000 | cpu, oracle | yes | oracle 221.0, cpu 4.0 |
| hi_medium | clean_default | range:29898238:31898238 | 2,000,000 | cpu, oracle | yes | oracle 326.6, cpu 9.6 |
| hi_medium | clean_wc_hi_medium | range:0:214967 | 214,967 | cpu, oracle | yes | oracle 3.0, cpu 3.5 |
| hi_medium | clean_wc_hi_medium | range:15841635:16056602 | 214,967 | cpu, oracle | yes | oracle 403.1, cpu 3.8 |
| hi_medium | clean_wc_hi_medium | range:31683271:31898238 | 214,967 | cpu, oracle | yes | oracle 486.1, cpu 6.1 |

| Dataset | Profile | Oracle selections | Oracle triggers checked | Share | Oracle s (planned) |
|---|---|---|---:|---:|---:|
| transxion | clean_default | all | 3,029,170 | 100.0% | 2.0 (2.1) |
| transxion | clean_wc_hi_small | all | 3,029,170 | 100.0% | 10.7 (11.0) |
| transxion | clean_all | all | 3,029,170 | 100.0% | 4.2 (6.5) |
| li_small | clean_default | all | 6,924,049 | 100.0% | 172.4 (178.0) |
| li_small | clean_wc_li_small | range:0:1772938, range:2575555:4348493, range:5151111:6924049 | 5,318,814 | 76.8% | 972.7 (1081.7) |
| li_small | clean_all | all | 6,924,049 | 100.0% | 491.2 (560.3) |
| hi_medium | clean_default | range:0:2000000, range:14949119:16949119, range:29898238:31898238 | 6,000,000 | 18.8% | 582.4 (678.6) |
| hi_medium | clean_wc_hi_medium | range:0:214967, range:15841635:16056602, range:31683271:31898238 | 644,901 | 2.0% | 907.7 (1081.7) |

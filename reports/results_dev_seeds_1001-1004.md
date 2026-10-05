# Development run (seeds 1001-1004 were looked at while fixing the closed-day bug)

| Dataset | Days x dishes | Baseline MAE | TabPFN MAE | Baseline WAPE | TabPFN WAPE | TabPFN vs baseline |
|---|---|---|---|---|---|---|
| sample_orders.csv (sample) | 90 x 5 | 3.62 | 3.33 | 30.3% | 27.9% | +7.9% |
| sample seed 1001 | 90 x 5 | 3.67 | 3.36 | 31.6% | 28.9% | +8.5% |
| sample seed 1002 | 90 x 5 | 3.94 | 3.48 | 32.2% | 28.4% | +11.7% |
| sample seed 1003 | 90 x 5 | 3.83 | 3.38 | 32.3% | 28.5% | +11.7% |
| sample seed 1004 | 90 x 5 | 3.25 | 2.94 | 28.1% | 25.5% | +9.5% |

Per dish (sample_orders.csv (sample)), MAE in plates:

| Dish | Avg sold/day | Baseline MAE | TabPFN MAE |
|---|---|---|---|
| Aloo Paratha | 17.1 | 3.75 | 3.49 |
| Dal Makhani | 12.3 | 3.67 | 3.51 |
| Gajar Halwa | 5.8 | 2.54 | 2.33 |
| Paneer Butter Masala | 11.0 | 3.44 | 3.20 |
| Rajma Chawal | 13.6 | 4.68 | 4.14 |

Holdout: last 28 days, one-day-ahead, TabPFN refit every 7 days. tabpfn 6.4.1, torch 2.2.2+cu121, Python 3.11.15, Linux-6.18.44-fc-v64-x86_64-with-glibc2.39.

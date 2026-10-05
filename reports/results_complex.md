| Dataset | Days x dishes | Baseline MAE | TabPFN MAE | Baseline WAPE | TabPFN WAPE | TabPFN vs baseline |
|---|---|---|---|---|---|---|
| sample_orders_complex.csv (sample) | 75 x 6 | 3.58 | 3.00 | 31.9% | 26.8% | +16.1% |

Per dish (sample_orders_complex.csv (sample)), MAE in plates:

| Dish | Avg sold/day | Baseline MAE | TabPFN MAE |
|---|---|---|---|
| Aloo Paratha | 15.1 | 2.95 | 3.05 |
| Chole Bhature | 10.7 | 6.76 | 3.28 |
| Dal Makhani | 11.9 | 2.71 | 2.54 |
| Kadhi Chawal | 7.4 | 2.97 | 2.88 |
| Paneer Butter Masala | 9.3 | 3.12 | 2.95 |
| Rajma Chawal | 12.9 | 2.98 | 3.32 |

Holdout: last 28 days, one-day-ahead, TabPFN refit every 7 days. tabpfn 6.4.1, torch 2.2.2+cu121, Python 3.11.15, Linux-6.18.44-fc-v70-x86_64-with-glibc2.39.

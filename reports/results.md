| Dataset | Days x dishes | Baseline MAE | TabPFN MAE | Baseline WAPE | TabPFN WAPE | TabPFN vs baseline |
|---|---|---|---|---|---|---|
| sample_orders.csv (sample) | 90 x 5 | 3.62 | 3.33 | 30.3% | 27.9% | +7.9% |
| sample seed 2001 | 90 x 5 | 3.77 | 3.11 | 32.1% | 26.4% | +17.6% |
| sample seed 2002 | 90 x 5 | 3.61 | 3.05 | 32.1% | 27.1% | +15.5% |
| sample seed 2003 | 90 x 5 | 4.34 | 3.66 | 37.2% | 31.3% | +15.7% |
| sample seed 2004 | 90 x 5 | 3.91 | 3.32 | 32.0% | 27.2% | +15.0% |
| sample seed 2005 | 90 x 5 | 4.18 | 3.62 | 33.3% | 28.9% | +13.4% |

Per dish (sample_orders.csv (sample)), MAE in plates:

| Dish | Avg sold/day | Baseline MAE | TabPFN MAE |
|---|---|---|---|
| Aloo Paratha | 17.1 | 3.75 | 3.49 |
| Dal Makhani | 12.3 | 3.67 | 3.51 |
| Gajar Halwa | 5.8 | 2.54 | 2.33 |
| Paneer Butter Masala | 11.0 | 3.44 | 3.20 |
| Rajma Chawal | 13.6 | 4.68 | 4.14 |

Holdout: last 28 days, one-day-ahead, TabPFN refit every 7 days. tabpfn 6.4.1, torch 2.2.2+cu121, Python 3.11.15, Linux-6.18.44-fc-v64-x86_64-with-glibc2.39.

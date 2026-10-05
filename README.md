# RasoiIQ: kal kitna banana hai?

**Built for a friend.** Simran runs a home food business in Chandigarh. Every night she guesses how many plates of dal makhani, rajma chawal and aloo paratha to prep for tomorrow. Guess high and food goes to waste; guess low and she turns customers away. RasoiIQ looks at her past orders and tells her, dish by dish, how much to prep tomorrow, what to buy, and how sure it is.

This was built for the DEV Hacktoberfest 2026 challenge, theme **Build for a Friend**, category **Best Use of TabPFN**.

> **Sample data notice.** Simran's real order history isn't in this repo. Everything you see by default (the page, the screenshots, the numbers below) uses `data/sample_orders.csv`, a 90-day **synthetic** dataset made by `scripts/generate_sample.py`. Drop a real `orders.csv` in the project root, or upload one on the page, to use real data.

**Built with PriorLabs-TabPFN.**

## Run it (one command)

```bash
./run.sh
```

Then open http://localhost:8000 (on a Mac it opens by itself).

On the first run, `run.sh` creates `.venv`, installs the pinned packages (about 1 GB, a few minutes), and generates the sample CSV if you don't have an `orders.csv`. The first forecast downloads the 44 MB TabPFN-2 model **once, with no login or token**. After that, everything runs offline.

**Requirements:** Python 3.11 or 3.12 (`brew install python@3.11` on a Mac). No GPU, no accounts, no API keys.

### Tested setup
The target machine is an **Intel iMac with 8 GB RAM and no GPU**. PyTorch stopped shipping Intel-Mac builds after **torch 2.2.2**, so the stack is pinned around it:

| Package | Version | Why |
|---|---|---|
| torch | 2.2.2 | last release with macOS x86_64 wheels |
| tabpfn | 6.4.1 | newest TabPFN that accepts torch < 2.5 |
| numpy | 1.26.4 | torch 2.2.2 is built against numpy 1.x |

`requirements-macos-intel.lock.txt` is the full dependency lock resolved for `x86_64-apple-darwin` + Python 3.11; every wheel in it exists for Intel Macs. The data stays under 1,000 rows (days × dishes) so TabPFN is comfortable on a CPU.

## Your data

Make a CSV like this and save it as `orders.csv` in the project folder, or upload it on the page:

```csv
date,item,quantity,notes
2026-09-01,Dal Makhani,12,
2026-09-01,Rajma Chawal,15,
2026-09-02,Dal Makhani,9,party order
```

- `date` can be `2026-09-01` or `01/09/2026` (day first).
- Several rows for the same dish on the same day are added up.
- `notes` is optional. Words like *festival, party, puja, function, shaadi* mark that day as special.
- A day with **no orders for any dish** is treated as "kitchen closed", not "zero demand" (see [DECISIONS.md](DECISIONS.md)).
- With more than 1,000 rows, the most recent days are used.
- Tick **"festival or party tomorrow?"** on the page if you know tomorrow is special.

### A messier sample to try

`data/sample_orders_complex.csv` (made by `scripts/generate_complex_sample.py`, also **synthetic**) looks more like a real order log, so you can see how RasoiIQ copes with untidy data:

- one row per customer order (2,093 rows that add up to 75 days × 6 dishes)
- columns named `Date, Dish, Qty, Notes`, with day-first dates (`04/10/2026`)
- the same dish spelled several ways (`dal makhani`, ` DAL MAKHANI `)
- a new dish (Chole Bhature) added on 1 September
- two days with nothing logged (kitchen closed), and bulk "party order" rows
- notes with commas inside quotes, and 4 broken rows (blank qty, `two`, a `-2` refund, no dish name)

Upload it on the page, or copy it to `orders.csv`. RasoiIQ merges the spellings, skips the 4 broken rows, treats the 2 empty days as closed, and tells you so.

## How it works

```
orders.csv ─► clean + daily grid ─► features ─► TabPFN-2 (CPU) ─► 10/50/90% forecast
                                        │                              │
                                        └─► same-weekday average ◄─────┤ backtest: last 28 days
                                                                       ▼
                                          prep plates + shopping list ─► one mobile page
                                          (optional: Gemma rewrites it in Hinglish)
```

| File | What it does |
|---|---|
| `app/data.py` | Reads the CSV, fixes dates, sums duplicates, fills a date × dish grid, marks closed days |
| `app/features.py` | Builds the features (each one uses only orders from **before** the day it predicts) |
| `app/forecast.py` | TabPFN fit and predict, the baseline, the backtest, the prep number |
| `app/planner.py` | Shopping list from per-plate recipes; optional Gemma rewrite |
| `app/main.py` | FastAPI server: `/`, `/api/forecast`, `/api/config`, `/api/sample.csv` |
| `app/static/index.html` | The single mobile-first page and its SVG chart (no external scripts, works offline) |
| `scripts/generate_sample.py` | Makes the synthetic sample data |
| `scripts/evaluate.py` | Produces every accuracy number in this README |

### How TabPFN is used

[TabPFN](https://github.com/PriorLabs/TabPFN) is a tabular foundation model from Prior Labs. It was pre-trained on millions of synthetic tables, so it predicts on a new small table in one forward pass, with no training loop or hyperparameter search. That suits a home kitchen with only 90 days of history.

Each row is one dish on one day. The target is plates sold. The features are:

| Feature | Meaning |
|---|---|
| `item_code` | which dish |
| `day_of_week`, `is_weekend` | Sunday paratha rush vs weekday office tiffins |
| `is_festival`, `is_festival_eve` | from a built-in Chandigarh/Punjab calendar (`app/festivals.py`), CSV notes, or the "special tomorrow" tick box |
| `lag_1`, `lag_7` | yesterday, and the same day last week |
| `mean_7`, `mean_28`, `trend_7` | recent level and whether it's rising or falling |
| `same_weekday_avg` | the average of the last 4 same weekdays (the baseline's own guess, given to TabPFN as a hint) |
| `day_index` | slow growth over time |

We call `TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cpu")`, fit on every past dish-day, and ask for **quantiles** (`output_type="quantiles"`, 10% / 50% / 90%) instead of a single number. That gives Simran a range ("likely 16 to 36") and lets the app add a buffer that grows with uncertainty:

> **prep = round up( median + ½ × (90th percentile − median) )**

So a dish TabPFN is unsure about gets more buffer than a steady one.

### How the comparison works

The **baseline** is what a careful person would do by hand: the average of the same weekday over the last 4 weeks.

The **backtest** hides the last 28 days and forecasts them one day at a time, as if it were the night before each one. TabPFN is refit every 7 days using only the days before that week. Both methods see exactly the same information and are scored on the same open days. We report **MAE** (average plates off per dish per day) and **WAPE** (total plates wrong ÷ total plates sold).

## Results: baseline vs TabPFN

All numbers below come from `scripts/evaluate.py` (copied from `reports/results.md`). They are from **synthetic sample data**, not Simran's real orders.

| Dataset | Days × dishes | Baseline MAE | TabPFN MAE | Baseline WAPE | TabPFN WAPE | TabPFN vs baseline |
|---|---|---|---|---|---|---|
| sample_orders.csv (the one shipped) | 90 × 5 | 3.62 | **3.33** | 30.3% | **27.9%** | +7.9% |
| sample seed 2001 | 90 × 5 | 3.77 | **3.11** | 32.1% | **26.4%** | +17.6% |
| sample seed 2002 | 90 × 5 | 3.61 | **3.05** | 32.1% | **27.1%** | +15.5% |
| sample seed 2003 | 90 × 5 | 4.34 | **3.66** | 37.2% | **31.3%** | +15.7% |
| sample seed 2004 | 90 × 5 | 3.91 | **3.32** | 32.0% | **27.2%** | +15.0% |
| sample seed 2005 | 90 × 5 | 4.18 | **3.62** | 33.3% | **28.9%** | +13.4% |

Per dish on the shipped sample (MAE in plates):

| Dish | Avg sold/day | Baseline | TabPFN |
|---|---|---|---|
| Aloo Paratha | 17.1 | 3.75 | **3.49** |
| Dal Makhani | 12.3 | 3.67 | **3.51** |
| Gajar Halwa | 5.8 | 2.54 | **2.33** |
| Paneer Butter Masala | 11.0 | 3.44 | **3.20** |
| Rajma Chawal | 13.6 | 4.68 | **4.14** |

Run on: tabpfn 6.4.1, torch 2.2.2, Python 3.11, 4-core Linux CPU (no GPU). A full backtest took about 13 to 19 seconds per dataset.

On the messier per-order sample (`.venv/bin/python scripts/evaluate.py --csv data/sample_orders_complex.csv --out results_complex`):

| Dataset | Days × dishes | Baseline MAE | TabPFN MAE | Baseline WAPE | TabPFN WAPE | TabPFN vs baseline |
|---|---|---|---|---|---|---|
| sample_orders_complex.csv | 75 × 6 | 3.58 | **3.00** | 31.9% | **26.8%** | +16.1% |

Most of that gain is one dish. Chole Bhature (added on 1 September and still growing) was 3.28 plates off with TabPFN vs 6.76 with the baseline, because the same-weekday average still counts the weeks before it was on the menu. TabPFN was slightly **worse** than the baseline on Aloo Paratha (3.05 vs 2.95) and Rajma Chawal (3.32 vs 2.98). Full per-dish table: `reports/results_complex.md`.

**What these numbers do and don't say:**
- TabPFN beat the same-weekday average on every dataset tested, by 8% to 18% in MAE.
- The data is synthetic, and I wrote the generator. It contains weekday patterns, festival spikes, growth and closures that the features are designed to see, so this is a check that the pipeline works, not proof it will help Simran by this much. Her real data will be noisier.
- An earlier version scored worse than the baseline on one dataset (seed 1004, −40%) because closed days counted as zero demand. That fix, and the results before and after it, are written up in [DECISIONS.md](DECISIONS.md). Seeds 2001 to 2005 were generated *after* the fix and weren't used to tune anything.
- Run `.venv/bin/python scripts/evaluate.py` on her real `orders.csv` to get the numbers that matter.

## Optional: Hinglish with Gemma

If the `GEMMA_API_KEY` environment variable is set (a free Google AI Studio key), the finished prep list is rewritten in simple Hinglish by Gemma (`gemma-3-27b-it` by default, change it with `GEMMA_MODEL`).

- Only the finished plan text (dish names and counts) is sent. Her order history is never sent.
- If Gemma changes any number, its version is thrown away and the plain plan is shown.
- If the key is missing or the network is down, the plain template is used. The app works the same either way.

```bash
GEMMA_API_KEY=your-key ./run.sh
```

## Why open matters here

- **Her sales data stays on her laptop.** The CSV is read and forecast locally on her CPU. Nothing is uploaded anywhere (unless she turns on the optional Gemma rewrite, which sends only the finished list).
- **It costs nothing to run.** No subscription, no per-prediction API bill, no cloud server. The model weights are free to use, including for her business, under the Prior Labs License (Apache 2.0 plus attribution).
- **It works offline after setup.** Once the 44 MB model is downloaded, RasoiIQ runs with no internet. I checked this by running a forecast inside a network namespace with no network at all. Telemetry in the TabPFN package is switched off (`TABPFN_DISABLE_TELEMETRY=1`).
- **Nothing is locked in.** The model is a file on disk. If a better open model comes out, swapping it means changing one function (`make_tabpfn` in `app/forecast.py`).

## Tests

```bash
.venv/bin/python -m pytest -q          # 35 tests, includes a real TabPFN CPU prediction
RASOIIQ_SKIP_REAL=1 .venv/bin/python -m pytest -q   # skip the real-model test
```

The tests cover CSV cleaning (aliases, day-first dates, bad rows, row cap, closed days), no-leakage feature building, weekly-refit backtest boundaries, the fallback to the baseline when TabPFN can't run, the shopping math, the Gemma number check, and the API.

## Personalise

`app/config.py` (or the `FRIEND_NAME`, `CITY`, `THEME` env vars) sets the name, city and theme shown on the page. Per-plate ingredients live in `RECIPES` in `app/planner.py`, and festival dates in `app/festivals.py`.

## Credits and licenses

- **Built with PriorLabs-TabPFN.** TabPFN-2 regressor weights: [Prior Labs License](https://huggingface.co/Prior-Labs/TabPFN-v2-reg/blob/main/LICENSE.txt) (Apache 2.0 with an attribution requirement). The `tabpfn` package is Apache 2.0.
- Optional Hinglish rewrite: Gemma, Google's open-weight model, via the Google AI Studio API.

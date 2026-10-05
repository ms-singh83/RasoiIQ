# Decisions

Choices made overnight without asking, with the reason for each. Change any of them freely.

## Setup and stack
1. **torch 2.2.2 + tabpfn 6.4.1 + numpy 1.26.4, Python 3.11.** torch 2.2.2 is the last PyTorch with Intel macOS wheels. tabpfn 7.0+ requires torch ≥ 2.5, so 6.4.1 is the newest TabPFN that works. numpy is held below 2 because torch 2.2.2 was built against numpy 1.x. Before any app code was written, I verified this with a real tiny TabPFN prediction on CPU. I also confirmed that every package in `requirements-macos-intel.lock.txt` has an Intel-Mac (x86_64) wheel for Python 3.11.
2. **TabPFN-2 weights, not 2.5.** v2 downloads with **no login**: Hugging Face first, then Prior Labs' public Google Cloud Storage mirror, which is built into the package. v2.5 is gated and needs an HF token, and its license is non-commercial, which doesn't suit a business. v2's license is Apache 2.0 plus an attribution line, so the page footer, README and post say "Built with PriorLabs-TabPFN". **HF_TOKEN is not needed.**
3. **TabPFN telemetry is off** (`TABPFN_DISABLE_TELEMETRY=1`, set in `app/__init__.py` and `run.sh`) so the app never contacts anything after the model is downloaded.
4. **The model cache lives in `.tabpfn_cache/`** inside the project, so it's easy to find and delete.
5. **Python 3.11 or 3.12 only.** torch 2.2.2 has no Python 3.13 build, and pandas 3 needs 3.11+. `run.sh` checks and tells you to `brew install python@3.11` if neither is found.

## Placeholders in the brief
6. The brief had `[NAME]`, `[ITEMS]`, `[CITY]` and `[PASTE THEME]` unfilled. I used what you told me earlier tonight: **Simran, Chandigarh, theme "Build for a Friend"**. They're set in `app/config.py` (or the `FRIEND_NAME` / `CITY` / `THEME` env vars). The page has no other place where the name is hard-coded, except the Gemma prompt, which reads the same setting.
7. **Sample dishes:** Dal Makhani, Rajma Chawal, Aloo Paratha, Paneer Butter Masala, Gajar Halwa. They're only in the sample generator and `RECIPES`; her real CSV can have any dishes.

## Data
8. **No `orders.csv` was present, so a 90-day synthetic sample is generated** (5 dishes × 90 days = 450 rows, under the 1,000-row limit). It ends 2026-10-03, so "tomorrow" is 2026-10-04. The page shows a yellow "Sample data" banner whenever it's in use, and the README says so at the top.
9. **The sample generator's patterns are my guesses**, not facts about Simran's business: office tiffins on weekdays, a Sunday paratha rush, festival spikes, about 20% growth over 90 days, 2 closed days, 3 bulk party orders, extra paratha on rainy monsoon days, and gajar halwa rising from mid-September.
10. **Real `orders.csv` beats the sample** automatically, and `orders.csv` is in `.gitignore` so her data can't be committed by accident.
11. **The 1,000-row cap** keeps the most recent days (`days = 1000 // number_of_dishes`) and shows a warning.
12. **A day with zero orders across every dish is "kitchen closed", stored as missing, not 0.** A single dish with 0 on an open day stays a real 0. This was found by testing, not planned. Before the fix, seed 1004 had a closed day right before the held-out window, and TabPFN read "yesterday everything was 0" as demand collapsing. It predicted almost nothing for a week and scored −40% vs the baseline. Real runs before and after:

    | Dataset | Before: baseline / TabPFN MAE | After: baseline / TabPFN MAE |
    |---|---|---|
    | shipped sample | 3.92 / 3.71 (+5.4%) | 3.62 / 3.33 (+7.9%) |
    | seed 1001 | 3.96 / 3.70 (+6.5%) | 3.67 / 3.36 (+8.5%) |
    | seed 1002 | 3.89 / 3.90 (−0.1%) | 3.94 / 3.48 (+11.7%) |
    | seed 1003 | 3.83 / 3.75 (+2.1%) | 3.83 / 3.38 (+11.7%) |
    | seed 1004 | 3.29 / 4.60 (−40.0%) | 3.25 / 2.94 (+9.5%) |

    Both methods' numbers changed, because closed days are no longer scored for either one (nobody needs a forecast for a day the kitchen is shut). Seeds 1001 to 1004 were looked at while fixing this, so the README reports **fresh seeds 2001 to 2005** that were generated only afterwards. The development run is kept in `reports/results_dev_seeds_1001-1004.md`.
13. **Dates** are read as ISO (`2026-09-01`) if at least 90% parse that way, otherwise day-first (`01/09/2026` = 1 September), the usual format in India.
14. **Festival calendar** covers 2025 and 2026 for Chandigarh/Punjab, and lunar dates are best-effort. A note containing festival / party / puja / function / shaadi / holiday / tyohar also marks the day.

## Forecasting
15. **One TabPFN model across all dishes**, with `item_code` as a feature, rather than one model per dish. It gives TabPFN more rows to learn from (shared weekday and festival effects).
16. **Point forecast = the TabPFN median (50th percentile)**, since median minimises absolute error. The 10th and 90th percentiles give the shown range.
17. **Prep = round up(median + ½ × (p90 − median))**: a buffer that grows when TabPFN is less sure. If TabPFN can't run, prep = round up(baseline × 1.1) and the page says TabPFN failed.
18. **Baseline = average of the last 4 same weekdays** (falls back to the 7-day average if there aren't 4 weeks yet).
19. **Backtest = last 28 days, one-day-ahead, TabPFN refit every 7 days** on only the earlier data. A single fit would have been faster but stale by the end of the window. A daily refit would be most faithful but about 4× slower on her CPU.
20. **First 7 days are not used as training targets** (not enough history for the lag features).
21. **The default dataset is forecast in the background at startup** and results are cached by file hash, so the page usually loads instantly. Only one TabPFN run happens at a time, to be gentle on 8 GB of RAM.

## Plan and Gemma
22. **All numbers in the plan are computed in Python** (per-plate grams in `RECIPES`, rounded up to 50 g, shown in kg from 1 kg). An LLM never does the arithmetic.
23. **Gemma is optional and gets only the finished list**, never the order history. Its output is discarded if any number in the plan is missing or changed. Default model `gemma-3-27b-it`, because Gemma 4 model IDs on AI Studio couldn't be checked without a key; change it with `GEMMA_MODEL`.
24. **No LLM runs locally**, per your instructions.

## Page
25. **One HTML file, no external scripts or CDNs** (hand-drawn SVG chart), so it works offline. Mobile-first, light and dark mode, chart with a hover/tap tooltip, a table view, and a per-dish switcher.
26. **Chart colours** come from a palette validated for colour-blind separation: blue = actual, orange = TabPFN, aqua dashed = baseline. The baseline is also dashed, so it isn't identified by colour alone.
27. **Copy is English with light Hinglish** ("Kal kitna banana hai?", "Forecast banao") so it reads naturally for her without being hard to follow for DEV readers.

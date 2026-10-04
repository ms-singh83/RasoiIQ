# Good morning ☕ Here's where RasoiIQ is

## TL;DR
The app is done and works end to end with **one command: `./run.sh`**. It uses TabPFN on CPU with no login, no API keys, and no network once the model is downloaded. On the 90-day sample data, TabPFN beat the same-weekday baseline: **MAE 3.33 vs 3.62 plates (7.9% closer)**, and 13–18% closer on 5 fresh synthetic datasets. All numbers come from real runs (`scripts/evaluate.py`), and all of it is **synthetic sample data**, not Simran's.

## What works (verified tonight)
- [x] **Stack check first, as you asked:** torch 2.2.2 + tabpfn 6.4.1 + numpy 1.26.4 on Python 3.11. A real tiny TabPFN prediction ran on CPU before any app code was written.
- [x] **Every dependency has an Intel-Mac wheel** for Python 3.11 (checked by cross-resolving for `x86_64-apple-darwin`). See `requirements-macos-intel.lock.txt`.
- [x] **TabPFN-2 weights download with no login** (44 MB, one time). **HF_TOKEN is not needed.**
- [x] **One-command start** from a fresh clone: `./run.sh` built the venv, installed everything, downloaded the model and served the page, all of which I checked.
- [x] **Full flow in a real browser** at phone size (390 px), light and dark: page loads → forecast per dish with a range → prep and shopping list → past vs predicted chart → TabPFN vs baseline score. CSV upload and the "festival tomorrow" box also work. No console errors and no sideways scrolling.
- [x] **Works offline:** a forecast ran inside a sandbox with no network at all, using the cached model.
- [x] **35 automated tests pass**, including one with real TabPFN.
- [x] **Sample data is labelled** with a yellow banner on the page and a notice at the top of the README.
- [x] **Data stays under 1,000 rows** (90 days × 5 dishes = 450).
- [x] **No local LLM.** Gemma is optional over the API, only if `GEMMA_API_KEY` is set.

## What doesn't work, or isn't proven
- **Not run on an actual Intel iMac.** It was built and tested on Linux x86_64 with the same versions. Speed on your iMac is unknown. Here, tomorrow's forecast took about 2–7 s and the accuracy check about 8–19 s; expect slower. The page shows a spinner and the real time taken.
- **The Gemma Hinglish rewrite was never called for real** (no key). It's tested with mocks only. Without a key, the page shows a plain English list with a few Hinglish words. If you add a key and it fails, the app quietly falls back to that list.
- **No real data, so no real accuracy number yet.** The synthetic data contains the patterns the features look for, so treat the results as "the pipeline works", not "Simran will save X%".
- **Festival dates for 2025–26 are best-effort.** Lunar dates may be off by a day; edit `app/festivals.py`.

## Decisions I made without you (full list in DECISIONS.md)
- **Simran / Chandigarh / "Build for a Friend"** filled the `[NAME]` / `[CITY]` / `[THEME]` placeholders, taken from what you told me earlier. Change them in `app/config.py`.
- **Sample dishes:** Dal Makhani, Rajma Chawal, Aloo Paratha, Paneer Butter Masala, Gajar Halwa (only in the sample and the recipes).
- **TabPFN-2, not 2.5:** v2 needs no login and allows commercial use with attribution; v2.5 is gated and non-commercial. The page, README and post say **"Built with PriorLabs-TabPFN"**, which the v2 license requires.
- **A day with zero orders for every dish counts as "kitchen closed".** This fixed a real failure (−40% on one dataset). Before/after numbers are in DECISIONS.md, and the README reports seeds generated *after* the fix.
- **Prep = median + half the gap to the 90th percentile**, rounded up.

## What's left for you
1. **On the iMac:**
   ```bash
   brew install python@3.11      # if `python3.11 --version` fails
   git clone https://github.com/ms-singh83/RasoiIQ && cd RasoiIQ
   git checkout claude/demand-forecaster-hacktoberfest-k6kp0q   # until merged
   ./run.sh
   ```
   The first run takes a few minutes to install. Tell me how long the forecast takes and I can tune it if it's slow (for example, set `REFIT_EVERY=14` to halve the accuracy-check time).
2. **Get Simran's real orders** as `orders.csv` (date,item,quantity[,notes]) in the project folder. It's git-ignored, so it won't be committed. Then run `.venv/bin/python scripts/evaluate.py` and paste the real table into the README and the post.
3. **Add her dishes' ingredients** to `RECIPES` in `app/planner.py` so the shopping list covers them (gram amounts per plate).
4. **Optional Gemma:** get a free key at https://aistudio.google.com/apikey and run `GEMMA_API_KEY=... ./run.sh`.
5. **POST_DRAFT.md:** fill in the two `[[ ... ]]` placeholders (her real dishes, and **her real reaction, which I didn't write**), add screenshots from `docs/`, and review the "what I'd add next" list.
6. **Open a PR / merge** the branch `claude/demand-forecaster-hacktoberfest-k6kp0q` when you're happy. I didn't open one.

## If install fails on the Mac
- `No matching distribution for torch==2.2.2`: you're on Python 3.13. Use 3.11 (`brew install python@3.11`, then `rm -rf .venv && ./run.sh`).
- Anything else: `rm -rf .venv && python3.11 -m venv .venv && .venv/bin/pip install -r requirements-macos-intel.lock.txt`, which installs the exact versions I resolved for Intel Macs.
- If the model download is blocked on your network, download `https://storage.googleapis.com/tabpfn-v2-model-files/05152025/tabpfn-v2-regressor.ckpt` and put it in `.tabpfn_cache/`.

## Commits on the branch
- 73fe403 Add README, decisions, blockers, post draft, screenshots and license attribution
- f23b71b Treat all-zero days as kitchen closed; add evaluation script and results
- 7bdd676 Add RasoiIQ app: TabPFN forecast, baseline backtest, prep plan, mobile page
- 89aa94b Pin a CPU-only TabPFN stack that works on Intel macOS
- (this summary)

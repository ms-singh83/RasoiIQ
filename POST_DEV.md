---
title: RasoiIQ: I built my friend a "kal kitna banana hai?" forecaster with TabPFN that runs offline on a laptop
published: false
tags: devchallenge, weekendchallenge, hf26challenge, machinelearning
---

*This is a submission for the [Hacktoberfest Weekend Challenge: Build for a Friend](https://dev.to/challenges/hacktoberfest-weekend-2026-10-01)*

## What I Built

My friend **Simran** runs a home food business out of her kitchen in **Chandigarh**. Every night she asks herself the same question: *kal kitna banana hai?* How much should I cook tomorrow?

She answers it by gut feeling. When she guesses high, food goes to waste and so does her money. When she guesses low, she runs out by lunch and has to turn regulars away. Festivals make it harder still: Karva Chauth, Navratri and Diwali are all coming up, plus the random bulk "party orders" that show up out of nowhere.

So I built **RasoiIQ** (*rasoi* means kitchen). She gives it a CSV of past orders, and it tells her for each dish:

- 🍛 **How many plates to prep tomorrow**, with a likely range ("likely 17 to 35")
- 🛒 **A shopping list** worked out from per-plate ingredients
- 📈 **A chart of the last 4 weeks**: what actually sold vs what the model would have predicted
- ✅ **An honest score**: how far off the model was on days it never saw, compared with plain "same weekday average" guessing

It runs **entirely on a laptop CPU**. No account, no API key, no cloud, no monthly bill, and it works with the Wi-Fi off.

![RasoiIQ on a phone: tomorrow's prep per dish with likely ranges](UPLOAD: docs/screenshot-phone.png)

<!-- PLACEHOLDER: SIMRAN'S REACTION. Bonus points per the challenge rules: "actually hand it over and tell us what they said". Paste what she really said, in her words (Hinglish is great!). Do NOT invent it. If you can't reach her before the deadline, delete this block. -->
> **What Simran said:** [[ her real words here ]]

## Demo

🎥 **Video (90 seconds):** [[ paste YouTube / Loom link, or upload the video to the post ]]

The demo shows: the empty page → uploading an orders CSV → tomorrow's forecast for each dish → ticking "festival or party tomorrow?" → uploading a messy order log → the chart and the accuracy score.

RasoiIQ is **local-first on purpose** (that's the point, see below), so there's no public URL. It runs on your machine with one command:

```bash
git clone https://github.com/ms-singh83/RasoiIQ
cd RasoiIQ
./run.sh        # then open http://localhost:8000
```

![Past vs predicted chart, with tomorrow's forecast and range on the right](UPLOAD: docs/screenshot-chart.png)

## Code

{% github ms-singh83/RasoiIQ %}

- `app/forecast.py`: TabPFN fit/predict, quantiles, rolling backtest
- `app/features.py`: leak-free features (each one only uses data from before the day it predicts)
- `app/data.py`: survives a real-world messy CSV
- `app/static/index.html`: one mobile-first page, no external scripts, so it works offline
- 36 tests, including one real TabPFN prediction on CPU

## How I Built It

### The open model at the core: TabPFN

Simran doesn't have years of sales data. She has a few months at most. Classic ML wants lots of data, a training loop and hyperparameter tuning. Prophet-style time-series tools want long seasonal history.

[TabPFN](https://github.com/PriorLabs/TabPFN), Prior Labs' **tabular foundation model**, flips that around. It was pre-trained on millions of synthetic tables, so you hand it a *small* table and it predicts in a single forward pass. There's no training loop and nothing to tune. A home kitchen with 90 days of orders is exactly what it was built for.

Each row is **one dish on one day**, and the features are things Simran already thinks about:

| Feature | Why it matters in her kitchen |
|---|---|
| `day_of_week`, `is_weekend` | Sunday aloo-paratha rush vs weekday office tiffins |
| `is_festival`, `is_festival_eve` | built-in Chandigarh/Punjab calendar + "party"/"puja" in her notes + a tick box for tomorrow |
| `lag_1`, `lag_7` | yesterday, and the same day last week |
| `mean_7`, `mean_28`, `trend_7` | is a dish catching on or fading? |
| `same_weekday_avg` | the "careful human" guess, given to TabPFN as a hint |
| `item_code`, `day_index` | which dish, and slow word-of-mouth growth |

### The part I'm proudest of: turning uncertainty into a prep number

I don't just ask TabPFN for one number. I ask for its **10th, 50th and 90th percentiles**:

```python
model = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cpu")
model.fit(X_train, y_train)
p10, p50, p90 = model.predict(X_tomorrow, output_type="quantiles",
                              quantiles=[0.1, 0.5, 0.9])
prep = ceil(p50 + 0.5 * (p90 - p50))   # buffer grows with uncertainty
```

A steady dish like dal makhani gets a small buffer. A volatile one, like a new dish that's still catching on, gets a bigger one. That's TabPFN's predictive distribution turned into a decision a home cook can act on.

### Is it actually better than guessing?

Judges hate made-up numbers, and so do I. Every number below comes from `scripts/evaluate.py`.

**Method:** hide the last 28 days and forecast them **one day at a time**, using only what was known the night before. TabPFN is refit every 7 days. The opponent is what a careful person would do by hand: **the average of the same weekday over the last 4 weeks.**

![TabPFN vs baseline accuracy card in the app](UPLOAD: docs/screenshot-accuracy.png)

| Dataset (synthetic) | Baseline MAE (plates) | TabPFN MAE | TabPFN vs baseline |
|---|---|---|---|
| Clean 90-day sample, 5 dishes | 3.62 | **3.33** | 7.9% closer |
| 5 fresh 90-day samples | 3.61–4.34 | **3.05–3.66** | 13–18% closer |
| Messy per-order log, 6 dishes | 3.58 | **3.00** | 16.1% closer |

**Being honest:** I don't have Simran's real order history in the repo yet, so these are on realistic **synthetic** order logs that I generated. They show the pipeline works; her real data will be noisier. On the messy log, most of the gain came from a *new dish* (Chole Bhature), where the weekday average still counted the weeks before it existed (6.76 plates off vs TabPFN's 3.28). TabPFN was actually slightly **worse** on two steady dishes. Small data, honest results.

### The bug that taught me the most

My first version lost to plain guessing by **40%** on one dataset. The cause: the kitchen had been **closed** the day before the test window, so every dish showed 0 orders. TabPFN read "yesterday everyone ordered zero" as demand collapsing, and predicted almost nothing for a week.

The fix was a domain rule, not a model tweak: **a day with zero orders across every dish means the kitchen was shut, not that nobody was hungry.** Those days become missing values, which TabPFN handles natively. After that, TabPFN beat the baseline on every dataset, including five fresh ones generated *after* the fix so I couldn't have tuned to them.

### Real-world messy data

Real order logs are messy, so RasoiIQ copes with:

- one row per customer order (it adds them up)
- `Date/Dish/Qty` instead of `date/item/quantity`
- Indian day-first dates (`04/10/2026`)
- `dal makhani` vs ` DAL MAKHANI ` (merged)
- refunds, blanks, `"two"` typed as a word (skipped, with a friendly warning)

### Making it run on an 8 GB Intel iMac with no GPU

This was the hardware constraint that shaped everything. PyTorch **stopped shipping Intel-Mac builds after 2.2.2**, and newer TabPFN releases need torch ≥ 2.5. Before writing any app code, I:

1. found the newest TabPFN that accepts torch 2.2.2 (**tabpfn 6.4.1**, with numpy < 2),
2. ran a **real tiny TabPFN prediction on CPU** to prove the stack works,
3. cross-resolved every dependency for `x86_64-apple-darwin` to confirm an Intel-Mac wheel exists,
4. kept the table **under 1,000 rows** so TabPFN stays quick on a CPU.

I also chose **TabPFN-2 weights**: they download **without any login**, and their license (Apache 2.0 + attribution) allows use in Simran's business.

**Stack:** Python · FastAPI · TabPFN-2 (CPU) · one HTML page with a hand-drawn SVG chart · pytest. Optional: Gemma via API to rewrite the prep list in Hinglish. It's off by default; when it's on, it gets only the finished list and its output is rejected if it changes a single number.

## Why Does Open Innovation Matter?

For a one-woman home kitchen, a closed forecasting API simply wouldn't work:

- 🔒 **Her sales data stays on her laptop.** Her order history is her customer list, her prices and her busiest days. With an open-weight model running locally, none of it ever leaves her machine.
- 💸 **It costs nothing to run.** No subscription and no per-prediction bill. Even a small monthly fee eats into the margins of a business that sells plates of rajma chawal.
- 📴 **It works offline.** After a one-time 44 MB download, RasoiIQ runs with no internet. I tested this by running a forecast inside a sandbox with **no network at all**. TabPFN's telemetry is switched off too.
- 🔁 **No lock-in.** The model is a file on disk. If a better open model comes out, it's a one-function swap.
- 🧠 **Open made the *right* model possible.** A closed general-purpose LLM is the wrong tool for "how many plates tomorrow". An open, specialised tabular foundation model gave calibrated ranges from 90 rows on an old iMac.

## My Agent Session

I built RasoiIQ with an AI coding agent (Claude Code). I set the constraints: an Intel iMac, CPU only, under 1,000 rows, no paid services, no invented numbers. Then I let it work overnight while logging every decision in [`DECISIONS.md`](https://github.com/ms-singh83/RasoiIQ/blob/claude/demand-forecaster-hacktoberfest-k6kp0q/DECISIONS.md) and every blocker in [`BLOCKERS.md`](https://github.com/ms-singh83/RasoiIQ/blob/claude/demand-forecaster-hacktoberfest-k6kp0q/BLOCKERS.md). The closed-day bug above was caught by its own evaluation runs, and the before/after numbers are in the repo.

<!-- If you saved the session with DevRelay, embed it here with the agent_session tag. Otherwise keep the paragraph above, or link the shared session. -->

## Prize Categories

- **Best Use of TabPFN.** TabPFN is the core of the project: quantile forecasts from a small CSV on CPU, turned into a prep decision, and benchmarked against a baseline with a leak-free rolling backtest.

---

*Built with PriorLabs-TabPFN.* TabPFN-2 weights are used under the [Prior Labs License](https://huggingface.co/Prior-Labs/TabPFN-v2-reg/blob/main/LICENSE.txt).

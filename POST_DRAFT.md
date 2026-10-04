---
title: I built my friend a "how much to cook tomorrow" app with TabPFN
published: false
tags: hacktoberfest, devchallenge, machinelearning, python
---

*This is a submission for the DEV Hacktoberfest 2026 challenge: Build for a Friend (Best Use of TabPFN).*

## Why I built this

My friend Simran runs a home food business in Chandigarh. She cooks [[ HER REAL DISHES, e.g. dal makhani, rajma chawal, aloo paratha ]].
<!-- PLACEHOLDER: replace with what she actually sells and how people order -->

Every night she has to decide how much to prep for tomorrow. She mostly goes by gut feeling. Some days she makes too much and the extra goes to waste. Other days she runs out by lunch and has to say no to people. On a festival or a random Sunday it's even harder to guess.

I wanted to give her something simple: open a page on her phone or laptop and see "tomorrow, make about this much of each dish, and buy this."

## What it does

RasoiIQ takes a CSV of her past orders (`date, item, quantity`) and shows:

- **Tomorrow's forecast for each dish**, with a likely range ("likely 16 to 36")
- **A prep number** with a small buffer that grows when the model is less sure
- **A shopping list** worked out from per-plate ingredients
- **A chart** of the last four weeks: what actually sold, what the model would have predicted, and a simple baseline
- **An honest score**: how far off the model was on days it hadn't seen, compared with a simple rule

<!-- SCREENSHOT: phone view of the forecast card -->
<!-- SCREENSHOT: chart + "Is TabPFN better than guessing?" card -->

## Why TabPFN

Simran doesn't have years of data. She has a few months at best. Most machine learning needs a lot of data, plus training and tuning.

[TabPFN](https://github.com/PriorLabs/TabPFN) from Prior Labs is a foundation model for tables. It was pre-trained on millions of synthetic tables, so you give it a small table and it predicts in one go. There's no training loop and nothing to tune. That was exactly the situation I had.

Each row is one dish on one day. The features are things she already thinks about:

- day of the week and weekend
- festival, or the day before a festival (Diwali, Karva Chauth, Lohri and so on), plus "party" or "festival" written in her notes
- yesterday's orders, the same day last week, the average of the last week and the last month, and whether orders are going up or down

The part I liked most is that TabPFN doesn't just give one number. I ask it for the 10th, 50th and 90th percentiles. The middle one is the forecast, and the gap tells me how unsure it is. The prep number is the middle guess plus half of the gap to the "busy day" number, so a steady dish gets a small buffer and an unpredictable one gets a bigger one.

## Does it actually beat guessing?

I compared it with what a careful person would do by hand: **take the average of the same weekday over the last four weeks.** I hid the last 28 days and had both methods predict them one day at a time, using only what was known the night before.

I don't have Simran's real data in the repo, so I tested on synthetic order data that I generated to look like a home kitchen (weekday office tiffins, a Sunday paratha rush, festival spikes, a couple of days the kitchen was shut, some bulk party orders).

| Dataset | Baseline (plates off per dish per day) | TabPFN | Difference |
|---|---|---|---|
| Sample shipped with the app | 3.62 | 3.33 | 7.9% closer |
| 5 more synthetic datasets | 3.61 to 4.34 | 3.05 to 3.66 | 13% to 18% closer |

All of these numbers come from the evaluation script in the repo.

To be honest about it: this is data I generated myself, so it has the patterns my features look for. It shows the pipeline works, not that Simran will see exactly this. I'll run it on her real orders next.

One mistake was worth sharing. My first version did *worse* than the simple rule on one dataset, by 40%. The kitchen had been closed the day before the test window, so every dish showed 0 orders, and the model read that as "nobody wants food any more." The fix was to treat a day with zero orders across every dish as "closed" and leave it out, rather than count it as zero demand. After that, it beat the baseline on every dataset I tried.

## Why open matters here

This was a big deal for me, because it's my friend's business data:

- **Her sales data stays on her laptop.** The forecast runs on her own CPU. Nothing gets uploaded.
- **It costs nothing to run.** No subscription and no API bill per prediction. The TabPFN-2 weights are free to use, including for her business, with an attribution line.
- **It works offline after setup.** The 44 MB model downloads once, without any login, and after that it runs with the Wi-Fi off. I tested this with no network at all.
- **It isn't locked to one company.** The model is a file on disk. If something better comes out, I can swap it in.

## Making it work on an old Intel iMac

My laptop for this is an Intel iMac with 8 GB of RAM and no GPU. That made it harder than it sounds. PyTorch stopped shipping Intel Mac builds after version 2.2.2, and newer TabPFN releases need a newer PyTorch. I ended up pinning torch 2.2.2, tabpfn 6.4.1 and numpy 1.26, and I checked that every package has an Intel Mac build before writing any app code. I also keep the data under 1,000 rows so TabPFN stays quick on a CPU.

## Simran's reaction

<!-- PLACEHOLDER: Add Simran's real reaction here, in her own words, after she has tried it. Don't publish without it, and don't make one up. -->
[[ SIMRAN'S REAL REACTION GOES HERE ]]

## Try it

```bash
git clone https://github.com/ms-singh83/RasoiIQ
cd RasoiIQ
./run.sh
```

Then open http://localhost:8000. It starts with sample data. Put your own `orders.csv` in the folder to use real orders.

## What I'd add next

<!-- Edit this list to match what you actually plan -->
- Run it on Simran's real order history and share the real numbers
- A way to log tomorrow's actual orders from the page, so the data grows without editing a CSV
- Better festival dates for future years

Built with PriorLabs-TabPFN.

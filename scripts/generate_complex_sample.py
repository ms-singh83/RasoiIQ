"""Generate a messier SAMPLE orders CSV, closer to what a real order log looks like.

Made-up data for demos and tests, not anyone's real sales. Compared with
generate_sample.py (one clean row per dish per day), this file has:

- one row per customer order (several rows per dish per day)
- different column names: Date, Dish, Qty, Notes
- day-first dates (04/10/2026 = 4 October)
- inconsistent dish spelling and spacing ("dal makhani", " DAL MAKHANI ")
- a dish added partway through (Chole Bhature from 1 September)
- two closed days with no rows at all
- bulk party orders and festival notes, some with commas inside quotes
- a few broken rows (blank qty, "two", a -2 refund, a missing dish)
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.festivals import FESTIVALS  # noqa: E402

# dish: (base plates/day, Mon..Sun multipliers, spellings seen in the log)
DISHES = {
    "Dal Makhani": (9, [1.0, 1.0, 1.0, 1.0, 1.1, 1.35, 1.45], ["Dal Makhani", "dal makhani", " DAL MAKHANI "]),
    "Rajma Chawal": (11, [1.25, 1.2, 1.2, 1.15, 1.1, 0.7, 0.6], ["Rajma Chawal", "rajma chawal", "Rajma chawal"]),
    "Aloo Paratha": (13, [0.9, 0.85, 0.85, 0.85, 0.9, 1.3, 1.7], ["Aloo Paratha", "aloo paratha", "ALOO PARATHA"]),
    "Paneer Butter Masala": (7, [0.85, 0.85, 0.9, 0.9, 1.1, 1.45, 1.5], ["Paneer Butter Masala", "paneer butter masala"]),
    "Kadhi Chawal": (6, [1.1, 1.1, 1.1, 1.1, 1.0, 0.8, 0.8], ["Kadhi Chawal", "kadhi chawal"]),
    "Chole Bhature": (8, [0.7, 0.7, 0.7, 0.8, 1.0, 1.6, 1.9], ["Chole Bhature", "chole bhature"]),
}
NEW_DISH, NEW_DISH_FROM = "Chole Bhature", date(2026, 9, 1)
FESTIVAL_LIFT = {"Dal Makhani": 1.6, "Rajma Chawal": 1.2, "Aloo Paratha": 1.3,
                 "Paneer Butter Masala": 1.9, "Kadhi Chawal": 1.1, "Chole Bhature": 1.7}
CUSTOMER_NOTES = ["", "", "", "", "less spicy", "extra butter", "pickup 1pm", "office tiffin",
                  "no onion, no garlic", "repeat customer", "deliver to Sector 22"]


def split_into_orders(plates: int, rng: np.random.Generator) -> list[int]:
    """Break a day's plates for one dish into individual orders of 1-4 plates."""
    orders = []
    while plates > 0:
        q = int(min(plates, rng.choice([1, 1, 2, 2, 2, 3, 4])))
        orders.append(q)
        plates -= q
    return orders


def generate(end: date, days: int, seed: int) -> list[list[str]]:
    rng = np.random.default_rng(seed)
    start = end - timedelta(days=days - 1)
    closed = {start + timedelta(days=int(d)) for d in rng.choice(np.arange(10, days - 10), 2, replace=False)}
    party = {start + timedelta(days=int(d)) for d in rng.choice(np.arange(5, days - 1), 4, replace=False)}
    rows: list[list[str]] = []
    for i in range(days):
        d = start + timedelta(days=i)
        if d in closed:
            continue  # nothing logged at all that day
        growth = 1.0 + 0.2 * i / days
        fest, eve = FESTIVALS.get(d), FESTIVALS.get(d + timedelta(days=1))
        party_dish = rng.choice(list(DISHES)) if d in party else None
        stamp = d.strftime("%d/%m/%Y")
        for dish, (base, dow, spellings) in DISHES.items():
            if dish == NEW_DISH and d < NEW_DISH_FROM:
                continue
            mu = base * dow[d.weekday()] * growth
            if dish == NEW_DISH:  # word spreads over the first few weeks
                mu *= min(1.0, 0.4 + (d - NEW_DISH_FROM).days / 30)
            if fest:
                mu *= FESTIVAL_LIFT[dish]
            elif eve:
                mu *= 1 + (FESTIVAL_LIFT[dish] - 1) * 0.4
            for q in split_into_orders(int(rng.poisson(mu)), rng):
                note = fest if fest and rng.random() < 0.5 else str(rng.choice(CUSTOMER_NOTES))
                rows.append([stamp, str(rng.choice(spellings)), str(q), note])
            if dish == party_dish:
                extra = int(rng.integers(12, 25))
                rows.append([stamp, dish, str(extra), f"party order, {extra} plates, kitty party"])
    # a handful of rows a real log would have
    rows += [
        [(end - timedelta(days=20)).strftime("%d/%m/%Y"), "Dal Makhani", "", "forgot to write qty"],
        [(end - timedelta(days=15)).strftime("%d/%m/%Y"), "Rajma Chawal", "two", "typed in words"],
        [(end - timedelta(days=9)).strftime("%d/%m/%Y"), "Paneer Butter Masala", "-2", "refund, cold delivery"],
        [(end - timedelta(days=4)).strftime("%d/%m/%Y"), "", "3", "dish not written"],
    ]
    rows.sort(key=lambda r: (r[0][6:], r[0][3:5], r[0][:2]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--end", default="2026-10-04")
    ap.add_argument("--days", type=int, default=75)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "sample_orders_complex.csv"))
    a = ap.parse_args()
    rows = generate(date.fromisoformat(a.end), a.days, a.seed)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Date", "Dish", "Qty", "Notes"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} order rows over {a.days} days of SAMPLE data to {a.out}")


if __name__ == "__main__":
    main()

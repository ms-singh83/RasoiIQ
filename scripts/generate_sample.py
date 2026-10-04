"""Generate a SAMPLE orders CSV for a Chandigarh home kitchen.

This is made-up data for demos and tests, not anyone's real sales. The patterns
are plausible guesses: office-tiffin weekdays, Sunday paratha rush, festival
spikes, slow word-of-mouth growth, a couple of closed days, bulk party orders,
and gajar halwa picking up as the weather cools.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.festivals import FESTIVALS  # noqa: E402

# item: (base plates per day, Mon..Sun multipliers)
ITEMS = {
    "Dal Makhani": (9, [1.0, 1.0, 1.0, 1.0, 1.1, 1.35, 1.45]),
    "Rajma Chawal": (11, [1.25, 1.2, 1.2, 1.15, 1.1, 0.7, 0.6]),
    "Aloo Paratha": (14, [0.9, 0.85, 0.85, 0.85, 0.9, 1.3, 1.7]),
    "Paneer Butter Masala": (7, [0.85, 0.85, 0.9, 0.9, 1.1, 1.45, 1.5]),
    "Gajar Halwa": (3, [0.9, 0.9, 0.9, 0.9, 1.0, 1.3, 1.4]),
}
FESTIVAL_LIFT = {"Dal Makhani": 1.6, "Rajma Chawal": 1.2, "Aloo Paratha": 1.3,
                 "Paneer Butter Masala": 1.9, "Gajar Halwa": 2.5}


def generate(end: date, days: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = end - timedelta(days=days - 1)
    closed = {start + timedelta(days=int(d)) for d in rng.choice(np.arange(15, days - 10), 2, replace=False)}
    party = {start + timedelta(days=int(d)) for d in rng.choice(np.arange(5, days), 3, replace=False)}
    rows = []
    for i in range(days):
        d = start + timedelta(days=i)
        growth = 1.0 + 0.2 * i / days
        fest, eve = FESTIVALS.get(d), FESTIVALS.get(d + timedelta(days=1))
        rainy = d.month in (7, 8) and rng.random() < 0.3
        party_item = rng.choice(list(ITEMS)) if d in party else None
        for item, (base, dow) in ITEMS.items():
            if d in closed:
                rows.append((d, item, 0, "closed - kitchen off"))
                continue
            mu = base * dow[d.weekday()] * growth
            if item == "Gajar Halwa":  # carrots get good from mid-September
                mu *= 0.5 if d.month in (6, 7, 8) else 1.0 + max(0, (d - date(d.year, 9, 15)).days) / 15
            if rainy and item == "Aloo Paratha":
                mu *= 1.4
            note = ""
            if fest:
                mu *= FESTIVAL_LIFT[item]
                note = fest
            elif eve:
                mu *= 1 + (FESTIVAL_LIFT[item] - 1) * 0.4
            q = int(rng.poisson(mu))
            if item == party_item:
                extra = int(rng.integers(12, 25))
                q += extra
                note = f"party order {extra} plates"
            rows.append((d.isoformat(), item, q, note))
    return pd.DataFrame(rows, columns=["date", "item", "quantity", "notes"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--end", default="2026-10-03")
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "sample_orders.csv"))
    a = ap.parse_args()
    df = generate(date.fromisoformat(a.end), a.days, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"Wrote {len(df)} rows ({a.days} days x {len(ITEMS)} items) of SAMPLE data to {a.out}")


if __name__ == "__main__":
    main()

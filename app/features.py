"""Turn daily order history into a feature table for TabPFN.

Every feature for day D uses only orders from days before D, so the same code
builds honest backtest rows and the row for tomorrow.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from .festivals import festival_name

FEATURES = [
    "item_code",        # which dish
    "day_of_week",      # 0 = Monday
    "is_weekend",
    "is_festival",      # calendar festival or notes say party/festival
    "is_festival_eve",  # day before a festival
    "lag_1",            # yesterday's orders
    "lag_7",            # same day last week
    "mean_7",           # average of the last 7 days
    "mean_28",          # average of the last 28 days
    "trend_7",          # last 7 days minus the 7 days before that
    "same_weekday_avg", # average of the last 4 same weekdays (also the baseline)
    "day_index",        # days since the start of the data (slow growth)
]
WARMUP_DAYS = 7


def _calendar(d: date, special: set[date]) -> tuple[int, int, int, int]:
    nxt = d + timedelta(days=1)
    return (
        d.weekday(),
        int(d.weekday() >= 5),
        int(festival_name(d) is not None or d in special),
        int(festival_name(nxt) is not None or nxt in special),
    )


def build_features(daily: pd.DataFrame, items: list[str], special_dates: set[date] | None = None,
                   extra_days: int = 0) -> pd.DataFrame:
    """Feature rows for every (date, item), plus `extra_days` future days with quantity NaN."""
    special = special_dates or set()
    wide = daily.pivot(index="date", columns="item", values="quantity").reindex(columns=items).astype(float)
    if extra_days:
        future = pd.date_range(wide.index.max() + pd.Timedelta(days=1), periods=extra_days, freq="D")
        wide = pd.concat([wide, pd.DataFrame(np.nan, index=future, columns=items)])
    start = wide.index.min()

    frames = []
    for code, item in enumerate(items):
        s = wide[item]
        past = s.shift(1)
        mean_7 = past.rolling(7, min_periods=1).mean()
        prev_7 = s.shift(8).rolling(7, min_periods=1).mean()
        same_wd = pd.concat([s.shift(7 * k) for k in (1, 2, 3, 4)], axis=1).mean(axis=1)
        frames.append(pd.DataFrame({
            "date": s.index, "item": item, "quantity": s.to_numpy(), "item_code": code,
            "lag_1": past.to_numpy(), "lag_7": s.shift(7).to_numpy(),
            "mean_7": mean_7.to_numpy(), "mean_28": past.rolling(28, min_periods=1).mean().to_numpy(),
            "trend_7": (mean_7 - prev_7).to_numpy(), "same_weekday_avg": same_wd.to_numpy(),
            "day_index": (s.index - start).days,
        }))
    feats = pd.concat(frames, ignore_index=True)
    cal = np.array([_calendar(d.date(), special) for d in feats["date"]])
    feats["day_of_week"], feats["is_weekend"], feats["is_festival"], feats["is_festival_eve"] = cal.T
    return feats[["date", "item", "quantity", *FEATURES]]


def training_rows(feats: pd.DataFrame, before: pd.Timestamp | None = None) -> pd.DataFrame:
    """Known-target rows with at least a week of history (and optionally before a date)."""
    start = feats["date"].min() + pd.Timedelta(days=WARMUP_DAYS)
    mask = feats["quantity"].notna() & (feats["date"] >= start)
    if before is not None:
        mask &= feats["date"] < before
    return feats[mask]


def baseline_prediction(feats: pd.DataFrame) -> np.ndarray:
    """Naive baseline: average of the same weekday over the last 4 weeks."""
    return feats["same_weekday_avg"].fillna(feats["mean_7"]).fillna(0.0).to_numpy(dtype=float)

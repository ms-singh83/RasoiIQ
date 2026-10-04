from datetime import date

import numpy as np
import pandas as pd

from app.data import load_orders
from app.features import FEATURES, baseline_prediction, build_features, training_rows
from tests.conftest import make_csv


def _feats(**kw):
    d = load_orders(make_csv(days=42))
    return d, build_features(d.daily, d.items, **kw)


def test_columns_and_shape():
    d, f = _feats(extra_days=1)
    assert list(f.columns) == ["date", "item", "quantity", *FEATURES]
    assert len(f) == 43 * 2
    assert f[f["date"] == f["date"].max()]["quantity"].isna().all()


def test_lags_and_same_weekday_average():
    d, f = _feats()
    dal = d.daily[d.daily["item"] == "Dal"].set_index("date")["quantity"]
    row = f[(f["item"] == "Dal") & (f["date"] == pd.Timestamp("2026-02-10"))].iloc[0]
    assert row["lag_1"] == dal[pd.Timestamp("2026-02-09")]
    assert row["lag_7"] == dal[pd.Timestamp("2026-02-03")]
    expected = np.mean([dal[pd.Timestamp("2026-02-10") - pd.Timedelta(days=7 * k)] for k in (1, 2, 3, 4)])
    assert row["same_weekday_avg"] == expected
    assert baseline_prediction(f[f.index == row.name])[0] == expected


def test_no_leakage_from_the_target_day_or_later():
    d, f = _feats()
    target = pd.Timestamp("2026-02-01")
    changed = d.daily.copy()
    changed.loc[changed["date"] >= target, "quantity"] = 999
    f2 = build_features(changed, d.items)
    cols = [c for c in FEATURES]
    a = f[f["date"] <= target][cols].reset_index(drop=True)
    b = f2[f2["date"] <= target][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_calendar_flags():
    d = load_orders(make_csv(days=30, start="2026-08-01"))
    f = build_features(d.daily, d.items, special_dates={date(2026, 8, 20)})
    by_day = f[f["item"] == "Dal"].set_index("date")
    assert by_day.loc["2026-08-15", "is_festival"] == 1        # Independence Day
    assert by_day.loc["2026-08-14", "is_festival_eve"] == 1
    assert by_day.loc["2026-08-20", "is_festival"] == 1        # marked via notes
    assert by_day.loc["2026-08-16", "is_weekend"] == 1         # Sunday
    assert by_day.loc["2026-08-17", "is_weekend"] == 0


def test_training_rows_skip_warmup_and_future():
    d, f = _feats(extra_days=1)
    t = training_rows(f)
    assert t["quantity"].notna().all()
    assert t["date"].min() == f["date"].min() + pd.Timedelta(days=7)
    assert (training_rows(f, before=pd.Timestamp("2026-02-01"))["date"] < "2026-02-01").all()

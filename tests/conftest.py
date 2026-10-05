import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("RASOIIQ_WARMUP", "0")
os.environ.pop("GEMMA_API_KEY", None)

from app.features import FEATURES  # noqa: E402


def make_csv(days=60, items=("Dal", "Rajma"), start="2026-01-05", seed=0) -> str:
    rng = np.random.default_rng(seed)
    rows = ["date,item,quantity,notes"]
    for i, d in enumerate(pd.date_range(start, periods=days, freq="D")):
        for j, it in enumerate(items):
            q = 10 + 5 * (d.weekday() >= 5) + j * 3 + int(rng.integers(0, 3))
            rows.append(f"{d.date()},{it},{q},")
    return "\n".join(rows) + "\n"


class FakeRegressor:
    """Stands in for TabPFN: predicts the same-weekday average, records what it saw."""

    fits: list = []

    def fit(self, X, y):
        FakeRegressor.fits.append((np.asarray(X).copy(), np.asarray(y).copy()))
        return self

    def predict(self, X, output_type="mean", quantiles=None):
        col = np.asarray(X)[:, FEATURES.index("same_weekday_avg")]
        mid = np.nan_to_num(col, nan=5.0)
        if output_type == "quantiles":
            return [mid - 2, mid, mid + 4]
        return mid


@pytest.fixture
def fake_factory():
    FakeRegressor.fits = []
    return FakeRegressor

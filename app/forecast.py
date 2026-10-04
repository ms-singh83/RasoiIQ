"""Tomorrow's forecast with TabPFN, checked against a same-weekday baseline."""
from __future__ import annotations

import logging
import warnings
import math
import os
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from .config import MODEL_CACHE
from .data import OrdersData
from .features import FEATURES, baseline_prediction, build_features, training_rows

log = logging.getLogger("rasoiiq.forecast")
# We keep the table under 1,000 rows on purpose; the CPU speed hint is noise here.
warnings.filterwarnings("ignore", message="Running on CPU with more than")

QUANTILES = [0.1, 0.5, 0.9]
HOLDOUT_DAYS = int(os.getenv("HOLDOUT_DAYS", "28"))
REFIT_EVERY = int(os.getenv("REFIT_EVERY", "7"))


class Regressor(Protocol):
    def fit(self, X, y): ...
    def predict(self, X, output_type: str = "mean", quantiles: list[float] | None = None): ...


def make_tabpfn() -> Regressor:
    """TabPFN-2 regressor on CPU. Weights download once (no login), then it runs offline."""
    os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", str(MODEL_CACHE))
    from tabpfn import TabPFNRegressor  # torch is heavy; import on first use
    from tabpfn.constants import ModelVersion

    return TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cpu", random_state=0)


@dataclass
class QuantileForecast:
    p10: np.ndarray
    p50: np.ndarray
    p90: np.ndarray


class TabPFNForecaster:
    """Fit TabPFN on feature rows and return the 10th / 50th / 90th percentile."""

    name = "TabPFN-2 (local CPU)"

    def __init__(self, factory: Callable[[], Regressor] = make_tabpfn):
        self._factory = factory

    def fit_predict(self, train: pd.DataFrame, test: pd.DataFrame) -> QuantileForecast:
        model = self._factory()
        model.fit(train[FEATURES].to_numpy(dtype=float), train["quantity"].to_numpy(dtype=float))
        qs = model.predict(test[FEATURES].to_numpy(dtype=float), output_type="quantiles", quantiles=QUANTILES)
        p10, p50, p90 = (np.clip(np.asarray(q, dtype=float), 0, None) for q in qs)
        return QuantileForecast(p10=np.minimum(p10, p50), p50=p50, p90=np.maximum(p90, p50))


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(y - p)))


def wape(y: np.ndarray, p: np.ndarray) -> float:
    """Weighted absolute percentage error: plates we got wrong / plates actually sold."""
    total = float(np.sum(np.abs(y)))
    return float(np.sum(np.abs(y - p)) / total * 100) if total else float("nan")


def prep_target(p50: float, p90: float) -> int:
    """Prep a little above the middle guess: halfway to the busy-day (90%) number."""
    return int(math.ceil(p50 + 0.5 * (p90 - p50) - 1e-9))


def backtest(feats: pd.DataFrame, forecaster: TabPFNForecaster | None,
             holdout_days: int = HOLDOUT_DAYS, refit_every: int = REFIT_EVERY) -> dict:
    """Rolling one-day-ahead backtest over the last `holdout_days` known days.

    TabPFN is refit every `refit_every` days on all history before that block,
    and each day's features only use orders up to the previous day, the same
    information Simran would have the night before.
    """
    known = feats[feats["quantity"].notna()]
    last = known["date"].max()
    cutoff = last - pd.Timedelta(days=holdout_days - 1)
    test = known[known["date"] >= cutoff].sort_values(["date", "item_code"])
    y = test["quantity"].to_numpy(dtype=float)
    base = baseline_prediction(test)
    rows = test[["date", "item"]].assign(actual=y, baseline=base)

    result = {
        "holdout_days": holdout_days,
        "refit_every_days": refit_every,
        "start": cutoff.date().isoformat(),
        "end": last.date().isoformat(),
        "n_predictions": int(len(test)),
        "baseline": {"mae": mae(y, base), "wape": wape(y, base)},
        "tabpfn": None,
    }
    if forecaster is not None:
        preds = []
        block_start = cutoff
        while block_start <= last:
            block_end = block_start + pd.Timedelta(days=refit_every)
            block = test[(test["date"] >= block_start) & (test["date"] < block_end)]
            q = forecaster.fit_predict(training_rows(feats, before=block_start), block)
            preds.append(pd.Series(q.p50, index=block.index))
            block_start = block_end
        tab = pd.concat(preds).reindex(test.index).to_numpy()
        rows = rows.assign(tabpfn=tab)
        result["tabpfn"] = {"mae": mae(y, tab), "wape": wape(y, tab)}

    per_item = {}
    for item, g in rows.groupby("item", sort=False):
        entry = {"actual_avg": float(g["actual"].mean()), "baseline_mae": mae(g["actual"].to_numpy(), g["baseline"].to_numpy())}
        if "tabpfn" in g:
            entry["tabpfn_mae"] = mae(g["actual"].to_numpy(), g["tabpfn"].to_numpy())
        per_item[item] = entry
    result["per_item"] = per_item
    result["rows"] = rows
    return result


@dataclass
class ItemForecast:
    item: str
    baseline: float
    p10: float | None
    p50: float | None
    p90: float | None
    prep: int
    last_week_same_day: float | None


@dataclass
class ForecastResult:
    target_date: date
    items: list[ItemForecast]
    model_used: str
    model_note: str | None
    backtest: dict | None
    chart: dict
    timings: dict = field(default_factory=dict)


def run_forecast(data: OrdersData, tomorrow_special: bool = False,
                 forecaster: TabPFNForecaster | None = None, with_backtest: bool = True) -> ForecastResult:
    target = data.end + timedelta(days=1)
    special = set(data.special_dates) | ({target} if tomorrow_special else set())
    feats = build_features(data.daily, data.items, special, extra_days=1)
    future = feats[feats["date"] == pd.Timestamp(target)].sort_values("item_code")
    base = baseline_prediction(future)
    forecaster = forecaster or TabPFNForecaster()
    timings: dict[str, float] = {}

    model_used, note, q = forecaster.name, None, None
    try:
        t0 = time.perf_counter()
        q = forecaster.fit_predict(training_rows(feats), future)
        timings["forecast_s"] = round(time.perf_counter() - t0, 2)
    except Exception as exc:  # weights not downloaded yet and offline, etc.
        log.exception("TabPFN failed; falling back to the baseline")
        model_used = "Same-weekday baseline (TabPFN failed)"
        note = f"TabPFN could not run ({type(exc).__name__}: {str(exc)[:200]}). Showing the baseline instead."
        forecaster = None

    bt = None
    if with_backtest:
        t0 = time.perf_counter()
        bt = backtest(feats, forecaster)
        timings["backtest_s"] = round(time.perf_counter() - t0, 2)

    items = []
    for i, item in enumerate(future["item"]):
        lw = data.daily[(data.daily["item"] == item) & (data.daily["date"] == pd.Timestamp(target - timedelta(days=7)))]
        b = float(base[i])
        if q is not None:
            p10, p50, p90 = float(q.p10[i]), float(q.p50[i]), float(q.p90[i])
            prep = prep_target(p50, p90)
        else:
            p10 = p50 = p90 = None
            prep = int(math.ceil(b * 1.1 - 1e-9))
        items.append(ItemForecast(item, b, p10, p50, p90, prep,
                                  float(lw["quantity"].iloc[0]) if len(lw) else None))

    return ForecastResult(target, items, model_used, note, bt, _chart(feats, bt, items, target), timings)


def _chart(feats: pd.DataFrame, bt: dict | None, items: list[ItemForecast], target: date) -> dict:
    """Last 28 days of actual orders, backtest predictions, and tomorrow (per item)."""
    known = feats[feats["quantity"].notna()]
    dates = sorted(known["date"].unique())[-28:]
    rows = bt["rows"].set_index(["date", "item"]) if bt else None
    series = {}
    for it in items:
        actual = known[known["item"] == it.item].set_index("date")["quantity"]
        entry = {"actual": [float(actual.get(d, 0.0)) for d in dates]}
        if rows is not None:
            for col in ("baseline", "tabpfn"):
                if col in rows:
                    entry[col] = [_num(rows[col].get((d, it.item))) for d in dates]
        entry["tomorrow"] = it.p50 if it.p50 is not None else it.baseline
        entry["tomorrow_low"], entry["tomorrow_high"] = it.p10, it.p90
        series[it.item] = entry
    return {"dates": [pd.Timestamp(d).date().isoformat() for d in dates],
            "target_date": target.isoformat(), "series": series}


def _num(v) -> float | None:
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)

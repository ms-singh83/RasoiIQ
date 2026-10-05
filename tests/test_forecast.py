import numpy as np
import pandas as pd
import pytest

from app.data import load_orders
from app.features import FEATURES
from app.forecast import TabPFNForecaster, backtest, mae, prep_target, run_forecast, wape
from app.features import build_features
from tests.conftest import make_csv


def test_metrics():
    y, p = np.array([10.0, 20.0]), np.array([12.0, 17.0])
    assert mae(y, p) == 2.5
    assert wape(y, p) == pytest.approx(5 / 30 * 100)


def test_prep_target_rounds_up_between_median_and_p90():
    assert prep_target(10.0, 14.0) == 12
    assert prep_target(10.2, 10.2) == 11
    assert prep_target(10.0, 10.0) == 10


def test_run_forecast_with_fake_model(fake_factory):
    d = load_orders(make_csv(days=60))
    r = run_forecast(d, forecaster=TabPFNForecaster(factory=fake_factory))
    assert str(r.target_date) == "2026-03-06"
    assert [i.item for i in r.items] == ["Dal", "Rajma"]
    for it in r.items:
        assert it.p10 <= it.p50 <= it.p90
        assert it.prep == prep_target(it.p50, it.p90)
    assert r.backtest["tabpfn"] is not None and r.model_note is None
    assert len(r.chart["dates"]) == 28 and len(r.chart["series"]["Dal"]["tabpfn"]) == 28


def test_backtest_refits_weekly_without_peeking(fake_factory):
    d = load_orders(make_csv(days=60))
    f = build_features(d.daily, d.items)
    bt = backtest(f, TabPFNForecaster(factory=fake_factory), holdout_days=28, refit_every=7)
    assert len(fake_factory.fits) == 4
    day_idx = FEATURES.index("day_index")
    cutoff_index = (pd.Timestamp(bt["start"]) - f["date"].min()).days
    for k, (X, _) in enumerate(fake_factory.fits):
        assert X[:, day_idx].max() < cutoff_index + 7 * k  # trained only on days before its block
    assert bt["n_predictions"] == 56


def test_falls_back_to_baseline_when_tabpfn_fails():
    def broken():
        raise RuntimeError("no weights and no internet")

    d = load_orders(make_csv(days=60))
    r = run_forecast(d, forecaster=TabPFNForecaster(factory=broken))
    assert "TabPFN could not run" in r.model_note
    assert r.backtest["tabpfn"] is None
    assert all(i.p50 is None and i.prep >= i.baseline for i in r.items)


def test_tomorrow_special_sets_festival_flag(fake_factory):
    d = load_orders(make_csv(days=60))
    seen = {}

    class Spy(fake_factory):
        def predict(self, X, output_type="mean", quantiles=None):
            seen["X"] = np.asarray(X)
            return super().predict(X, output_type, quantiles)

    run_forecast(d, tomorrow_special=True, forecaster=TabPFNForecaster(factory=Spy), with_backtest=False)
    assert (seen["X"][:, FEATURES.index("is_festival")] == 1).all()

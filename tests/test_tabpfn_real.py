"""Real TabPFN on CPU (downloads the 44 MB model on first run). Skip with RASOIIQ_SKIP_REAL=1."""
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.getenv("RASOIIQ_SKIP_REAL") == "1", reason="real model test skipped")


def test_tiny_real_prediction():
    from app.forecast import make_tabpfn

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    y = 2 * X[:, 0] - X[:, 1] + rng.normal(scale=0.1, size=60)
    m = make_tabpfn()
    m.fit(X[:50], y[:50])
    p = m.predict(X[50:])
    assert np.mean(np.abs(p - y[50:])) < 0.5
    q10, q50, q90 = m.predict(X[50:], output_type="quantiles", quantiles=[0.1, 0.5, 0.9])
    assert np.all(q10 <= q50) and np.all(q50 <= q90)

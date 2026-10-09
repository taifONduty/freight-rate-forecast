import numpy as np
import pandas as pd
import pytest

from ratecast.intervals import interval, leave_one_fold_out_coverage, residual_quantiles


@pytest.fixture
def predictions() -> pd.DataFrame:
    """Out-of-time predictions whose second forecast month is five times noisier than the first."""
    rng = np.random.default_rng(0)
    n = 6000
    month = rng.integers(1, 3, n)
    fold = np.repeat(["a", "b", "c"], n // 3)
    start = pd.Series(fold).map({"a": "2025-01-01", "b": "2025-03-01", "c": "2025-05-01"})
    predicted = rng.uniform(200, 5000, n)
    return pd.DataFrame({
        "fold": fold,
        "date": pd.to_datetime(start) + pd.to_timedelta((month - 1) * 31 + rng.integers(0, 28, n), unit="D"),
        "horizon_month": month,
        "predicted": predicted,
        "actual": predicted * np.exp(rng.normal(0, np.where(month == 1, 0.01, 0.05))),
        "clean": True,
    })


def test_interval_width_follows_the_forecast_month(predictions):
    month = predictions["horizon_month"]
    lower, upper = interval(predictions["predicted"].to_numpy(), month, residual_quantiles(predictions))
    inside = (predictions["actual"] >= lower) & (predictions["actual"] <= upper)
    assert inside[month == 1].mean() == pytest.approx(0.9, abs=0.02)
    assert inside[month == 2].mean() == pytest.approx(0.9, abs=0.02)


def test_intervals_hold_their_coverage_out_of_fold(predictions):
    assert leave_one_fold_out_coverage(predictions, coverage=0.9).between(0.87, 0.93).all()

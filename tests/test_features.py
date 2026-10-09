import numpy as np
import pandas as pd
import pytest

from ratecast.features import build_features, daily_market_index, days_to_quarter_end


def test_days_to_quarter_end():
    dates = pd.Series(pd.to_datetime(["2025-10-01", "2025-11-01", "2025-12-31", "2025-03-30"]))
    assert days_to_quarter_end(dates).tolist() == [91, 60, 0, 1]


def test_quarter_end_ramp_is_zero_until_the_last_month(loads):
    features = build_features(loads, daily_market_index(loads))
    outside = features["days_to_quarter_end"] >= 30
    last_day = features["days_to_quarter_end"] == 0
    not_flatbed = features["equipment"] != "Flatbed"
    assert outside.any() and last_day.any() and not_flatbed.any()
    assert (features.loc[outside, "ramp_30"] == 0).all()
    assert (features.loc[last_day, "ramp_7"] == 1).all()
    assert (features.loc[not_flatbed, "ramp_30_flatbed"] == 0).all()


def test_daily_index_averages_loads_and_extends_history(loads):
    history = daily_market_index(loads)
    later = pd.DataFrame({
        "date": pd.to_datetime([history.index.max() + pd.DateOffset(days=1)] * 3),
        "market_index": [1.0, np.nan, 1.2],
    })
    extended = daily_market_index(later, history)
    assert extended.iloc[-1] == pytest.approx(1.1)
    assert extended.iloc[:-1].equals(history)

    gap = later.assign(date=pd.to_datetime([history.index.max() + pd.DateOffset(days=2)] * 3))
    with pytest.raises(ValueError, match="no market index"):
        daily_market_index(gap, history)


def test_smoothed_index_ignores_future_days(loads):
    index = daily_market_index(loads)
    shocked = index.copy()
    shocked.iloc[-10:] *= 1.5
    before = loads["date"] < index.index[-10]
    original = build_features(loads[before], index)["log_market_smooth"]
    assert build_features(loads[before], shocked)["log_market_smooth"].equals(original)

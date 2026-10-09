import numpy as np
import pandas as pd
import pytest

from ratecast.features import build_features, daily_market_index
from ratecast.labels import flag_corrupted_labels
from ratecast.model import LaneEffects, RateModel


def test_planted_label_corruption_is_flagged(loads):
    rates = loads["posted_rate"].to_numpy().copy()
    rates[:20] *= 3
    rates[20:40] /= 3
    features = build_features(loads.assign(posted_rate=rates), daily_market_index(loads))
    assert np.flatnonzero(flag_corrupted_labels(features)).tolist() == list(range(40))


def test_level_keeps_following_the_trend_after_training_ends(loads):
    flat_market = loads.assign(market_index=1.0)
    model = RateModel().fit(build_features(flat_market, daily_market_index(flat_market)))

    probe = flat_market.iloc[[0, 0]].assign(date=pd.to_datetime(["2025-07-20", "2025-11-12"]))
    future_index = pd.Series(1.0, index=pd.date_range("2025-01-01", "2025-12-31"))
    early, late = model.predict(build_features(probe, future_index))

    assert late / early == pytest.approx(np.exp(0.10 * 115 / 365), abs=0.005)


def test_lane_effects_are_shrunk_by_volume():
    rng = np.random.default_rng(0)
    lanes = np.repeat([f"lane-{i}" for i in range(50)], 100)
    residual = np.repeat(rng.normal(0, 0.02, 50), 100) + rng.normal(0, 0.02, lanes.size)
    lanes = np.concatenate([lanes, ["busy"] * 200, ["rare"]])
    residual = np.concatenate([residual, 0.05 + rng.normal(0, 0.02, 200), [0.05]])

    effects = LaneEffects().fit(pd.Series(lanes), residual)
    busy, rare, unseen = effects.predict(pd.Series(["busy", "rare", "never seen"]))

    assert busy == pytest.approx(0.05, abs=0.005)
    # Lane spread and noise are both 0.02, so a single load keeps half its mean.
    assert rare == pytest.approx(0.05 * 0.5, abs=0.005)
    assert unseen == 0

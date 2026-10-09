import numpy as np

from ratecast.evaluation import backtest, time_folds
from ratecast.features import build_features, daily_market_index
from ratecast.labels import flag_corrupted_labels
from tests.synthetic import make_loads


def test_backtest_trains_only_on_screened_past_loads():
    loads = make_loads(days=304)
    planted = loads.index[:40]
    rates = loads["posted_rate"].to_numpy().copy()
    rates[:40] *= 4
    frame = build_features(loads.assign(posted_rate=rates), daily_market_index(loads))
    fitted = []

    class Recorder:
        def fit(self, fit_frame):
            fitted.append(fit_frame)
            return self

        def predict(self, test_frame):
            return test_frame["posted_rate"].to_numpy()

    _, predictions = backtest(frame, flag_corrupted_labels(frame), {"recorder": Recorder})

    for (_, _, test_mask), fit_frame in zip(time_folds(frame["date"]), fitted, strict=True):
        assert fit_frame["date"].max() < frame.loc[test_mask, "date"].min()
        assert not fit_frame.index.isin(planted).any()
    assert np.array_equal(np.unique(predictions["horizon_month"]), [1, 2])

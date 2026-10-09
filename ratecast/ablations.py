"""Variants of the final model that were tried and left out (report section 4.6)."""

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from ratecast.data import LABEL
from ratecast.evaluation import Regressor, backtest
from ratecast.features import build_features
from ratecast.model import LOAD_FEATURES, LaneEffects, RateModel
from ratecast.pipeline import TrainingData

FINAL = "final model"
RECENCY_HALF_LIFE_DAYS = 120
SMOOTHING_HALF_LIVES = (3, 7, 10, 14)


class RecencyWeighted(RateModel):
    """Each load counts half as much for every RECENCY_HALF_LIFE_DAYS before the end of training."""

    def fit(self, frame: pd.DataFrame, sample_weight: np.ndarray | None = None) -> "RecencyWeighted":
        age = (frame["date"].max() - frame["date"]).dt.days.to_numpy()
        return super().fit(frame, sample_weight=0.5 ** (age / RECENCY_HALF_LIFE_DAYS))


class Backfitted(RateModel):
    """A second round: the ridge refitted to the rate minus the boosted part, then the corrections again."""

    def fit(self, frame: pd.DataFrame, sample_weight: np.ndarray | None = None) -> "Backfitted":
        super().fit(frame, sample_weight)
        log_rate = np.log(frame[LABEL].to_numpy())
        boosted = self.boosting_.predict(frame[LOAD_FEATURES])
        self.linear_ = self._fit_linear(frame, log_rate - boosted, sample_weight)
        self._fit_corrections(frame, log_rate - self.linear_.predict(frame), sample_weight)
        return self


class Anchored(RateModel):
    """Every forecast shifted by the mean residual of the last `days` days of training."""

    def __init__(self, days: int):
        super().__init__()
        self.days = days

    def fit(self, frame: pd.DataFrame, sample_weight: np.ndarray | None = None) -> "Anchored":
        super().fit(frame, sample_weight)
        recent = frame[frame["date"] > frame["date"].max() - pd.DateOffset(days=self.days)]
        self.offset_ = float(np.mean(np.log(recent[LABEL].to_numpy()) - super().predict_log(recent)))
        return self

    def predict_log(self, frame: pd.DataFrame) -> np.ndarray:
        return super().predict_log(frame) + self.offset_


class OutOfFoldLanes(RateModel):
    """Lane effects estimated from out-of-fold boosted residuals instead of in-sample ones."""

    def _fit_corrections(
        self, frame: pd.DataFrame, residual: np.ndarray, sample_weight: np.ndarray | None
    ) -> None:
        features = frame[LOAD_FEATURES]
        out_of_fold = np.zeros(len(frame))
        for fit, held in KFold(5, shuffle=True, random_state=0).split(features):
            booster = self._booster().fit(features.iloc[fit], residual[fit])
            out_of_fold[held] = booster.predict(features.iloc[held])
        self.boosting_ = self._booster().fit(features, residual, sample_weight=sample_weight)
        self.lane_effects_ = LaneEffects().fit(frame["lane"], residual - out_of_fold)


VARIANTS: dict[str, Callable[[], Regressor]] = {
    FINAL: RateModel,
    f"recent loads weighted ({RECENCY_HALF_LIFE_DAYS}-day half-life)": RecencyWeighted,
    "second backfitting round": Backfitted,
    "larger trees (63 leaves, 1,000 rounds)": lambda: RateModel(
        gbm_params={"max_leaf_nodes": 63, "max_iter": 1000}
    ),
    "level anchored to the last 14 days": lambda: Anchored(14),
    "level anchored to the last 28 days": lambda: Anchored(28),
    "lane effects from out-of-fold residuals": OutOfFoldLanes,
}


def run_ablations(data: TrainingData) -> pd.DataFrame:
    """Mean clean-label MAE and MAPE over the backtest folds, and the MAE change against the final model."""
    rows = [_score(name, data.frame, data.corrupted, make) for name, make in VARIANTS.items()]
    for half_life in SMOOTHING_HALF_LIVES:
        frame = build_features(data.frame, data.market, half_life=half_life)
        rows.append(_score(f"smoothed index half-life {half_life} days", frame, data.corrupted, RateModel))
    table = pd.DataFrame(rows).set_index("variant")
    return table.assign(MAE_change=table["MAE"] - table.loc[FINAL, "MAE"])


def _score(name: str, frame: pd.DataFrame, corrupted: pd.Series, make: Callable[[], Regressor]) -> dict:
    scores, _ = backtest(frame, corrupted, {name: make})
    return {"variant": name, "MAE": scores["MAE"].mean(), "MAPE": scores["MAPE"].mean()}

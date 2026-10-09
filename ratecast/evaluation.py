from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ratecast.config import FOLD_STARTS, GBM_PARAMS, HORIZON_MONTHS
from ratecast.data import LABEL
from ratecast.features import months_after
from ratecast.labels import flag_corrupted_labels
from ratecast.model import LOAD_FEATURES, RateModel


class Regressor(Protocol):
    def fit(self, frame: pd.DataFrame) -> "Regressor": ...

    def predict(self, frame: pd.DataFrame) -> np.ndarray: ...


Candidates = Mapping[str, Callable[[], Regressor]]
FINAL_MODEL = "Final: + lane effects"
PLAIN_BOOSTING = "Gradient boosting alone"
WITHOUT_SMOOTHING = "Final without smoothed index"


class RatePerMileBaseline:
    """Median rate per mile for the equipment type, times distance."""

    def fit(self, frame: pd.DataFrame) -> "RatePerMileBaseline":
        self.per_mile_ = (frame[LABEL] / frame["distance"]).groupby(frame["equipment"]).median()
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return (frame["equipment"].map(self.per_mile_) * frame["distance"]).to_numpy()


class BoostingOnly:
    """Gradient boosting on log rate with calendar and market features as ordinary inputs."""

    def __init__(self, extra: Sequence[str] = ()):
        self.features = [*LOAD_FEATURES, "log_market", "trend_years", "days_to_quarter_end", *extra]

    def fit(self, frame: pd.DataFrame) -> "BoostingOnly":
        self.model_ = HistGradientBoostingRegressor(categorical_features=["equipment_code"], **GBM_PARAMS)
        self.model_.fit(frame[self.features], np.log(frame[LABEL]))
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.exp(self.model_.predict(frame[self.features]))


CANDIDATES: Candidates = {
    "Rate per mile by equipment": RatePerMileBaseline,
    PLAIN_BOOSTING: BoostingOnly,
    "Linear terms only": lambda: RateModel(use_boosting=False, use_lane_effects=False),
    "+ boosted load correction": lambda: RateModel(use_lane_effects=False),
    FINAL_MODEL: RateModel,
    WITHOUT_SMOOTHING: lambda: RateModel(smoothed_market=False),
}


def time_folds(dates: pd.Series) -> list[tuple[str, pd.Series, pd.Series]]:
    folds = []
    for start in map(pd.Timestamp, FOLD_STARTS):
        end = start + pd.DateOffset(months=HORIZON_MONTHS)
        name = f"{start:%b}-{end - pd.DateOffset(days=1):%b}"
        folds.append((name, dates < start, (dates >= start) & (dates < end)))
    return folds


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAPE": float(100 * np.mean(np.abs(error) / actual)),
        "bias_pct": float(100 * np.mean(np.log(predicted / actual))),
    }


def backtest(
    frame: pd.DataFrame, corrupted: pd.Series, candidates: Candidates = CANDIDATES
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every candidate on every fold.

    Labels are screened inside each training window only. Test metrics are
    reported on clean labels and, separately, on all labels, since the hidden
    November-December labels presumably carry the same corruption.
    """
    scores, predictions = [], []
    for fold, fit_mask, test_mask in time_folds(frame["date"]):
        fit = frame[fit_mask]
        fit = fit[~flag_corrupted_labels(fit)]
        test = frame[test_mask]
        clean = ~corrupted[test_mask].to_numpy()
        actual = test[LABEL].to_numpy()
        horizon_month = months_after(test["date"], fit["date"].max()).to_numpy()
        for name, make in candidates.items():
            predicted = make().fit(fit).predict(test)
            scores.append({
                "model": name, "fold": fold, "n_test": int(clean.sum()),
                **metrics(actual[clean], predicted[clean]),
                **{f"{key}_all": value for key, value in metrics(actual, predicted).items()},
            })
            predictions.append(pd.DataFrame({
                "model": name, "fold": fold, "date": test["date"].to_numpy(),
                "distance": test["distance"].to_numpy(), "horizon_month": horizon_month,
                "actual": actual, "predicted": predicted, "clean": clean,
            }))
    return pd.DataFrame(scores), pd.concat(predictions, ignore_index=True)


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    models = list(dict.fromkeys(scores["model"]))
    folds = list(dict.fromkeys(scores["fold"]))
    mae = scores.pivot(index="model", columns="fold", values="MAE").loc[models, folds]
    means = scores.groupby("model")[["MAE", "RMSE", "MAPE", "MAE_all", "MAPE_all"]].mean().loc[models]
    return mae.join(means.add_prefix("mean_"))

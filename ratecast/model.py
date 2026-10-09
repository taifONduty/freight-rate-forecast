import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer

from ratecast.config import GBM_PARAMS
from ratecast.data import LABEL
from ratecast.features import RAMP_COLUMNS

COORDINATES = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
LOAD_FEATURES = ["log_distance", "weight_lb", "equipment_code", *COORDINATES]
LINEAR_PASSTHROUGH = ["weight_missing", "is_flatbed", "is_reefer", "trend_years", *RAMP_COLUMNS, *COORDINATES]


def linear_component(smoothed_market: bool = True) -> Pipeline:
    """Every term that has to extrapolate past the training window lives here."""
    market_columns = ["log_market", "log_market_smooth"] if smoothed_market else ["log_market"]
    weight = make_pipeline(
        SimpleImputer(strategy="median"), SplineTransformer(n_knots=4, extrapolation="linear")
    )
    columns = ColumnTransformer([
        ("distance", SplineTransformer(n_knots=6, extrapolation="linear"), ["log_distance"]),
        ("weight", weight, ["weight_lb"]),
        *[(column, PolynomialFeatures(degree=2, include_bias=False), [column]) for column in market_columns],
        ("other", "passthrough", LINEAR_PASSTHROUGH),
    ])
    return make_pipeline(columns, Ridge(alpha=1.0))


class LaneEffects:
    """Mean residual per lane, shrunk toward zero according to how many loads support it."""

    def fit(self, lanes: pd.Series, residual: np.ndarray) -> "LaneEffects":
        data = pd.DataFrame({"lane": lanes.to_numpy(), "residual": residual})
        by_lane = data.groupby("lane")["residual"]
        stats = by_lane.agg(["size", "mean"])
        within = data["residual"] - by_lane.transform("mean")
        noise = float((within**2).sum()) / max(len(data) - len(stats), 1)
        between = max(float(stats["mean"].var(ddof=0)) - float((noise / stats["size"]).mean()), 0.0)
        self.effects_ = stats["mean"] * between / (between + noise / stats["size"])
        return self

    def predict(self, lanes: pd.Series) -> np.ndarray:
        return lanes.map(self.effects_).fillna(0.0).to_numpy()


class RateModel:
    """log(rate) = linear market, calendar and load terms + boosted load correction + lane effect.

    The boosted stage never sees a date-derived feature, so the level beyond
    October comes only from the linear terms and the market index.
    """

    def __init__(
        self,
        use_boosting: bool = True,
        use_lane_effects: bool = True,
        smoothed_market: bool = True,
        gbm_params: dict | None = None,
    ):
        self.use_boosting = use_boosting
        self.use_lane_effects = use_lane_effects
        self.smoothed_market = smoothed_market
        self.gbm_params = {**GBM_PARAMS, **(gbm_params or {})}

    def fit(self, frame: pd.DataFrame, sample_weight: np.ndarray | None = None) -> "RateModel":
        """sample_weight, if given, applies to the ridge and boosted stages but not the lane means."""
        log_rate = np.log(frame[LABEL].to_numpy())
        self.linear_ = self._fit_linear(frame, log_rate, sample_weight)
        self._fit_corrections(frame, log_rate - self.linear_.predict(frame), sample_weight)
        return self

    def _fit_linear(
        self, frame: pd.DataFrame, target: np.ndarray, sample_weight: np.ndarray | None
    ) -> Pipeline:
        return linear_component(self.smoothed_market).fit(frame, target, ridge__sample_weight=sample_weight)

    def _fit_corrections(
        self, frame: pd.DataFrame, residual: np.ndarray, sample_weight: np.ndarray | None
    ) -> None:
        if self.use_boosting:
            self.boosting_ = self._booster().fit(frame[LOAD_FEATURES], residual, sample_weight=sample_weight)
            residual = residual - self.boosting_.predict(frame[LOAD_FEATURES])
        if self.use_lane_effects:
            self.lane_effects_ = LaneEffects().fit(frame["lane"], residual)

    def _booster(self) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(categorical_features=["equipment_code"], **self.gbm_params)

    def predict_log(self, frame: pd.DataFrame) -> np.ndarray:
        log_rate = self.linear_.predict(frame)
        if self.use_boosting:
            log_rate = log_rate + self.boosting_.predict(frame[LOAD_FEATURES])
        if self.use_lane_effects:
            log_rate = log_rate + self.lane_effects_.predict(frame["lane"])
        return log_rate

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log(frame))

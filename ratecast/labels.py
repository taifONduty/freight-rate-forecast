import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from ratecast.config import LABEL_RATIO_LIMIT
from ratecast.data import LABEL


def reference_log_rate(features: pd.DataFrame, keep: np.ndarray | None = None) -> np.ndarray:
    """A deliberately plain log-linear fit of the rate, with no calendar terms besides the trend."""
    weight = features["weight_lb"].fillna(features["weight_lb"].median())
    X = np.column_stack([
        features["log_distance"],
        features["log_distance"] ** 2,
        features["is_flatbed"],
        features["is_reefer"],
        weight / 1e4,
        features["log_market"],
        features["trend_years"],
    ])
    y = np.log(features[LABEL].to_numpy())
    keep = np.ones(len(y), dtype=bool) if keep is None else keep
    return LinearRegression().fit(X[keep], y[keep]).predict(X)


def flag_corrupted_labels(features: pd.DataFrame) -> pd.Series:
    """True where posted_rate is off from the reference by more than LABEL_RATIO_LIMIT.

    The first pass is pulled by the corrupted rows themselves, so the
    reference is refitted once on the rows that survive it.
    """
    log_rate = np.log(features[LABEL].to_numpy())
    limit = np.log(LABEL_RATIO_LIMIT)
    keep = np.ones(len(features), dtype=bool)
    for _ in range(2):
        keep = np.abs(log_rate - reference_log_rate(features, keep)) < limit
    return pd.Series(~keep, index=features.index, name="corrupted")

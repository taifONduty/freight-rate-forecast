import numpy as np
import pandas as pd


def residual_quantiles(predictions: pd.DataFrame, coverage: float = 0.9) -> pd.DataFrame:
    """Quantiles of log(actual / predicted) on clean out-of-time rows, per forecast month."""
    clean = predictions[predictions["clean"]]
    log_error = np.log(clean["actual"] / clean["predicted"])
    tail = (1 - coverage) / 2
    quantiles = log_error.groupby(clean["horizon_month"]).quantile([tail, 1 - tail]).unstack()
    return quantiles.set_axis(["lower", "upper"], axis=1)


def interval(
    predicted: np.ndarray, horizon_month: pd.Series, quantiles: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    bounds = quantiles.loc[horizon_month.to_numpy()]
    return predicted * np.exp(bounds["lower"].to_numpy()), predicted * np.exp(bounds["upper"].to_numpy())


def leave_one_fold_out_coverage(predictions: pd.DataFrame, coverage: float = 0.9) -> pd.Series:
    """Share of each fold's clean loads inside intervals calibrated on other folds' other dates.

    Consecutive folds share test months, so rows dated inside the held-out
    window are excluded from the calibration set as well.
    """
    covered = {}
    for fold in predictions["fold"].unique():
        held = predictions[(predictions["fold"] == fold) & predictions["clean"]]
        overlapping = predictions["date"].between(held["date"].min(), held["date"].max())
        quantiles = residual_quantiles(predictions[(predictions["fold"] != fold) & ~overlapping], coverage)
        lower, upper = interval(held["predicted"].to_numpy(), held["horizon_month"], quantiles)
        covered[fold] = float(np.mean((held["actual"] >= lower) & (held["actual"] <= upper)))
    return pd.Series(covered, name="coverage")

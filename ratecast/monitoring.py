import numpy as np
import pandas as pd

NUMERIC_DRIFT_COLUMNS = [
    "distance", "weight_lb", "log_market", "log_market_smooth", "pickup_lat", "delivery_lat",
]
CATEGORICAL_DRIFT_COLUMNS = ["equipment"]
REFERENCE_COLUMNS = [*NUMERIC_DRIFT_COLUMNS, *CATEGORICAL_DRIFT_COLUMNS, "pickup", "delivery", "lane"]
PSI_WARNING = 0.25
SHARE_FLOOR = 1e-4


def stability_index(expected: np.ndarray, actual: np.ndarray) -> float:
    """Population stability index of two share vectors; above 0.25 is conventionally a major shift."""
    expected, actual = np.clip(expected, SHARE_FLOOR, None), np.clip(actual, SHARE_FLOOR, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def numeric_stability(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    reference, current = reference.dropna(), current.dropna()
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    return stability_index(
        np.histogram(reference, edges)[0] / len(reference), np.histogram(current, edges)[0] / len(current)
    )


def categorical_stability(reference: pd.Series, current: pd.Series) -> float:
    shares = pd.concat(
        [reference.value_counts(normalize=True), current.value_counts(normalize=True)], axis=1
    ).fillna(0.0)
    return stability_index(shares.iloc[:, 0].to_numpy(), shares.iloc[:, 1].to_numpy())


def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """PSI flags a changed distribution; outside_range says whether the model must extrapolate."""
    rows = []
    for column in NUMERIC_DRIFT_COLUMNS:
        before, after = reference[column], current[column]
        outside = (~after.dropna().between(before.min(), before.max())).mean()
        rows.append(_drift_row(column, numeric_stability(before, after), outside, before, after))
    for column in CATEGORICAL_DRIFT_COLUMNS:
        before, after = reference[column], current[column]
        outside = (~after.isin(set(before))).mean()
        rows.append(_drift_row(column, categorical_stability(before, after), outside, before, after))
    return pd.DataFrame(rows)


def _drift_row(feature: str, psi: float, outside: float, before: pd.Series, after: pd.Series) -> dict:
    return {
        "feature": feature,
        "psi": psi,
        "outside_range": float(outside),
        "missing_reference": float(before.isna().mean()),
        "missing_current": float(after.isna().mean()),
    }


def novelty(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, float]:
    """Share of current loads whose city or lane the model has never been trained on."""
    cities = set(reference["pickup"]).union(reference["delivery"])
    known_city = current["pickup"].isin(cities) & current["delivery"].isin(cities)
    known_lane = current["lane"].isin(set(reference["lane"]))
    return {
        "unseen_city_share": float(1 - known_city.mean()),
        "unseen_lane_share": float(1 - known_lane.mean()),
    }

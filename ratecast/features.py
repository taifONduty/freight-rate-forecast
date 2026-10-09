import numpy as np
import pandas as pd

from ratecast.config import EPOCH, EQUIPMENT_TYPES, MARKET_HALF_LIFE_DAYS, QUARTER_END_WINDOWS

EQUIPMENT_CODES = {name: code for code, name in enumerate(EQUIPMENT_TYPES)}
RAMP_COLUMNS = [
    f"ramp_{window}{suffix}"
    for window in QUARTER_END_WINDOWS
    for suffix in ("", "_flatbed", "_reefer")
]


def daily_market_index(loads: pd.DataFrame, history: pd.Series | None = None) -> pd.Series:
    """Mean market_index per calendar day, appended to any earlier history.

    Per-load values scatter around the daily level with no relation to the
    rate, so the daily mean is the usable signal, and it also covers loads
    whose own value is missing.
    """
    daily = loads.groupby("date")["market_index"].mean()
    if history is not None:
        daily = daily.combine_first(history)
    daily = daily.dropna().sort_index()
    calendar = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    missing = calendar.difference(daily.index)
    if len(missing):
        raise ValueError(f"no market index for {len(missing)} days, first {missing[0].date()}")
    return daily


def days_to_quarter_end(dates: pd.Series) -> pd.Series:
    quarter_end = dates.dt.to_period("Q").dt.end_time.dt.normalize()
    return (quarter_end - dates).dt.days


def months_after(dates: pd.Series, anchor: pd.Timestamp) -> pd.Series:
    """Calendar months between anchor and each date: 1 for the month after the anchor's month."""
    return (dates.dt.year - anchor.year) * 12 + dates.dt.month - anchor.month


def build_features(
    loads: pd.DataFrame, market_index: pd.Series, half_life: float = MARKET_HALF_LIFE_DAYS
) -> pd.DataFrame:
    log_index = np.log(market_index)
    smoothed = log_index.ewm(halflife=half_life).mean()
    unknown_dates = ~loads["date"].isin(market_index.index)
    if unknown_dates.any():
        raise ValueError(f"market index does not cover {loads.loc[unknown_dates, 'date'].min().date()}")

    # Negative weights have the magnitudes of normal ones and price like them once flipped.
    weight = loads["weight"].abs()
    is_flatbed = (loads["equipment"] == "Flatbed").astype(float)
    is_reefer = (loads["equipment"] == "Reefer").astype(float)
    to_quarter_end = days_to_quarter_end(loads["date"])
    ramps = {}
    for window in QUARTER_END_WINDOWS:
        ramp = np.clip(window - to_quarter_end, 0, None) / window
        ramps[f"ramp_{window}"] = ramp
        ramps[f"ramp_{window}_flatbed"] = ramp * is_flatbed
        ramps[f"ramp_{window}_reefer"] = ramp * is_reefer

    return loads.assign(
        weight_lb=weight,
        weight_missing=weight.isna().astype(float),
        log_distance=np.log(loads["distance"]),
        equipment_code=loads["equipment"].map(EQUIPMENT_CODES),
        is_flatbed=is_flatbed,
        is_reefer=is_reefer,
        trend_years=(loads["date"] - EPOCH).dt.days / 365.0,
        days_to_quarter_end=to_quarter_end,
        log_market=loads["date"].map(log_index),
        log_market_smooth=loads["date"].map(smoothed),
        lane=loads["pickup"] + " > " + loads["delivery"],
        **ramps,
    )

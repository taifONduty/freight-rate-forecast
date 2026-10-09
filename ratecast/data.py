from pathlib import Path

import pandas as pd

from ratecast.config import EQUIPMENT_TYPES

LOAD_COLUMNS = (
    "load_id", "pickup", "delivery", "pickup_lat", "pickup_lon", "delivery_lat",
    "delivery_lon", "distance", "equipment", "weight", "date", "market_index",
)
NUMERIC_COLUMNS = (
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "distance", "weight", "market_index",
)
LABEL = "posted_rate"
DECEMBER_COLUMNS = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]


class SchemaError(ValueError):
    pass


def read_loads(path: Path, labelled: bool = False) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_loads(frame, labelled)
    return frame.assign(date=pd.to_datetime(frame["date"]))


def validate_loads(frame: pd.DataFrame, labelled: bool = False) -> None:
    required = LOAD_COLUMNS + ((LABEL,) if labelled else ())
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise SchemaError(f"missing columns: {missing}")
    if frame["load_id"].duplicated().any():
        raise SchemaError("duplicate load_id values")
    for column in NUMERIC_COLUMNS + ((LABEL,) if labelled else ()):
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise SchemaError(f"column {column} is not numeric")
    unknown = set(frame["equipment"]) - set(EQUIPMENT_TYPES)
    if unknown:
        raise SchemaError(f"unknown equipment types: {sorted(unknown)}")
    if pd.to_datetime(frame["date"], errors="coerce").isna().any():
        raise SchemaError("unparseable dates")
    if not (frame["distance"] > 0).all():
        raise SchemaError("distance must be positive and present")
    if labelled and not (frame[LABEL] > 0).all():
        raise SchemaError(f"{LABEL} must be positive and present")


def city_coordinates(*frames: pd.DataFrame) -> pd.DataFrame:
    """One (lat, lon) per city, gathered from both ends of every load."""
    ends = [
        frame[[city, f"{city}_lat", f"{city}_lon"]].set_axis(["city", "lat", "lon"], axis=1)
        for frame in frames
        for city in ("pickup", "delivery")
    ]
    return pd.concat(ends).drop_duplicates("city").set_index("city").sort_index()


def read_december(path: Path, coordinates: pd.DataFrame) -> pd.DataFrame:
    """The chart template only names its cities, so coordinates are looked up."""
    frame = pd.read_csv(path)
    if list(frame.columns) != DECEMBER_COLUMNS:
        raise SchemaError(f"expected columns {DECEMBER_COLUMNS}")
    unknown = sorted(set(frame["pickup"]).union(frame["delivery"]) - set(coordinates.index))
    if unknown:
        raise SchemaError(f"no coordinates for cities: {unknown}")
    located = frame.assign(
        load_id=[f"DEC-{i:02d}" for i in range(1, len(frame) + 1)],
        date=pd.to_datetime(frame["date"]),
        market_index=float("nan"),
        **{
            f"{end}_{axis}": frame[end].map(coordinates[axis])
            for end in ("pickup", "delivery")
            for axis in ("lat", "lon")
        },
    )
    validate_loads(located)
    return located

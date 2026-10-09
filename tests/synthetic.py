import numpy as np
import pandas as pd

from ratecast.config import EPOCH, EQUIPMENT_TYPES

CITIES = {
    "A": (30.0, -95.0), "B": (34.0, -84.0), "C": (40.0, -75.0), "D": (42.0, -88.0), "E": (36.0, -115.0),
}


def make_loads(
    n: int = 4000, days: int = 240, trend: float = 0.10, market_noise: float = 0.01, seed: int = 0
) -> pd.DataFrame:
    """Synthetic loads priced by a known log-linear rule."""
    rng = np.random.default_rng(seed)
    pickup, delivery = rng.choice(list(CITIES), n), rng.choice(list(CITIES), n)
    day = rng.integers(0, days, n)
    distance = np.exp(rng.uniform(np.log(80), np.log(2500), n))
    equipment = rng.choice(EQUIPMENT_TYPES, n)
    weight = rng.uniform(5000, 45000, n)
    market_level = np.exp(0.1 * np.sin(2 * np.pi * day / 90))
    log_rate = (
        0.6 + 0.87 * np.log(distance) + 0.08 * (equipment == "Flatbed") + 0.12 * (equipment == "Reefer")
        + 0.02 * weight / 1e4 + 0.15 * np.log(market_level) + trend * day / 365 + rng.normal(0, 0.01, n)
    )
    return pd.DataFrame({
        "load_id": [f"L-{i:05d}" for i in range(n)],
        "pickup": pickup,
        "delivery": delivery,
        "pickup_lat": [CITIES[c][0] for c in pickup],
        "pickup_lon": [CITIES[c][1] for c in pickup],
        "delivery_lat": [CITIES[c][0] for c in delivery],
        "delivery_lon": [CITIES[c][1] for c in delivery],
        "distance": distance,
        "equipment": equipment,
        "weight": weight,
        "date": EPOCH + pd.to_timedelta(day, unit="D"),
        "market_index": market_level + rng.normal(0, market_noise, n),
        "posted_rate": np.exp(log_rate),
    })

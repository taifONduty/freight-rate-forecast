import numpy as np
import pandas as pd
import pytest

from ratecast.data import SchemaError, city_coordinates, read_december, validate_loads


@pytest.mark.parametrize(
    "corrupt, message",
    [
        (lambda f: f.drop(columns="distance"), "missing columns"),
        (lambda f: f.assign(equipment="Tanker"), "unknown equipment"),
        (lambda f: f.assign(distance=0.0), "distance"),
        (lambda f: f.assign(load_id="L-1"), "duplicate"),
        (lambda f: f.assign(posted_rate=np.where(f.index == 0, np.nan, f["posted_rate"])), "posted_rate"),
        (lambda f: f.assign(posted_rate=0.0), "posted_rate"),
    ],
)
def test_validate_rejects_malformed_input(loads, corrupt, message):
    with pytest.raises(SchemaError, match=message):
        validate_loads(corrupt(loads), labelled=True)


def test_december_rows_get_city_coordinates(loads, tmp_path):
    template = tmp_path / "december.csv"
    pd.DataFrame({
        "pickup": ["A", "A"], "delivery": ["C", "C"], "distance": [360, 360],
        "equipment": ["Dry Van"] * 2, "weight": [32000, 32000],
        "date": ["2025-12-01", "2025-12-02"], "predicted_rate": [None, None],
    }).to_csv(template, index=False)

    december = read_december(template, city_coordinates(loads))

    coordinates = december[["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]].iloc[0]
    assert coordinates.tolist() == [30.0, -95.0, 40.0, -75.0]
    without_c = loads[(loads["pickup"] != "C") & (loads["delivery"] != "C")]
    with pytest.raises(SchemaError, match="no coordinates"):
        read_december(template, city_coordinates(without_c))

from dataclasses import dataclass
from typing import NamedTuple

import pandas as pd

from ratecast import config
from ratecast.data import DECEMBER_COLUMNS, city_coordinates, read_december, read_loads
from ratecast.evaluation import CANDIDATES, FINAL_MODEL, backtest
from ratecast.features import build_features, daily_market_index, months_after
from ratecast.intervals import interval, residual_quantiles
from ratecast.labels import flag_corrupted_labels
from ratecast.model import RateModel
from ratecast.monitoring import REFERENCE_COLUMNS


class TrainingData(NamedTuple):
    frame: pd.DataFrame
    corrupted: pd.Series
    market: pd.Series


@dataclass(frozen=True)
class TrainedModel:
    model: RateModel
    market_history: pd.Series
    coordinates: pd.DataFrame
    interval_quantiles: pd.DataFrame
    reference: pd.DataFrame
    trained_through: pd.Timestamp


@dataclass(frozen=True)
class Predictions:
    validation: pd.DataFrame
    submission: pd.DataFrame
    december: pd.DataFrame
    december_chart: pd.DataFrame


def load_training_data() -> TrainingData:
    loads = read_loads(config.TRAIN_PATH, labelled=True)
    market = daily_market_index(loads)
    frame = build_features(loads, market)
    return TrainingData(frame, flag_corrupted_labels(frame), market)


def train() -> TrainedModel:
    """Fit on every clean label; interval widths come from the backtest folds."""
    data = load_training_data()
    _, predictions = backtest(data.frame, data.corrupted, {FINAL_MODEL: CANDIDATES[FINAL_MODEL]})
    clean = data.frame[~data.corrupted]
    return TrainedModel(
        model=RateModel().fit(clean),
        market_history=data.market,
        coordinates=city_coordinates(data.frame),
        interval_quantiles=residual_quantiles(predictions),
        reference=clean[REFERENCE_COLUMNS],
        trained_through=data.frame["date"].max(),
    )


def score(trained: TrainedModel, loads: pd.DataFrame, market: pd.Series) -> pd.DataFrame:
    features = build_features(loads, market)
    predicted = trained.model.predict(features)
    horizon = months_after(features["date"], trained.trained_through)
    horizon = horizon.clip(1, trained.interval_quantiles.index.max())
    lower, upper = interval(predicted, horizon, trained.interval_quantiles)
    return features.assign(predicted_rate=predicted, lower_90=lower, upper_90=upper)


def align_to_template(template: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    rates = scored.set_index("load_id")["predicted_rate"]
    unscored = template.loc[~template["load_id"].isin(rates.index), "load_id"]
    if len(unscored):
        raise ValueError(f"{len(unscored)} template load_ids were not scored, e.g. {unscored.iloc[0]}")
    return pd.DataFrame({
        "load_id": template["load_id"],
        "predicted_rate": rates.loc[template["load_id"]].round(2).to_numpy(),
    })


def predict(trained: TrainedModel) -> Predictions:
    validation = read_loads(config.VALIDATION_PATH)
    market = daily_market_index(validation, trained.market_history)
    scored = score(trained, validation, market)

    coordinates = trained.coordinates.combine_first(city_coordinates(validation))
    december = score(trained, read_december(config.DECEMBER_PATH, coordinates), market)
    chart = december[DECEMBER_COLUMNS].assign(predicted_rate=december["predicted_rate"].round(2))

    return Predictions(
        validation=scored,
        submission=align_to_template(pd.read_csv(config.TEMPLATE_PATH), scored),
        december=december,
        december_chart=chart,
    )

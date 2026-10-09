from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter
from sklearn.linear_model import LinearRegression

from ratecast import config
from ratecast.analysis import quote_signal_by_month
from ratecast.data import LABEL, city_coordinates
from ratecast.evaluation import CANDIDATES, FINAL_MODEL, PLAIN_BOOSTING, WITHOUT_SMOOTHING, backtest
from ratecast.features import daily_market_index
from ratecast.labels import reference_log_rate
from ratecast.model import COORDINATES
from ratecast.pipeline import TrainedModel, TrainingData, load_training_data, predict

BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
GRAY, INK, MUTED, GRID = "#a3a29c", "#0b0b0b", "#52514e", "#e4e3df"
EQUIPMENT_COLORS = {"Dry Van": BLUE, "Flatbed": ORANGE, "Reefer": AQUA}
REGIME_EXAMPLES = [(9, "September: copy"), (10, "October: mirror"), (12, "December: noise")]


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.edgecolor": GRAY, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.7,
        "lines.linewidth": 1.6, "lines.solid_capstyle": "round", "legend.frameon": False,
    })


def _save(figure: plt.Figure, name: str) -> Path:
    path = config.FIGURE_DIR / name
    figure.savefig(path)
    plt.close(figure)
    return path


def label_screening(data: TrainingData) -> Path:
    keep = ~data.corrupted.to_numpy()
    ratio = data.frame[LABEL].to_numpy() / np.exp(reference_log_rate(data.frame, keep))
    figure, axis = plt.subplots(figsize=(7, 2.6))
    axis.hist(ratio, bins=np.geomspace(0.12, 8, 160), color=BLUE)
    axis.set_xscale("log")
    axis.set_yscale("log")
    for limit in (1 / config.LABEL_RATIO_LIMIT, config.LABEL_RATIO_LIMIT):
        axis.axvline(limit, color=MUTED, linewidth=0.8)
    axis.set_xticks([0.2, 0.5, 1, 2, 5], ["0.2", "0.5", "1", "2", "5"])
    axis.set_xlabel("posted rate / reference fit")
    axis.set_ylabel("loads")
    axis.set_title(f"{keep.sum():,} plausible labels, {(~keep).sum()} outside the cut-offs")
    return _save(figure, "label_screening.png")


def quote_signal(data: TrainingData, trained: TrainedModel, validation: pd.DataFrame) -> Path:
    labelled = data.frame[~data.corrupted].assign(predicted_rate=lambda f: trained.model.predict(f))
    loads = pd.concat([labelled, validation], ignore_index=True)
    table = quote_signal_by_month(loads)

    figure = plt.figure(figsize=(7, 4.6))
    grid = figure.add_gridspec(2, 3, height_ratios=(1.15, 1), hspace=0.55, wspace=0.3)
    bars = figure.add_subplot(grid[0, :])
    x = pd.Series(np.arange(len(table)), index=table.index)
    bars.bar(x - 0.2, table["corr_actual"], width=0.36, color=BLUE, label="vs actual rate per mile")
    bars.bar(x + 0.2, table["corr_predicted"], width=0.36, color=ORANGE,
             label="vs predicted rate per mile (no labels needed)")
    bars.axhline(0, color=GRAY, linewidth=0.8)
    for month in table.index[table["corr_actual"].isna()]:
        value = table.loc[month, "corr_predicted"]
        bars.text(x[month] + 0.2, 0.08, f"{value:+.2f}", ha="center", fontsize=7.5, color=MUTED)
    bars.set_xticks(x, table.index)
    bars.set_ylim(-1.1, 1.1)
    bars.set_ylabel("correlation")
    bars.legend(loc="lower left", ncol=2, bbox_to_anchor=(0, 1.0))
    bars.set_title("quote_signal by month", pad=22)

    for column, (month, title) in enumerate(REGIME_EXAMPLES):
        axis = figure.add_subplot(grid[1, column])
        group = loads[loads["date"].dt.month == month]
        group = group.sample(min(1500, len(group)), random_state=0)
        axis.scatter(group["predicted_rate"] / group["distance"], group["quote_signal"],
                     s=3, color=BLUE, alpha=0.35, linewidths=0)
        axis.set_title(title, fontsize=9)
        axis.set_xlabel("predicted $/mile")
        axis.set_xlim(1.4, 3.6)
        axis.set_ylim(1.0, 3.2)
        axis.grid(axis="both")
        if column == 0:
            axis.set_ylabel("quote_signal")
    return _save(figure, "quote_signal.png")


def market_level(data: TrainingData, market: pd.Series, forecast_start: pd.Timestamp) -> Path:
    clean = data.frame[~data.corrupted]
    load_terms = np.column_stack([
        clean["log_distance"], clean["log_distance"] ** 2, clean["is_flatbed"], clean["is_reefer"],
        clean["weight_lb"].fillna(clean["weight_lb"].median()), clean[COORDINATES],
    ])
    log_rate = np.log(clean[LABEL].to_numpy())
    residual = log_rate - LinearRegression().fit(load_terms, log_rate).predict(load_terms)
    level = pd.Series(100 * residual, index=clean.index).groupby(clean["date"]).mean()
    smoothed = np.exp(np.log(market).ewm(halflife=config.MARKET_HALF_LIFE_DAYS).mean())

    figure, (top, bottom) = plt.subplots(2, 1, figsize=(7, 4.2), sharex=True, gridspec_kw={"hspace": 0.35})
    top.plot(level.index, level, color=BLUE, linewidth=1.2)
    top.set_ylabel("% vs load-only fit")
    top.set_title("Daily rate level after distance, equipment, weight and geography")
    bottom.plot(market.index, market, color=GRAY, linewidth=0.9, label="daily mean")
    smoothed_label = f"{config.MARKET_HALF_LIFE_DAYS}-day half-life average"
    bottom.plot(smoothed.index, smoothed, color=BLUE, label=smoothed_label)
    bottom.axvspan(forecast_start, market.index.max(), color=GRAY, alpha=0.12, linewidth=0)
    bottom.text(forecast_start, 0.92, " prediction window", transform=bottom.get_xaxis_transform(),
                color=MUTED, fontsize=8)
    bottom.set_ylabel("market_index")
    bottom.set_title("Market index")
    bottom.legend(loc="lower right", bbox_to_anchor=(1, 1), ncol=2)
    bottom.xaxis.set_major_locator(mdates.MonthLocator())
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    return _save(figure, "market_level.png")


def quarter_end(data: TrainingData) -> Path:
    last_complete_quarter_end = data.frame["date"].max().to_period("Q").start_time
    clean = data.frame[~data.corrupted & (data.frame["date"] < last_complete_quarter_end)]
    residual = 100 * (np.log(clean[LABEL].to_numpy()) - reference_log_rate(clean))
    week = (clean["days_to_quarter_end"] // 7) * 7 + 3.5
    by_week = pd.Series(residual, index=clean.index).groupby([clean["equipment"], week]).mean()

    figure, axis = plt.subplots(figsize=(7, 2.8))
    for equipment, color in EQUIPMENT_COLORS.items():
        series = by_week.loc[equipment]
        axis.plot(series.index, series, color=color, marker="o", markersize=3.5, label=equipment)
    axis.set_xlim(92, -1)
    axis.axvspan(30, -1, color=GRAY, alpha=0.1, linewidth=0)
    axis.set_xlabel("days to quarter end (complete quarters pooled)")
    axis.set_ylabel("% vs reference fit")
    axis.set_title("The last month of each quarter prices higher, most of all for flatbeds")
    axis.legend(loc="upper left")
    return _save(figure, "quarter_end.png")


def horizon_bias(predictions: pd.DataFrame) -> Path:
    models = {PLAIN_BOOSTING: (ORANGE, "gradient boosting alone"), FINAL_MODEL: (BLUE, "final model")}
    clean = predictions[predictions["clean"] & predictions["model"].isin(models)]
    fold_start = clean.groupby("fold")["date"].transform("min")
    week = (clean["date"] - fold_start).dt.days // 7 + 1
    bias = (100 * np.log(clean["predicted"] / clean["actual"])).groupby([clean["model"], week]).mean()

    figure, axis = plt.subplots(figsize=(7, 2.6))
    for name, (color, label) in models.items():
        series = bias.loc[name]
        axis.plot(series.index, series, color=color, marker="o", markersize=3.5, label=label)
    axis.axhline(0, color=GRAY, linewidth=0.8)
    axis.set_xlabel("week of the two-month forecast window")
    axis.set_ylabel("mean error, %")
    axis.set_title("Bias across the forecast window, averaged over the five folds")
    axis.legend(loc="lower left")
    return _save(figure, "horizon_bias.png")


def december_interval(december: pd.DataFrame) -> Path:
    lane = december.iloc[0]
    figure, axis = plt.subplots(figsize=(7, 2.6))
    axis.fill_between(december["date"], december["lower_90"], december["upper_90"], color=BLUE, alpha=0.12,
                      linewidth=0, label="90% interval for a single load")
    axis.plot(december["date"], december["predicted_rate"], color=BLUE, marker="o", markersize=3,
              label="predicted rate")
    axis.set_ylabel("$")
    axis.set_title(f"{lane['pickup']} to {lane['delivery']}, {lane['equipment'].lower()}, "
                   f"{lane['distance']:,.0f} miles, {lane['weight']:,.0f} lb")
    axis.legend(loc="upper left")
    axis.xaxis.set_major_locator(mdates.DayLocator(bymonthday=(1, 8, 15, 22, 29)))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: _day_label(mdates.num2date(value))))
    return _save(figure, "december_interval.png")


def _day_label(day: pd.Timestamp) -> str:
    return f"{day:%b} {day.day}"


def city_premiums(data: TrainingData, validation: pd.DataFrame) -> Path:
    clean = data.frame[~data.corrupted]
    residual = 100 * (np.log(clean[LABEL].to_numpy()) - reference_log_rate(clean))
    ends = pd.concat([
        pd.DataFrame({"city": clean[end], "lat": clean[f"{end}_lat"], "lon": clean[f"{end}_lon"],
                      "residual": residual})
        for end in ("pickup", "delivery")
    ])
    premium = ends.groupby(["city", "lat", "lon"])["residual"].mean().reset_index()
    unseen = city_coordinates(validation).reset_index()
    unseen = unseen[~unseen["city"].isin(premium["city"])]

    figure, axis = plt.subplots(figsize=(7, 3.6))
    colors = LinearSegmentedColormap.from_list("premium", [BLUE, "#f0efec", RED])
    limit = premium["residual"].abs().max()
    points = axis.scatter(premium["lon"], premium["lat"], c=premium["residual"], cmap=colors, s=42,
                          norm=TwoSlopeNorm(0, -limit, limit), edgecolors="white", linewidths=0.8)
    axis.scatter(unseen["lon"], unseen["lat"], s=42, facecolors="none", edgecolors=INK, linewidths=1.1,
                 label="only in validation")
    for city in unseen.itertuples():
        axis.annotate(city.city, (city.lon, city.lat), xytext=(5, 3), textcoords="offset points",
                      fontsize=7.5, color=MUTED)
    figure.colorbar(points, ax=axis, label="city premium, % vs reference fit", shrink=0.85)
    axis.set_xlabel("longitude")
    axis.set_ylabel("latitude")
    axis.grid(axis="both")
    axis.legend(loc="lower left")
    axis.set_title("City premiums rise to the south")
    return _save(figure, "city_premiums.png")


def forecast_tracking(predictions: pd.DataFrame) -> Path:
    models = {FINAL_MODEL: (BLUE, "final model"), WITHOUT_SMOOTHING: (ORANGE, "without the smoothed index")}
    folds = {"Jul-Aug": "Trained to June, forecasting July and August",
             "Sep-Oct": "Trained to August, forecasting September and October"}
    figure, axes = plt.subplots(2, 1, figsize=(7, 4.8), gridspec_kw={"hspace": 0.45, "top": 0.88})
    for axis, (fold, title) in zip(axes, folds.items(), strict=True):
        rows = predictions[predictions["clean"] & (predictions["fold"] == fold)]
        actual = (rows["actual"] / rows["distance"]).groupby(rows["date"]).mean()
        axis.plot(actual.index, actual, color=INK, linewidth=1.2, label="actual")
        for name, (color, label) in models.items():
            model_rows = rows[rows["model"] == name]
            daily = (model_rows["predicted"] / model_rows["distance"]).groupby(model_rows["date"]).mean()
            axis.plot(daily.index, daily, color=color, linewidth=1.2, label=label)
        axis.set_ylabel("mean $/mile")
        axis.set_title(title)
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: _day_label(mdates.num2date(value))))
    figure.legend(*axes[0].get_legend_handles_labels(), loc="upper right", ncol=3, bbox_to_anchor=(0.9, 1.0))
    return _save(figure, "forecast_tracking.png")


def make_figures() -> list[Path]:
    _style()
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data = load_training_data()
    trained: TrainedModel = joblib.load(config.MODEL_PATH)
    result = predict(trained)
    market = daily_market_index(result.validation, trained.market_history)
    forecast_start = trained.trained_through + pd.DateOffset(days=1)
    compared = {name: CANDIDATES[name] for name in (PLAIN_BOOSTING, FINAL_MODEL, WITHOUT_SMOOTHING)}
    _, predictions = backtest(data.frame, data.corrupted, compared)
    return [
        label_screening(data),
        quote_signal(data, trained, result.validation),
        market_level(data, market, forecast_start),
        quarter_end(data),
        city_premiums(data, result.validation),
        horizon_bias(predictions),
        forecast_tracking(predictions),
        december_interval(result.december),
    ]

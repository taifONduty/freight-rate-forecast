import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from ratecast.config import MARKET_HALF_LIFE_DAYS
from ratecast.data import LABEL
from ratecast.evaluation import BoostingOnly, metrics, time_folds
from ratecast.labels import flag_corrupted_labels
from ratecast.model import RateModel
from ratecast.pipeline import TrainingData


def split_comparison(clean: pd.DataFrame) -> pd.DataFrame:
    """Clean-label MAE of plain boosting with and without quote_signal, under two split designs."""
    variants = {"without quote_signal": (), "with quote_signal": ("quote_signal",)}
    table = {}
    for name, extra in variants.items():
        random_split = [
            _mae(BoostingOnly(extra).fit(clean.iloc[fit]), clean.iloc[test])
            for fit, test in KFold(5, shuffle=True, random_state=0).split(clean)
        ]
        time_split = [
            _mae(BoostingOnly(extra).fit(clean[fit]), clean[test])
            for _, fit, test in time_folds(clean["date"])
        ]
        table[name] = {"random 5-fold": np.mean(random_split), "time-based folds": np.mean(time_split)}
    return pd.DataFrame(table)


def _mae(model: BoostingOnly, test: pd.DataFrame) -> float:
    return metrics(test[LABEL].to_numpy(), model.predict(test))["MAE"]


def quote_signal_by_month(loads: pd.DataFrame) -> pd.DataFrame:
    """How quote_signal relates to the rate per mile, month by month.

    The correlation with the predicted rate needs no labels, so it also
    covers November and December, where posted_rate is missing.
    """
    rows = []
    for month, group in loads.groupby(loads["date"].dt.to_period("M")):
        by_equipment = group.groupby("equipment")["quote_signal"].mean()
        rows.append({
            "month": month.strftime("%b"),
            "corr_actual": group["quote_signal"].corr(group[LABEL] / group["distance"]),
            "corr_predicted": group["quote_signal"].corr(group["predicted_rate"] / group["distance"]),
            "equipment_spread": by_equipment.max() - by_equipment.min(),
        })
    return pd.DataFrame(rows).set_index("month")


def learned_time_effects(model: RateModel) -> pd.Series:
    """The ridge stage's market, trend and quarter-end terms, translated into readable effects.

    A one-day move shifts the daily index fully but the smoothed index only by
    its same-day weight; a sustained move shifts both.
    """
    coef = pd.Series(model.linear_[-1].coef_, index=model.linear_[0].get_feature_names_out())
    same_day_weight = 1 - 0.5 ** (1 / MARKET_HALF_LIFE_DAYS)

    def elasticity(index_level: float, sustained: bool) -> float:
        log_level = np.log(index_level)
        daily = coef["log_market__log_market"] + 2 * coef["log_market__log_market^2"] * log_level
        smooth = (coef["log_market_smooth__log_market_smooth"]
                  + 2 * coef["log_market_smooth__log_market_smooth^2"] * log_level)
        return daily + (1.0 if sustained else same_day_weight) * smooth

    def premium(equipment: str, days_to_quarter_end: int) -> float:
        suffixes = {"Dry Van": [""], "Flatbed": ["", "_flatbed"], "Reefer": ["", "_reefer"]}[equipment]
        ramps = {window: max(window - days_to_quarter_end, 0) / window for window in (30, 14, 7)}
        log_premium = sum(coef[f"other__ramp_{w}{s}"] * r for w, r in ramps.items() for s in suffixes)
        return 100 * np.expm1(log_premium)

    effects = {
        f"elasticity, {kind} move at index {level}": elasticity(level, kind == "sustained")
        for level in (0.9, 1.0, 1.3)
        for kind in ("one-day", "sustained")
    }
    effects |= {
        "trend, % per month": 100 * np.expm1(coef["other__trend_years"] / 12),
        "flatbed premium over dry van, %": 100 * np.expm1(coef["other__is_flatbed"]),
        "reefer premium over dry van, %": 100 * np.expm1(coef["other__is_reefer"]),
    }
    effects |= {
        f"{equipment.lower()} quarter-end premium {days} days out, %": premium(equipment, days)
        for days in (14, 7, 0)
        for equipment in ("Dry Van", "Flatbed", "Reefer")
    }
    return pd.Series(effects, name="value")


def unseen_city_check(data: TrainingData, cities_per_draw: int = 6, draws: int = 3) -> pd.DataFrame:
    """Refit the last fold without a few training cities and compare the loads that touch them.

    Validation has eight cities the model never saw, so this measures what
    pricing a city from its coordinates alone costs.
    """
    _, fit_mask, test_mask = time_folds(data.frame["date"])[-1]
    history = data.frame[fit_mask]
    test = data.frame[test_mask & ~data.corrupted]
    cities = sorted(set(history["pickup"]).union(history["delivery"]))
    rng = np.random.default_rng(0)
    rows = []
    for draw in range(draws):
        held_out = set(rng.choice(cities, cities_per_draw, replace=False))
        fit = history[~(history["pickup"].isin(held_out) | history["delivery"].isin(held_out))]
        fit = fit[~flag_corrupted_labels(fit)]
        error = np.abs(RateModel().fit(fit).predict(test) / test[LABEL].to_numpy() - 1)
        touches = (test["pickup"].isin(held_out) | test["delivery"].isin(held_out)).to_numpy()
        rows.append({
            "held_out": ", ".join(sorted(held_out)),
            "loads_touching": int(touches.sum()),
            "MAPE_touching": 100 * error[touches].mean(),
            "MAPE_other": 100 * error[~touches].mean(),
        })
    return pd.DataFrame(rows)

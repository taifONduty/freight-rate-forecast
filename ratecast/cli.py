import argparse
import time

import joblib
import pandas as pd

from ratecast import config
from ratecast.ablations import run_ablations
from ratecast.analysis import learned_time_effects, split_comparison, unseen_city_check
from ratecast.evaluation import FINAL_MODEL, backtest, summarize
from ratecast.intervals import leave_one_fold_out_coverage
from ratecast.monitoring import PSI_WARNING, drift_report, novelty
from ratecast.pipeline import load_training_data, predict, train


def run_backtest() -> None:
    data = load_training_data()
    print(f"{len(data.frame):,} labelled loads, {int(data.corrupted.sum())} corrupted labels screened out")
    scores, predictions = backtest(data.frame, data.corrupted)
    summary = summarize(scores)
    coverage = leave_one_fold_out_coverage(predictions[predictions["model"] == FINAL_MODEL])
    splits = split_comparison(data.frame[~data.corrupted])
    cities = unseen_city_check(data)

    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(config.TABLE_DIR / "backtest_scores.csv", index=False)
    summary.to_csv(config.TABLE_DIR / "backtest_summary.csv")
    coverage.to_csv(config.TABLE_DIR / "interval_coverage.csv")
    splits.to_csv(config.TABLE_DIR / "split_comparison.csv")
    cities.to_csv(config.TABLE_DIR / "unseen_cities.csv", index=False)

    print("\nMAE ($) on clean labels per fold, and means over folds:")
    print(summary.round(2).to_string())
    print("\n90% interval coverage, each fold calibrated on the others:")
    print(coverage.round(3).to_string())
    print("\nPlain boosting MAE ($) under two validation schemes:")
    print(splits.round(2).to_string())
    print("\nLast fold refitted without a few training cities, MAPE (%):")
    print(cities.round(2).to_string(index=False))


def run_train() -> None:
    trained = train()
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(trained, config.MODEL_PATH)
    effects = learned_time_effects(trained.model)
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    effects.to_csv(config.TABLE_DIR / "time_effects.csv")
    print(f"trained on {len(trained.reference):,} clean loads through {trained.trained_through:%Y-%m-%d}")
    print(f"90% interval, log scale, by forecast month:\n{trained.interval_quantiles.round(4).to_string()}")
    print(f"what the ridge stage learned:\n{effects.round(3).to_string()}")
    print(f"saved {config.MODEL_PATH.relative_to(config.ROOT)}")


def run_predict() -> None:
    trained = joblib.load(config.MODEL_PATH)
    result = predict(trained)
    drift = drift_report(trained.reference, result.validation)
    new = novelty(trained.reference, result.validation)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.submission.to_csv(config.OUTPUT_DIR / "validation_predictions.csv", index=False)
    result.december_chart.to_csv(
        config.OUTPUT_DIR / "december_chart_inputs.csv", index=False, date_format="%Y-%m-%d"
    )
    result.december[["date", "predicted_rate", "lower_90", "upper_90"]].round(2).to_csv(
        config.OUTPUT_DIR / "december_intervals.csv", index=False, date_format="%Y-%m-%d"
    )
    drift.to_csv(config.OUTPUT_DIR / "input_drift.csv", index=False)

    rates = result.submission["predicted_rate"]
    print(f"validation: {len(rates):,} loads, median ${rates.median():,.0f}, "
          f"range ${rates.min():,.0f} to ${rates.max():,.0f}")
    chart = result.december_chart["predicted_rate"]
    print(f"december chart: ${chart.iloc[0]:,.0f} on the 1st, ${chart.iloc[-1]:,.0f} on the 31st")
    print(f"unseen cities touch {new['unseen_city_share']:.1%} of loads, "
          f"unseen lanes {new['unseen_lane_share']:.1%}")
    for row in drift[drift["psi"] > PSI_WARNING].itertuples():
        print(f"warning: {row.feature} has shifted (PSI {row.psi:.2f}), "
              f"{row.outside_range:.1%} outside the training range")


def run_ablation_study() -> None:
    table = run_ablations(load_training_data())
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.TABLE_DIR / "ablations.csv")
    print("Mean over the backtest folds, clean labels:")
    print(table.round(2).to_string())


def run_figures() -> None:
    from ratecast.figures import make_figures

    for path in make_figures():
        print(f"wrote {path.relative_to(config.ROOT)}")


COMMANDS = {
    "backtest": run_backtest,
    "ablations": run_ablation_study,
    "train": run_train,
    "predict": run_predict,
    "figures": run_figures,
}


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m ratecast", description="Freight rate forecasting.")
    parser.add_argument("command", choices=COMMANDS)
    args = parser.parse_args()
    pd.set_option("display.width", 160)
    started = time.time()
    COMMANDS[args.command]()
    print(f"done in {time.time() - started:.0f}s")

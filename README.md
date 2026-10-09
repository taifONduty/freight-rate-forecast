# Freight rate forecasting

My solution to the Spotter machine learning engineer assessment. The task is to predict
the posted rate of 12,000 truckload shipments dated November and December 2025 from
48,000 labelled loads dated January to October 2025, and to price one fixed lane for
every day of December.

The write-up is in [`report/report.pdf`](report/report.pdf).

## Results

Every load to predict is dated after the last labelled load, so I validated with five
expanding-window folds that each train on the past and forecast the next two months.
Mean over the folds, on clean labels:

| Model | MAE | MAPE |
|---|---|---|
| Rate per mile by equipment | $168.2 | 8.33% |
| Gradient boosting alone | $48.8 | 2.08% |
| Linear terms only | $45.6 | 1.94% |
| + boosted load correction | $33.0 | 1.40% |
| **Final: + lane effects** | **$31.0** | **1.31%** |

The final model works on log rate. A ridge regression carries everything that has to
extrapolate past October: the daily market index and a 5-day smoothed version of it, a
linear trend, and a quarter-end premium that differs by equipment. A scikit-learn
gradient-boosted model then corrects the load-level pricing (distance, weight, equipment,
coordinates) without seeing any date feature, and per-lane effects shrunk by volume take
what is left.

## Running it

Python 3.10 or newer.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python -m pytest -q              # 18 tests, a few seconds
.venv/bin/python -m ratecast backtest      # model comparison, ~3 min
.venv/bin/python -m ratecast ablations     # variants left out (report section 4.6), ~9 min
.venv/bin/python -m ratecast train         # final fit, ~30 s
.venv/bin/python -m ratecast predict       # writes outputs/
.venv/bin/python score.py --predictions outputs/validation_predictions.csv \
    --december-predictions outputs/december_chart_inputs.csv \
    --output-dir outputs/scorer_results
```

`make setup` then `make all` runs the same sequence, except the ablations (`make ablations`),
and regenerates the report figures.
`make report` rebuilds the PDF with [Tectonic](https://tectonic-typesetting.github.io).
Results are deterministic.

## Outputs

| File | Contents |
|---|---|
| `outputs/validation_predictions.csv` | `load_id,predicted_rate` for the 12,000 validation loads, in template order |
| `outputs/december_chart_inputs.csv` | the December template with `predicted_rate` filled in |
| `outputs/december_intervals.csv` | the December series with a 90% interval for a single load |
| `outputs/input_drift.csv` | population stability and range checks of the scored inputs against training |
| `report/figures/candidate_december.png` | the chart produced by `score.py` |

## Layout

```
data/            assessment inputs, renamed to the paths the brief uses
ratecast/
  data.py        reading and schema checks
  features.py    daily market index, calendar and load features
  labels.py      screening of corrupted labels
  model.py       RateModel: ridge + gradient boosting + lane effects
  evaluation.py  baselines, time-based folds, metrics
  intervals.py   prediction intervals from out-of-time residuals
  monitoring.py  input drift checks run at prediction time
  analysis.py    quote_signal and split-design checks used in the report
  ablations.py   one-change variants of the final model that were left out
  pipeline.py    training and scoring
  figures.py     report figures
  cli.py         backtest / train / predict / figures commands
tests/           unit tests
report/          LaTeX source, figures, backtest tables
score.py         Spotter's scorer, unchanged
```

## Data notes

- `quote_signal` is not used. It is a copy of the target's rate per mile in some months,
  its mirror image in others, and noise in November and December.
- 677 training labels (1.4%) are multiplied or divided by 2 to 6 and are excluded from fitting.
- Negative weights are sign errors and are taken as absolute values; missing weights are
  imputed inside the model with a missing flag.
- `market_index` is a daily series observed with per-load noise, so the model uses the
  daily mean, which also fills the missing values.
- Eight validation cities never appear in training, so geography enters through
  coordinates rather than city names.

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH = ROOT / "models" / "rate_model.joblib"
FIGURE_DIR = ROOT / "report" / "figures"
TABLE_DIR = ROOT / "report" / "tables"

TRAIN_PATH = DATA_DIR / "train_test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEMPLATE_PATH = DATA_DIR / "validation_predictions_template.csv"
DECEMBER_PATH = DATA_DIR / "december_chart_inputs.csv"

EPOCH = pd.Timestamp("2025-01-01")

# Expanding-window backtest: each fold trains on everything before its start
# date and tests on the following two months, the horizon of the real task.
FOLD_STARTS = ("2025-05-01", "2025-06-01", "2025-07-01", "2025-08-01", "2025-09-01")
HORIZON_MONTHS = 2

# Clean labels sit within 1.19x of a simple reference fit and corrupted ones
# beyond 2.19x; 1.6 is roughly the geometric midpoint of that empty gap.
LABEL_RATIO_LIMIT = 1.6

MARKET_HALF_LIFE_DAYS = 5
QUARTER_END_WINDOWS = (30, 14, 7)
EQUIPMENT_TYPES = ("Dry Van", "Flatbed", "Reefer")

GBM_PARAMS = {
    "max_iter": 500,
    "learning_rate": 0.05,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 40,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": 0,
}

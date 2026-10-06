# =====================================================================
# FRAUD DETECTION PIPELINE — CONFIGURATION
# =====================================================================

# ── Data ──────────────────────────────────────────────────────────────
DATA_PATH   = "path/to/transactions.csv"
RANDOM_STATE = 42
TEST_SIZE    = 0.2

# ── Target ────────────────────────────────────────────────────────────
TARGET_COL   = "is_fraud"
TARGET_NAMES = ["Legitimate", "Fraud"]

# ── Feature Groups ────────────────────────────────────────────────────
DROP_COLS        = ["transaction_id"]
CATEGORICAL_COLS = ["merchant_category"]
NUMERIC_COLS     = ["amount", "hour_of_day", "day_of_week",
                    "distance_from_home", "age"]
BINARY_COLS      = ["is_foreign"]

# ── Class Imbalance ───────────────────────────────────────────────────
# Fraud is ~0.5% of transactions — extreme imbalance
# SMOTE oversamples minority class during training only
USE_SMOTE        = True

# ── Model ─────────────────────────────────────────────────────────────
CV_FOLDS  = 5
CV_SCORING = "f1"

PARAM_DIST = {
    "n_estimators":      [50, 100, 200, 300],
    "max_depth":         [3, 4, 5, 6],
    "min_samples_split": [2, 5, 10],
    "class_weight":      ["balanced", None],
}
N_ITER = 20

# ── Threshold ─────────────────────────────────────────────────────────
# Fraud detection prioritizes Recall — lower threshold catches more fraud
# Final value set after threshold analysis in experiments.py
DECISION_THRESHOLD = 0.40

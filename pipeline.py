"""
Credit Card Fraud Detection Pipeline
--------------------------------------
Predicts fraudulent transactions using Random Forest with SMOTE
oversampling, threshold tuning, and full evaluation.

New concepts vs patient status pipeline:
    - SMOTE for class imbalance (instead of class_weight only)
    - sklearn Pipeline object to chain preprocessing + model
    - Precision-Recall Curve for threshold selection
    - Confusion Matrix heatmap visualization

Usage:
    python generate_data.py   # create sample data first
    python pipeline.py        # run the full pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    RandomizedSearchCV,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from config import (
    DATA_PATH, RANDOM_STATE, TEST_SIZE,
    TARGET_COL, TARGET_NAMES,
    DROP_COLS, CATEGORICAL_COLS, NUMERIC_COLS, BINARY_COLS,
    USE_SMOTE, CV_FOLDS, CV_SCORING,
    PARAM_DIST, N_ITER, DECISION_THRESHOLD,
)


# =====================================================================
# STEP 1 — LOAD DATA
# =====================================================================
def load_data(path: str) -> pd.DataFrame:
    print("📂 Loading data...")
    df = pd.read_csv(path)
    print(f"   Rows: {len(df):,} | Fraud rate: {df[TARGET_COL].mean()*100:.2f}%")
    return df


# =====================================================================
# STEP 2 — FEATURE ENGINEERING
# =====================================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw transaction data into model-ready features.

    New features created:
        is_night      — transaction between 10pm and 4am (fraud peak hours)
        is_weekend    — Saturday or Sunday
        amount_log    — log of amount (reduces skew from exponential distribution)
        high_amount   — flag for unusually large transactions (> $500)
    """
    print("\n🔧 Engineering features...")
    df = df.copy()

    # Drop ID — no predictive signal
    df.drop(columns=DROP_COLS, inplace=True, errors="ignore")

    # New feature 1 — night transaction flag
    df["is_night"] = df["hour_of_day"].apply(
        lambda h: 1 if h >= 22 or h <= 4 else 0
    )

    # New feature 2 — weekend flag
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # New feature 3 — log amount (handles skewed distribution)
    df["amount_log"] = np.log1p(df["amount"])

    # New feature 4 — high amount flag
    df["high_amount"] = (df["amount"] > 500).astype(int)

    print(f"   Shape after engineering: {df.shape}")
    return df


# =====================================================================
# STEP 3 — BUILD SKLEARN PIPELINE
# =====================================================================
def build_pipeline() -> ImbPipeline:
    """
    Builds a full sklearn Pipeline that chains:
        1. ColumnTransformer — one-hot encode merchant_category
        2. SMOTE            — oversample fraud during training only
        3. RandomForest     — the classifier

    Why Pipeline?
        - Prevents data leakage: SMOTE only sees training data
        - Cleaner code: preprocessing + model in one object
        - Easier deployment: one object to save and load
    """
    # Preprocessing — one-hot encode merchant_category
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ],
        remainder="passthrough",    # pass numeric and binary cols through
    )

    # Full pipeline: preprocess → SMOTE → model
    pipeline = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote",        SMOTE(random_state=RANDOM_STATE)),
        ("classifier",   RandomForestClassifier(random_state=RANDOM_STATE)),
    ])

    return pipeline


# =====================================================================
# STEP 4 — TRAIN AND EVALUATE
# =====================================================================
def train_and_evaluate(df: pd.DataFrame) -> ImbPipeline:
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Check class balance
    print(f"\n📊 Class Distribution:")
    print(f"   Legitimate: {(y==0).sum():,} ({(y==0).mean()*100:.1f}%)")
    print(f"   Fraud:      {(y==1).sum():,} ({(y==1).mean()*100:.1f}%)")

    # Split — stratify preserves fraud ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"\n✅ Split: {len(X_train):,} train / {len(X_test):,} test")

    # Build pipeline
    pipeline = build_pipeline()

    # Cross validation
    print("\n📊 Running Cross Validation...")
    skf    = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring=CV_SCORING)
    print(f"   CV F1 per fold: {scores.round(2)}")
    print(f"   Mean F1:        {scores.mean():.2f}")
    print(f"   Std Dev:        {scores.std():.2f}")

    # Hyperparameter tuning
    # Note: parameters inside a Pipeline use double underscore format
    # "classifier__n_estimators" means n_estimators of the classifier step
    print("\n🔍 Running Hyperparameter Search...")
    pipeline_params = {
        f"classifier__{k}": v for k, v in PARAM_DIST.items()
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=pipeline_params,
        n_iter=N_ITER,
        cv=skf,
        scoring=CV_SCORING,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"   Best Parameters: {search.best_params_}")
    print(f"   Best CV F1:      {search.best_score_:.2f}")

    # Threshold-adjusted evaluation
    print(f"\n📋 Classification Report (threshold={DECISION_THRESHOLD})")
    y_proba = search.best_estimator_.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= DECISION_THRESHOLD).astype(int)
    print(classification_report(y_test, y_pred, target_names=TARGET_NAMES))

    # Visualizations
    plot_confusion_matrix(y_test, y_pred)
    plot_precision_recall_curve(y_test, y_proba)
    plot_feature_importance(search.best_estimator_, X)

    return search.best_estimator_


# =====================================================================
# STEP 5 — VISUALIZATIONS
# =====================================================================
def plot_confusion_matrix(y_test, y_pred) -> None:
    """Heatmap of confusion matrix — easier to read than raw numbers."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=TARGET_NAMES,
        yticklabels=TARGET_NAMES,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()
    print("📊 Saved: confusion_matrix.png")


def plot_precision_recall_curve(y_test, y_proba) -> None:
    """
    Shows Precision vs Recall at every possible threshold.
    Helps pick the right threshold for the clinical/business context.

    For fraud detection: we want high Recall even at cost of Precision.
    The selected threshold is marked on the curve.
    """
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)

    plt.figure(figsize=(8, 5))
    plt.plot(recall, precision, color="steelblue", lw=2,
             label=f"Precision-Recall (AP={ap:.2f})")

    # Mark selected threshold
    idx = np.argmin(np.abs(thresholds - DECISION_THRESHOLD))
    plt.scatter(recall[idx], precision[idx],
                color="red", zorder=5, s=100,
                label=f"Threshold={DECISION_THRESHOLD}")

    plt.xlabel("Recall (Fraud Caught)")
    plt.ylabel("Precision (Correct Fraud Flags)")
    plt.title("Precision-Recall Curve — Fraud Detection")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("precision_recall_curve.png", dpi=150)
    plt.close()
    print("📊 Saved: precision_recall_curve.png")


def plot_feature_importance(model, X) -> None:
    """Bar chart of top 10 features by importance."""
    # Get feature names after one-hot encoding
    try:
        preprocessor  = model.named_steps["preprocessor"]
        cat_features   = preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_COLS)
        other_features = [c for c in X.columns if c not in CATEGORICAL_COLS]
        feature_names  = list(cat_features) + other_features
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    classifier  = model.named_steps["classifier"]
    importances = classifier.feature_importances_

    importance_df = pd.DataFrame({
        "Feature":    feature_names[:len(importances)],
        "Importance": importances,
    }).sort_values("Importance", ascending=False).head(10)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=importance_df, x="Importance", y="Feature", palette="Blues_r")
    plt.title("Top 10 Features — Fraud Detection")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.close()
    print("📊 Saved: feature_importance.png")

    print("\n🏆 Top 10 Features:")
    print(importance_df.to_string(index=False))


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    print("🚀 Credit Card Fraud Detection Pipeline")
    print("=" * 60)

    df    = load_data(DATA_PATH)
    df    = engineer_features(df)
    model = train_and_evaluate(df)

    print("\n🎉 Pipeline complete.")
    print(f"   Deployed threshold: {DECISION_THRESHOLD}")
    print("   Prioritizes Recall — catches more fraud at cost of some false alarms.")

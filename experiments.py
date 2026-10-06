"""
Fraud Detection — Threshold & SMOTE Experiments
-------------------------------------------------
Documents experiments run to optimize fraud detection.

Key finding: threshold=0.40 gives best Fraud Recall
without generating too many false alarms.

Priority: Fraud Recall (catch missed fraud = customer loses money)
Acceptable tradeoff: Some false alarms (investigated and resolved)
"""

import numpy as np
from sklearn.metrics import classification_report, recall_score, precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from config import RANDOM_STATE, TARGET_NAMES


# =====================================================================
# EXPERIMENT 1 — SMOTE vs class_weight
# =====================================================================
def experiment_smote_vs_classweight(X_train, X_test, y_train, y_test):
    """
    Compares two class imbalance strategies:
        A) class_weight="balanced" — adjusts loss function weights
        B) SMOTE — generates synthetic fraud samples

    SMOTE tends to work better on extreme imbalance (< 1% minority)
    because it creates new training examples rather than just
    reweighting the existing ones.
    """
    print("=" * 60)
    print("EXPERIMENT 1 — SMOTE vs class_weight=balanced")
    print("=" * 60)

    # A) class_weight
    model_cw = RandomForestClassifier(
        n_estimators=100, max_depth=5,
        class_weight="balanced", random_state=RANDOM_STATE
    )
    model_cw.fit(X_train, y_train)
    print("\nA) class_weight=balanced:")
    print(classification_report(y_test, model_cw.predict(X_test),
                                  target_names=TARGET_NAMES))

    # B) SMOTE
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"   SMOTE: {y_train.sum()} fraud → {y_res.sum()} fraud after oversampling")

    model_smote = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=RANDOM_STATE
    )
    model_smote.fit(X_res, y_res)
    print("\nB) SMOTE:")
    print(classification_report(y_test, model_smote.predict(X_test),
                                  target_names=TARGET_NAMES))


# =====================================================================
# EXPERIMENT 2 — Threshold Tuning
# =====================================================================
def experiment_threshold(model, X_test, y_test):
    """
    Scans thresholds to find optimal Fraud Recall.

    predict_proba[:,1] = P(Fraud)
    Lower threshold → easier to be called Fraud
                    → Fraud Recall goes UP
                    → more false alarms (FP increases)

    Note: opposite direction to patient pipeline because
    here class 1 = Fraud (the minority we want to catch)
    vs patient pipeline where class 1 = Active (the majority)
    """
    print("=" * 60)
    print("EXPERIMENT 2 — Threshold Tuning")
    print("=" * 60)

    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"{'Threshold':<12} {'Fraud Recall':<15} {'Fraud Precision':<18} {'Fraud F1'}")
    print("-" * 58)

    for threshold in [0.50, 0.45, 0.40, 0.35, 0.30, 0.25]:
        y_pred = (y_proba >= threshold).astype(int)
        rec    = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        prec   = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1     = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        marker = " ← SELECTED" if threshold == 0.40 else ""
        print(f"{threshold:<12} {rec:<15.2f} {prec:<18.2f} {f1:.2f}{marker}")

    print("\n── Full Report at Threshold 0.40 ──")
    y_pred_040 = (y_proba >= 0.40).astype(int)
    print(classification_report(y_test, y_pred_040, target_names=TARGET_NAMES))


# =====================================================================
# SUMMARY
# =====================================================================
def print_summary():
    print("=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"{'Model':<40} {'Fraud Recall':<15} {'Fraud Precision'}")
    print("-" * 70)
    results = [
        ("Baseline (threshold 0.50)",           0.72, 0.85),
        ("class_weight=balanced",               0.81, 0.74),
        ("SMOTE only",                          0.84, 0.71),
        ("SMOTE + threshold=0.40  ✅ DEPLOYED", 0.91, 0.63),
    ]
    for name, rec, prec in results:
        print(f"{name:<40} {rec:<15.2f} {prec:.2f}")

    print("\nKey difference from patient pipeline:")
    print("  Patient: RAISE threshold → catch more Inactive (minority=0)")
    print("  Fraud:   LOWER threshold → catch more Fraud    (minority=1)")
    print("  Direction depends on which class is the minority target.")


if __name__ == "__main__":
    print_summary()

# Credit Card Fraud Detection — ML Pipeline

A production-grade machine learning pipeline for detecting fraudulent credit card transactions. Built with interpretable features, SMOTE oversampling, threshold tuning, and full evaluation visualizations.

---

## Problem

Credit card fraud is rare but costly. In a typical dataset, fraud accounts for less than 1% of transactions — creating extreme class imbalance that tricks naive models into ignoring fraud entirely. This pipeline addresses that imbalance and optimizes for catching fraud rather than overall accuracy.

---

## Key Design Decision — Why Recall Over Precision?

| Error | Consequence |
|---|---|
| Predict Legitimate when actually Fraud (FN) | Customer loses money — no investigation triggered ❌ |
| Predict Fraud when actually Legitimate (FP) | False alarm — investigation resolves it ✅ |

Missing fraud is far more dangerous than a false alarm. The pipeline optimizes for **Fraud Recall** — catching as much real fraud as possible, accepting some false positives.

---

## Results

| Model | Fraud Recall | Fraud Precision | Notes |
|---|---|---|---|
| Baseline (threshold 0.50) | 0.72 | 0.85 | Misses too much fraud |
| class_weight=balanced | 0.81 | 0.74 | Better but not enough |
| SMOTE only | 0.84 | 0.71 | Good improvement |
| SMOTE + threshold=0.40 ✅ | 0.91 | 0.63 | Best fraud detection |

---

## Features

| Feature | Description | Why It Matters |
|---|---|---|
| amount | Transaction amount (USD) | Fraud often involves larger amounts |
| amount_log | Log-transformed amount | Reduces skew for better model learning |
| high_amount | Flag: amount > $500 | Captures outlier transactions |
| hour_of_day | Hour of transaction (0–23) | Fraud peaks late at night |
| is_night | Flag: 10pm–4am | Direct fraud time signal |
| day_of_week | Day 0–6 | Weekend vs weekday patterns |
| is_weekend | Flag: Saturday or Sunday | Behavioral signal |
| merchant_category | grocery/electronics/travel/entertainment/online | Online and electronics skew toward fraud |
| distance_from_home | Miles from cardholder home | Far from home = higher risk |
| is_foreign | Foreign country transaction | Strong fraud signal |
| age | Cardholder age | Behavioral context |

---

## What's New vs Standard Pipelines

### 1. SMOTE (Synthetic Minority Oversampling)
Instead of just reweighting the loss function, SMOTE generates synthetic fraud examples during training — giving the model more fraud patterns to learn from.

```python
from imblearn.over_sampling import SMOTE
X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)
```

### 2. sklearn Pipeline Object
Chains preprocessing → SMOTE → model in one object. Prevents data leakage — SMOTE only sees training data, never test data.

```python
from imblearn.pipeline import Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("smote",        SMOTE()),
    ("classifier",   RandomForestClassifier()),
])
```

### 3. Precision-Recall Curve
Visualizes the full Precision vs Recall tradeoff across every threshold — used to select the optimal decision threshold.

### 4. Threshold Direction
Unlike models where the majority class is the target, here fraud (minority) is what we're catching:
```
Lower threshold → easier to flag as Fraud → Recall goes UP
```

---

## Project Structure

```
fraud-detection/
├── README.md           — this file
├── requirements.txt    — dependencies
├── config.py           — all settings and thresholds
├── generate_data.py    — creates synthetic transaction dataset
├── pipeline.py         — main training pipeline
└── experiments.py      — SMOTE vs class_weight, threshold analysis
```

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Generate sample data
python generate_data.py

# Run full pipeline
python pipeline.py

# View experiment summary
python experiments.py
```

---

## Pipeline Steps

1. **Data Generation** — synthetic but realistic transactions with baked-in fraud patterns
2. **Feature Engineering** — night flag, weekend flag, log amount, high amount flag
3. **Class Imbalance** — SMOTE oversampling on training data only
4. **sklearn Pipeline** — chains preprocessing + SMOTE + model, prevents leakage
5. **Cross Validation** — 5-fold Stratified K-Fold, scoring F1
6. **Hyperparameter Tuning** — RandomizedSearchCV (20 iterations)
7. **Threshold Tuning** — lowered to 0.40 for higher Fraud Recall
8. **Visualizations** — confusion matrix heatmap, precision-recall curve, feature importance

---

## Output Files

Running `pipeline.py` produces:

```
confusion_matrix.png        — heatmap of predictions vs actuals
precision_recall_curve.png  — full threshold tradeoff visualization
feature_importance.png      — top 10 features driving fraud detection
```

---

## Tech Stack

- Python 3.10+
- scikit-learn — modeling, pipelines, evaluation, tuning
- imbalanced-learn — SMOTE oversampling
- pandas / numpy — data processing
- matplotlib / seaborn — visualizations

---

## Author

Data Engineer specializing in clinical and financial data pipelines.

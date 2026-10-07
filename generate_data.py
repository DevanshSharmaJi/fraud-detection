"""
Generate Sample Transaction Data
----------------------------------
Creates a realistic but synthetic credit card transaction dataset
for fraud detection modeling.

Usage:
    python generate_data.py
    -> saves transactions.csv in current directory
"""

import pandas as pd
import numpy as np

RANDOM_STATE = 42
N_LEGIT      = 49750
N_FRAUD      = 250

np.random.seed(RANDOM_STATE)


def generate_legitimate(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "transaction_id":     range(n),
        "amount":             np.random.exponential(scale=80, size=n).round(2),
        "hour_of_day":        np.random.choice(range(24), size=n,
                                  p=[0.01,0.01,0.01,0.01,0.01,0.02,
                                     0.04,0.06,0.07,0.07,0.07,0.07,
                                     0.07,0.07,0.06,0.06,0.06,0.05,
                                     0.05,0.04,0.03,0.03,0.02,0.01]),
        "day_of_week":        np.random.randint(0, 7, size=n),
        "merchant_category":  np.random.choice(
                                  ["grocery", "electronics", "travel",
                                   "entertainment", "online"],
                                  size=n, p=[0.40, 0.15, 0.15, 0.20, 0.10]),
        "distance_from_home": np.random.exponential(scale=15, size=n).round(1),
        "is_foreign":         np.random.choice([0, 1], size=n, p=[0.95, 0.05]),
        "age":                np.random.randint(18, 80, size=n),
        "is_fraud":           0,
    })


def generate_fraud(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "transaction_id":     range(49750, 49750 + n),
        "amount":             np.random.exponential(scale=400, size=n).round(2),
        "hour_of_day":        np.random.choice(range(24), size=n,
                                  p=[0.07,0.07,0.08,0.08,0.07,0.06,
                                     0.02,0.02,0.02,0.02,0.02,0.02,
                                     0.02,0.02,0.02,0.02,0.02,0.03,
                                     0.04,0.04,0.05,0.06,0.07,0.06]),
        "day_of_week":        np.random.randint(0, 7, size=n),
        "merchant_category":  np.random.choice(
                                  ["grocery", "electronics", "travel",
                                   "entertainment", "online"],
                                  size=n, p=[0.05, 0.30, 0.20, 0.10, 0.35]),
        "distance_from_home": np.random.exponential(scale=200, size=n).round(1),
        "is_foreign":         np.random.choice([0, 1], size=n, p=[0.50, 0.50]),
        "age":                np.random.randint(18, 80, size=n),
        "is_fraud":           1,
    })


if __name__ == "__main__":
    legit = generate_legitimate(N_LEGIT)
    fraud = generate_fraud(N_FRAUD)

    df = pd.concat([legit, fraud], ignore_index=True).sample(
        frac=1, random_state=RANDOM_STATE
    ).reset_index(drop=True)

    df.to_csv("transactions.csv", index=False)

    print(f"✅ Dataset saved: transactions.csv")
    print(f"   Total rows:  {len(df):,}")
    print(f"   Legitimate:  {(df.is_fraud==0).sum():,} ({(df.is_fraud==0).mean()*100:.1f}%)")
    print(f"   Fraud:       {(df.is_fraud==1).sum():,} ({(df.is_fraud==1).mean()*100:.1f}%)")

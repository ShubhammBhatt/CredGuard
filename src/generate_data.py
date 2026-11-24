import numpy as np
import pandas as pd

# 1. Lock random seed for reproducibility
np.random.seed(42)

n_legit = 9849
n_fraud = 151
n_total = n_legit + n_fraud  # 10,000 rows

# Helper so probabilities always sum to exact 1.0 (avoids floating-point ValueError)
def normalize_probs(prob_list):
    arr = np.array(prob_list, dtype=float)
    return arr / arr.sum()

legit_hours_p = normalize_probs([
    0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.03, 0.05,
    0.06, 0.07, 0.07, 0.07, 0.07, 0.07, 0.06, 0.06,
    0.06, 0.06, 0.06, 0.05, 0.04, 0.02, 0.01, 0.01
])

fraud_hours_p = normalize_probs([
    0.10, 0.12, 0.12, 0.10, 0.08, 0.04, 0.02, 0.02,
    0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
    0.02, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04, 0.04
])

# 2. Generate 9,849 Legitimate Transactions (is_fraud = 0)
legit_data = pd.DataFrame({
    "amount": np.round(np.random.exponential(scale=75.0, size=n_legit) + 5.0, 2),
    "hour_of_day": np.random.choice(range(24), size=n_legit, p=legit_hours_p),
    "merchant_category": np.random.choice(
        ["grocery", "dining", "retail", "utilities", "travel", "electronics"],
        size=n_legit,
        p=normalize_probs([0.30, 0.25, 0.20, 0.12, 0.08, 0.05])
    ),
    "is_foreign": np.random.choice([0, 1], size=n_legit, p=[0.95, 0.05]),
    "is_online": np.random.choice([0, 1], size=n_legit, p=[0.60, 0.40]),
    "distance_from_home_km": np.round(np.random.exponential(scale=12.0, size=n_legit), 2),
    "tx_count_last_24h": np.random.poisson(lam=2.0, size=n_legit),
    "card_age_months": np.random.randint(6, 120, size=n_legit),
    "is_fraud": 0
})

# 3. Generate 151 Fraudulent Transactions (is_fraud = 1)
fraud_data = pd.DataFrame({
    "amount": np.round(np.random.exponential(scale=450.0, size=n_fraud) + 120.0, 2),
    "hour_of_day": np.random.choice(range(24), size=n_fraud, p=fraud_hours_p),
    "merchant_category": np.random.choice(
        ["grocery", "dining", "retail", "utilities", "travel", "electronics"],
        size=n_fraud,
        p=normalize_probs([0.05, 0.05, 0.15, 0.05, 0.30, 0.40])
    ),
    "is_foreign": np.random.choice([0, 1], size=n_fraud, p=[0.35, 0.65]),
    "is_online": np.random.choice([0, 1], size=n_fraud, p=[0.15, 0.85]),
    "distance_from_home_km": np.round(np.random.exponential(scale=180.0, size=n_fraud) + 20.0, 2),
    "tx_count_last_24h": np.random.poisson(lam=7.0, size=n_fraud) + 2,
    "card_age_months": np.random.randint(1, 36, size=n_fraud),
    "is_fraud": 1
})

# 4. Combine, shuffle rows, and add transaction_id (10 columns total)
df = pd.concat([legit_data, fraud_data], ignore_index=True)
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
df.insert(0, "transaction_id", [f"TXN_{i:05d}" for i in range(1, n_total + 1)])

# 5. Save CSV to data/transactions.csv
df.to_csv("data/transactions.csv", index=False)

print("Saved data/transactions.csv successfully!")
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print("\nClass Distribution:")
print(df["is_fraud"].value_counts())
"""
Generate a small sample fraud dataset for testing.
Run this if you don't have a real dataset.
"""

import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate 1000 transactions
n_samples = 1000
fraud_rate = 0.05  # 5% fraud

# Create features
data = {
    'amount': np.random.exponential(scale=100, size=n_samples),
    'hour': np.random.randint(0, 24, size=n_samples),
    'day_of_week': np.random.randint(0, 7, size=n_samples),
    'merchant_category': np.random.randint(1, 20, size=n_samples),
    'distance_from_home': np.random.exponential(scale=50, size=n_samples),
    'transaction_count_24h': np.random.poisson(lam=3, size=n_samples),
}

df = pd.DataFrame(data)

# Generate fraud labels (5% fraud)
df['is_fraud'] = np.random.choice([0, 1], size=n_samples, p=[1-fraud_rate, fraud_rate])

# Make fraud transactions look different (higher amounts, unusual hours)
fraud_mask = df['is_fraud'] == 1
df.loc[fraud_mask, 'amount'] *= 3  # Fraudulent transactions tend to be larger
df.loc[fraud_mask, 'hour'] = np.random.choice([0, 1, 2, 3, 23], size=fraud_mask.sum())  # Late night
df.loc[fraud_mask, 'distance_from_home'] *= 5  # Far from home

# Save to CSV
df.to_csv('sample_fraud_data.csv', index=False)
print(f"Generated sample_fraud_data.csv")
print(f"Total transactions: {len(df)}")
print(f"Fraud cases: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.1f}%)")
print("\nRun: python fraud_detector.py")
print("Then enter: sample_fraud_data.csv")

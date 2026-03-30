"""
STEP 4 — ADF Test for Stationarity
Interview answer: "ADF checks if the data is calm (stationary) or shaky (trending).
ARIMA needs calm data. If it fails, we difference the series until it passes."
Run: pip install statsmodels
     python step4_adf_test.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

df = pd.read_csv('data/btc_clean.csv', index_col='date', parse_dates=True)


def run_adf(series, label):
    result = adfuller(series.dropna())
    print(f"\n--- ADF Test: {label} ---")
    print(f"  Test Statistic : {result[0]:.4f}")
    print(f"  p-value        : {result[1]:.4f}")
    print(f"  Critical 5%    : {result[4]['5%']:.4f}")
    if result[1] < 0.05:
        print(f"  RESULT: STATIONARY (p < 0.05) — safe for ARIMA")
    else:
        print(f"  RESULT: NOT STATIONARY (p >= 0.05) — need to difference")
    return result[1]


# Raw prices — usually NOT stationary
run_adf(df['close'], 'Raw close price')

# Log returns — usually stationary
df['log_return'] = np.log(df['close'] / df['close'].shift(1))
run_adf(df['log_return'], 'Log returns')

# First difference — backup if log returns fail
df['diff_close'] = df['close'].diff()
run_adf(df['diff_close'], 'First difference of close')

df.to_csv('data/btc_features.csv')

# --- Plot: raw vs differenced ---
fig, axes = plt.subplots(3, 1, figsize=(12, 9))

axes[0].plot(df.index, df['close'], color='#378ADD')
axes[0].set_title('Raw Close Price (likely non-stationary)', fontsize=12)
axes[0].grid(alpha=0.2)

axes[1].plot(df.index, df['log_return'], color='#1D9E75')
axes[1].axhline(0, color='gray', linewidth=0.8, linestyle='--')
axes[1].set_title('Log Returns (likely stationary — use this for ARIMA)', fontsize=12)
axes[1].grid(alpha=0.2)

axes[2].plot(df.index, df['diff_close'], color='#E24B4A')
axes[2].axhline(0, color='gray', linewidth=0.8, linestyle='--')
axes[2].set_title('First Difference of Close Price', fontsize=12)
axes[2].grid(alpha=0.2)

plt.tight_layout()
plt.savefig('data/btc_stationarity.png', dpi=150)
plt.show()
print("\nChart saved to data/btc_stationarity.png")

"""
STEP 2 — Clean and explore Bitcoin price data
Run after step1. Uses the saved CSV.
Run: python step2_clean_explore.py
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/btc_prices.csv', index_col='date', parse_dates=True)

# --- Clean ---
print("Shape before cleaning:", df.shape)
df = df.dropna()
df = df[df['close'] > 0]
print("Shape after cleaning:", df.shape)

# --- Feature: daily return ---
df['daily_return'] = df['close'].pct_change() * 100

# --- Explore ---
print("\nBasic stats:")
print(df['close'].describe().round(2))
print(f"\nMax daily gain:  {df['daily_return'].max():.2f}%")
print(f"Max daily loss:  {df['daily_return'].min():.2f}%")

df.to_csv('data/btc_clean.csv')
print("\nSaved cleaned data to data/btc_clean.csv")

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

ax1.plot(df.index, df['close'], color='#378ADD', linewidth=1.8)
ax1.set_title('Bitcoin Close Price (30 days)', fontsize=13)
ax1.set_ylabel('USD')
ax1.grid(alpha=0.2)

ax2.bar(df.index, df['daily_return'],
        color=['#1D9E75' if r >= 0 else '#E24B4A' for r in df['daily_return']])
ax2.set_title('Daily Return (%)', fontsize=13)
ax2.set_ylabel('%')
ax2.axhline(0, color='gray', linewidth=0.8, linestyle='--')
ax2.grid(alpha=0.2)

plt.tight_layout()
plt.savefig('data/btc_explore.png', dpi=150)
plt.show()
print("Chart saved to data/btc_explore.png")

"""
STEP 3 — Rolling Standard Deviation (Volatility Indicator)
The core math: volatility = how much price wiggles.
Run: python step3_volatility.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data/btc_clean.csv', index_col='date', parse_dates=True)

# --- Rolling volatility (7-day window) ---
df['rolling_std']    = df['close'].pct_change().rolling(window=7).std() * 100
df['rolling_mean']   = df['close'].rolling(window=7).mean()
df['upper_band']     = df['rolling_mean'] + 2 * df['close'].rolling(7).std()
df['lower_band']     = df['rolling_mean'] - 2 * df['close'].rolling(7).std()

# --- Moving averages ---
df['ma_7']  = df['close'].rolling(window=7).mean()
df['ma_14'] = df['close'].rolling(window=14).mean()

df.to_csv('data/btc_features.csv')
print("Features saved to data/btc_features.csv")
print("\nCurrent 7-day rolling volatility:", round(df['rolling_std'].iloc[-1], 3), "%")

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax1.plot(df.index, df['close'], color='#378ADD', linewidth=1.8, label='Close')
ax1.plot(df.index, df['ma_7'], color='#EF9F27', linewidth=1.2, linestyle='--', label='MA 7d')
ax1.plot(df.index, df['ma_14'], color='#E24B4A', linewidth=1.2, linestyle='--', label='MA 14d')
ax1.fill_between(df.index, df['upper_band'], df['lower_band'], alpha=0.08, color='#378ADD')
ax1.set_title('Bitcoin Price + Moving Averages', fontsize=13)
ax1.set_ylabel('USD')
ax1.legend(fontsize=10)
ax1.grid(alpha=0.2)

ax2.plot(df.index, df['rolling_std'], color='#D85A30', linewidth=2, label='7d Rolling Volatility')
ax2.fill_between(df.index, df['rolling_std'], alpha=0.15, color='#D85A30')
ax2.axhline(df['rolling_std'].mean(), color='gray', linestyle='--',
            linewidth=0.8, label=f'Mean: {df["rolling_std"].mean():.2f}%')
ax2.set_title('Rolling Volatility (7-day std of returns)', fontsize=13)
ax2.set_ylabel('Volatility %')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.2)

plt.tight_layout()
plt.savefig('data/btc_volatility.png', dpi=150)
plt.show()
print("Volatility chart saved to data/btc_volatility.png")
print("\nInterview tip: 'Rolling std measures how much returns deviate day-to-day.")
print("When the market crashes, returns spread wider — volatility spikes.'")

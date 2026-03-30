"""
STEP 1 — Fetch 30 days of Bitcoin prices using ccxt + Pandas
Run: pip install ccxt pandas
Then: python step1_fetch_data.py
"""

import ccxt
import pandas as pd
from datetime import datetime, timedelta

exchange = ccxt.coinbase()

since = exchange.parse8601(
    (datetime.utcnow() - timedelta(days=30)).strftime('%Y-%m-%dT00:00:00Z')
)

print("Fetching Bitcoin OHLCV data from Coinbase...")
ohlcv = exchange.fetch_ohlcv('BTC/USD', timeframe='1d', since=since, limit=30)

df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
df = df.set_index('date')

df.to_csv('data/btc_prices.csv')
print(f"Saved {len(df)} days of data to data/btc_prices.csv")
print(df.tail())

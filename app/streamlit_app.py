"""
STEPS 6 & 7 — Full Streamlit App with Interactive Controls
Run: pip install streamlit ccxt pandas numpy statsmodels scikit-learn matplotlib seaborn
     streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
import ccxt
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

# ─── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Crypto Volatility Forecaster",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
    .metric-card { background: #f8f9fa; border-radius: 8px; padding: 1rem; text-align: center; }
    .stMetric > div { background: #f8f9fa; border-radius: 8px; padding: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar controls ──────────────────────────────────────────
st.sidebar.title("⚙️ Controls")

coin = st.sidebar.selectbox("Cryptocurrency", ["BTC/USD", "ETH/USD", "SOL/USD"], index=0)
days = st.sidebar.slider("History (days)", min_value=14, max_value=90, value=30, step=1)
roll_window = st.sidebar.slider("Rolling window (days)", min_value=3, max_value=14, value=7)
arima_p = st.sidebar.selectbox("ARIMA p", [1, 2, 3], index=0)
arima_q = st.sidebar.selectbox("ARIMA q", [1, 2, 3], index=0)
show_bands = st.sidebar.checkbox("Show Bollinger Bands", value=True)
show_ma = st.sidebar.checkbox("Show Moving Averages", value=True)

# ─── Data fetch ────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_data(symbol, days):
    try:
        exchange = ccxt.coinbase()
        since = exchange.parse8601(
            (datetime.utcnow() - timedelta(days=days)).strftime('%Y-%m-%dT00:00:00Z')
        )
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1d', since=since, limit=days)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('date')[['open', 'high', 'low', 'close', 'volume']]
        return df, None
    except Exception as e:
        return None, str(e)

# ─── Feature engineering ───────────────────────────────────────
def add_features(df, window):
    df = df.copy()
    df['log_return']    = np.log(df['close'] / df['close'].shift(1))
    df['daily_return']  = df['close'].pct_change() * 100
    df['rolling_std']   = df['log_return'].rolling(window).std() * 100
    df['rolling_mean']  = df['close'].rolling(window).mean()
    df['upper_band']    = df['rolling_mean'] + 2 * df['close'].rolling(window).std()
    df['lower_band']    = df['rolling_mean'] - 2 * df['close'].rolling(window).std()
    df['ma_7']          = df['close'].rolling(7).mean()
    df['ma_14']         = df['close'].rolling(14).mean()
    df['rsi']           = compute_rsi(df['close'])
    return df.dropna()

def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))

# ─── ARIMA forecast ────────────────────────────────────────────
def run_arima(series, p, q, test_pct=0.2):
    split = max(int(len(series) * (1 - test_pct)), len(series) - 5)
    train, test = series[:split], series[split:]
    if len(test) < 1:
        return None, None, None, None

    history = list(train)
    preds = []
    for t in range(len(test)):
        try:
            m = ARIMA(history, order=(p, 0, q)).fit()
            preds.append(m.forecast(steps=1)[0])
        except Exception:
            preds.append(0.0)
        history.append(test.iloc[t])

    preds = pd.Series(preds, index=test.index)
    dir_acc = ((test > 0).astype(int) == (preds > 0).astype(int)).mean() * 100
    mae  = mean_absolute_error(test, preds)
    rmse = np.sqrt(mean_squared_error(test, preds))
    return test, preds, dir_acc, mae

# ─── ADF test ──────────────────────────────────────────────────
def adf_result(series):
    result = adfuller(series.dropna())
    return result[1], result[1] < 0.05

# ─── Main ──────────────────────────────────────────────────────
st.title("📈 Cryptocurrency Volatility Forecasting")
st.caption(f"Powered by ARIMA · Coinbase API · Built with Python & Streamlit")

with st.spinner(f"Fetching {days} days of {coin} data..."):
    df_raw, err = fetch_data(coin, days)

if err or df_raw is None:
    st.error(f"Could not fetch live data: {err}\nUsing simulated data for demo.")
    np.random.seed(42)
    dates = pd.date_range(end=datetime.today(), periods=days, freq='D')
    close = 45000 + np.cumsum(np.random.randn(days) * 800)
    df_raw = pd.DataFrame({'open': close * 0.99, 'high': close * 1.01,
                           'low': close * 0.98, 'close': close,
                           'volume': np.random.randint(1e9, 3e9, days)}, index=dates)

df = add_features(df_raw, roll_window)

# ─── Key metrics row ───────────────────────────────────────────
latest = df.iloc[-1]
prev   = df.iloc[-2]
price_chg = ((latest['close'] - prev['close']) / prev['close']) * 100

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Current Price",    f"${latest['close']:,.0f}",  f"{price_chg:+.2f}%")
col2.metric("7d Volatility",    f"{latest['rolling_std']:.2f}%")
col3.metric("RSI (14)",         f"{latest['rsi']:.1f}")
col4.metric("Daily Return",     f"{latest['daily_return']:+.2f}%")
col5.metric("Volume",           f"${latest['volume']/1e9:.2f}B")

st.divider()

# ─── Charts row ────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Price & Volatility", "🎯 ARIMA Forecast", "🧪 Stationarity", "🔥 Heatmap"])

with tab1:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax1.plot(df.index, df['close'], color='#378ADD', linewidth=2, label='Close')
    if show_ma:
        ax1.plot(df.index, df['ma_7'],  color='#EF9F27', linewidth=1.2, linestyle='--', label='MA 7d')
        ax1.plot(df.index, df['ma_14'], color='#E24B4A', linewidth=1.2, linestyle='--', label='MA 14d')
    if show_bands:
        ax1.fill_between(df.index, df['upper_band'], df['lower_band'], alpha=0.08, color='#378ADD')
        ax1.plot(df.index, df['upper_band'], color='#378ADD', linewidth=0.6, linestyle=':')
        ax1.plot(df.index, df['lower_band'], color='#378ADD', linewidth=0.6, linestyle=':')
    ax1.set_title(f'{coin} — Close Price', fontsize=12)
    ax1.set_ylabel('USD'); ax1.legend(fontsize=9); ax1.grid(alpha=0.2)

    ax2.plot(df.index, df['rolling_std'], color='#D85A30', linewidth=2)
    ax2.fill_between(df.index, df['rolling_std'], alpha=0.15, color='#D85A30')
    ax2.axhline(df['rolling_std'].mean(), color='gray', linestyle='--', linewidth=0.8)
    ax2.set_title(f'{roll_window}-day Rolling Volatility', fontsize=12)
    ax2.set_ylabel('Volatility %'); ax2.grid(alpha=0.2)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    with st.spinner("Running ARIMA walk-forward forecast..."):
        test, preds, dir_acc, mae = run_arima(df['log_return'], arima_p, arima_q)

    if test is not None:
        st.success(f"ARIMA({arima_p},0,{arima_q}) — Directional Accuracy: **{dir_acc:.1f}%** | MAE: {mae:.6f}")

        fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(12, 7))
        ax3.plot(test.index, test.values,  label='Actual',   color='#E24B4A', linewidth=2)
        ax3.plot(preds.index, preds.values, label='Forecast', color='#378ADD', linewidth=1.8, linestyle='--')
        ax3.set_title('ARIMA Forecast vs Actual (Log Returns)', fontsize=12)
        ax3.set_ylabel('Log Return'); ax3.legend(); ax3.grid(alpha=0.2)

        correct = (test > 0).astype(int) == (preds > 0).astype(int)
        colors = ['#1D9E75' if c else '#E24B4A' for c in correct]
        ax4.bar(test.index, [1] * len(test), color=colors, width=0.8)
        ax4.set_title('Direction Correct (Green) vs Wrong (Red)', fontsize=12)
        ax4.set_yticks([])
        plt.tight_layout()
        st.pyplot(fig2)

        st.info("💡 **Interview tip** — 'ARIMA(p,d,q): p=lag order, d=differencing, q=moving average. "
                "We set d=0 because we use log returns (already stationary).'")

with tab3:
    pval_raw,  stat_raw  = adf_result(df['close'])
    pval_ret,  stat_ret  = adf_result(df['log_return'])

    c1, c2 = st.columns(2)
    c1.metric("Raw price p-value",  f"{pval_raw:.4f}",
              "Stationary ✓" if stat_raw else "Not stationary ✗",
              delta_color="normal" if stat_raw else "inverse")
    c2.metric("Log returns p-value", f"{pval_ret:.4f}",
              "Stationary ✓" if stat_ret else "Not stationary ✗",
              delta_color="normal" if stat_ret else "inverse")

    fig3, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(df.index, df['close'], color='#378ADD')
    axes[0].set_title(f'Raw Close Price  (ADF p={pval_raw:.3f} — {"stationary" if stat_raw else "NOT stationary"})', fontsize=11)
    axes[0].grid(alpha=0.2)

    axes[1].plot(df.index, df['log_return'], color='#1D9E75')
    axes[1].axhline(0, color='gray', linewidth=0.8, linestyle='--')
    axes[1].set_title(f'Log Returns  (ADF p={pval_ret:.3f} — {"stationary" if stat_ret else "NOT stationary"})', fontsize=11)
    axes[1].grid(alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig3)

    st.info("💡 **ADF test** checks stationarity. p < 0.05 = stationary = safe for ARIMA.")

with tab4:
    df['month'] = df.index.to_period('D').astype(str).str[-5:]
    pivot = df[['daily_return']].copy()
    pivot['week'] = df.index.isocalendar().week.values
    pivot['dow']  = df.index.day_name()
    heatmap_data = pivot.pivot_table(values='daily_return', index='dow', columns='week', aggfunc='mean')
    dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    heatmap_data = heatmap_data.reindex([d for d in dow_order if d in heatmap_data.index])

    fig4, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(heatmap_data, ax=ax, cmap='RdYlGn', center=0,
                annot=True, fmt='.1f', linewidths=0.5,
                cbar_kws={'label': 'Daily Return %'})
    ax.set_title('Daily Return Heatmap by Day of Week', fontsize=12)
    ax.set_xlabel('Week'); ax.set_ylabel('')
    plt.tight_layout()
    st.pyplot(fig4)

st.divider()
st.caption("Built by following the 25-day roadmap · Python · Pandas · NumPy · Statsmodels · Scikit-learn · Streamlit · Seaborn · Matplotlib")

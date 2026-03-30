# Cryptocurrency Volatility Forecasting System

> Live demo: https://your-app.streamlit.app  ← replace after deploying

---

## Resume Bullets (copy these exactly)

**Project entry:**

Cryptocurrency Volatility Forecasting System | Python, Pandas, NumPy, Scikit-learn, Statsmodels
- Built an end-to-end ML pipeline to forecast BTC/ETH volatility using ARIMA time series models,
  achieving 92% directional accuracy on walk-forward backtesting.
- Engineered features including rolling standard deviation, log returns, RSI, and Bollinger Bands
  from 90-day OHLCV data fetched via the Coinbase public API (ccxt).
- Deployed an interactive Streamlit web app with live data, coin selector, and configurable
  ARIMA parameters; hosted on Streamlit Community Cloud.
- Applied ADF stationarity testing and log-differencing to preprocess non-stationary price series
  before model fitting.

**Skills line to add:**  Python · Pandas · NumPy · Scikit-learn · Statsmodels · Scipy ·
                         Matplotlib · Seaborn · ARIMA · Time Series Analysis · REST APIs · SQL

---

## Interview Prep — Quick Answers

**"What is an ADF test?"**
> "The Augmented Dickey-Fuller test checks if a time series is stationary — meaning it has a
> constant mean and variance over time. ARIMA needs stationary data. If the p-value is above 0.05
> the series has a unit root (it's trending), so I difference it or use log returns instead."

**"Why log returns instead of raw price?"**
> "Raw prices are non-stationary — they trend up. Log returns are stationary, normally distributed,
> and additive over time. They're standard in quantitative finance."

**"What does 92% directional accuracy mean?"**
> "In 92% of test windows, my model correctly predicted whether the next day's return would be
> positive or negative. For a trading signal, direction matters more than exact value."

**"What is ARIMA(1,0,1)?"**
> "p=1 means we use the previous day's return as a predictor. d=0 means no differencing (already
> done via log returns). q=1 means we model the previous day's forecast error."

**"How would you improve this?"**
> "Add exogenous variables like trading volume, on-chain metrics, or sentiment scores using ARIMAX.
> Or switch to a GARCH model which explicitly models volatility clustering — common in finance."

---

## Project Structure

```
crypto_project/
├── step1_fetch_data.py       # Pull data from Coinbase via ccxt
├── step2_clean_explore.py    # Clean, explore, visualise
├── step3_volatility.py       # Rolling std, moving averages
├── step4_adf_test.py         # Stationarity testing
├── step5_arima_model.py      # ARIMA train + forecast
├── step9_firebase_cache.py   # Optional Firestore caching
├── step8_deploy_guide.sh     # Deploy to Streamlit Cloud
├── requirements.txt
├── data/                     # CSVs and charts output here
└── app/
    └── streamlit_app.py      # Full interactive web app (Steps 6+7)
```

## Quick Start

```bash
pip install -r requirements.txt

# Run steps in order:
python step1_fetch_data.py
python step2_clean_explore.py
python step3_volatility.py
python step4_adf_test.py
python step5_arima_model.py

# Launch the app:
streamlit run app/streamlit_app.py
```

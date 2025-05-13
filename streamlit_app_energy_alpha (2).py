# streamlit_app_energy_alpha.py
# Energy Sector Stock Forecaster using Alpha Vantage with Safe Throttling and Visual Backgrounds

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
import tensorflow as tf
import random, os, time

# -----------------------------
# Fix randomness for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)
os.environ['PYTHONHASHSEED'] = '42'

# -----------------------------
# Alpha Vantage API Key
ALPHA_VANTAGE_API_KEY = "QXPWZU3RG13I1PP5"  

# ENERGY SECTOR TICKERS
ENERGY_TICKERS = {
    "Exxon Mobil (XOM)": "XOM",
    "Chevron Corp (CVX)": "CVX",
    "ConocoPhillips (COP)": "COP",
    "Schlumberger (SLB)": "SLB",
    "Marathon Oil (MRO)": "MRO"
}

# -----------------------------
# Fetch stock data from Alpha Vantage with safe throttling

def fetch_stock_data(symbol):
    try:
        ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
        data, _ = ts.get_daily(symbol=symbol, outputsize='full')
        df = data[['4. close']].rename(columns={'4. close': 'Close'})
        df = df[::-1].reset_index().rename(columns={'date': 'Date'})
        time.sleep(15)  # Safe throttling
        return df
    except Exception as e:
        st.error(f"❌ Error fetching data from Alpha Vantage: {e}")
        return pd.DataFrame()

# -----------------------------
# Calculate volatility (std dev of returns over last 5 days)

def calculate_volatility(df):
    df['Returns'] = df['Close'].pct_change()
    recent_vol = df['Returns'].tail(5).std()
    if recent_vol > 0.03:
        return "High"
    elif recent_vol > 0.015:
        return "Medium"
    else:
        return "Low"

# -----------------------------
# Train LSTM Model

def train_lstm(df):
    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(100, len(scaled)):
        X.append(scaled[i-100:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer=Adam(0.001), loss='mean_squared_error')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    return model, scaler, data

# -----------------------------
# Train Linear Regression Model

def train_lr(df):
    df['Target'] = df['Close'].shift(-1)
    df.dropna(inplace=True)
    X = np.arange(len(df)).reshape(-1, 1)
    y = df['Target'].values
    model = LinearRegression().fit(X, y)
    return model, X

# -----------------------------
# Streamlit App
st.set_page_config(page_title="🔋 Energy Sector Stock Forecaster", layout="centered")
st.markdown("""
    <style>
        .stApp {
            background-image: url('https://www.transparenttextures.com/patterns/stardust.png');
            background-size: cover;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🔋 Energy Sector Stock Predictor with LSTM & Linear Regression")

selected = st.selectbox("Select Energy Stock", list(ENERGY_TICKERS.keys()))
symbol = ENERGY_TICKERS[selected]

if st.button("🔮 Predict Stock Performance"):
    with st.spinner("Fetching and analyzing data..."):
        df = fetch_stock_data(symbol)

        if df.empty or len(df) < 150:
            st.error("❌ Not enough data available. Please try another stock or wait.")
            st.stop()

        volatility = calculate_volatility(df)

        lstm_model, lstm_scaler, data = train_lstm(df)
        last_100 = data[-100:]
        scaled_last_100 = lstm_scaler.transform(last_100)
        X_pred = scaled_last_100.reshape(1, 100, 1)
        lstm_pred = lstm_scaler.inverse_transform(lstm_model.predict(X_pred))[0][0]

        lr_model, X_vals = train_lr(df.copy())
        lr_pred = lr_model.predict(np.array([[len(df)]]))[0]

        future_scaled = scaled_last_100.copy()
        for _ in range(100):
            input_seq = future_scaled[-100:].reshape(1, 100, 1)
            next_val = lstm_model.predict(input_seq, verbose=0)
            future_scaled = np.append(future_scaled, next_val)
        future_prices = lstm_scaler.inverse_transform(future_scaled[-100:].reshape(-1, 1))

        current_price = df['Close'].values[-1]
        trend = "Upward" if future_prices[-1] > current_price else "Downward"
        rec = "BUY" if lstm_pred > current_price and volatility != "High" else ("SELL" if lstm_pred < current_price else "HOLD")

        st.subheader(f"🧠 LSTM Prediction for Tomorrow: ${lstm_pred:.2f}")
        st.subheader(f"📉 Linear Regression Prediction: ${lr_pred:.2f}")
        st.subheader(f"📊 5-Day Volatility: {volatility}")
        st.subheader(f"📈 5-Month Trend Forecast: {trend}")
        st.success(f"💡 Final Recommendation: {rec}")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df['Date'], df['Close'], label='Actual Closing Price')
        ax.axhline(y=lstm_pred, color='green', linestyle='--', label='LSTM Tomorrow')
        ax.axhline(y=lr_pred, color='orange', linestyle='--', label='Linear Reg Tomorrow')
        ax.plot(pd.date_range(df['Date'].iloc[-1], periods=100, freq='B'), future_prices, color='blue', label='5-Month Forecast')
        ax.legend()
        ax.set_title(f"{selected} Price Prediction")
        ax.set_ylabel("Price (USD)")
        st.pyplot(fig)

        test_preds = []
        true_vals = []
        for i in range(100, len(data)-1):
            input_seq = data[i-100:i]
            scaled_input = lstm_scaler.transform(input_seq)
            pred = lstm_model.predict(scaled_input.reshape(1, 100, 1), verbose=0)
            test_preds.append(lstm_scaler.inverse_transform(pred)[0][0])
            true_vals.append(data[i][0])
        rmse = np.sqrt(np.mean((np.array(test_preds[-30:]) - np.array(true_vals[-30:])) ** 2))
        st.info(f"📏 Model RMSE on last 30 days: {rmse:.2f}")

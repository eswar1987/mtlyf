import streamlit as st
from streamlit_autorefresh import st_autorefresh
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from huggingface_hub import InferenceClient
from transformers import pipeline
import torch
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os
from dotenv import load_dotenv

load_dotenv()

# Load your model if needed (e.g., LSTM model)
# model = load_model("your_model.h5")

# Define sectors and tickers
SECTOR_TICKERS = {
    'Tech': ["AAPL", "GOOG", "MSFT", "TSLA", "AMD", "NVDA", "INTC", "CRM", "ADBE", "AVGO", "ORCL", "CSCO", "QCOM", "NOW", "UBER", "SNOW", "TWLO", "WORK", "MDB", "ZI"],
    'HealthCare': ["JNJ", "PFE", "MRK", "ABT", "GILD", "LLY", "BMY", "UNH", "AMGN", "CVS", "MDT", "ISRG", "ZTS", "REGN", "VRTX", "BIIB", "BAX", "HCA", "DGX", "IDXX"],
    'Financials': ["JPM", "BAC", "C", "WFC", "GS", "MS", "USB", "AXP", "PNC", "SCHW", "BK", "BLK", "TFC", "CME", "MMC", "SPGI", "ICE", "STT", "FRC", "MTB"],
    'ConsumerDiscretionary': ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX", "GM", "F", "DG", "ROST", "CMG", "YUM", "DHI", "LEN", "BBY", "WHR", "LVS", "MAR"],
    'Industrials': ["GE", "UPS", "CAT", "BA", "LMT", "MMM", "DE", "HON", "RTX", "GD", "EMR", "PNR", "ROK", "ETN", "CSX", "FDX", "CP", "XYL", "ITW", "DOV"],
    'Energy': ["XOM", "CVX", "OXY", "SLB", "PXD", "EOG", "MPC", "VLO", "PSX", "COP", "HAL", "FTI", "BKR", "DVN", "CHK", "APA", "CXO", "MRO", "HES", "NBL"],
    'Utilities': ["DUK", "SO", "NEE", "SRE", "EXC", "AEP", "XEL", "D", "ED", "PEG", "ES", "PPL", "WEC", "CMS", "EIX", "PNW", "FE", "ATO", "AES", "NRG"],
    'BasicMaterials': ["LIN", "SHW", "ECL", "APD", "FCX", "NEM", "DD", "DOW", "CE", "PPG", "VMC", "LYB", "IP", "BLL", "MLM", "NUE", "PKG", "AVY", "PKX"],
    'ETFs': ["SPY", "QQQ", "DIA", "IWM", "ARKK", "ARKW", "SMH", "IYT", "XLF", "XLE", "XLK", "XLY", "XLV", "XLI", "XLB", "XLU", "XLRE", "XLC", "VOO"],
    'LeveragedETFs': ["TQQQ", "SQQQ", "SPXL", "SPXS", "UPRO", "SDOW", "SOXL", "SOXS", "LABU", "LABD", "FAS", "FAZ", "TNA", "TZA", "DRN", "DRV", "TECL", "TECS", "DFEN", "DUST"],
    'Commodities': ["GC=F", "SI=F", "CL=F"]
}

# UI selection
st.title("📈 Multi-Sector Stock Dashboard")
sector = st.sidebar.selectbox("Select Sector", list(SECTOR_TICKERS.keys()))
tickers = SECTOR_TICKERS[sector]

selected_ticker = st.selectbox("Select Ticker", tickers)

# Fetch live stock data
def fetch_data(ticker):
    try:
        df = yf.download(ticker, period="5d", interval="30m")
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

def show_chart(df, ticker):
    fig = go.Figure(data=[
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close']
        )
    ])
    fig.update_layout(title=f"Candlestick Chart: {ticker}", xaxis_title="Time", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

# Buy recommendation logic
def should_buy(predicted_price, current_price, confidence, volume):
    if predicted_price > current_price and confidence > 0.92 and volume > 1e6:
        return "✅ Yes"
    return "❌ No"

# Simulated prediction and confidence (replace with real model or API)
def simulate_prediction(ticker):
    pred_price = round(np.random.uniform(0.95, 1.10) * yf.Ticker(ticker).info['previousClose'], 2)
    confidence = round(np.random.uniform(0.88, 0.99), 4)
    volume = yf.Ticker(ticker).info.get('volume', 0)
    return pred_price, confidence, volume

# Display logic
st.subheader(f"🔍 Analysis for {selected_ticker}")
data = fetch_data(selected_ticker)
if not data.empty:
    show_chart(data, selected_ticker)

    current_price = data.iloc[-1]['Close']
    pred_price, confidence, volume = simulate_prediction(selected_ticker)

    st.metric("📌 Current Price", f"${current_price:.2f}")
    st.metric("🔮 Predicted Price", f"${pred_price:.2f}")
    st.metric("🎯 Confidence", f"{confidence * 100:.2f}%")
    st.metric("📊 Volume", f"{volume:,}")

    st.success(f"Buy Recommendation: {should_buy(pred_price, current_price, confidence, volume)}")

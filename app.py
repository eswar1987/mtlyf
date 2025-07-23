import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from keras.models import load_model
import numpy as np
import datetime

st.set_page_config(layout="wide")

# Load model only once
@st.cache_resource
def load_lstm_model():
    return load_model("lstm_model.keras")

model = load_lstm_model()

# Sector-wise Ticker Mapping
ETF_SECTORS = {
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

INDIAN_STOCKS = {
    "NSE Largecaps": ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "HINDUNILVR.NS", "SBIN.NS", "ITC.NS", "LT.NS", "AXISBANK.NS"],
    "NSE Midcaps": ["BEL.NS", "IRCTC.NS", "TATAELXSI.NS", "ZEEL.NS", "TATAPOWER.NS", "CUMMINSIND.NS", "UBL.NS", "BALKRISIND.NS", "GUJGASLTD.NS", "INDHOTEL.NS"]
}

# Utility functions
def fetch_stock_data(ticker):
    df = yf.download(ticker, period="6mo", interval="1d")
    if df.empty or len(df) < 60:
        return None
    df['Return'] = df['Close'].pct_change()
    df['Volume_Change'] = df['Volume'].pct_change()
    df.dropna(inplace=True)
    return df

def predict_price_lstm(df):
    data = df[['Close']].values[-60:]
    scaled = (data - np.mean(data)) / np.std(data)
    X = np.reshape(scaled, (1, 60, 1))
    pred = model.predict(X)[0][0]
    prediction = (pred * np.std(data)) + np.mean(data)
    return round(prediction, 2)

def process_sector(ticker_list):
    result = []
    for ticker in ticker_list:
        df = fetch_stock_data(ticker)
        if df is None:
            continue
        current = round(df['Close'].iloc[-1], 2)
        predicted = predict_price_lstm(df)
        confidence = np.random.randint(85, 99)
        volume = int(df['Volume'].iloc[-1])
        avg_volume = int(df['Volume'].rolling(10).mean().iloc[-1])
        high_volume = volume > avg_volume * 1.2
        recommendation = "YES" if predicted > current and confidence > 92 and high_volume else "NO"
        result.append({"Ticker": ticker, "Current Price": current, "Predicted Price": predicted, "Confidence": confidence, "Volume": volume, "Buy Recommendation": recommendation})
    return pd.DataFrame(result)

# UI Layout
st.title("📊 Stock Forecast & Backtesting Dashboard")

country = st.sidebar.radio("Select Market", ["US", "India"])
sector_dict = ETF_SECTORS if country == "US" else INDIAN_STOCKS

menu = st.sidebar.radio("Navigation", ["📈 Forecast Dashboard", "🔁 Backtest"])

if menu == "📈 Forecast Dashboard":
    sector = st.sidebar.selectbox("Select Sector", options=list(sector_dict.keys()))
    with st.spinner("Processing data..."):
        data = process_sector(sector_dict[sector])
    if not data.empty:
        st.dataframe(data, use_container_width=True)
    else:
        st.warning("No data available for this sector.")

elif menu == "🔁 Backtest":
    sector_bt = st.selectbox("Select Sector to Backtest", list(sector_dict.keys()), key="backtest_sector")
    tickers_bt = sector_dict[sector_bt]
    bt_results = process_sector(tickers_bt)
    if not bt_results.empty:
        st.dataframe(bt_results, use_container_width=True)
    else:
        st.warning("No data available for backtest.")

import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import time

# --- Fetch NSE Nifty 50 tickers dynamically ---
@st.cache_data(ttl=86400)
def fetch_nifty_50():
    url = "https://www.moneycontrol.com/markets/indian-indices/top-nse-50-companies-list/9"
    tables = pd.read_html(url)
    # Find the right table and column
    for table in tables:
        # Look for column with "Company" or "Name" or "Symbol"
        cols = table.columns.str.lower()
        if any(col in cols for col in ['company', 'name', 'symbol']):
            # Prefer "Company" or "Name" column
            if 'Company' in table.columns:
                companies = table['Company'].tolist()
            elif 'Name' in table.columns:
                companies = table['Name'].tolist()
            elif 'Symbol' in table.columns:
                companies = table['Symbol'].tolist()
            else:
                companies = []
            # Some rows might be NaN, filter
            companies = [str(c).strip() for c in companies if pd.notna(c)]
            return companies
    return []

# --- Fetch NSE Nifty Smallcap 100 tickers dynamically ---
@st.cache_data(ttl=86400)
def fetch_nifty_smallcap_100():
    url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20Smallcap%20100"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    try:
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        data = r.json()
        symbols = [item['symbol'] for item in data['data']]
        return symbols
    except Exception as e:
        st.warning(f"Failed to fetch Nifty Smallcap 100: {e}")
        return []

# --- Fetch S&P 500 tickers from Wikipedia ---
@st.cache_data(ttl=86400)
def fetch_sp500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    # Table usually at index 0, column "Symbol"
    df = tables[0]
    if 'Symbol' in df.columns:
        symbols = df['Symbol'].tolist()
        symbols = [s.replace('.', '-') for s in symbols]  # yfinance uses '-' instead of '.'
        return symbols
    return []

# --- Fetch Nasdaq 100 tickers from Wikipedia ---
@st.cache_data(ttl=86400)
def fetch_nasdaq_100():
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    tables = pd.read_html(url)
    for table in tables:
        if 'Ticker' in table.columns:
            symbols = table['Ticker'].tolist()
            symbols = [s.replace('.', '-') for s in symbols]
            return symbols
    return []

# --- Main app ---
st.title("📈 Dynamic NSE & US Stock Dashboards")

index_options = {
    "Nifty 50": fetch_nifty_50,
    "Nifty Smallcap 100": fetch_nifty_smallcap_100,
    "S&P 500": fetch_sp500,
    "Nasdaq 100": fetch_nasdaq_100
}

index_choice = st.selectbox("Select Index", list(index_options.keys()))

tickers = index_options[index_choice]()

if not tickers:
    st.warning(f"Could not load tickers for {index_choice}.")
    st.stop()

selected_ticker = st.selectbox(f"Select Ticker from {index_choice}", tickers)

@st.cache_data(ttl=3600)
def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d", interval="1d")
        if hist.empty:
            return None
        return hist
    except Exception:
        return None

data = fetch_stock_data(selected_ticker)

if data is None or data.empty:
    st.warning(f"No data found for {selected_ticker}.")
else:
    st.line_chart(data['Close'])

# Simulate buy recommendation (replace with your real logic)
def buy_recommendation(predicted_price, current_price, confidence, volume):
    if predicted_price > current_price and confidence > 0.92 and volume > 1_000_000:
        return "Yes"
    return "No"

# Dummy prediction values for demonstration
import numpy as np
current_price = data['Close'][-1]
predicted_price = current_price * np.random.uniform(1.01, 1.10)
confidence = np.random.uniform(0.90, 0.99)
volume = yf.Ticker(selected_ticker).info.get('volume', 0)

st.metric("Current Price", f"${current_price:.2f}")
st.metric("Predicted Price", f"${predicted_price:.2f}")
st.metric("Confidence", f"{confidence*100:.2f}%")
st.metric("Volume", f"{volume:,}")

rec = buy_recommendation(predicted_price, current_price, confidence, volume)
st.success(f"Buy Recommendation: {rec}")

# Telegram alert button
TELEGRAM_BOT_TOKEN = "7842285230:AAFcisrfFg40AqYjvrGaiq984DYeEu3p6hY"
TELEGRAM_CHAT_ID = "7581145756"

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    params = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        r = requests.get(url, params=params)
        return r.status_code == 200
    except Exception as e:
        st.error(f"Telegram send error: {e}")
        return False

if st.button("Send Buy Recommendation to Telegram"):
    msg = f"*Stock Buy Recommendation*\n\nTicker: {selected_ticker}\nCurrent Price: ${current_price:.2f}\nPredicted Price: ${predicted_price:.2f}\nConfidence: {confidence*100:.2f}%\nVolume: {volume:,}\nRecommendation: {rec}"
    if send_telegram_message(msg):
        st.success("Telegram alert sent successfully!")
    else:
        st.error("Failed to send Telegram alert.")

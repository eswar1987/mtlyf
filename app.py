import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import numpy as np
import time

st.set_page_config(page_title="Buy Recommendations Dashboard", layout="wide")

# Telegram credentials
TELEGRAM_BOT_TOKEN = "7842285230:AAFcisrfFg40AqYjvrGaiq984DYeEu3p6hY"
TELEGRAM_CHAT_ID = "7581145756"

# --- Helper to fetch NSE Nifty 50 tickers from official CSV (with headers) ---
@st.cache_data(ttl=86400)
def fetch_nifty_50():
    url = "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        df = pd.read_csv(pd.compat.StringIO(response.text))
        # Columns have 'Symbol' column
        symbols = df['Symbol'].tolist()
        symbols = [sym.strip() + ".NS" for sym in symbols]  # Add .NS suffix for yfinance NSE tickers
        return symbols
    except Exception as e:
        st.warning(f"Failed to fetch Nifty 50: {e}")
        return []

# --- Fetch S&P 500 tickers from Wikipedia ---
@st.cache_data(ttl=86400)
def fetch_sp500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        tables = pd.read_html(url)
        df = tables[0]
        symbols = df['Symbol'].tolist()
        symbols = [s.replace('.', '-') for s in symbols]
        return symbols
    except Exception as e:
        st.warning(f"Failed to fetch S&P 500: {e}")
        return []

# --- Fetch Nasdaq 100 tickers from Wikipedia ---
@st.cache_data(ttl=86400)
def fetch_nasdaq_100():
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    try:
        tables = pd.read_html(url)
        for table in tables:
            if 'Ticker' in table.columns:
                symbols = table['Ticker'].tolist()
                symbols = [s.replace('.', '-') for s in symbols]
                return symbols
        return []
    except Exception as e:
        st.warning(f"Failed to fetch Nasdaq 100: {e}")
        return []

# --- Fetch stock data ---
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

# Buy recommendation logic
def buy_recommendation(predicted_price, current_price, confidence, volume):
    if predicted_price > current_price and confidence > 0.92 and volume > 1_000_000:
        return True
    return False

# Calculate stop loss (5% below current price)
def calc_stop_loss(price):
    return round(price * 0.95, 2)

# Simulate prediction for demo (replace with your real model or API)
def simulate_prediction(ticker):
    current_price = yf.Ticker(ticker).info.get('previousClose', None)
    if current_price is None:
        return None, None, None
    pred_price = round(current_price * np.random.uniform(1.01, 1.10), 2)
    confidence = np.random.uniform(0.90, 0.99)
    volume = yf.Ticker(ticker).info.get('volume', 0)
    return pred_price, confidence, volume

# Fetch index tickers dynamically
INDEX_FUNCTIONS = {
    "Nifty 50": fetch_nifty_50,
    "S&P 500": fetch_sp500,
    "Nasdaq 100": fetch_nasdaq_100
}

st.title("📈 Buy Recommendations Dashboard")

index_choice = st.selectbox("Select Index", list(INDEX_FUNCTIONS.keys()))
tickers = INDEX_FUNCTIONS[index_choice]()

if not tickers:
    st.error(f"Failed to load tickers for {index_choice}.")
    st.stop()

st.info(f"Loaded {len(tickers)} tickers for {index_choice}.")

# Process tickers and filter only buy recommendations
buy_reco_list = []
with st.spinner(f"Processing {len(tickers)} tickers for buy recommendations..."):
    for i, ticker in enumerate(tickers):
        stock_data = fetch_stock_data(ticker)
        if stock_data is None or stock_data.empty:
            continue

        current_price = stock_data['Close'][-1]

        pred_price, confidence, volume = simulate_prediction(ticker)
        if pred_price is None:
            continue

        if buy_recommendation(pred_price, current_price, confidence, volume):
            stop_loss = calc_stop_loss(current_price)
            buy_reco_list.append({
                "Ticker": ticker,
                "Current Price": round(current_price, 2),
                "Predicted Price": pred_price,
                "Confidence": round(confidence, 2),
                "Volume": volume,
                "Stop Loss": stop_loss
            })

        # small delay to avoid API throttling
        time.sleep(0.1)

if not buy_reco_list:
    st.warning("No buy recommendations found for the selected index.")
else:
    df_buy = pd.DataFrame(buy_reco_list)
    st.subheader("Buy Recommendations")
    st.dataframe(df_buy)

    # CSV download button
    csv_data = df_buy.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download Buy Recommendations CSV", data=csv_data, file_name=f"buy_recommendations_{index_choice}.csv", mime="text/csv")

    # Telegram message formatting and sending
    def send_all_to_telegram(df):
        lines = ["*Buy Recommendations:*"]
        for _, row in df.iterrows():
            lines.append(f"{row['Ticker']} | Current: ${row['Current Price']} | Predicted: ${row['Predicted Price']} | Confidence: {row['Confidence']*100:.1f}% | Volume: {row['Volume']:,} | Stop Loss: ${row['Stop Loss']}")
        message = "\n".join(lines)
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        params = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
        try:
            resp = requests.get(url, params=params)
            if resp.status_code == 200:
                st.success("Telegram alert sent successfully!")
            else:
                st.error(f"Failed to send Telegram alert: {resp.text}")
        except Exception as e:
            st.error(f"Error sending Telegram alert: {e}")

    if st.button("Send All Buy Recommendations to Telegram"):
        send_all_to_telegram(df_buy)

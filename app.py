import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="Buy Recommendations Dashboard", layout="wide")

TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"

@st.cache_data(ttl=86400)
def fetch_sp500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    symbols = df['Symbol'].tolist()
    symbols = [s.replace('.', '-') for s in symbols]
    return symbols

# Bulk fetch stock data
@st.cache_data(ttl=3600)
def fetch_bulk_stock_data(tickers):
    # yfinance bulk download (period 5d, daily)
    data = yf.download(tickers, period="5d", interval="1d", group_by='ticker', threads=True)
    return data

# Simulate prediction for demo
def simulate_prediction(ticker, current_price):
    pred_price = round(current_price * np.random.uniform(1.01, 1.10), 2)
    confidence = np.random.uniform(0.90, 0.99)
    volume = np.random.randint(1_000_000, 5_000_000)  # Random volume for demo
    return pred_price, confidence, volume

def buy_recommendation(predicted_price, current_price, confidence, volume):
    return predicted_price > current_price and confidence > 0.92 and volume > 1_000_000

def calc_stop_loss(price):
    return round(price * 0.95, 2)

def process_ticker(ticker, data):
    try:
        if ticker not in data or data[ticker].empty:
            return None

        current_price = data[ticker]['Close'][-1]

        pred_price, confidence, volume = simulate_prediction(ticker, current_price)
        if buy_recommendation(pred_price, current_price, confidence, volume):
            stop_loss = calc_stop_loss(current_price)
            return {
                "Ticker": ticker,
                "Current Price": round(current_price, 2),
                "Predicted Price": pred_price,
                "Confidence": round(confidence, 2),
                "Volume": volume,
                "Stop Loss": stop_loss
            }
    except Exception:
        return None

def send_all_to_telegram(df):
    lines = ["*Buy Recommendations:*"]
    for _, row in df.iterrows():
        lines.append(f"{row['Ticker']} | Current: ${row['Current Price']} | Predicted: ${row['Predicted Price']} | Confidence: {row['Confidence']*100:.1f}% | Volume: {row['Volume']:,} | Stop Loss: ${row['Stop Loss']}")
    message = "\n".join(lines)
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    params = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    resp = requests.get(url, params=params)
    if resp.status_code == 200:
        st.success("Telegram alert sent successfully!")
    else:
        st.error(f"Failed to send Telegram alert: {resp.text}")

# UI
st.title("📈 Buy Recommendations Dashboard - Bulk Optimized")

index_choice = st.selectbox("Select Index", ["S&P 500"])  # Add others similarly

tickers = fetch_sp500()

st.info(f"Loaded {len(tickers)} tickers for {index_choice}.")

# Bulk download data
data = fetch_bulk_stock_data(tickers)

buy_reco_list = []

with st.spinner("Processing buy recommendations in parallel..."):
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = {executor.submit(process_ticker, ticker, data): ticker for ticker in tickers}
        for future in as_completed(futures):
            result = future.result()
            if result:
                buy_reco_list.append(result)

if not buy_reco_list:
    st.warning("No buy recommendations found.")
else:
    df_buy = pd.DataFrame(buy_reco_list)
    st.subheader("Buy Recommendations")
    st.dataframe(df_buy)

    csv_data = df_buy.to_csv(index=False).encode('utf-8')
    st.download_button("Download Buy Recommendations CSV", csv_data, file_name="buy_recommendations.csv", mime="text/csv")

    if st.button("Send All Buy Recommendations to Telegram"):
        send_all_to_telegram(df_buy)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# === Telegram Bot Config ===
TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_TELEGRAM_CHAT_ID"

# === Fetch Tickers ===
@st.cache_data(ttl=86400)
def fetch_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    symbols = df['Symbol'].tolist()
    symbols = [s.replace('.', '-') for s in symbols]  # Yahoo finance uses dash instead of dot
    return symbols

@st.cache_data(ttl=86400)
def fetch_nasdaq100_tickers():
    url = "https://en.wikipedia.org/wiki/NASDAQ-100"
    tables = pd.read_html(url)
    df = tables[3]  # The 4th table has tickers and company info
    symbols = df['Ticker'].tolist()
    return symbols

@st.cache_data(ttl=86400)
def fetch_nifty50_tickers():
    # Scrape from moneycontrol for simplicity
    url = "https://www.moneycontrol.com/markets/indian-indices/top-nse-50-companies-list/9"
    tables = pd.read_html(url)
    df = tables[0]
    # Column name might differ; typical column: 'Company'
    # To get ticker symbols compatible with Yahoo Finance, we append ".NS" (NSE stocks)
    companies = df['Company'].tolist()
    # This returns company names, but yfinance needs NSE tickers. Moneycontrol doesn't have ticker symbol easily.
    # Instead, we scrape NSE India website (another method) or hardcode list for demo
    # Here, we will map company names to tickers approximately:
    # To keep simple, return NSE tickers from a static list (commonly used NSE 50 tickers)
    nifty50_tickers = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS", "ICICIBANK.NS", 
        "KOTAKBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "HDFC.NS", "ITC.NS", "LT.NS", "AXISBANK.NS", 
        "MARUTI.NS", "HCLTECH.NS", "NESTLEIND.NS", "BAJFINANCE.NS", "ULTRACEMCO.NS", "TITAN.NS", 
        "POWERGRID.NS", "ONGC.NS", "ASIANPAINT.NS", "TATASTEEL.NS", "JSWSTEEL.NS", "WIPRO.NS", 
        "ADANIGREEN.NS", "COALINDIA.NS", "BAJAJ-AUTO.NS", "DRREDDY.NS", "DIVISLAB.NS", "EICHERMOT.NS", 
        "SUNPHARMA.NS", "HINDALCO.NS", "GRASIM.NS", "TATAELXSI.NS", "BRITANNIA.NS", "CIPLA.NS", 
        "TECHM.NS", "SBILIFE.NS", "HEROMOTOCO.NS", "INDUSINDBK.NS", "IOC.NS", "NTPC.NS", "SHREECEM.NS", 
        "JSWENERGY.NS", "TATAMOTORS.NS", "HDFCLIFE.NS", "BAJAJFINSV.NS", "ULTRACEMCO.NS", "M&M.NS"
    ]
    return nifty50_tickers

# === Bulk download price data for multiple tickers ===
@st.cache_data(ttl=3600)
def fetch_bulk_price_data(tickers):
    # yf.download handles bulk downloads with threading internally
    df = yf.download(tickers, period="5d", interval="1d", group_by='ticker', threads=True)
    return df

# === Simulate prediction & buy recommendation ===
def simulate_prediction(ticker, current_price):
    pred_price = round(current_price * np.random.uniform(1.01, 1.10), 2)
    confidence = np.random.uniform(0.90, 0.99)
    volume = np.random.randint(1_000_000, 10_000_000)
    return pred_price, confidence, volume

def buy_recommendation(predicted_price, current_price, confidence, volume):
    return predicted_price > current_price and confidence > 0.92 and volume > 1_000_000

def calc_stop_loss(price):
    return round(price * 0.95, 2)

# === Process single ticker ===
def process_ticker(ticker, price_data):
    try:
        # Check if ticker data available
        if ticker not in price_data or price_data[ticker].empty:
            return None

        current_price = price_data[ticker]['Close'][-1]

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
    except Exception as e:
        return None

def send_telegram(df):
    if df.empty:
        return
    lines = ["*Buy Recommendations:*"]
    for _, row in df.iterrows():
        lines.append(f"{row['Ticker']} | Current: ${row['Current Price']} | Predicted: ${row['Predicted Price']} | Confidence: {row['Confidence']*100:.1f}% | Volume: {row['Volume']:,} | Stop Loss: ${row['Stop Loss']}")
    message = "\n".join(lines)
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    params = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    resp = requests.get(url, params=params)
    return resp.status_code == 200

# === Streamlit UI ===

st.title("🚀 Buy Recommendations for S&P 500, Nifty 50 & Nasdaq 100")

index_choice = st.selectbox("Select Index", ["S&P 500", "Nifty 50", "Nasdaq 100"])

if index_choice == "S&P 500":
    tickers = fetch_sp500_tickers()
elif index_choice == "Nifty 50":
    tickers = fetch_nifty50_tickers()
else:
    tickers = fetch_nasdaq100_tickers()

st.write(f"Loaded {len(tickers)} tickers for {index_choice}")

# Bulk fetch prices once
price_data = fetch_bulk_price_data(tickers)

buy_recommendations = []

with st.spinner("Processing buy recommendations..."):
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(process_ticker, ticker, price_data): ticker for ticker in tickers}
        for future in as_completed(futures):
            res = future.result()
            if res:
                buy_recommendations.append(res)

if buy_recommendations:
    df_buy = pd.DataFrame(buy_recommendations)
    st.subheader("✅ Buy Recommendations")
    st.dataframe(df_buy)

    csv_bytes = df_buy.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv_bytes, file_name=f"{index_choice}_buy_recommendations.csv")

    if st.button("Send All Buy Recommendations to Telegram"):
        success = send_telegram(df_buy)
        if success:
            st.success("Telegram message sent!")
        else:
            st.error("Failed to send Telegram message.")
else:
    st.warning("No buy recommendations found.")


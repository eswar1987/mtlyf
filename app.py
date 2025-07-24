import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Telegram config
TELEGRAM_BOT_TOKEN = "7842285230:AAFcisrfFg40AqYjvrGaiq984DYeEu3p6hY"
TELEGRAM_CHAT_ID = "7581145756"

# Cache tickers for 24 hours
@st.cache_data(ttl=86400)
def fetch_nifty50_tickers():
    url = "https://hi-imcodeman.github.io/stock-nse-india/classes/index.nseindia.html#getallstocksymbols"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        # Filter symbols in Nifty 50 group
        nifty_50_symbols = [item['symbol'] + ".NS" for item in data if item.get('index') == "NIFTY 50"]
        if not nifty_50_symbols:
            # Fallback list if empty
            nifty_50_symbols = [
                "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
                "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "HDFC.NS"
            ]
        return nifty_50_symbols
    except Exception as e:
        st.error(f"Error fetching Nifty 50 tickers: {e}")
        return [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
            "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "HDFC.NS"
        ]

@st.cache_data(ttl=86400)
def fetch_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        tables = pd.read_html(url)
        df = tables[0]
        return df['Symbol'].tolist()
    except Exception as e:
        st.error(f"Failed to fetch S&P 500 tickers: {e}")
        return ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'FB']

@st.cache_data(ttl=86400)
def fetch_nasdaq100_tickers():
    url = "https://en.wikipedia.org/wiki/NASDAQ-100"
    try:
        tables = pd.read_html(url)
        df = tables[3]
        return df['Ticker'].tolist()
    except Exception as e:
        st.error(f"Failed to fetch Nasdaq 100 tickers: {e}")
        return ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA']

# Fetch stock info concurrently for speed
def fetch_stock_info_concurrent(tickers):
    results = []

    def fetch_one(ticker):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d")
            price = hist['Close'].iloc[-1] if not hist.empty else None
            volume = hist['Volume'].iloc[-1] if not hist.empty else None
            confidence = np.random.uniform(0.85, 0.99)  # Fake confidence for demo
            return ticker, price, volume, confidence
        except Exception:
            return ticker, None, None, 0.0

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {executor.submit(fetch_one, t): t for t in tickers}
        for future in as_completed(future_to_ticker):
            ticker, price, volume, confidence = future.result()
            results.append({
                'Ticker': ticker,
                'Price': price,
                'Volume': volume,
                'Confidence': confidence
            })
    return results

def buy_recommendation(price, predicted_price, confidence, volume):
    if predicted_price and price and predicted_price > price and confidence > 0.92 and volume and volume > 1_000_000:
        return True
    return False

def calc_stop_loss(price):
    if price:
        return round(price * 0.95, 2)
    return None

# For simplicity, simulated predicted price as +5% over current price
def simulate_predicted_price(price):
    if price:
        return round(price * 1.05, 2)
    return None

def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    params = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        resp = requests.get(url, params=params)
        return resp.status_code == 200
    except Exception as e:
        st.error(f"Telegram send error: {e}")
        return False

# --- Streamlit App ---
st.title("📈 Buy Recommendations Dashboard with Fast Data Fetch")

index_option = st.selectbox("Select Index:", ["Nifty 50", "S&P 500", "Nasdaq 100"])

if index_option == "Nifty 50":
    tickers = fetch_nifty50_tickers()
elif index_option == "S&P 500":
    tickers = fetch_sp500_tickers()
else:
    tickers = fetch_nasdaq100_tickers()

st.write(f"Total tickers fetched: {len(tickers)}")

with st.spinner("Fetching stock data..."):
    stocks_data = fetch_stock_info_concurrent(tickers)

# Compute buy recommendations
buy_recos = []
for stock in stocks_data:
    ticker = stock['Ticker']
    price = stock['Price']
    volume = stock['Volume']
    confidence = stock['Confidence']
    pred_price = simulate_predicted_price(price)
    if buy_recommendation(price, pred_price, confidence, volume):
        buy_recos.append({
            "Ticker": ticker,
            "Current Price": price,
            "Predicted Price": pred_price,
            "Confidence": round(confidence, 3),
            "Volume": volume,
            "Stop Loss": calc_stop_loss(price)
        })

if not buy_recos:
    st.warning("No buy recommendations at this time.")
else:
    df_buys = pd.DataFrame(buy_recos)
    st.dataframe(df_buys)

    csv = df_buys.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Buy Recommendations CSV", data=csv, file_name="buy_recommendations.csv")

    if st.button("🚀 Send Buy Recommendations to Telegram"):
        msg_lines = ["*Buy Recommendations:*"]
        for r in buy_recos:
            msg_lines.append(f"{r['Ticker']} | Price: ${r['Current Price']:.2f} | Predicted: ${r['Predicted Price']:.2f} | Confidence: {r['Confidence']*100:.1f}% | Stop Loss: ${r['Stop Loss']}")
        message = "\n".join(msg_lines)
        success = send_telegram_message(message)
        if success:
            st.success("Telegram alert sent!")
        else:
            st.error("Failed to send Telegram alert.")

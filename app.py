import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import time
from bs4 import BeautifulSoup
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

# --------- Setup logging ---------
logging.basicConfig(level=logging.INFO)

# --------- Telegram config ---------
TELEGRAM_BOT_TOKEN = "7842285230:AAFcisrfFg40AqYjvrGaiq984DYeEu3p6hY"
TELEGRAM_CHAT_ID = "7581145756"

# --------- Cache data for 1 day ---------
@st.cache_data(ttl=86400)
def fetch_nifty50_tickers():
    url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.nseindia.com/get-quotes/equity?symbol=RELIANCE"
    }
    session = requests.Session()
    for attempt in range(5):
        try:
            session.get("https://www.nseindia.com", headers=headers, timeout=10)
            response = session.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            symbols = [item['symbol'] + ".NS" for item in data['data']]
            return symbols
        except Exception as e:
            if attempt < 4:
                time.sleep(2)
                continue
            else:
                st.error(f"Failed to fetch Nifty 50: {e}")
                fallback = [
                    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
                    "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "HDFC.NS"
                ]
                st.warning("Using fallback Nifty 50 tickers")
                return fallback

@st.cache_data(ttl=86400)
def fetch_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        tables = pd.read_html(url)
        df = tables[0]
        return df['Symbol'].tolist()
    except Exception as e:
        st.error(f"Failed to fetch S&P 500 tickers: {e}")
        fallback = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'FB']  # Minimal fallback
        st.warning("Using fallback S&P 500 tickers")
        return fallback

@st.cache_data(ttl=86400)
def fetch_nasdaq100_tickers():
    url = "https://en.wikipedia.org/wiki/NASDAQ-100"
    try:
        tables = pd.read_html(url)
        df = tables[3]  # The 4th table on the page contains tickers
        symbols = df['Ticker'].tolist()
        return symbols
    except Exception as e:
        st.error(f"Failed to fetch Nasdaq 100 tickers: {e}")
        fallback = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA']
        st.warning("Using fallback Nasdaq 100 tickers")
        return fallback

# --------- LSTM Model ---------
class StockPriceLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(StockPriceLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.linear(out[:, -1, :])
        return out

def prepare_data(df, sequence_length=20):
    close_prices = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_prices)
    X, y = [], []
    for i in range(len(scaled) - sequence_length):
        X.append(scaled[i:i+sequence_length])
        y.append(scaled[i+sequence_length])
    return np.array(X), np.array(y), scaler

@st.cache_data(ttl=3600)
def predict_price_lstm(ticker):
    try:
        df = yf.Ticker(ticker).history(period="120d")
        if df.empty or len(df) < 40:
            return None
        X, y, scaler = prepare_data(df)
        device = torch.device('cpu')
        model = StockPriceLSTM()
        model.to(device)
        model.eval()

        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            predictions = model(X_tensor).cpu().numpy()

        pred_prices = scaler.inverse_transform(predictions)
        return float(pred_prices[-1][0])
    except Exception as e:
        logging.error(f"Price prediction error for {ticker}: {e}")
        return None

# --------- Buy recommendation logic ---------
def buy_recommendation(predicted_price, current_price, confidence, volume):
    if predicted_price and current_price and predicted_price > current_price and confidence > 0.92 and volume > 1_000_000:
        return True
    return False

# --------- Fetch stock data ---------
@st.cache_data(ttl=300)
def fetch_stock_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        price = hist['Close'].iloc[-1] if not hist.empty else None
        volume = hist['Volume'].iloc[-1] if not hist.empty else None
        # Fake confidence for demo (replace with your model's output)
        confidence = np.random.uniform(0.85, 0.99)
        return price, volume, confidence
    except Exception as e:
        logging.error(f"Error fetching stock info {ticker}: {e}")
        return None, None, 0.0

# --------- Stop loss calculation ---------
def calc_stop_loss(price):
    if price:
        return round(price * 0.95, 2)
    return None

# --------- Telegram message ---------
def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    params = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        resp = requests.get(url, params=params)
        return resp.status_code == 200
    except Exception as e:
        logging.error(f"Telegram send error: {e}")
        return False

# --------- Main app ---------
st.set_page_config(page_title="Stock Buy Recommendations Dashboard", layout="wide")
st.title("📈 Stock Buy Recommendations Dashboard")

index_option = st.selectbox("Select Index:", ["Nifty 50", "S&P 500", "Nasdaq 100"])

with st.spinner("Fetching tickers..."):
    if index_option == "Nifty 50":
        tickers = fetch_nifty50_tickers()
    elif index_option == "S&P 500":
        tickers = fetch_sp500_tickers()
    else:
        tickers = fetch_nasdaq100_tickers()

st.write(f"Total tickers fetched: {len(tickers)}")

# Process buy recommendations
results = []
progress_bar = st.progress(0)
total = len(tickers)
batch_size = 20  # to avoid blocking, process in batches

for i in range(0, total, batch_size):
    batch = tickers[i:i+batch_size]
    for ticker in batch:
        price, volume, confidence = fetch_stock_info(ticker)
        pred_price = predict_price_lstm(ticker)
        if buy_recommendation(pred_price, price, confidence, volume):
            stop_loss = calc_stop_loss(price)
            results.append({
                "Ticker": ticker,
                "Current Price": price,
                "Predicted Price": pred_price,
                "Confidence": round(confidence, 3),
                "Volume": volume,
                "Stop Loss": stop_loss
            })
    progress_bar.progress(min((i + batch_size) / total, 1.0))

if not results:
    st.warning("No buy recommendations at this time.")
else:
    df_results = pd.DataFrame(results)
    st.dataframe(df_results)

    # Download CSV
    csv = df_results.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Buy Recommendations CSV", data=csv, file_name="buy_recommendations.csv")

    # Telegram alert button
    if st.button("🚀 Send Buy Recommendations to Telegram"):
        msg_lines = ["*Buy Recommendations:*"]
        for r in results:
            msg_lines.append(f"{r['Ticker']} | Price: ${r['Current Price']:.2f} | Predicted: ${r['Predicted Price']:.2f} | Confidence: {r['Confidence']*100:.1f}% | Stop Loss: ${r['Stop Loss']}")
        message = "\n".join(msg_lines)
        success = send_telegram_message(message)
        if success:
            st.success("Telegram alert sent!")
        else:
            st.error("Failed to send Telegram alert.")


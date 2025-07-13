import streamlit as st
import yfinance as yf
from huggingface_hub import InferenceClient
import pandas as pd
import time
import logging
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# === Secrets / Tokens ===
HF_API_TOKEN = "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"
TELEGRAM_BOT_TOKEN = "7842285230:AAFcisrfFg40AqYjvrGaiq984DYeEu3p6hY"
TELEGRAM_CHAT_ID = "7581145756"
client = InferenceClient(token=HF_API_TOKEN)

# === Models ===
MODELS = {
    "buy_recommendation": "fuchenru/Trading-Hero-LLM"
}

# === Logging ===
logging.basicConfig(level=logging.INFO)

# === Load FinBERT Sentiment Model ===
@st.cache_resource
def load_sentiment_model():
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, sentiment_model = load_sentiment_model()

# === Load Buy Recommendation Classification Pipeline ===
@st.cache_resource
def load_buy_rec_pipeline():
    model_name = MODELS["buy_recommendation"]
    return pipeline("text-classification", model=model_name, tokenizer=model_name)

buy_rec_pipeline = load_buy_rec_pipeline()

# === Sector Dictionary ===
ETF_SECTORS = {
    'Tech': ["AAPL", "GOOG", "MSFT", "TSLA", "AMD", "NVDA", "INTC"],
    'HealthCare': ["JNJ", "PFE", "MRK", "ABT"],
    'Financials': ["JPM", "BAC", "C", "WFC"],
    'Energy': ["XOM", "CVX", "SLB"]
}

# === Helper Functions ===
def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if hist.empty:
            raise ValueError("No historical data")
        price = hist['Close'].iloc[-1]
        volume = hist['Volume'].iloc[-1]
        return {"price": price, "volume": volume}
    except Exception as e:
        logging.error(f"Error fetching stock data for {ticker}: {e}")
        return {"price": None, "volume": None}

def fetch_recent_headline(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if news and len(news) > 0:
            return news[0].get('title', f"{ticker} stock")
    except Exception as e:
        logging.warning(f"No news for {ticker}: {e}")
    return f"{ticker} stock"

def call_local_sentiment_with_score(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = sentiment_model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            score, label_id = torch.max(probs, dim=1)
            label = ["Negative", "Neutral", "Positive"][label_id.item()]
            return label, round(score.item(), 2)
    except Exception as e:
        logging.error(f"Sentiment model error: {e}")
        return "Neutral", 0.0

def call_hf_model_buy(ticker):
    prompt = f"Should I buy {ticker} stock? One word answer."
    try:
        results = buy_rec_pipeline(prompt)
        if results and len(results) > 0:
            label = results[0]['label'].lower()
            if label in ['buy', 'yes', 'strong buy']:
                return "Yes"
            else:
                return "No"
    except Exception as e:
        logging.error(f"Buy recommendation error for {ticker}: {e}")
    return "No"

def calc_stop_loss(price):
    return round(price * 0.95, 2) if price else None

def is_strong_signal(pred_price, current_price, buy_recommendation):
    return pred_price and current_price and pred_price > current_price and buy_recommendation.lower() == "yes"

# === LSTM Model Definition ===
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
    X, y = np.array(X), np.array(y)
    return X, y, scaler

def process_sector(tickers):
    results = []
    for ticker in tickers:
        stock = fetch_stock_data(ticker)
        if not stock or stock['price'] is None:
            continue
        headline = fetch_recent_headline(ticker)
        sentiment, confidence = call_local_sentiment_with_score(headline)
        pred_price = stock['price'] * 1.01  # placeholder for predicted price
        buy = call_hf_model_buy(ticker)
        stop_loss = calc_stop_loss(stock['price'])
        strong_signal = "✅" if is_strong_signal(pred_price, stock['price'], buy) else ""

        results.append({
            "Ticker": ticker,
            "Price": round(stock['price'], 2),
            "Volume": int(stock['volume']) if stock['volume'] else 0,
            "Predicted Price": round(pred_price, 2),
            "Headline": headline,
            "Sentiment": sentiment,
            "Confidence": confidence,
            "Buy Recommendation": buy,
            "Stop Loss": stop_loss,
            "Strong Signal": strong_signal
        })

        time.sleep(0.5)
    return results

def backtest_yesterday_lstm_signal(ticker, window=20, days=60):
    df = yf.Ticker(ticker).history(period=f"{days+window}d")
    df = df.reset_index()
    if df.empty or len(df) < window + 2:
        return None, "Not enough historical data."

    returns = []
    signals = []

    for i in range(window, len(df)-1):
        prev_data = df.loc[i-window:i-1]
        if prev_data['Close'].isnull().any():
            continue

        try:
            X, _, scaler = prepare_data(prev_data)
            model = StockPriceLSTM()
            model.eval()
            input_seq = torch.tensor(X[-1:]).float()
            with torch.no_grad():
                pred_scaled = model(input_seq).numpy()
            pred_price = scaler.inverse_transform(pred_scaled)[0][0]
        except Exception as e:
            continue

        yesterday_close = df.loc[i-1, 'Close']
        today_open = df.loc[i, 'Open']
        today_close = df.loc[i, 'Close']

        buy_signal = pred_price > yesterday_close
        signals.append((df.loc[i, 'Date'], pred_price, yesterday_close, buy_signal))

        if buy_signal:
            ret = (today_close - today_open) / today_open
            returns.append(ret)
        else:
            returns.append(0)

    returns = np.array(returns)
    if len(returns) == 0:
        return None, "No trades were made."

    result = {
        "Total Return (%)": round((np.prod(1 + returns) - 1) * 100, 2),
        "Win Rate (%)": round(np.sum(returns > 0) / len(returns) * 100, 2),
        "Avg Daily Return (%)": round(np.mean(returns) * 100, 2),
        "Max Drawdown (%)": round(
            np.min((np.cumprod(1 + returns) - np.maximum.accumulate(np.cumprod(1 + returns))) /
                   np.maximum.accumulate(np.cumprod(1 + returns))) * 100, 2
        )
    }
    return result, signals

# === Streamlit UI ===
st.set_page_config(page_title="📊 AI Stock Sentiment Dashboard", layout="wide")
st.title("📊 AI Stock Sentiment Dashboard")

sector = st.sidebar.selectbox("Select Sector", options=list(ETF_SECTORS.keys()))
st.sidebar.markdown("Data powered by PyTorch FinBERT + HuggingFace + Yahoo Finance")

with st.spinner(f"Processing data for {sector} sector..."):
    tickers = ETF_SECTORS[sector]
    data = process_sector(tickers)
    df = pd.DataFrame(data)

if df.empty:
    st.warning("No data available for this sector.")
    st.stop()

buy_yes = df[df["Buy Recommendation"] == "Yes"].shape[0]
buy_no = df[df["Buy Recommendation"] == "No"].shape[0]
col1, col2 = st.columns(2)
col1.metric("🟢 Buy Recommendations", buy_yes)
col2.metric("🔴 Not Buy", buy_no)

def highlight_buy(val):
    return "color: green; font-weight: bold" if val == "Yes" else "color: red; font-weight: bold"

def highlight_signal(val):
    return "background-color: #c8f7c5; font-weight: bold" if val == "✅" else ""

def highlight_volume(val):
    if val > 10_000_000:
        return "background-color: lightgreen"
    elif val > 1_000_000:
        return "background-color: lightyellow"
    return "background-color: lightcoral"

styled_df = (
    df.style
    .applymap(highlight_buy, subset=["Buy Recommendation"])
    .applymap(highlight_signal, subset=["Strong Signal"])
    .applymap(highlight_volume, subset=["Volume"])
    .format({
        "Price": "${:,.2f}",
        "Predicted Price": lambda x: f"${x:.2f}" if isinstance(x, (float, int)) else x,
        "Stop Loss": lambda x: f"${x:.2f}" if isinstance(x, (float, int)) else x,
        "Volume": "{:,}",
        "Confidence": "{:.2f}"
    })
)

st.dataframe(styled_df, height=700)

st.subheader("📉 Backtest LSTM Buy Strategy")
backtest_ticker = st.selectbox("Select Ticker to Backtest", ETF_SECTORS[sector], key="backtest")

if st.button("Run Backtest"):
    with st.spinner("Running backtest on past 60 days..."):
        backtest_result, signals = backtest_yesterday_lstm_signal(backtest_ticker)

    if isinstance(backtest_result, dict):
        st.success("Backtest Complete")
        st.json(backtest_result)
    else:
        st.warning(signals)

    if isinstance(signals, list):
        st.write("Buy Signals Preview:")
        preview_df = pd.DataFrame(signals, columns=["Date", "Predicted", "Previous Close", "Buy Signal"])
        st.dataframe(preview_df.tail(20))

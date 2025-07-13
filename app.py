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

def calc_stop_loss(price):
    return round(price * 0.95, 2) if price else None

def is_strong_signal(pred_price, current_price, confidence, volume):
    return (
        pred_price and current_price and
        pred_price > current_price and
        confidence >= 0.6 and
        volume > 1_000_000
    )

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    params = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        response = requests.get(url, params=params)
        return response.status_code == 200
    except Exception as e:
        logging.error(f"Telegram send error: {e}")
        return False

# === LSTM Class ===
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

def predict_price_lstm_backtest(ticker):
    try:
        df = yf.Ticker(ticker).history(period="120d")
        if df.empty or len(df) < 40:
            return None, None

        X, y, scaler = prepare_data(df)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = StockPriceLSTM()
        model.to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

        model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            predictions = model(X_tensor).cpu().numpy()
            actuals = y_tensor.cpu().numpy()

        predictions = np.array(predictions).reshape(-1, 1)
        actuals = np.array(actuals).reshape(-1, 1)
        pred_prices = scaler.inverse_transform(predictions)
        actual_prices = scaler.inverse_transform(actuals)

        accuracy = 100 - np.mean(np.abs(pred_prices - actual_prices) / actual_prices) * 100
        return round(float(pred_prices[-1][0]), 2), round(accuracy, 2)

    except Exception as e:
        logging.error(f"Backtest error for {ticker}: {e}")
        return None, None

# === Streamlit Tabs ===
st.set_page_config(page_title="📊 AI Stock Sentiment Dashboard", layout="wide")
tabs = st.tabs(["📊 Dashboard", "🔁 Backtest"])

# === DASHBOARD TAB ===
with tabs[0]:
    st.title("📊 AI Stock Sentiment Dashboard")
    sector = st.selectbox("Select Sector", options=list(ETF_SECTORS.keys()))
    tickers = ETF_SECTORS[sector]

    results = []
    with st.spinner("Analyzing tickers..."):
        for ticker in tickers:
            stock = fetch_stock_data(ticker)
            if not stock or stock['price'] is None:
                continue
            headline = fetch_recent_headline(ticker)
            sentiment, confidence = call_local_sentiment_with_score(headline)
            pred_price, _ = predict_price_lstm_backtest(ticker)
            stop_loss = calc_stop_loss(stock['price'])
            strong_signal = "✅" if is_strong_signal(pred_price, stock['price'], confidence, stock['volume']) else ""

            results.append({
                "Ticker": ticker,
                "Price": round(stock['price'], 2),
                "Volume": int(stock['volume']) if stock['volume'] else 0,
                "Predicted Price": pred_price if pred_price else "N/A",
                "Headline": headline,
                "Sentiment": sentiment,
                "Confidence": confidence,
                "Buy Recommendation": "Yes" if strong_signal == "✅" else "No",
                "Stop Loss": stop_loss,
                "Strong Signal": strong_signal
            })
            time.sleep(0.5)

    df = pd.DataFrame(results)

    buy_yes = df[df["Buy Recommendation"] == "Yes"]
    buy_no = df[df["Buy Recommendation"] == "No"]

    col1, col2 = st.columns(2)
    col1.metric("🟢 Buy Recommendations", len(buy_yes))
    col2.metric("🔴 Not Buy", len(buy_no))

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

    if st.button("Send Buy Summary to Telegram"):
        message_lines = [f"*Buy Summary for {sector} Sector:*"]
        for _, row in buy_yes.iterrows():
            message_lines.append(f"{row['Ticker']} @ ${row['Price']} → Pred: ${row['Predicted Price']}, Conf: {row['Confidence']*100:.1f}%")
        message = "\n".join(message_lines)
        st.success("Summary sent!") if send_telegram_message(message) else st.error("Failed to send.")

# === BACKTEST TAB ===
with tabs[1]:
    st.title("🔁 Backtest Accuracy")
    sector = st.selectbox("Select Sector to Backtest", list(ETF_SECTORS.keys()), key="backtest_sector")
    tickers = ETF_SECTORS[sector]
    backtest_data = []

    with st.spinner("Running backtest predictions..."):
        for ticker in tickers:
            pred_price, accuracy = predict_price_lstm_backtest(ticker)
            if pred_price and accuracy:
                backtest_data.append({"Ticker": ticker, "Predicted Price": pred_price, "Accuracy (%)": accuracy})

    if backtest_data:
        df_bt = pd.DataFrame(backtest_data)
        st.dataframe(df_bt)
    else:
        st.warning("No backtest results available for this sector.")

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
    "buy_recommendation": "fuchenru/Trading-Hero-LLM",
    "price_transformer": "microsoft/ts-mbart-large"  # example transformer model
}

logging.basicConfig(level=logging.INFO)

@st.cache_resource
def load_sentiment_model():
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, sentiment_model = load_sentiment_model()

@st.cache_resource
def load_buy_rec_pipeline():
    model_name = MODELS["buy_recommendation"]
    return pipeline("text-classification", model=model_name, tokenizer=model_name)

buy_rec_pipeline = load_buy_rec_pipeline()

ETF_SECTORS = {
    'Tech': ["AAPL", "GOOG", "MSFT", "TSLA", "AMD", "NVDA", "INTC"],
    'HealthCare': ["JNJ", "PFE", "MRK", "ABT"],
    'Financials': ["JPM", "BAC", "C", "WFC"],
    'Energy': ["XOM", "CVX", "SLB"]
}

def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="30d")
        if hist.empty:
            raise ValueError("No historical data")
        price = hist['Close'].iloc[-1]
        volume = hist['Volume'].iloc[-1]
        return {"price": price, "volume": volume, "history": hist}
    except Exception as e:
        logging.error(f"Error fetching stock data for {ticker}: {e}")
        return {"price": None, "volume": None, "history": None}

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

# === LSTM Model for fallback price prediction ===
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

def predict_price_lstm(df):
    try:
        X, y, scaler = prepare_data(df)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = StockPriceLSTM()
        model.to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

        model.train()
        epochs = 30
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

        model.eval()
        last_seq = torch.tensor(X[-1:], dtype=torch.float32).to(device)
        with torch.no_grad():
            pred_scaled = model(last_seq).cpu().numpy()

        pred_price = scaler.inverse_transform(pred_scaled)[0][0]
        return round(float(pred_price), 2)
    except Exception as e:
        logging.error(f"LSTM price prediction error: {e}")
        return None

# === Hugging Face Time-Series Transformer Price Prediction ===
def predict_price_transformer(ticker):
    try:
        # Get recent price history
        df = yf.Ticker(ticker).history(period="30d")
        if df.empty or len(df) < 20:
            return None

        # Prepare input sequence (just closing prices normalized)
        close_prices = df['Close'].values[-20:]
        min_p, max_p = close_prices.min(), close_prices.max()
        norm_prices = (close_prices - min_p) / (max_p - min_p + 1e-9)

        # The transformer expects input as list of floats
        inputs = {
            "inputs": norm_prices.tolist()
        }
        # Call Hugging Face inference API
        response = client.text_generation(MODELS["price_transformer"], inputs)
        # The model's output is expected as a predicted normalized price (float)
        pred_norm_price = float(response.get('generated_text', '0').strip())
        # Rescale back to original price range
        pred_price = pred_norm_price * (max_p - min_p) + min_p

        return round(pred_price, 2)
    except Exception as e:
        logging.error(f"Transformer price prediction error for {ticker}: {e}")
        return None

def aggregate_price_prediction(ticker):
    data = fetch_stock_data(ticker)
    df_hist = data.get("history")
    if df_hist is None or len(df_hist) < 20:
        return None
    pred1 = predict_price_transformer(ticker)
    pred2 = predict_price_lstm(df_hist)
    preds = [p for p in [pred1, pred2] if p is not None]
    if not preds:
        return None
    avg_pred = sum(preds) / len(preds)
    return round(avg_pred, 2)

def is_strong_signal(pred_price, current_price, buy_recommendation, confidence, volume):
    return (
        pred_price is not None and current_price is not None and
        pred_price > current_price and
        buy_recommendation.lower() == "yes" and
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

# === Streamlit UI ===
st.set_page_config(page_title="📊 AI Stock Sentiment Dashboard", layout="wide")
tabs = st.tabs(["📊 Dashboard", "🔁 Backtest"])

with tabs[0]:
    st.title("📊 AI Stock Sentiment Dashboard")
    sector = st.sidebar.selectbox("Select Sector", options=list(ETF_SECTORS.keys()))
    st.sidebar.markdown("Data powered by PyTorch FinBERT + HuggingFace + Yahoo Finance")

    tickers = ETF_SECTORS[sector]
    results = []
    with st.spinner("Processing data..."):
        for ticker in tickers:
            stock = fetch_stock_data(ticker)
            if not stock or stock['price'] is None:
                continue
            headline = fetch_recent_headline(ticker)
            sentiment, confidence = call_local_sentiment_with_score(headline)
            pred_price = aggregate_price_prediction(ticker)
            buy = call_hf_model_buy(ticker)
            stop_loss = calc_stop_loss(stock['price'])
            strong = "✅" if is_strong_signal(pred_price, stock['price'], buy, confidence, stock['volume']) else ""

            results.append({
                "Ticker": ticker,
                "Price": round(stock['price'], 2),
                "Volume": int(stock['volume']) if stock['volume'] else 0,
                "Predicted Price": pred_price if pred_price else "N/A",
                "Headline": headline,
                "Sentiment": sentiment,
                "Confidence": confidence,
                "Buy Recommendation": buy,
                "Stop Loss": stop_loss,
                "Strong Signal": strong
            })
            time.sleep(0.5)

    df = pd.DataFrame(results)
    if df.empty:
        st.warning("No data available.")
        st.stop()

    col1, col2 = st.columns(2)
    col1.metric("✅ Strong Signals", df[df["Strong Signal"] == "✅"].shape[0])
    col2.metric("Total Stocks", df.shape[0])

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

    if st.button("Send Top 5 Strong Signals to Telegram"):
        top5 = df[df["Strong Signal"] == "✅"].sort_values(by="Confidence", ascending=False).head(5)
        if top5.empty:
            st.warning("No strong signals to send.")
        else:
            lines = ["*Top 5 Strong Buy Signals:*"]
            for _, row in top5.iterrows():
                lines.append(f"{row['Ticker']} | {row['Sentiment']} ({row['Confidence']*100:.1f}%) | Price: ${row['Price']} | Predicted: {row['Predicted Price']}")
            msg = "\n".join(lines)
            if send_telegram_message(msg):
                st.success("Telegram message sent.")
            else:
                st.error("Failed to send message.")

    st.download_button("📥 Download Signals CSV", df.to_csv(index=False), "signals.csv")

with tabs[1]:
    st.title("🔁 Backtest Accuracy")
    sector_bt = st.selectbox("Select Sector to Backtest", list(ETF_SECTORS.keys()), key="backtest_sector")
    tickers_bt = ETF_SECTORS[sector_bt]
    backtest_data = []

    with st.spinner("Running backtest predictions..."):
        for ticker in tickers_bt:
            # Use LSTM for backtest accuracy for simplicity
            df_hist = yf.Ticker(ticker).history(period="120d")
            if df_hist.empty or len(df_hist) < 40:
                continue
            pred_price = predict_price_lstm(df_hist)
            actual_price = df_hist['Close'].iloc[-1]
            accuracy = 100 - abs(pred_price - actual_price) / actual_price * 100 if pred_price else 0
            backtest_data.append({
                "Ticker": ticker,
                "Predicted Price": pred_price,
                "Actual Price": round(actual_price, 2),
                "Accuracy (%)": round(accuracy, 2)
            })

    if backtest_data:
        df_bt = pd.DataFrame(backtest_data)
        st.dataframe(df_bt)
        st.download_button("📥 Download Backtest CSV", df_bt.to_csv(index=False), "backtest.csv")
    else:
        st.warning("No backtest results available for this sector.")

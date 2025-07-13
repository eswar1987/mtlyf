import streamlit as st
import yfinance as yf
from huggingface_hub import InferenceClient
import pandas as pd
import time
import re
import logging
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# === Secrets / Tokens ===
HF_API_TOKEN = "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"
TELEGRAM_BOT_TOKEN = "7842285230:AAFcisrfFg40AqYjvrGaiq984DYeEu3p6hY"
TELEGRAM_CHAT_ID = "7581145756"
client = InferenceClient(token=HF_API_TOKEN)

# === Models ===
MODELS = {
    "price_prediction": "SelvaprakashV/stock-prediction-model",
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

# === Sectors ===
ETF_SECTORS = {
    'Tech': ["AAPL", "GOOG", "MSFT", "TSLA", "AMD", "NVDA", "INTC"],
    'HealthCare': ["JNJ", "PFE", "MRK", "ABT"],
    'Financials': ["JPM", "BAC", "C", "WFC"],
    'ConsumerDiscretionary': ["AMZN", "MCD", "NKE"],
    'Energy': ["XOM", "CVX", "SLB"]
}

# === Helper Functions ===
def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "price": info.get('regularMarketPrice') or info.get('previousClose'),
            "volume": info.get('volume')
        }
    except Exception as e:
        logging.error(f"Error fetching stock data for {ticker}: {e}")
        return {"price": None, "volume": None}

def fetch_recent_headline(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if news and len(news) > 0:
            return news[0]['title']
    except Exception as e:
        logging.warning(f"No news for {ticker}: {e}")
    return f"{ticker} stock"  # fallback

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

def call_hf_model_price(ticker, retries=3):
    for _ in range(retries):
        try:
            output = client.text_generation(MODELS["price_prediction"], ticker)
            text = output.get("generated_text", "") if isinstance(output, dict) else output
            numbers = re.findall(r"\d+\.\d+", text)
            if numbers:
                return float(numbers[0])
            time.sleep(1)
        except Exception as e:
            logging.error(f"Price prediction error for {ticker}: {e}")
    return None

def call_hf_model_buy(ticker, retries=3):
    prompt = f"Should I buy {ticker} stock? One word answer."
    for _ in range(retries):
        try:
            output = client.text_generation(MODELS["buy_recommendation"], prompt)
            text = output.get("generated_text", "") if isinstance(output, dict) else output
            match = re.search(r"\b(yes|no|buy|hold|sell|strong buy|strong sell)\b", text, re.I)
            if match:
                return "Yes" if match.group(0).lower() in ["yes", "buy", "strong buy"] else "No"
            time.sleep(1)
        except Exception as e:
            logging.error(f"Buy recommendation error for {ticker}: {e}")
    return "No"

def calc_stop_loss(price):
    return round(price * 0.95, 2) if price else None

def is_strong_signal(pred_price, current_price, buy_recommendation):
    return pred_price and current_price and pred_price > current_price and buy_recommendation.lower() == "yes"

def process_sector(tickers):
    results = []
    for ticker in tickers:
        stock = fetch_stock_data(ticker)
        if not stock or stock['price'] is None:
            continue
        headline = fetch_recent_headline(ticker)
        sentiment, confidence = call_local_sentiment_with_score(headline)
        pred_price = call_hf_model_price(ticker)
        buy = call_hf_model_buy(ticker)
        stop_loss = calc_stop_loss(stock['price'])
        strong_signal = "✅" if is_strong_signal(pred_price, stock['price'], buy) else ""

        results.append({
            "Ticker": ticker,
            "Price": round(stock['price'], 2),
            "Volume": stock['volume'] or 0,
            "Predicted Price": round(pred_price, 2) if pred_price else "N/A",
            "Headline": headline,
            "Sentiment": sentiment,
            "Confidence": confidence,
            "Buy Recommendation": buy,
            "Stop Loss": stop_loss,
            "Strong Signal": strong_signal
        })

        time.sleep(0.5)
    return results

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

# Metrics
buy_yes = df[df["Buy Recommendation"] == "Yes"].shape[0]
buy_no = df[df["Buy Recommendation"] == "No"].shape[0]
col1, col2 = st.columns(2)
col1.metric("🟢 Buy Recommendations", buy_yes)
col2.metric("🔴 Not Buy", buy_no)

# Styling
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
    message = f"*Buy Summary for {sector} Sector:*\nYes: {buy_yes}\nNo: {buy_no}"
    st.success("Message sent!") if send_telegram_message(message) else st.error("Failed to send message.")

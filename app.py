import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
from transformers import pipeline
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load your LSTM model (make sure model.h5 is in your folder)
@st.cache_resource
def load_price_model():
    return load_model("model.h5")

price_model = load_price_model()

# Load sentiment pipeline (FinBERT or generic sentiment model)
@st.cache_resource
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

sentiment_analyzer = load_sentiment_pipeline()

# Preprocessing scaler for price input
scaler = MinMaxScaler(feature_range=(0, 1))

# News API config
NEWS_API_KEY = "745d4e372a4848fca701b7d5d6787bec"
NEWS_ENDPOINT = "https://newsapi.org/v2/everything"

# Telegram config
TELEGRAM_BOT_TOKEN = "7842285230:AAFcisrfFg40AqYjvrGaiq984DYeEu3p6hY"
TELEGRAM_CHAT_ID = "7581145756"

# Fetch news headlines for ticker
def fetch_news(ticker, max_articles=5):
    params = {
        'q': ticker,
        'apiKey': NEWS_API_KEY,
        'language': 'en',
        'sortBy': 'relevance',
        'pageSize': max_articles
    }
    try:
        response = requests.get(NEWS_ENDPOINT, params=params)
        articles = response.json().get('articles', [])
        headlines = [article['title'] for article in articles]
        return headlines
    except Exception as e:
        st.error(f"News fetch error: {e}")
        return []

# Get average sentiment score of headlines
def analyze_sentiment(headlines):
    if not headlines:
        return None, "No news"
    results = sentiment_analyzer(headlines)
    # FinBERT returns labels: Positive, Neutral, Negative; map them to scores
    label_map = {"Positive":1, "Neutral":0.5, "Negative":0}
    scores = [label_map.get(r['label'], 0.5) for r in results]
    avg_score = np.mean(scores)
    return avg_score, results

# Prepare data for price prediction - example: last 60 days close
def prepare_price_data(ticker, days=60):
    df = yf.download(ticker, period=f"{days}d", interval="1d")
    if df.empty or 'Close' not in df.columns:
        return None
    close_prices = df['Close'].values.reshape(-1,1)
    scaled = scaler.fit_transform(close_prices)
    # Model expects (1,60,1) shape
    if len(scaled) < days:
        return None
    input_data = scaled[-days:].reshape(1, days, 1)
    return input_data, close_prices[-1][0]

# Predict next day price using LSTM
def predict_price(ticker):
    data = prepare_price_data(ticker)
    if data is None:
        return None, None
    input_data, last_price = data
    pred_scaled = price_model.predict(input_data)[0][0]
    # Reverse scale prediction (approximate)
    pred_price = scaler.inverse_transform([[pred_scaled]])[0][0]
    confidence = 0.95  # you can build a better confidence measure
    return pred_price, confidence

# Buy recommendation combining price & sentiment
def combined_buy_recommendation(current_price, predicted_price, confidence, sentiment_score, volume):
    if current_price is None or predicted_price is None:
        return "No"
    # Basic logic: price up, confidence high, positive sentiment and volume high
    if (
        predicted_price > current_price and
        confidence > 0.9 and
        sentiment_score is not None and sentiment_score > 0.6 and
        volume > 1_000_000
    ):
        return "Yes"
    return "No"

# Telegram message
def send_telegram_message(message: str) -> bool:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    params = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        r = requests.get(url, params=params)
        return r.status_code == 200
    except Exception as e:
        st.error(f"Telegram error: {e}")
        return False

# Main app
st.title("📊 Real ML + Sentiment Stock Dashboard")

# Choose index & fetch tickers (reuse your previous fetch functions)

index_option = st.selectbox("Select Index", ["S&P 500", "Nifty 50", "Nasdaq 100"])
tickers = []  # fetch your tickers dynamically or static

# For demo, small list:
tickers = ["AAPL", "MSFT", "RELIANCE.NS"]

max_tickers = st.sidebar.slider("Max tickers to scan", 5, 30, 10)
tickers = tickers[:max_tickers]

results = []
progress_bar = st.progress(0)
status_text = st.empty()

for i, ticker in enumerate(tickers):
    current_price, volume = fetch_stock_data(ticker)
    if current_price is None:
        continue

    pred_price, confidence = predict_price(ticker)
    headlines = fetch_news(ticker)
    sentiment_score, sentiment_results = analyze_sentiment(headlines)
    buy = combined_buy_recommendation(current_price, pred_price, confidence, sentiment_score, volume)

    if buy == "Yes":
        stop_loss = round(current_price * 0.95, 2)
        results.append({
            "Ticker": ticker,
            "Current Price": current_price,
            "Predicted Price": pred_price,
            "Confidence": confidence,
            "Sentiment Score": sentiment_score,
            "Volume": volume,
            "Stop Loss": stop_loss,
            "News Headlines": headlines[:3]
        })

    status_text.text(f"Processed {i+1}/{len(tickers)} tickers")
    progress_bar.progress((i+1)/len(tickers))
    time.sleep(1)

progress_bar.empty()
status_text.empty()

if not results:
    st.warning("No buy recommendations found.")
else:
    df = pd.DataFrame(results)
    st.dataframe(df)

    csv = df.to_csv(index=False).encode()
    st.download_button("Download CSV", csv, "buy_recommendations.csv")

    if st.button("Send to Telegram"):
        lines = ["*Buy Recommendations:*"]
        for _, row in df.iterrows():
            lines.append(
                f"{row['Ticker']} | Price: ${row['Current Price']:.2f} | Predicted: ${row['Predicted Price']:.2f} | Conf: {row['Confidence']*100:.1f}% | Sentiment: {row['Sentiment Score']:.2f} | Stop Loss: ${row['Stop Loss']:.2f}"
            )
        message = "\n".join(lines)
        if send_telegram_message(message):
            st.success("Telegram message sent!")
        else:
            st.error("Failed to send Telegram message.")

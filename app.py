!pip install yfinance streamlit requests

import yfinance as yf
import requests
import streamlit as st
from datetime import datetime

# Telegram bot credentials
BOT_TOKEN = '7842285230:AAFcisrfFg40AqYjvrGaiq984DYeEu3p6hY'
CHAT_ID = '7581145756'

st.set_page_config(page_title="MotleyBot - Stock Picks", layout="wide")
st.title("📈 MotleyBot: AI Stock Recommender")

# Define function to send Telegram alerts
def send_telegram(message):
    url = f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage'
    requests.post(url, json={'chat_id': CHAT_ID, 'text': message, 'parse_mode': 'HTML'})

# Get sentiment from Alpha Vantage News API
def fetch_sentiment(ticker):
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey=demo"
    res = requests.get(url).json()
    feed = res.get('feed', [])
    bull, bear = 0, 0
    headlines = []
    for i in feed[:3]:
        label = i.get('overall_sentiment_label', 'Neutral')
        if label.lower() == 'bullish': bull += 1
        if label.lower() == 'bearish': bear += 1
        headlines.append(f"{i['title']} ({label})")
    return bull, bear, headlines

# Recommendation logic
def generate_recommendation(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        price = info.get('regularMarketPrice', 0)
        volume = info.get('volume', 0)
        name = info.get('shortName', ticker)

        bull, bear, headlines = fetch_sentiment(ticker)
        sentiment_score = bull - bear

        if sentiment_score > 0:
            reco = "BUY"
            color = "green"
            reason = "Strong bullish sentiment and healthy volume."
        elif sentiment_score < 0:
            reco = "SELL"
            color = "red"
            reason = "Bearish media tone indicates caution."
        else:
            reco = "HOLD"
            color = "gray"
            reason = "Neutral outlook, no major movement expected."

        send_telegram(f"<b>{name}</b> ({ticker})\n<b>Price:</b> ${price:.2f}\n<b>Reco:</b> {reco}")

        return {
            "name": name,
            "ticker": ticker,
            "price": price,
            "volume": volume,
            "bullish": bull,
            "bearish": bear,
            "recommendation": reco,
            "reason": reason,
            "color": color,
            "headlines": headlines
        }
    except Exception as e:
        return {"error": str(e), "ticker": ticker}

# Sidebar options
tickers = st.sidebar.multiselect("Select Tickers", ["AAPL", "TSLA", "MSFT", "GOOG", "NVDA", "SPY", "VOO", "TQQQ", "SQQQ"], default=["AAPL", "TSLA"])

st.sidebar.markdown("---")
if st.sidebar.button("🔍 Analyze Selected"):
    results = [generate_recommendation(t) for t in tickers]
    for r in results:
        if 'error' in r:
            st.error(f"{r['ticker']}: {r['error']}")
        else:
            st.markdown(f"""
                <div style='border:1px solid #ccc; padding:10px; border-radius:10px; margin-bottom:15px;'>
                <h3>{r['name']} ({r['ticker']})</h3>
                <b>Price:</b> ${r['price']:.2f} | <b>Volume:</b> {r['volume']:,}<br>
                <b>Sentiment:</b> Bullish {r['bullish']} / Bearish {r['bearish']}<br>
                <b>Recommendation:</b> <span style='color:{r['color']}; font-size:18px;'><b>{r['recommendation']}</b></span><br>
                <b>Reason:</b> {r['reason']}<br>
                <ul>{''.join([f'<li>{h}</li>' for h in r['headlines']])}</ul>
                </div>
            """, unsafe_allow_html=True)
else:
    st.info("Select tickers from the sidebar and click 'Analyze Selected'")

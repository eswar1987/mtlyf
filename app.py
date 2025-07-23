import streamlit as st
import pandas as pd
import requests
from transformers import pipeline
import datetime
import time
import os

# Load environment or direct keys
NEWSAPI_KEY = "745d4e372a4848fca701b7d5d6787bec"
TELEGRAM_BOT_TOKEN = "7842285230:AAFcisrfFg40AqYjvrGaiq984DYeEu3p6hY"
TELEGRAM_CHAT_ID = "7581145756"

# Load sentiment pipeline
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

sentiment_model = load_sentiment_model()

def get_index_tickers():
    # Fetching latest tickers (you can update CSVs if needed)
    nifty_50 = pd.read_html("https://www.moneycontrol.com/markets/indian-indices/top-nse-50-companies-list/9")[0]['Company'].tolist()
    smallcap = pd.read_html("https://www.moneycontrol.com/markets/indian-indices/top-nse-smallcap-100-companies-list/14")[0]['Company'].tolist()
    sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]["Symbol"].tolist()
    nasdaq100 = pd.read_html("https://en.wikipedia.org/wiki/NASDAQ-100")[4]["Ticker"].tolist()
    return {
        "NIFTY 50": nifty_50,
        "NIFTY Smallcap 100": smallcap,
        "S&P 500": sp500,
        "NASDAQ 100": nasdaq100
    }

# Get top news from NewsAPI
def fetch_news(company_name):
    url = f"https://newsapi.org/v2/everything?q={company_name}+order+profit&language=en&sortBy=publishedAt&apiKey={NEWSAPI_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("articles", [])
    return []

def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    requests.post(url, data=payload)

def analyze_and_alert(companies):
    st.subheader("🔍 Scanning News for Orders & Profits")
    for company in companies:
        articles = fetch_news(company)
        for article in articles[:3]:  # Only top 3 per company
            title = article["title"]
            description = article.get("description", "")
            combined_text = title + " " + description
            sentiment = sentiment_model(combined_text)[0]
            if sentiment['label'] == "positive":
                msg = f"🚀 *{company}*\n\n📰 {title}\n🗾 {description}\n📈 Sentiment: {sentiment['label']} ({sentiment['score']:.2f})"
                send_telegram_alert(msg)
                st.success(msg)
            time.sleep(1)

# UI
st.title("📊 Order & Profit Tracker - India & US")
st.write("Tracks companies winning orders and posting profits with positive sentiment.")

index_data = get_index_tickers()
selected_indices = st.multiselect("Select Indices", list(index_data.keys()), default=["NIFTY 50", "S&P 500"])
if st.button("Scan & Alert"):
    combined_companies = []
    for idx in selected_indices:
        combined_companies.extend(index_data[idx])
    analyze_and_alert(combined_companies[:50])  # limit for speed

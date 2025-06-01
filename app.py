import streamlit as st
import yfinance as yf
from huggingface_hub import InferenceClient
import pandas as pd
import time
import requests
import re
from streamlit_autorefresh import st_autorefresh
from dotenv import load_dotenv
import os

# Load env variables
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

client = InferenceClient(token=HF_API_TOKEN)

MODELS = {
    "price_prediction": "SelvaprakashV/stock-prediction-model",
    "news_sentiment": "cg1026/financial-news-sentiment-lora",
    "buy_recommendation": "fuchenru/Trading-Hero-LLM"
}

ETF_SECTORS = {
    'Tech': ["AAPL", "GOOG", "MSFT", "TSLA", "AMD", "NVDA", "INTC", "CRM", "ADBE", "AVGO",
             "ORCL", "CSCO", "QCOM", "NOW", "UBER", "SNOW", "TWLO", "WORK", "MDB", "ZI"],
    'HealthCare': ["JNJ", "PFE", "MRK", "ABT", "GILD", "LLY", "BMY", "UNH", "AMGN", "CVS",
                   "MDT", "ISRG", "ZTS", "REGN", "VRTX", "BIIB", "BAX", "HCA", "DGX", "IDXX"],
    'Financials': ["JPM", "BAC", "C", "WFC", "GS", "MS", "USB", "AXP", "PNC", "SCHW",
                   "BK", "BLK", "TFC", "CME", "MMC", "SPGI", "ICE", "STT", "FRC", "MTB"],
    'ConsumerDiscretionary': ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX",
                             "GM", "F", "DG", "ROST", "CMG", "YUM", "DHI", "LEN", "BBY", "WHR", "LVS", "MAR"],
    'Industrials': ["GE", "UPS", "CAT", "BA", "LMT", "MMM", "DE", "HON", "RTX", "GD",
                    "EMR", "PNR", "ROK", "ETN", "CSX", "FDX", "CP", "XYL", "ITW", "DOV"],
    'Energy': ["XOM", "CVX", "OXY", "SLB", "PXD", "EOG", "MPC", "VLO", "PSX", "COP",
               "HAL", "FTI", "BKR", "DVN", "CHK", "APA", "CXO", "MRO", "HES", "NBL"],
    'Utilities': ["DUK", "SO", "NEE", "SRE", "EXC", "AEP", "XEL", "D", "ED", "PEG",
                  "ES", "PPL", "WEC", "CMS", "EIX", "PNW", "FE", "ATO", "AES", "NRG"],
    'BasicMaterials': ["LIN", "SHW", "ECL", "APD", "FCX", "NEM", "DD", "DOW", "CE", "PPG",
                       "VMC", "LYB", "IP", "BLL", "MLM", "NUE", "PKG", "AVY", "PKX"],
    'ETFs': ["SPY", "QQQ", "DIA", "IWM", "ARKK", "ARKW", "SMH", "IYT", "XLF", "XLE",
             "XLK", "XLY", "XLV", "XLI", "XLB", "XLU", "XLRE", "XLC", "VOO"],
    'LeveragedETFs': ["TQQQ", "SQQQ", "SPXL", "SPXS", "UPRO", "SDOW", "SOXL", "SOXS",
                      "LABU", "LABD", "FAS", "FAZ", "TNA", "TZA", "DRN", "DRV", "TECL", "TECS", "DFEN", "DUST"],
    'Commodities': ["GC=F", "SI=F", "CL=F"]
}

PENNY_STOCKS = [
    "SENS", "SNDL", "GEVO", "FIZZ", "PLUG", "KNDI", "NIO", "NOK", "VSTM", "OCGN",
    "CLOV", "AEMD", "ACHV", "BLNK", "CNET", "CERE", "FCEL", "IPHA", "KOSS", "MARA"
]

@st.cache_data(ttl=900, show_spinner=False)
def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        price = info.get('regularMarketPrice') or info.get('previousClose')
        volume = info.get('volume')
        return price, volume
    except Exception:
        return None, None

def call_hf_model_price(ticker):
    try:
        output = client.text_generation(MODELS["price_prediction"], ticker)
        if output and isinstance(output, str):
            numbers = re.findall(r"\d+\.\d+", output)
            if numbers:
                return float(numbers[0])
        return None
    except Exception:
        return None

def call_hf_model_sentiment(ticker):
    try:
        result = client.text_classification(MODELS["news_sentiment"], ticker)
        if result and isinstance(result, list) and len(result) > 0:
            return result[0]["label"]
        return None
    except Exception:
        return None

def call_hf_model_buy(ticker):
    try:
        prompt = f"Should I buy {ticker} stock? One word answer."
        output = client.text_generation(MODELS["buy_recommendation"], prompt)
        if output and isinstance(output, str):
            # Extract first word (yes/no/hold)
            return output.strip().split()[0].lower()
        return None
    except Exception:
        return None

def calc_stop_loss(price):
    return round(price * 0.95, 2) if isinstance(price, (int, float)) else None

@st.cache_data(ttl=900, show_spinner=False)
def process_sector(tickers):
    results = []
    for ticker in tickers:
        price, volume = fetch_stock_data(ticker)
        if price is None:
            continue
        pred_price = call_hf_model_price(ticker)
        sentiment = call_hf_model_sentiment(ticker)
        buy = call_hf_model_buy(ticker)
        stop_loss = calc_stop_loss(price)

        strong_signal = ""
        if pred_price and pred_price > price and buy == "yes":
            strong_signal = "✅"

        results.append({
            "Ticker": ticker,
            "Price": round(price, 2),
            "Volume": volume,
            "Predicted Price": round(pred_price, 2) if pred_price else None,
            "Sentiment": sentiment.capitalize() if sentiment else "N/A",
            "Buy Recommendation": buy.capitalize() if buy else "N/A",
            "Stop Loss": stop_loss,
            "Strong Signal": strong_signal
        })
        time.sleep(0.2)
    return results

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    params = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        resp = requests.get(url, params=params)
        return resp.status_code == 200
    except Exception:
        return False

# Streamlit Layout & UI
st.set_page_config(page_title="📊 Sector-wise Stock Dashboard", layout="wide")
st.title("📊 Sector-wise Stock Dashboard with LLM Predictions")

# Auto refresh every 5 minutes
st_autorefresh(interval=300000, key="datarefresh")

sectors = list(ETF_SECTORS.keys()) + ["Penny Stocks"]
sector = st.sidebar.selectbox("Select Sector", sectors)

tickers = PENNY_STOCKS if sector == "Penny Stocks" else ETF_SECTORS[sector]

search_ticker = st.sidebar.text_input("Filter tickers (comma separated)")

if search_ticker:
    filter_list = [t.strip().upper() for t in search_ticker.split(",")]
    tickers = [t for t in tickers if t in filter_list]

with st.spinner(f"Fetching data for {sector}..."):
    data = process_sector(tickers)

if not data:
    st.warning("No data found.")
else:
    df = pd.DataFrame(data)

    # KPIs summary
    avg_price = df["Price"].mean()
    total_volume = df["Volume"].sum()
    strong_buys = df["Strong Signal"].value_counts().get("✅", 0)

    k1, k2, k3 = st.columns(3)
    k1.metric("Average Price", f"${avg_price:.2f}")
    k2.metric("Total Volume", f"{total_volume:,}")
    k3.metric("Strong Buy Signals", f"{strong_buys}")

    # Style functions
    def sentiment_color(val):
        if val == "Positive":
            color = "green"
        elif val == "Negative":
            color = "red"
        elif val == "Neutral":
            color = "orange"
        else:
            color = "gray"
        return f"color: {color}; font-weight: bold;"

    def buy_color(val):
        colors = {"Yes": "green", "No": "red", "Hold": "orange", "N/A": "gray"}
        return f"background-color: {colors.get(val, 'white')}; color: white; font-weight: bold; border-radius: 4px; text-align: center;"

    def strong_signal_style(val):
        return "background-color: #d4edda; font-weight: bold;" if val == "✅" else ""

    def volume_bar(val):
        max_vol = df["Volume"].max()
        if pd.isna(val):
            return ""
        percentage = (val / max_vol) * 100 if max_vol > 0 else 0
        color = "#007bff"  # bootstrap blue
        bar = f"""
        background: linear-gradient(90deg, {color} {percentage}%, transparent {percentage}%);
        """
        return bar

    styled_df = df.style \
        .applymap(sentiment_color, subset=["Sentiment"]) \
        .applymap(buy_color, subset=["Buy Recommendation"]) \
        .applymap(strong_signal_style, subset=["Strong Signal"]) \
        .applymap(volume_bar, subset=["Volume"]) \
        .format({"Price": "${:.2f}", "Predicted Price": "${:.2f}", "Stop Loss": "${:.2f}"})

    st.dataframe(styled_df, height=600)

    st.download_button(
        label="📥 Download CSV",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name=f"{sector}_stocks.csv",
        mime="text/csv"
    )

    if st.button("🚀 Send Sector Overview to Telegram"):
        msg = f"*{sector} Sector Overview*\n\n"
        for row in data:
            msg += (f"{row['Ticker']}: Price ${row['Price']}, Predicted ${row['Predicted Price']}, "
                    f"Sentiment: {row['Sentiment']}, Buy: {row['Buy Recommendation']}, "
                    f"Stop Loss: ${row['Stop Loss']}, Signal: {row['Strong Signal']}\n")
        if send_telegram_message(msg):
            st.success("Message sent to Telegram!")
        else:
            st.error("Failed to send message to Telegram.")

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

# Models
MODELS = {
    "price_prediction": "SelvaprakashV/stock-prediction-model",
    "news_sentiment": "cg1026/financial-news-sentiment-lora",
    "buy_recommendation": "fuchenru/Trading-Hero-LLM"
}

# Sector Stocks
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

# Cache per sector for 15 minutes
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
        return "N/A"
    except Exception:
        return "Error"

def call_hf_model_sentiment(ticker):
    try:
        result = client.text_classification(MODELS["news_sentiment"], ticker)
        if result and isinstance(result, list) and len(result) > 0:
            return result[0]["label"]
        return "N/A"
    except Exception:
        return "Error"

def call_hf_model_buy(ticker):
    try:
        prompt = f"Should I buy {ticker} stock? One word answer."
        output = client.text_generation(MODELS["buy_recommendation"], prompt)
        if output and isinstance(output, str):
            return output.strip().split()[0]
        return "N/A"
    except Exception:
        return "Error"

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
        if (isinstance(pred_price, float) and pred_price > price) and buy.lower() == "yes":
            strong_signal = "✅"

        results.append({
            "Ticker": ticker,
            "Price": round(price, 2),
            "Volume": volume,
            "Predicted Price": pred_price,
            "Sentiment": sentiment,
            "Buy Recommendation": buy,
            "Stop Loss": stop_loss,
            "Strong Signal": strong_signal
        })
        time.sleep(0.2)  # slight delay to avoid rate limits
    return results

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    params = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        resp = requests.get(url, params=params)
        return resp.status_code == 200
    except Exception:
        return False

# === Streamlit UI ===
st.set_page_config(page_title="📊 Sector-wise Stock Dashboard", layout="wide")
st.title("📊 Sector-wise Stock Dashboard with LLM Predictions")

# Auto refresh every 5 minutes (300000 ms)
st_autorefresh(interval=300000, key="datarefresh")

sectors = list(ETF_SECTORS.keys()) + ["Penny Stocks"]
sector = st.sidebar.selectbox("Select Sector", sectors)

tickers = PENNY_STOCKS if sector == "Penny Stocks" else ETF_SECTORS[sector]

with st.spinner(f"Processing {sector}... This may take a moment for large sectors."):
    data = process_sector(tickers)

if not data:
    st.warning("No data found for this sector.")
else:
    df = pd.DataFrame(data)

    # Styling function
    def style_df(df):
        # Highlight max volume
        df_styled = df.style.highlight_max(subset=["Volume"], color="lightblue") \
            .applymap(lambda v: "color: green;" if str(v).lower() == "yes" else "", subset=["Buy Recommendation"]) \
            .applymap(lambda v: "background-color: #d4edda; font-weight: bold;" if v == "✅" else "", subset=["Strong Signal"])
        return df_styled

    st.write(style_df(df))

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


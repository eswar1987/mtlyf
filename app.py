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

# Load env variables (if you use .env)
load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN") or "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"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or "7842285230:AAFcisrfFg40AqYjvrGaiq984DYeEu3p6hY"
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") or "7581145756"

client = InferenceClient(token=HF_API_TOKEN)

MODELS = {
    "price_prediction": "SelvaprakashV/stock-prediction-model",
    "news_sentiment": "cg1026/financial-news-sentiment-lora",
    "buy_recommendation": "fuchenru/Trading-Hero-LLM"
}

ETF_SECTORS = {
    'Tech': ["AAPL", "GOOG", "MSFT", "TSLA", "AMD", "NVDA", "INTC", "CRM", "ADBE", "AVGO", "ORCL", "CSCO", "QCOM", "NOW", "UBER", "SNOW", "TWLO", "WORK", "MDB", "ZI"],
    'HealthCare': ["JNJ", "PFE", "MRK", "ABT", "GILD", "LLY", "BMY", "UNH", "AMGN", "CVS", "MDT", "ISRG", "ZTS", "REGN", "VRTX", "BIIB", "BAX", "HCA", "DGX", "IDXX"],
    'Financials': ["JPM", "BAC", "C", "WFC", "GS", "MS", "USB", "AXP", "PNC", "SCHW", "BK", "BLK", "TFC", "CME", "MMC", "SPGI", "ICE", "STT", "FRC", "MTB"],
    'ConsumerDiscretionary': ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX", "GM", "F", "DG", "ROST", "CMG", "YUM", "DHI", "LEN", "BBY", "WHR", "LVS", "MAR"],
    'Industrials': ["GE", "UPS", "CAT", "BA", "LMT", "MMM", "DE", "HON", "RTX", "GD", "EMR", "PNR", "ROK", "ETN", "CSX", "FDX", "CP", "XYL", "ITW", "DOV"],
    'Energy': ["XOM", "CVX", "OXY", "SLB", "PXD", "EOG", "MPC", "VLO", "PSX", "COP", "HAL", "FTI", "BKR", "DVN", "CHK", "APA", "CXO", "MRO", "HES", "NBL"],
    'Utilities': ["DUK", "SO", "NEE", "SRE", "EXC", "AEP", "XEL", "D", "ED", "PEG", "ES", "PPL", "WEC", "CMS", "EIX", "PNW", "FE", "ATO", "AES", "NRG"],
    'BasicMaterials': ["LIN", "SHW", "ECL", "APD", "FCX", "NEM", "DD", "DOW", "CE", "PPG", "VMC", "LYB", "IP", "BLL", "MLM", "NUE", "PKG", "AVY", "PKX"],
    'ETFs': ["SPY", "QQQ", "DIA", "IWM", "ARKK", "ARKW", "SMH", "IYT", "XLF", "XLE", "XLK", "XLY", "XLV", "XLI", "XLB", "XLU", "XLRE", "XLC", "VOO"],
    'LeveragedETFs': ["TQQQ", "SQQQ", "SPXL", "SPXS", "UPRO", "SDOW", "SOXL", "SOXS", "LABU", "LABD", "FAS", "FAZ", "TNA", "TZA", "DRN", "DRV", "TECL", "TECS", "DFEN", "DUST"],
    'Commodities': ["GC=F", "SI=F", "CL=F"]
}

PENNY_STOCKS = [
    "SENS", "SNDL", "GEVO", "FIZZ", "PLUG", "KNDI", "NIO", "NOK", "VSTM", "OCGN",
    "CLOV", "AEMD", "ACHV", "BLNK", "CNET", "CERE", "FCEL", "IPHA", "KOSS", "MARA"
]

# Helpers

def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        price = info.get('regularMarketPrice') or info.get('previousClose')
        volume = info.get('volume')
        return {"price": price, "volume": volume}
    except Exception:
        return {"price": None, "volume": None}

def call_hf_model_price(ticker):
    try:
        output = client.text_generation(MODELS["price_prediction"], ticker)
        numbers = re.findall(r"\d+\.\d+", output)
        return float(numbers[0]) if numbers else "N/A"
    except Exception:
        return "Error"

def call_hf_model_sentiment(ticker):
    try:
        result = client.text_classification(MODELS["news_sentiment"], ticker)
        if result and isinstance(result, list):
            return result[0]["label"]
        return "N/A"
    except Exception:
        return "Error"

def call_hf_model_buy(ticker):
    try:
        prompt = f"Should I buy {ticker} stock? One word answer."
        output = client.text_generation(MODELS["buy_recommendation"], prompt)
        return output.strip().split()[0]
    except Exception:
        return "Error"

def calc_stop_loss(price):
    return round(price * 0.95, 2) if isinstance(price, (int, float)) else None


@st.cache_data(ttl=3600, show_spinner=False)
def process_sector(tickers):
    results = []
    for ticker in tickers:
        stock = fetch_stock_data(ticker)
        if stock['price'] is None:
            continue

        pred_price = call_hf_model_price(ticker)
        sentiment = call_hf_model_sentiment(ticker)
        buy = call_hf_model_buy(ticker)
        stop_loss = calc_stop_loss(stock['price'])

        strong_signal = ""
        if isinstance(pred_price, float) and pred_price > stock['price'] and buy.lower() == "yes":
            strong_signal = "✅"

        results.append({
            "Ticker": ticker,
            "Price": round(stock['price'], 2),
            "Volume": stock['volume'],
            "Predicted Price": pred_price,
            "Sentiment": sentiment,
            "Buy Recommendation": buy,
            "Stop Loss": stop_loss,
            "Strong Signal": strong_signal
        })

        time.sleep(0.5)  # Reduce delay for speed but still polite to API
    return results


def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    params = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        response = requests.get(url, params=params)
        return response.status_code == 200
    except Exception:
        return False


# UI Start
st.set_page_config(page_title="Stock Dashboard", layout="wide")
st.title("📊 Sector-wise Stock Dashboard with LLM Predictions")

# Auto-refresh every 5 minutes (300000 ms)
st_autorefresh(interval=300000, key="datarefresh")

sectors = list(ETF_SECTORS.keys()) + ["Penny Stocks"]
sector = st.sidebar.selectbox("Select Sector", sectors)

tickers = PENNY_STOCKS if sector == "Penny Stocks" else ETF_SECTORS[sector]

with st.spinner(f"Processing {sector}..."):
    data = process_sector(tickers)

if not data:
    st.warning("No data found.")
else:
    df = pd.DataFrame(data)

    # Styling function
    def highlight_volume_max(s):
        is_max = s == s.max()
        return ['background-color: #add8e6' if v else '' for v in is_max]

    def highlight_buy(val):
        color = 'green' if str(val).lower() == 'yes' else ''
        return f'color: {color}'

    def highlight_strong_signal(val):
        return 'background-color: #d4edda' if val == "✅" else ''

    styled_df = (
        df.style
        .apply(highlight_volume_max, subset=["Volume"])
        .applymap(highlight_buy, subset=["Buy Recommendation"])
        .applymap(highlight_strong_signal, subset=["Strong Signal"])
        .format({"Price": "${:.2f}", "Predicted Price": "${}"})
    )

    st.dataframe(styled_df, height=600)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download CSV",
        csv,
        f"{sector}_stocks.csv",
        "text/csv"
    )

    if st.button("🚀 Send to Telegram"):
        msg = f"*{sector} Sector Overview*\n\n"
        for row in data:
            msg += f"{row['Ticker']}: Price ${row['Price']}, Predicted ${row['Predicted Price']}, Sentiment: {row['Sentiment']}, Buy: {row['Buy Recommendation']}, SL: ${row['Stop Loss']}, Signal: {row['Strong Signal']}\n"
        if send_telegram_message(msg):
            st.success("Sent!")
        else:
            st.error("Failed to send.")

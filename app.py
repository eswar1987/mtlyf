import streamlit as st
import yfinance as yf
from huggingface_hub import InferenceClient
import pandas as pd
import time
import re
import logging
import requests

# Setup logging
logging.basicConfig(level=logging.INFO)

# === Secrets / Tokens ===
HF_API_TOKEN = "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"
TELEGRAM_BOT_TOKEN = "7842285230:AAFcisrfFg40AqYjvrGaiq984DYeEu3p6hY"
TELEGRAM_CHAT_ID = "7581145756"

client = InferenceClient(token=HF_API_TOKEN)

# === Models ===
MODELS = {
    "price_prediction": "SelvaprakashV/stock-prediction-model",
    "news_sentiment": "cg1026/financial-news-sentiment-lora",
    "buy_recommendation": "fuchenru/Trading-Hero-LLM"
}

# === Sector Data ===
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

# === Helper Functions ===

def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        price = info.get('regularMarketPrice') or info.get('previousClose')
        volume = info.get('volume')
        return {"price": price, "volume": volume}
    except Exception as e:
        logging.error(f"Fetch data error for {ticker}: {e}")
        return {"price": None, "volume": None}

def call_hf_model_price(ticker, retries=3):
    for _ in range(retries):
        try:
            output = client.text_generation(MODELS["price_prediction"], ticker)
            if isinstance(output, dict):
                text = output.get("generated_text", "")
            else:
                text = output
            logging.info(f"Price prediction output for {ticker}: {text}")
            numbers = re.findall(r"\d+\.\d+", text)
            if numbers:
                price = float(numbers[0])
                if price > 0:
                    return price
            time.sleep(1)
        except Exception as e:
            logging.error(f"Price prediction error for {ticker}: {e}")
    return None

def call_hf_model_sentiment(ticker, retries=3):
    for _ in range(retries):
        try:
            output = client.text_classification(MODELS["news_sentiment"], ticker)
            if output and isinstance(output, list) and "label" in output[0]:
                sentiment = output[0]["label"]
                logging.info(f"Sentiment output for {ticker}: {sentiment}")
                return sentiment
            time.sleep(1)
        except Exception as e:
            logging.error(f"Sentiment error for {ticker}: {e}")
    return "Neutral"

def call_hf_model_buy(ticker, retries=3):
    prompt = f"Should I buy {ticker} stock? One word answer."
    for _ in range(retries):
        try:
            output = client.text_generation(MODELS["buy_recommendation"], prompt)
            if isinstance(output, dict):
                text = output.get("generated_text", "")
            else:
                text = output
            logging.info(f"Buy recommendation output for {ticker}: {text}")
            match = re.search(r"\b(yes|no|buy|hold|sell|strong buy|strong sell)\b", text, re.I)
            if match:
                answer = match.group(0).lower()
                if answer in ["yes", "buy", "strong buy"]:
                    return "Yes"
                else:
                    return "No"
            time.sleep(1)
        except Exception as e:
            logging.error(f"Buy recommendation error for {ticker}: {e}")
    return "No"

def calc_stop_loss(price):
    if isinstance(price, (int, float)):
        return round(price * 0.95, 2)
    return None

def is_strong_signal(pred_price, current_price, buy_recommendation):
    try:
        if pred_price is not None and current_price is not None and buy_recommendation:
            return (pred_price > current_price) and (buy_recommendation.lower() == "yes")
    except:
        return False
    return False

@st.cache_data(show_spinner=False)
def process_sector(tickers):
    results = []
    for ticker in tickers:
        stock = fetch_stock_data(ticker)
        if not stock or stock['price'] is None:
            continue

        pred_price = call_hf_model_price(ticker)
        sentiment = call_hf_model_sentiment(ticker)
        buy = call_hf_model_buy(ticker)
        stop_loss = calc_stop_loss(stock['price'])
        strong_signal = "✅" if is_strong_signal(pred_price, stock['price'], buy) else ""

        results.append({
            "Ticker": ticker,
            "Price": round(stock['price'], 2),
            "Volume": stock['volume'] or 0,
            "Predicted Price": round(pred_price, 2) if pred_price is not None else "N/A",
            "Sentiment": sentiment,
            "Buy Recommendation": buy,
            "Stop Loss": stop_loss,
            "Strong Signal": strong_signal
        })

        time.sleep(0.5)  # To avoid rate limits

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

st.set_page_config(page_title="Stock Sector Dashboard", layout="wide")

st.title("📊 Stock Dashboard with AI Signals")

sector = st.sidebar.selectbox("Select Sector", options=list(ETF_SECTORS.keys()))

st.sidebar.markdown("## Info")
st.sidebar.write("Data refreshed with caching and HuggingFace model calls.")

# Process selected sector
with st.spinner(f"Fetching and processing {sector} data..."):
    data = process_sector(ETF_SECTORS[sector])
    df = pd.DataFrame(data)

if df.empty:
    st.warning("No data available for selected sector.")
    st.stop()

# Summary: Only Buy Recommendations count on top
buy_yes = df[df["Buy Recommendation"] == "Yes"].shape[0]
buy_no = df[df["Buy Recommendation"] == "No"].shape[0]

col1, col2 = st.columns(2)
col1.metric("🟢 Buy Recommendations", buy_yes)
col2.metric("🔴 Not Buy", buy_no)

# Style function for table
def highlight_volume(val):
    if val > 10_000_000:
        color = 'lightgreen'
    elif val > 1_000_000:
        color = 'lightyellow'
    else:
        color = 'lightcoral'
    return f'background-color: {color}'

def highlight_strong_signal(val):
    if val == "✅":
        return "background-color: #90ee90; font-weight: bold; color: green"
    return ""

def highlight_buy_rec(val):
    if val == "Yes":
        return "color: green; font-weight: bold"
    elif val == "No":
        return "color: red; font-weight: bold"
    else:
        return ""

# Apply styling
styled_df = (
    df.style
    .applymap(highlight_buy_rec, subset=["Buy Recommendation"])
    .applymap(highlight_strong_signal, subset=["Strong Signal"])
    .applymap(highlight_volume, subset=["Volume"])
    .format({
        "Price": "${:,.2f}",
        "Predicted Price": lambda x: f"${x:.2f}" if isinstance(x, (float,int)) else x,
        "Stop Loss": lambda x: f"${x:.2f}" if isinstance(x, (float,int)) else x,
        "Volume": "{:,}"
    })
    .set_properties(subset=["Ticker"], **{'font-weight': 'bold'})
    .set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#f0f0f0'), ('color', '#333'), ('font-weight', 'bold')]},
        {'selector': 'td', 'props': [('text-align', 'center')]}
    ])
)

st.dataframe(styled_df, height=650)

# Optional: Button to send summary to Telegram
if st.button("Send Buy Summary to Telegram"):
    message = f"*Buy Recommendations in {sector} Sector:*\nYes: {buy_yes}\nNo: {buy_no}"
    if send_telegram_message(message):
        st.success("Telegram message sent!")
    else:
        st.error("Failed to send Telegram message.")


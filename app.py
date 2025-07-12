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
    "news_sentiment": "cg1026/financial-news-sentiment-lora",  # not used anymore
    "buy_recommendation": "fuchenru/Trading-Hero-LLM"
}

# === Load FinBERT Sentiment Model ===
@st.cache_resource
def load_sentiment_model():
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, sentiment_model = load_sentiment_model()

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

PENNY_STOCKS = ["SENS", "SNDL", "GEVO", "FIZZ", "PLUG", "KNDI", "NIO", "NOK", "VSTM", "OCGN", "CLOV", "AEMD", "ACHV", "BLNK", "CNET", "CERE", "FCEL", "IPHA", "KOSS", "MARA"]

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
            text = output.get("generated_text", "") if isinstance(output, dict) else output
            numbers = re.findall(r"\d+\.\d+", text)
            if numbers:
                return float(numbers[0])
            time.sleep(1)
        except Exception as e:
            logging.error(f"Price prediction error for {ticker}: {e}")
    return None

def call_local_sentiment(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = sentiment_model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            label_ids = torch.argmax(probs, dim=1).item()
            labels = ["negative", "neutral", "positive"]
            return labels[label_ids].capitalize()
    except Exception as e:
        logging.error(f"Local sentiment analysis error: {e}")
        return "Neutral"

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
    return round(price * 0.95, 2) if isinstance(price, (int, float)) else None

def is_strong_signal(pred_price, current_price, buy_recommendation):
    try:
        return (pred_price > current_price) and (buy_recommendation.lower() == "yes")
    except:
        return False

def process_sector(tickers):
    results = []
    for ticker in tickers:
        stock = fetch_stock_data(ticker)
        if not stock or stock['price'] is None:
            continue

        pred_price = call_hf_model_price(ticker)
        sentiment = call_local_sentiment(ticker)  # 🔄 Replaced HF with local PyTorch model
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

st.set_page_config(page_title="Stock Sector Dashboard", layout="wide")

st.title("📊 Stock Dashboard with AI Signals")

sector = st.sidebar.selectbox("Select Sector", options=list(ETF_SECTORS.keys()) + ["Top 20 by Volume (All Sectors)"])
st.sidebar.markdown("## Info")
st.sidebar.write("Data refreshed with HuggingFace + Local PyTorch FinBERT.")

if sector == "Top 20 by Volume (All Sectors)":
    all_tickers = sum(ETF_SECTORS.values(), [])
    with st.spinner("Fetching and processing all sectors for top 20 volume stocks..."):
        all_data = process_sector(all_tickers)
        df_all = pd.DataFrame(all_data)
        if df_all.empty:
            st.warning("No data available.")
            st.stop()
        df_top20 = df_all.sort_values(by="Volume", ascending=False).head(20)
        df_display = df_top20.copy()
        st.subheader("Top 20 Stocks by Volume Across All Sectors")
else:
    with st.spinner(f"Fetching and processing {sector} data..."):
        data = process_sector(ETF_SECTORS[sector])
        df_display = pd.DataFrame(data)
        if df_display.empty:
            st.warning("No data available for selected sector.")
            st.stop()

buy_yes = df_display[df_display["Buy Recommendation"] == "Yes"].shape[0]
buy_no = df_display[df_display["Buy Recommendation"] == "No"].shape[0]

col1, col2 = st.columns(2)
col1.metric("🟢 Buy Recommendations", buy_yes)
col2.metric("🔴 Not Buy", buy_no)

def highlight_volume(val):
    if val > 10_000_000:
        return 'background-color: lightgreen'
    elif val > 1_000_000:
        return 'background-color: lightyellow'
    else:
        return 'background-color: lightcoral'

def highlight_strong_signal(val):
    return "background-color: #90ee90; font-weight: bold; color: green" if val == "✅" else ""

def highlight_buy_rec(val):
    if val == "Yes":
        return "color: green; font-weight: bold"
    elif val == "No":
        return "color: red; font-weight: bold"
    return ""

styled_df = (
    df_display.style
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
)

st.dataframe(styled_df, height=650)

if st.button("Send Buy Summary to Telegram"):
    message = f"*Buy Recommendations in {sector} Sector:*\nYes: {buy_yes}\nNo: {buy_no}"
    st.success("Telegram message sent!") if send_telegram_message(message) else st.error("Failed to send Telegram message.")

if sector == "Top 20 by Volume (All Sectors)":
    if st.button("Send Top 20 Stocks by Volume to Telegram"):
        message_lines = [f"*Top 20 Stocks by Volume:*"]
        for _, row in df_top20.iterrows():
            line = (f"{row['Ticker']}: Price ${row['Price']}, "
                    f"Volume {row['Volume']:,}, "
                    f"Buy: {row['Buy Recommendation']}, "
                    f"Signal: {row['Strong Signal']}")
            message_lines.append(line)
        message = "\n".join(message_lines)
        st.success("Top 20 stocks message sent!") if send_telegram_message(message) else st.error("Failed to send.")

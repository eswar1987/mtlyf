import streamlit as st
import yfinance as yf
from huggingface_hub import InferenceClient
import pandas as pd
import numpy as np
import time
import requests
import re

# === Secrets ===
HF_API_TOKEN = "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"  # Replace with your token or use secrets
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
        return {
            "price": info.get('regularMarketPrice') or info.get('previousClose'),
            "volume": info.get('volume')
        }
    except Exception as e:
        return {"price": None, "volume": None}

def call_hf_model_price(ticker):
    try:
        output = client.text_generation(MODELS["price_prediction"], ticker)
        text = output.get("generated_text", "") if isinstance(output, dict) else output
        numbers = re.findall(r"\d+\.\d+", text)
        return float(numbers[0]) if numbers else "N/A"
    except:
        return "N/A"

def call_hf_model_sentiment(ticker):
    try:
        output = client.text_classification(MODELS["news_sentiment"], ticker)
        if output and isinstance(output, list) and "label" in output[0]:
            return output[0]["label"]
        return "N/A"
    except:
        return "N/A"

def call_hf_model_buy(ticker):
    try:
        prompt = f"Should I buy {ticker} stock? One word answer."
        output = client.text_generation(MODELS["buy_recommendation"], prompt)
        text = output.get("generated_text", "") if isinstance(output, dict) else output
        return text.strip().split()[0].capitalize()
    except:
        return "N/A"

def calc_stop_loss(price):
    return round(price * 0.95, 2) if isinstance(price, (int, float, np.float64)) else "N/A"

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
        try:
            if (isinstance(pred_price, (int, float, np.float64)) and pred_price > stock['price'] 
                and buy.lower() == "yes"):
                strong_signal = "✅"
        except:
            strong_signal = ""

        results.append({
            "Ticker": ticker,
            "Price": stock['price'],
            "Volume": stock['volume'] or 0,
            "Predicted Price": pred_price,
            "Sentiment": sentiment,
            "Buy Recommendation": buy,
            "Stop Loss": stop_loss,
            "Strong Signal": strong_signal
        })

        time.sleep(0.5)  # Small delay to avoid rate limits
    return results

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    params = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        r = requests.get(url, params=params)
        return r.status_code == 200
    except:
        return False

# === Formatting helpers ===
def safe_money_format(x):
    try:
        return f"${float(x):,.2f}"
    except:
        return str(x)

def highlight_volume_max(s):
    is_max = s == s.max()
    return ['background-color: lightblue' if v else '' for v in is_max]

def color_buy(val):
    if isinstance(val, str) and val.lower() == "yes":
        return 'color: green; font-weight: bold'
    elif isinstance(val, str) and val.lower() == "no":
        return 'color: red; font-weight: bold'
    return ''

def highlight_strong_signal(val):
    if val == "✅":
        return 'background-color: #d4edda; font-weight: bold; color: green'
    return ''

# === Streamlit UI ===
st.set_page_config(page_title="Stock Dashboard with LLM Signals", layout="wide")
st.title("📊 Sector-wise Stock Dashboard with LLM Predictions")

# Sidebar select
sectors = list(ETF_SECTORS.keys()) + ["Penny Stocks"]
sector = st.sidebar.selectbox("Select Sector", sectors)

tickers = PENNY_STOCKS if sector == "Penny Stocks" else ETF_SECTORS[sector]

@st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
def cached_sector_data(tickers):
    return process_sector(tickers)

with st.spinner(f"Fetching data for {sector}... This may take a moment."):
    data = cached_sector_data(tickers)

if not data:
    st.warning("No data found for the selected sector.")
else:
    df = pd.DataFrame(data)

    # KPIs
    avg_price = np.mean([x for x in df["Price"] if isinstance(x, (int, float, np.float64))])
    total_vol = np.sum(df["Volume"])
    strong_buy_count = df["Strong Signal"].apply(lambda x: x == "✅").sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Average Price", f"${avg_price:,.2f}")
    col2.metric("Total Volume", f"{total_vol:,}")
    col3.metric("Strong Buy Signals", f"{strong_buy_count}")

    # Styling
    styled_df = df.style \
        .apply(highlight_volume_max, subset=["Volume"]) \
        .applymap(color_buy, subset=["Buy Recommendation"]) \
        .applymap(highlight_strong_signal, subset=["Strong Signal"]) \
        .format({
            "Price": safe_money_format,
            "Predicted Price": safe_money_format,
            "Stop Loss": safe_money_format,
            "Volume": "{:,}"
        })

    st.dataframe(styled_df, height=600)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV", csv, f"{sector}_stocks.csv", "text/csv")

    if st.button("🚀 Send to Telegram"):
        msg = f"*{sector} Sector Overview*\n\n"
        for row in data:
            msg += (f"{row['Ticker']}: Price ${row['Price']}, Predicted ${row['Predicted Price']}, "
                    f"Sentiment: {row['Sentiment']}, Buy: {row['Buy Recommendation']}, "
                    f"SL: ${row['Stop Loss']}, Signal: {row['Strong Signal']}\n")
        if send_telegram_message(msg):
            st.success("Telegram message sent!")
        else:
            st.error("Failed to send Telegram message.")

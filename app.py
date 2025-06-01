import streamlit as st
import yfinance as yf
from huggingface_hub import InferenceClient
import pandas as pd
import time
import requests
import re

# Install matplotlib if missing: pip install matplotlib

# === Embed secrets directly ===
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
@st.cache_data(show_spinner=False)
def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "price": info.get('regularMarketPrice') or info.get('previousClose'),
            "volume": info.get('volume')
        }
    except:
        return {"price": None, "volume": None}

@st.cache_data(show_spinner=False)
def call_hf_model_price(ticker):
    try:
        output = client.text_generation(MODELS["price_prediction"], ticker)
        numbers = re.findall(r"\d+\.\d+", output)
        return float(numbers[0]) if numbers else None
    except:
        return None

@st.cache_data(show_spinner=False)
def call_hf_model_sentiment(ticker):
    try:
        result = client.text_classification(MODELS["news_sentiment"], ticker)
        if result and len(result) > 0:
            return result[0]["label"]
        else:
            return "N/A"
    except:
        return "N/A"

@st.cache_data(show_spinner=False)
def call_hf_model_buy(ticker):
    try:
        prompt = f"Should I buy {ticker} stock? One word answer."
        output = client.text_generation(MODELS["buy_recommendation"], prompt)
        # Extract first word and lowercase it for consistency
        answer = output.strip().split()[0].lower()
        if answer in ["yes", "no"]:
            return answer.capitalize()
        return "N/A"
    except:
        return "N/A"

def calc_stop_loss(price):
    return round(price * 0.95, 2) if isinstance(price, (int, float)) else None

@st.cache_data(show_spinner=False)
def process_sector(tickers):
    results = []
    for ticker in tickers:
        stock = fetch_stock_data(ticker)
        price = stock.get('price')
        volume = stock.get('volume')

        if price is None:
            # Skip tickers with no price data
            continue

        pred_price = call_hf_model_price(ticker)
        sentiment = call_hf_model_sentiment(ticker)
        buy = call_hf_model_buy(ticker)
        stop_loss = calc_stop_loss(price)

        strong_signal = ""
        if (pred_price is not None and pred_price > price) and (buy.lower() == "yes"):
            strong_signal = "✅"

        results.append({
            "Ticker": ticker,
            "Price": round(price, 2),
            "Volume": volume,
            "Predicted Price": round(pred_price, 2) if pred_price is not None else "N/A",
            "Sentiment": sentiment,
            "Buy Recommendation": buy,
            "Stop Loss": stop_loss if stop_loss else "N/A",
            "Strong Signal": strong_signal
        })
        time.sleep(0.8)  # Slight delay to avoid API rate limit
    return results

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    params = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        resp = requests.get(url, params=params)
        return resp.status_code == 200
    except:
        return False


# === Streamlit UI ===
st.set_page_config(page_title="📊 Sector-wise Stock Dashboard", layout="wide")

st.title("📊 Sector-wise Stock Dashboard with LLM Predictions")

# Autorefresh every 5 minutes (300000 ms)
st_autorefresh = st.experimental_memo.clear()  # Clear cache to force refresh if needed, or skip
refresh_interval = 300  # seconds
st_autorefresh_id = st.experimental_get_query_params().get("refresh", ["0"])[0]

if st.button("Refresh Now 🔄"):
    st.experimental_rerun()

sectors = list(ETF_SECTORS.keys()) + ["Penny Stocks"]
sector = st.sidebar.selectbox("Select Sector", sectors)

tickers = PENNY_STOCKS if sector == "Penny Stocks" else ETF_SECTORS[sector]

with st.spinner(f"Loading data for {sector}..."):
    data = process_sector(tickers)

if not data:
    st.warning("No data found for this sector.")
else:
    df = pd.DataFrame(data)

    # Defensive check for columns
    columns_to_display = ["Ticker", "Price", "Volume", "Predicted Price", "Sentiment", "Buy Recommendation", "Stop Loss", "Strong Signal"]

    # Apply styling to dataframe
    def highlight_buy(val):
        if isinstance(val, str):
            if val.lower() == "yes":
                return 'color: green; font-weight: bold;'
            elif val.lower() == "no":
                return 'color: red; font-weight: bold;'
        return ''

    def highlight_strong_signal(val):
        if val == "✅":
            return 'background-color: #d4edda; font-weight: bold;'
        return ''

    styled_df = (
        df.style
        .applymap(highlight_buy, subset=['Buy Recommendation'])
        .applymap(highlight_strong_signal, subset=['Strong Signal'])
        .background_gradient(subset=['Volume'], cmap='Blues')
        .format({"Price": "${:.2f}", "Predicted Price": "${}", "Stop Loss": "${}"})
    )

    st.dataframe(styled_df, use_container_width=True)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"{sector}_stocks.csv",
        mime="text/csv"
    )

    if st.button("🚀 Send Overview to Telegram"):
        msg = f"*{sector} Sector Overview*\n\n"
        for row in data:
            msg += (
                f"{row['Ticker']}: Price ${row['Price']}, Predicted ${row['Predicted Price']}, "
                f"Sentiment: {row['Sentiment']}, Buy: {row['Buy Recommendation']}, "
                f"Stop Loss: ${row['Stop Loss']}, Signal: {row['Strong Signal']}\n"
            )
        if send_telegram_message(msg):
            st.success("Sent to Telegram!")
        else:
            st.error("Failed to send message.")

# Note: For full auto-refresh, consider using:
#   `st.experimental_rerun()` with a timer or JavaScript injected,
# or run with Streamlit’s native rerun feature on file change or manually press Refresh.

import streamlit as st
import yfinance as yf
import pandas as pd
from huggingface_hub import InferenceClient
import requests
import os
from dotenv import load_dotenv

# Load secrets
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Initialize HF client
client = InferenceClient(model="SelvaprakashV/stock-prediction-model", token=HF_API_TOKEN)

# Define sectors and tickers
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

st.set_page_config(layout="wide")
st.title("📊 Comprehensive Stock Market Dashboard")
st.markdown("""
<style>
    .stDataFrame div {
        font-size: 14px;
    }
    .css-1aumxhk {
        overflow: scroll;
    }
</style>
""", unsafe_allow_html=True)

sector = st.selectbox("Select Sector", options=list(ETF_SECTORS.keys()) + ["Penny Stocks"])
tickers = PENNY_STOCKS if sector == "Penny Stocks" else ETF_SECTORS[sector]

results = []

for ticker in tickers:
    try:
        data = yf.Ticker(ticker).info
        price = round(data.get('regularMarketPrice', 0), 2)
        volume = data.get('volume', 0)

        prompt = f"Predict the next price for {ticker} currently priced at {price}"
        pred_response = client.text_generation(prompt, max_new_tokens=10)
        predicted_price = float(''.join(filter(str.isdigit, pred_response)) or 0)

        sentiment_prompt = f"Is {ticker} bullish or bearish today?"
        sentiment_response = client.text_generation(sentiment_prompt, max_new_tokens=10)
        sentiment = "Bullish" if "bullish" in sentiment_response.lower() else "Bearish"

        recommendation = "Buy" if predicted_price > price else "Hold"
        stop_loss = round(price * 0.95, 2)
        status_icon = "🟢" if predicted_price > price else "🔴"

        results.append({
            "Ticker": ticker,
            "Price": f"${price}",
            "Volume": f"{volume:,}",
            "Predicted Price": f"${predicted_price}",
            "Sentiment": sentiment,
            "Buy Recommendation": f"{status_icon} {recommendation}",
            "Stop Loss": f"${stop_loss}"
        })

    except Exception as e:
        results.append({
            "Ticker": ticker,
            "Price": "Error",
            "Volume": "Error",
            "Predicted Price": str(e),
            "Sentiment": "Error",
            "Buy Recommendation": "Error",
            "Stop Loss": "Error"
        })

# Convert to DataFrame
df = pd.DataFrame(results)
st.dataframe(df, use_container_width=True)

# Download CSV
st.download_button("📥 Download CSV", data=df.to_csv(index=False), file_name=f"{sector}_analysis.csv")

# Send Telegram
if st.button("📤 Send Updates to Telegram"):
    try:
        message = df.to_markdown(index=False)
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT

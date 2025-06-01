# streamlit_stock_dashboard.py
import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import re
import time
from huggingface_hub import InferenceClient

# === Load HuggingFace token ===
HF_API_TOKEN = st.secrets.get("HUGGINGFACE_TOKEN", "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")
client = InferenceClient(token=HF_API_TOKEN)

# === Models ===
MODELS = {
    "price_prediction": "SelvaprakashV/stock-prediction-model",
    "news_sentiment": "cg1026/financial-news-sentiment-lora",
    "buy_recommendation": "fuchenru/Trading-Hero-LLM"
}

# === Sector Data ===
SECTORS = {
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
    'Commodities': ["GC=F", "SI=F", "CL=F"],
    'PENNY_STOCKS': ["SENS", "SNDL", "GEVO", "FIZZ", "PLUG", "KNDI", "NIO", "NOK", "VSTM", "OCGN","CLOV", "AEMD", "ACHV", "BLNK", "CNET", "CERE", "FCEL", "IPHA", "KOSS", "MARA"]
}

# === Helper Functions ===
def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        price = stock.info.get('regularMarketPrice')
        volume = stock.info.get('volume')
        return price, volume
    except:
        return None, None

def call_model_prediction(ticker):
    try:
        output = client.text_generation(MODELS["price_prediction"], ticker)
        numbers = re.findall(r"\d+\.\d+", output)
        return float(numbers[0]) if numbers else None
    except:
        return None

def call_model_sentiment(ticker):
    try:
        return client.text_classification(MODELS["news_sentiment"], ticker)[0]["label"]
    except:
        return "N/A"

def call_model_buy(ticker):
    try:
        output = client.text_generation(MODELS["buy_recommendation"], f"Should I buy {ticker} stock? One word answer.")
        return output.strip().split()[0]
    except:
        return "N/A"

def calc_stop_loss(price):
    return round(price * 0.95, 2) if isinstance(price, (int, float)) else None

def process_sector(tickers):
    results = []
    for ticker in tickers:
        price, volume = fetch_stock_data(ticker)
        if price is None:
            continue

        pred_price = call_model_prediction(ticker)
        sentiment = call_model_sentiment(ticker)
        buy = call_model_buy(ticker)
        stop_loss = calc_stop_loss(price)

        strong_signal = "✅" if pred_price and pred_price > price and buy.lower() == "yes" else ""

        results.append({
            "Ticker": ticker,
            "Price": round(price, 2),
            "Volume": volume,
            "Predicted Price": pred_price if pred_price else "N/A",
            "Sentiment": sentiment,
            "Buy Recommendation": buy,
            "Stop Loss": stop_loss,
            "Strong Signal": strong_signal
        })

        time.sleep(0.5)
    return results

# === Streamlit UI ===
st.set_page_config(page_title="📊 Stock Dashboard", layout="wide")
st.title("📈 Sector-wise Stock Dashboard with AI Insights")

selected_sector = st.sidebar.selectbox("Select Sector", list(SECTORS.keys()))
tickers = SECTORS[selected_sector]

with st.spinner(f"Fetching data for {selected_sector}..."):
    data = process_sector(tickers)

if data:
    df = pd.DataFrame(data)
    st.subheader(f"📋 Stocks in {selected_sector} Sector")

    def highlight_row(row):
        return ["background-color: #d1f0d1" if row["Strong Signal"] == "✅" else "" for _ in row]

    def safe_format(val, fmt):
        try:
            return fmt.format(val)
        except:
            return val

    styled_df = df.style.apply(highlight_row, axis=1).format({
        "Price": lambda x: safe_format(x, "${:,.2f}"),
        "Predicted Price": lambda x: safe_format(x, "${:,.2f}"),
        "Stop Loss": lambda x: safe_format(x, "${:,.2f}"),
        "Volume": lambda x: safe_format(x, "{:,}")
    })

    st.dataframe(styled_df, height=600)

    st.download_button("📥 Download CSV", df.to_csv(index=False).encode(), f"{selected_sector}_stocks.csv")

    strong_buys = df[df['Strong Signal'] == "✅"]
    if not strong_buys.empty:
        st.markdown("### 🚀 Strong Buy Recommendations")
        for idx, row in strong_buys.iterrows():
            st.markdown(f"- **{row['Ticker']}**: ${row['Price']} → Predicted ${row['Predicted Price']}")
else:
    st.warning("No data found or failed to fetch.")

# === Summary Button ===
if st.button("📊 Sector Summary Report"):
    st.markdown("## 📢 Sector-wise Buy Summary")
    for sector, symbols in SECTORS.items():
        rows = process_sector(symbols)
        buy_recs = [r['Ticker'] for r in rows if r['Buy Recommendation'].lower() == 'yes']
        st.write(f"**{sector}**: {', '.join(buy_recs) if buy_recs else 'No buys suggested'}")

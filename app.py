import streamlit as st
import yfinance as yf
from huggingface_hub import InferenceClient
import pandas as pd
import time
import re

# === Secrets (move to env or secrets in production) ===
HF_API_TOKEN = "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"
client = InferenceClient(token=HF_API_TOKEN)

# === Models ===
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

# --- Helper functions ---

def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "price": info.get('regularMarketPrice') or info.get('previousClose'),
            "volume": info.get('volume') or 0
        }
    except:
        return {"price": None, "volume": None}

def call_hf_model_price(ticker):
    try:
        output = client.text_generation(MODELS["price_prediction"], ticker)
        # Extract first float number from the output
        numbers = re.findall(r"\d+\.\d+", output.generated_text if hasattr(output, 'generated_text') else output)
        return float(numbers[0]) if numbers else None
    except:
        return None

def call_hf_model_sentiment(ticker):
    try:
        output = client.text_classification(MODELS["news_sentiment"], ticker)
        return output[0]["label"] if output else None
    except:
        return None

def call_hf_model_buy(ticker):
    try:
        prompt = f"Should I buy {ticker} stock? One word answer."
        output = client.text_generation(MODELS["buy_recommendation"], prompt)
        # Take first word from generated text
        text = output.generated_text if hasattr(output, 'generated_text') else output
        return text.strip().split()[0]
    except:
        return None

def calc_stop_loss(price):
    return round(price * 0.95, 2) if isinstance(price, (int, float)) else None

@st.cache_data(show_spinner=False)
def process_sector(tickers):
    results = []
    for ticker in tickers:
        stock = fetch_stock_data(ticker)
        price = stock['price']
        if price is None:
            continue
        volume = stock['volume']

        pred_price = call_hf_model_price(ticker) or 0
        sentiment = call_hf_model_sentiment(ticker) or "N/A"
        buy = call_hf_model_buy(ticker) or "No"
        stop_loss = calc_stop_loss(price)

        strong_signal = "✅" if pred_price > price and buy.lower() == "yes" else ""

        results.append({
            "Ticker": ticker,
            "Price": round(price, 2),
            "Volume": volume,
            "Predicted Price": round(pred_price, 2) if pred_price else "N/A",
            "Sentiment": sentiment,
            "Buy Recommendation": buy,
            "Stop Loss": stop_loss,
            "Strong Signal": strong_signal
        })
        time.sleep(0.5)  # Reduce to avoid rate limits; adjust as needed
    return results

# === Streamlit UI ===
st.set_page_config(page_title="📊 Sector Stock Dashboard with LLM Signals", layout="wide")
st.title("📈 Sector-wise Stock Dashboard with LLM Predictions")

sectors = list(ETF_SECTORS.keys()) + ["Penny Stocks"]
sector = st.sidebar.selectbox("Select Sector", sectors)

tickers = PENNY_STOCKS if sector == "Penny Stocks" else ETF_SECTORS[sector]

# Button for manual refresh
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()

with st.spinner(f"Fetching data for {sector}..."):
    data = process_sector(tickers)

if not data:
    st.warning("No data found for this sector.")
    st.stop()

df = pd.DataFrame(data)

# Summary stats
avg_price = df["Price"].mean()
total_volume = df["Volume"].sum()
strong_buy_count = df["Strong Signal"].apply(lambda x: 1 if x == "✅" else 0).sum()

col1, col2, col3 = st.columns(3)
col1.metric("Average Price", f"${avg_price:,.2f}")
col2.metric("Total Volume", f"{total_volume:,}")
col3.metric("Strong Buy Signals", f"{strong_buy_count}")

# Styling functions
def highlight_volume_max(s):
    is_max = s == s.max()
    return ['background-color: lightblue' if v else '' for v in is_max]

def color_buy(val):
    if str(val).lower() == "yes":
        return 'color: green; font-weight: bold'
    elif str(val).lower() == "no":
        return 'color: red'
    return ''

def highlight_strong_signal(val):
    return 'background-color: #d4edda; font-weight: bold;' if val == "✅" else ''

# Apply styling
styled_df = df.style \
    .apply(highlight_volume_max, subset=["Volume"]) \
    .applymap(color_buy, subset=["Buy Recommendation"]) \
    .applymap(highlight_strong_signal, subset=["Strong Signal"]) \
    .format({"Price": "${:,.2f}", "Predicted Price": "${:,.2f}", "Stop Loss": "${:,.2f}", "Volume": "{:,}"})

# Display with st.write for stable Styler support
st.write(styled_df, height=600)

# CSV download
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    "📥 Download CSV",
    csv,
    f"{sector}_stocks.csv",
    "text/csv"
)

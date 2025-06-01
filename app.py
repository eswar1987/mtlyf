import streamlit as st
import yfinance as yf
from huggingface_hub import InferenceClient
import pandas as pd
import time
import requests
import re

# === Embed secrets directly (replace with your tokens or better use env variables) ===
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
        return {
            "price": info.get('regularMarketPrice') or info.get('previousClose'),
            "volume": info.get('volume')
        }
    except Exception as e:
        print(f"fetch_stock_data error for {ticker}: {e}")
        return {"price": None, "volume": None}

def call_hf_model_price(ticker):
    try:
        output = client.text_generation(MODELS["price_prediction"], ticker)
        text_output = output if isinstance(output, str) else output.get('generated_text', '')
        
        numbers = re.findall(r"\d+\.\d+", text_output)
        if numbers:
            return float(numbers[0])
        else:
            int_numbers = re.findall(r"\d+", text_output)
            if int_numbers:
                return float(int_numbers[0])
            else:
                return "N/A"
    except Exception as e:
        print(f"call_hf_model_price error for {ticker}: {e}")
        return "N/A"

def call_hf_model_sentiment(ticker):
    try:
        output = client.text_classification(MODELS["news_sentiment"], ticker)
        if output and isinstance(output, list) and "label" in output[0]:
            return output[0]["label"]
        return "N/A"
    except Exception as e:
        print(f"call_hf_model_sentiment error for {ticker}: {e}")
        return "N/A"

def call_hf_model_buy(ticker):
    try:
        prompt = f"Should I buy {ticker} stock? One word answer."
        output = client.text_generation(MODELS["buy_recommendation"], prompt)
        text_output = output if isinstance(output, str) else output.get('generated_text', '')
        first_word = text_output.strip().split()[0].lower()
        if first_word in ["yes", "no", "maybe"]:
            return first_word.capitalize()
        else:
            return "N/A"
    except Exception as e:
        print(f"call_hf_model_buy error for {ticker}: {e}")
        return "N/A"

def calc_stop_loss(price):
    try:
        return round(price * 0.95, 2) if isinstance(price, (int, float)) else None
    except Exception as e:
        print(f"calc_stop_loss error: {e}")
        return None

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
        if (isinstance(pred_price, float) and pred_price > stock['price'] and buy.lower() == "yes"):
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

        time.sleep(1)  # be kind to API

    return results

def style_dataframe(df):
    def highlight_volume(val):
        if val == df['Volume'].max():
            return 'background-color: lightblue; font-weight: bold;'
        return ''

    def highlight_buy(val):
        if val == "Yes":
            return 'color: green; font-weight: bold;'
        return ''

    def highlight_strong_signal(val):
        if val == "✅":
            return 'background-color: #d4edda; font-weight: bold;'
        return ''

    styled = df.style.applymap(highlight_volume, subset=['Volume']) \
                     .applymap(highlight_buy, subset=['Buy Recommendation']) \
                     .applymap(highlight_strong_signal, subset=['Strong Signal'])
    return styled

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    params = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    return requests.get(url, params=params).status_code == 200

# === Streamlit UI ===
st.set_page_config(page_title="Stock Dashboard", layout="wide")
st.title("📊 Sector-wise Stock Dashboard with LLM Predictions")

sectors = list(ETF_SECTORS.keys()) + ["Penny Stocks"]
sector = st.sidebar.selectbox("Select Sector", sectors)

tickers = PENNY_STOCKS if sector == "Penny Stocks" else ETF_SECTORS[sector]

with st.spinner(f"Processing {sector}..."):
    data = process_sector(tickers)

if not data:
    st.warning("No data found.")
else:
    df = pd.DataFrame(data)

    # Show only Buy Recommendations on top
    buy_df = df[df["Buy Recommendation"] == "Yes"]

    st.subheader(f"Stocks Recommended to Buy in {sector} Sector")
    if not buy_df.empty:
        st.dataframe(style_dataframe(buy_df), height=400)
    else:
        st.write("No Buy Recommendations currently.")

    st.subheader(f"All Stocks in {sector} Sector")
    st.dataframe(style_dataframe(df), height=600)

    st.download_button(
        "📥 Download CSV",
        df.to_csv(index=False).encode('utf-8'),
        f"{sector}_stocks.csv",
        "text/csv"
    )

    if st.button("🚀 Send to Telegram"):
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


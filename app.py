import streamlit as st
import yfinance as yf
from huggingface_hub import InferenceClient
import pandas as pd
import time
import requests
import matplotlib.pyplot as plt

# ===== Direct API tokens (replace with your tokens) =====
HF_API_TOKEN = "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"
TELEGRAM_BOT_TOKEN = "7842285230:AAFcisrfFg40AqYjvrGaiq984DYeEu3p6hY"
TELEGRAM_CHAT_ID = "7581145756"

# Initialize Huggingface client
client = InferenceClient(token=HF_API_TOKEN)

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

MODELS = {
    "price_prediction": "SelvaprakashV/stock-prediction-model",
    "news_sentiment": "cg1026/financial-news-sentiment-lora",
    "buy_recommendation": "fuchenru/Trading-Hero-LLM"
}

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
        output = client.text_generation(model=MODELS["price_prediction"], inputs=ticker)
        text = output[0]['generated_text'] if output else None
        if text:
            import re
            numbers = re.findall(r"\d+\.\d+", text)
            if numbers:
                return float(numbers[0])
        return None
    except Exception as e:
        return f"Error: {e}"

def call_hf_model_sentiment(ticker):
    try:
        output = client.text_classification(model=MODELS["news_sentiment"], inputs=ticker)
        if output and isinstance(output, list):
            return output[0].get("label", "N/A")
        return "N/A"
    except Exception as e:
        return f"Error: {e}"

def call_hf_model_buy(ticker):
    try:
        prompt = f"Should I buy {ticker} stock? Please answer in one word."
        output = client.text_generation(model=MODELS["buy_recommendation"], inputs=prompt)
        if output:
            text = output[0]['generated_text']
            return text.strip().split()[0]
        return "N/A"
    except Exception as e:
        return f"Error: {e}"

def calc_stop_loss(price):
    if price and isinstance(price, (float, int)):
        return round(price * 0.95, 2)
    return None

def process_sector(sector_name, tickers):
    results = []
    for t in tickers:
        data = fetch_stock_data(t)
        price = data['price']
        volume = data['volume']
        if price is None:
            continue

        pred_price = call_hf_model_price(t)
        sentiment = call_hf_model_sentiment(t)
        buy_rec = call_hf_model_buy(t)
        stop_loss = calc_stop_loss(price)

        results.append({
            "Ticker": t,
            "Price": price,
            "Volume": volume,
            "Predicted Price": pred_price,
            "Sentiment": sentiment,
            "Buy Recommendation": buy_rec,
            "Stop Loss": stop_loss
        })

        time.sleep(1)  # To avoid rate limiting
    return results

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    params = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    resp = requests.get(url, params=params)
    return resp.status_code == 200

# ===== Streamlit UI =====
st.title("Sector-wise Stock Dashboard with Predictions & Sentiments")

sectors = list(ETF_SECTORS.keys()) + ["Penny Stocks"]
selected_sector = st.sidebar.selectbox("Select Sector", sectors)

if selected_sector == "Penny Stocks":
    tickers = PENNY_STOCKS
else:
    tickers = ETF_SECTORS[selected_sector]

with st.spinner(f"Fetching data for {selected_sector}..."):
    data = process_sector(selected_sector, tickers)

if not data:
    st.warning("No data available for this sector.")
else:
    df = pd.DataFrame(data)

    # Format prices nicely
    def format_price(x):
        if isinstance(x, (float, int)):
            return f"${x:.2f}"
        return x

    df['Price_display'] = df['Price'].apply(format_price)
    df['Stop Loss_display'] = df['Stop Loss'].apply(format_price)

    # Strong Buy Signal if predicted price > price and Buy Recommendation contains "buy"
    def strong_buy(row):
        try:
            pred = float(row['Predicted Price'])
            price = float(row['Price'])
            rec = str(row['Buy Recommendation']).lower()
            if pred > price and ("buy" in rec):
                return "✅ Strong Buy"
            else:
                return ""
        except Exception:
            return ""

    df['Strong Buy Signal'] = df.apply(strong_buy, axis=1)

    # Highlight rows where predicted price > price
    def highlight_row(row):
        try:
            pred = float(row['Predicted Price'])
            price = float(row['Price'])
            if pred > price:
                return ['background-color: #d4edda'] * len(row)
        except:
            pass
        return [''] * len(row)

    styled_df = df.style.apply(highlight_row, axis=1)
    st.dataframe(styled_df)

    # Price vs Predicted Price chart for top 10 by volume
    top10 = df.sort_values('Volume', ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(top10['Ticker'], top10['Price'], label='Current Price', alpha=0.7)
    ax.bar(top10['Ticker'], top10['Predicted Price'], label='Predicted Price', alpha=0.5)
    ax.set_ylabel('Price ($)')
    ax.set_title(f"Current vs Predicted Prices - Top 10 by Volume in {selected_sector}")
    ax.legend()
    st.pyplot(fig)

    # Download CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"{selected_sector}_stocks.csv",
        mime='text/csv'
    )

    # Telegram send button
    if st.button("📤 Send to Telegram"):
        message = f"*Sector:* {selected_sector}\n\n"
        for _, row in df.iterrows():
            message += (
                f"{row['Ticker']}: Price {row['Price_display']}, Predicted {row['Predicted Price']}, "
                f"Sentiment {row['Sentiment']}, Buy Rec: {row['Buy Recommendation']}, "
                f"Stop Loss: {row['Stop Loss_display']}, {row['Strong Buy Signal']}\n"
            )
        sent = send_telegram_message(message)
        if sent:
            st.success("✅ Telegram message sent!")
        else:
            st.error("❌ Failed to send Telegram message.")

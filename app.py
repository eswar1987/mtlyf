import streamlit as st
import yfinance as yf
from huggingface_hub import InferenceClient
import time

HF_API_TOKEN = "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"  # Replace this with your token
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
    except Exception as e:
        st.error(f"Error fetching {ticker}: {e}")
        return {"price": None, "volume": None}

def call_hf_model(model_name, input_text):
    try:
        response = client.model_inference(MODELS[model_name], {"inputs": input_text})
        if isinstance(response, list) and len(response) > 0:
            return response[0].get('label', str(response))
        elif isinstance(response, dict):
            return response.get('label', str(response))
        else:
            return str(response)
    except Exception as e:
        return f"Error: {e}"

def calc_stop_loss(price):
    if price:
        return round(price * 0.95, 2)
    return None

def process_sector(sector_name, tickers):
    results = []
    for t in tickers:
        data = fetch_stock_data(t)
        if not data['price']:
            continue
        pred = call_hf_model("price_prediction", t)
        sentiment = call_hf_model("news_sentiment", t)
        buy_rec = call_hf_model("buy_recommendation", f"Should I buy {t}?")
        stop_loss = calc_stop_loss(data['price'])
        results.append({
            "ticker": t,
            "price": data['price'],
            "volume": data['volume'],
            "predicted_price": pred,
            "sentiment": sentiment,
            "buy_recommendation": buy_rec,
            "stop_loss": stop_loss
        })
        time.sleep(1)  # To avoid rate limits
    return results

st.title("Sector-wise Stock Dashboard with Predictions & Sentiments")

all_data = {}

sectors = list(ETF_SECTORS.keys()) + ["Penny Stocks"]
selected_sector = st.sidebar.selectbox("Select Sector", sectors)

if selected_sector == "Penny Stocks":
    all_data[selected_sector] = process_sector(selected_sector, PENNY_STOCKS)
else:
    all_data[selected_sector] = process_sector(selected_sector, ETF_SECTORS[selected_sector])

for sector, stocks in all_data.items():
    st.header(f"{sector} ({len(stocks)} stocks)")
    if stocks:
        st.table([{
            "Ticker": s['ticker'],
            "Price": f"${s['price']:.2f}",
            "Volume": s['volume'],
            "Predicted Price": s['predicted_price'],
            "Sentiment": s['sentiment'],
            "Buy Recommendation": s['buy_recommendation'],
            "Stop Loss": f"${s['stop_loss']:.2f}"
        } for s in stocks])
    else:
        st.write("No data available.")

import streamlit as st
import yfinance as yf
from huggingface_hub import InferenceClient
import pandas as pd
import time, re, logging, requests
from io import BytesIO

logging.basicConfig(level=logging.INFO)

# === Secrets & Models ===
HF_API_TOKEN = "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"
TELEGRAM_BOT_TOKEN = "7842285230:AAFcisrfFg40AqYjvrGaiq984DYeEu3p6hY"
TELEGRAM_CHAT_ID = "7581145756"
client = InferenceClient(token=HF_API_TOKEN)
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

# === Helper Functions ===
def fetch_stock_data(ticker):
    try:
        info = yf.Ticker(ticker).info
        return {"price": info.get('regularMarketPrice') or info.get('previousClose'), "volume": info.get('volume')}
    except: return {"price": None, "volume": None}

def call_hf_model_price(ticker, retries=3):
    for _ in range(retries):
        o = client.text_generation(MODELS["price_prediction"], ticker)
        text = o.get("generated_text", o) if isinstance(o, dict) else o
        nums = re.findall(r"\d+\.\d+", text)
        if nums and (price:=float(nums[0]))>0: return price
        time.sleep(1)
    return None

def call_hf_model_sentiment(ticker, retries=3):
    for _ in range(retries):
        o = client.text_classification(MODELS["news_sentiment"], ticker)
        if isinstance(o, list) and "label" in o[0]:
            return o[0]["label"]
        time.sleep(1)
    return "Neutral"

def call_hf_model_buy(ticker, retries=3):
    prompt=f"Should I buy {ticker} stock? One word answer."
    for _ in range(retries):
        o = client.text_generation(MODELS["buy_recommendation"], prompt)
        text = o.get("generated_text", o) if isinstance(o, dict) else o
        m = re.search(r"\b(yes|no|buy|hold|sell|strong buy|strong sell)\b", text, re.I)
        if m:
            return "Yes" if m.group(0).lower() in ["yes","buy","strong buy"] else "No"
        time.sleep(1)
    return "No"

def calc_stop_loss(price):
    return round(price*0.95, 2) if isinstance(price, (int,float)) else None

def is_strong_signal(pred, current, buy):
    return bool(pred and current and buy.lower()=="yes" and pred > current)

@st.cache_data
def process_sector(tickers):
    res=[]
    for t in tickers:
        s=fetch_stock_data(t)
        if not s or s["price"] is None: continue
        p=call_hf_model_price(t)
        sent=call_hf_model_sentiment(t)
        b=call_hf_model_buy(t)
        sl=calc_stop_loss(s["price"])
        sig="✅" if is_strong_signal(p,s["price"],b) else ""
        res.append({"Ticker":t, "Price":round(s["price"],2), "Volume":s["volume"] or 0,
                    "Predicted Price":round(p,2) if p else "N/A", "Sentiment":sent,
                    "Buy Recommendation":b, "Stop Loss":sl, "Strong Signal":sig})
        time.sleep(0.3)
    return pd.DataFrame(res)

def send_telegram_message(msg):
    url=f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    params={"chat_id":TELEGRAM_CHAT_ID, "text":msg, "parse_mode":"Markdown"}
    try:
        resp=requests.get(url,params=params)
        return resp.status_code==200
    except: return False

def process_all_sectors_top_volume(limit=20):
    all_tickers = [t for lst in ETF_SECTORS.values() for t in lst]
    df = process_sector(all_tickers)
    return df.sort_values("Volume", ascending=False).head(limit)

def send_top_volume_to_telegram(limit=20):
    df = process_all_sectors_top_volume(limit)
    if df.empty:
        return send_telegram_message("⚠️ No data for top volume stocks.")
    lines=[f"*🔥 Top {limit} High‑Volume Stocks (All Sectors)*"]
    for _, r in df.iterrows():
        lines.append(
            f"• *{r['Ticker']}* – ${r['Price']:.2f} | Pred: {r['Predicted Price']} "
            f"| Buy: {r['Buy Recommendation']} | Vol: {int(r['Volume']):,} {r['Strong Signal']}"
        )
    return send_telegram_message("\n".join(lines))

# === Streamlit UI ===
st.set_page_config(page_title="Stock Sector Dashboard", layout="wide")
st.title("📊 Stock Dashboard with AI Signals")

sector = st.sidebar.selectbox("Select Sector", list(ETF_SECTORS.keys()))
vol_thresh = st.sidebar.slider("Minimum Volume (M)", 0, 50, 10)*1_000_000
st.sidebar.markdown("Model-powered signals | Export data | Telegram")

df_sector = process_sector(ETF_SECTORS[sector])
df = df_sector[df_sector["Volume"] > vol_thresh]

if df.empty:
    st.warning("No data for chosen filters.")
    st.stop()

buy_yes, buy_no = df[df["Buy Recommendation"]=="Yes"].shape[0], df[df["Buy Recommendation"]=="No"].shape[0]
c1, c2 = st.columns(2)
c1.metric("🟢 Buy", buy_yes)
c2.metric("🔴 Not Buy", buy_no)

def fmt_vol(v):
    if v>10_000_000: return 'background-color: lightgreen'
    if v>1_000_000: return 'background-color: lightyellow'
    return 'background-color: lightcoral'

styled = (
    df.style.applymap(lambda v: "color: green; font-weight: bold" if v=="Yes" else ("color: red; font-weight: bold" if v=="No" else ""),
                     subset=["Buy Recommendation"])
            .applymap(lambda v: "background-color: #90ee90; font-weight: bold; color: green" if v=="✅" else "", subset=["Strong Signal"])
            .applymap(fmt_vol, subset=["Volume"])
            .format({"Price":"${:,.2f}", "Predicted Price":lambda x: f"${x:.2f}" if isinstance(x,(int,float)) else x,
                     "Stop Loss":lambda x: f"${x:.2f}" if isinstance(x,(int,float)) else x, "Volume":"{:,}"})
            .set_properties(subset=["Ticker"], **{"font-weight":"bold"})
)

st.dataframe(styled, height=600)

# Exports
st.markdown("### 📥 Export")
csv_data = df.to_csv(index=False).encode()
buf = BytesIO(); df.to_excel(buf, index=False, engine="openpyxl"); xlsx_data = buf.getvalue()
c_csv, c_xl = st.columns(2)
c_csv.download_button("⬇️ CSV", csv_data, "stocks.csv","text/csv")
c_xl.download_button("⬇️ Excel", xlsx_data, "stocks.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# Telegram Buttons
if st.button("📤 Send Buy Summary to Telegram"):
    msg=(
        f"*Buy Recs in {sector} Sector*\n"
        f"Yes: {buy_yes}  |  No: {buy_no}\n"
        f"Filtered Volume > {vol_thresh:,}"
    )
    st.success("Sent!" if send_telegram_message(msg) else "Failed.")

if st.button("📤 Send Top 20 High‑Volume Stocks to Telegram"):
    with st.spinner("Fetching top 20..."):
        st.success("Sent!" if send_top_volume_to_telegram(20) else "Failed.")

# Stub for future automation
def automated_telegram_alert():
    dft = process_all_sectors_top_volume(20)
    msg = (
        f"*Daily Top 20 High‑Volume Stocks*\n" +
        "\n".join([
            f"{r['Ticker']}: ${r['Price']:.2f}, Buy: {r['Buy Recommendation']}, Vol: {int(r['Volume']):,} {r['Strong Signal']}"
            for _, r in dft.iterrows()
        ])
    )
    send_telegram_message(msg)

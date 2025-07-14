import streamlit as st
import yfinance as yf
import pandas as pd
import time
import logging
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# === Secrets / Tokens ===
HF_API_TOKEN = "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"
TELEGRAM_BOT_TOKEN = "7842285230:AAFcisrfFg40AqYjvrGaiq984DYeEu3p6hY"
TELEGRAM_CHAT_ID = "7581145756"

# === Logging ===
logging.basicConfig(level=logging.INFO)

# === Cache Models ===
@st.cache_resource
def load_sentiment_model():
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

@st.cache_resource
def load_buy_rec_pipeline():
    model_name = "fuchenru/Trading-Hero-LLM"
    return pipeline("text-classification", model=model_name, tokenizer=model_name)

tokenizer, sentiment_model = load_sentiment_model()
buy_rec_pipeline = load_buy_rec_pipeline()

# === Sectors and tickers ===
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
@st.cache_data(ttl=3600)
def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if hist.empty:
            raise ValueError("No historical data")
        price = hist['Close'].iloc[-1]
        volume = hist['Volume'].iloc[-1]
        return {"price": price, "volume": volume}
    except Exception as e:
        logging.error(f"Error fetching stock data for {ticker}: {e}")
        return {"price": None, "volume": None}

@st.cache_data(ttl=3600)
def fetch_recent_headline(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if news and len(news) > 0:
            return news[0].get('title', f"{ticker} stock")
    except Exception as e:
        logging.warning(f"No news for {ticker}: {e}")
    return f"{ticker} stock"

def call_local_sentiment_with_score(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = sentiment_model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            score, label_id = torch.max(probs, dim=1)
            label = ["Negative", "Neutral", "Positive"][label_id.item()]
            return label, round(score.item(), 2)
    except Exception as e:
        logging.error(f"Sentiment model error: {e}")
        return "Neutral", 0.0

def calc_stop_loss(price):
    return round(price * 0.95, 2) if price else None

def enhanced_buy_recommendation(pred_price, current_price, confidence, volume):
    if (
        pred_price is not None and current_price is not None and
        pred_price > current_price and
        confidence > 0.92 and
        volume > 1_000_000
    ):
        return "Yes"
    return "No"

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    params = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        response = requests.get(url, params=params)
        return response.status_code == 200
    except Exception as e:
        logging.error(f"Telegram send error: {e}")
        return False

# === LSTM Model ===
class StockPriceLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(StockPriceLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.linear(out[:, -1, :])
        return out

def prepare_data(df, sequence_length=20):
    close_prices = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_prices)
    X, y = [], []
    for i in range(len(scaled) - sequence_length):
        X.append(scaled[i:i+sequence_length])
        y.append(scaled[i+sequence_length])
    X, y = np.array(X), np.array(y)
    return X, y, scaler

@st.cache_data(ttl=3600)
def predict_price_lstm_backtest(ticker):
    try:
        df = yf.Ticker(ticker).history(period="120d")
        if df.empty or len(df) < 40:
            return None, None

        X, y, scaler = prepare_data(df)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = StockPriceLSTM()
        model.to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

        model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            predictions = model(X_tensor).cpu().numpy()
            actuals = y_tensor.cpu().numpy()

        predictions = np.array(predictions).reshape(-1, 1)
        actuals = np.array(actuals).reshape(-1, 1)
        pred_prices = scaler.inverse_transform(predictions)
        actual_prices = scaler.inverse_transform(actuals)

        accuracy = 100 - np.mean(np.abs(pred_prices - actual_prices) / actual_prices) * 100
        return round(float(pred_prices[-1][0]), 2), round(accuracy, 2)

    except Exception as e:
        logging.error(f"Backtest error for {ticker}: {e}")
        return None, None

# === Process sector data ===
def process_sector(tickers):
    results = []
    for ticker in tickers:
        stock = fetch_stock_data(ticker)
        if not stock or stock['price'] is None:
            continue
        headline = fetch_recent_headline(ticker)
        sentiment, confidence = call_local_sentiment_with_score(headline)
        pred_price, _ = predict_price_lstm_backtest(ticker)
        buy = enhanced_buy_recommendation(pred_price, stock['price'], confidence, stock['volume'])
        stop_loss = calc_stop_loss(stock['price'])
        strong = "✅" if buy == "Yes" else ""

        results.append({
            "Ticker": ticker,
            "Price": round(stock['price'], 2),
            "Volume": int(stock['volume']) if stock['volume'] else 0,
            "Predicted Price": pred_price if pred_price else "N/A",
            "Headline": headline,
            "Sentiment": sentiment,
            "Confidence": confidence,
            "Buy Recommendation": buy,
            "Stop Loss": stop_loss,
            "Strong Signal": strong
        })
        time.sleep(0.3)  # Slight delay to avoid API rate limits
    return results

# === Streamlit UI ===
st.set_page_config(page_title="📊 AI Stock Sentiment Dashboard", layout="wide")
tabs = st.tabs(["📊 Dashboard", "🔁 Backtest"])

with tabs[0]:
    st.title("📊 AI Stock Sentiment Dashboard")
    sector = st.sidebar.selectbox("Select Sector", options=list(ETF_SECTORS.keys()))
    st.sidebar.markdown("Data powered by PyTorch FinBERT + HuggingFace + Yahoo Finance")

    with st.spinner("Processing data..."):
        data = process_sector(ETF_SECTORS[sector])
    df = pd.DataFrame(data)

    if df.empty:
        st.warning("No data available.")
        st.stop()

    col1, col2 = st.columns(2)
    col1.metric("✅ Strong Buy Signals", df[df["Strong Signal"] == "✅"].shape[0])
    col2.metric("Total Stocks", df.shape[0])

    def highlight_signal(val):
        return "background-color: #c8f7c5; font-weight: bold" if val == "✅" else ""

    def highlight_volume(val):
        if val > 10_000_000:
            return "background-color: lightgreen"
        elif val > 1_000_000:
            return "background-color: lightyellow"
        return ""

    styled_df = (
        df.style
        .applymap(highlight_signal, subset=["Strong Signal"])
        .applymap(highlight_volume, subset=["Volume"])
        .format({
            "Price": "${:,.2f}",
            "Predicted Price": lambda x: f"${x:.2f}" if isinstance(x, (float, int)) else x,
            "Stop Loss": lambda x: f"${x:.2f}" if isinstance(x, (float, int)) else x,
            "Volume": "{:,}",
            "Confidence": "{:.2f}"
        })
    )

    st.dataframe(styled_df, height=700)

    if st.button("Send Top 5 Strong Signals to Telegram"):
        top5 = df[df["Strong Signal"] == "✅"].sort_values(by="Confidence", ascending=False).head(5)
        if top5.empty:
            st.warning("No strong signals to send.")
        else:
            lines = ["*Top 5 Strong Buy Signals:*"]
            for _, row in top5.iterrows():
                lines.append(
                    f"{row['Ticker']} | {row['Sentiment']} ({row['Confidence']*100:.1f}%) | "
                    f"Price: ${row['Price']} | Predicted: ${row['Predicted Price']}"
                )
            msg = "\n".join(lines)
            if send_telegram_message(msg):
                st.success("Telegram message sent.")
            else:
                st.error("Failed to send message.")

with tabs[1]:
    st.title("🔁 Backtest Accuracy")
    sector_bt = st.selectbox("Select Sector to Backtest", list(ETF_SECTORS.keys()), key="backtest_sector")
    tickers_bt = ETF_SECTORS[sector_bt]
    backtest_data = []

    with st.spinner("Running backtest predictions..."):
        for ticker in tickers_bt:
            pred_price, accuracy = predict_price_lstm_backtest(ticker)
            if pred_price and accuracy:
                backtest_data.append({"Ticker": ticker, "Predicted Price": pred_price, "Accuracy (%)": accuracy})

    if backtest_data:
        df_bt = pd.DataFrame(backtest_data)
        st.dataframe(df_bt)
        st.download_button("📥 Download Backtest CSV", df_bt.to_csv(index=False), "backtest.csv")
    else:
        st.warning("No backtest results available for this sector.")

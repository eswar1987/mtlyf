import streamlit as st
import yfinance as yf
from huggingface_hub import InferenceClient
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
from concurrent.futures import ThreadPoolExecutor

# === Secrets / Tokens ===
HF_API_TOKEN = "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"
TELEGRAM_BOT_TOKEN = "7842285230:AAFcisrfFg40AqYjvrGaiq984DYeEu3p6hY"
TELEGRAM_CHAT_ID = "7581145756"
client = InferenceClient(token=HF_API_TOKEN)

# === Models ===
MODELS = {
    "buy_recommendation": "fuchenru/Trading-Hero-LLM"
}

# === Logging ===
logging.basicConfig(level=logging.INFO)

# === Load FinBERT Sentiment Model ===
@st.cache_resource
def load_sentiment_model():
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, sentiment_model = load_sentiment_model()

@st.cache_resource
def load_buy_rec_pipeline():
    model_name = MODELS["buy_recommendation"]
    return pipeline("text-classification", model=model_name, tokenizer=model_name)

buy_rec_pipeline = load_buy_rec_pipeline()

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

def call_hf_model_buy(ticker):
    prompt = f"Should I buy {ticker} stock? One word answer."
    try:
        results = buy_rec_pipeline(prompt)
        if results and len(results) > 0:
            label = results[0]['label'].lower()
            if label in ['buy', 'yes', 'strong buy']:
                return "Yes"
            else:
                return "No"
    except Exception as e:
        logging.error(f"Buy recommendation error for {ticker}: {e}")
    return "No"

def calc_stop_loss(price):
    return round(price * 0.95, 2) if price else None

def enhanced_buy_recommendation(pred_price, current_price, confidence, volume, model_label):
    # Thresholds can be adjusted
    if (
        pred_price and current_price and
        pred_price > current_price * 1.01 and
        confidence >= 0.6 and
        volume > 1_000_000 and
        model_label.lower() in ['yes', 'buy', 'strong buy']
    ):
        return "Yes"
    return "No"

# === LSTM Model Definition for price prediction ===
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

def predict_price_lstm(ticker):
    try:
        df = yf.Ticker(ticker).history(period="100d")
        if df.empty or len(df) < 30:
            return None

        X, y, scaler = prepare_data(df)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = StockPriceLSTM()
        model.to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

        model.train()
        epochs = 50
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

        model.eval()
        last_seq = torch.tensor(X[-1:], dtype=torch.float32).to(device)
        with torch.no_grad():
            pred_scaled = model(last_seq).cpu().numpy()

        pred_price = scaler.inverse_transform(pred_scaled)[0][0]
        return round(float(pred_price), 2)
    except Exception as e:
        logging.error(f"LSTM price prediction error for {ticker}: {e}")
        return None

# Cache the stock info processing for faster UI switching sectors
@st.cache_data(show_spinner=False)
def process_ticker_data(tickers):
    results = []

    def process(ticker):
        stock = fetch_stock_data(ticker)
        if not stock or stock['price'] is None:
            return None
        headline = fetch_recent_headline(ticker)
        sentiment, confidence = call_local_sentiment_with_score(headline)
        pred_price = predict_price_lstm(ticker)
        model_label = call_hf_model_buy(ticker)
        buy = enhanced_buy_recommendation(pred_price, stock['price'], confidence, stock['volume'], model_label)
        stop_loss = calc_stop_loss(stock['price'])
        strong_signal = "✅" if buy == "Yes" else ""

        return {
            "Ticker": ticker,
            "Price": round(stock['price'], 2),
            "Volume": int(stock['volume']) if stock['volume'] else 0,
            "Predicted Price": pred_price if pred_price else "N/A",
            "Headline": headline,
            "Sentiment": sentiment,
            "Confidence": confidence,
            "Buy Recommendation": buy,
            "Stop Loss": stop_loss,
            "Strong Signal": strong_signal
        }

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process, ticker) for ticker in tickers]
        for future in futures:
            res = future.result()
            if res:
                results.append(res)
    return results

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    params = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        response = requests.get(url, params=params)
        return response.status_code == 200
    except Exception as e:
        logging.error(f"Telegram send error: {e}")
        return False

# === Streamlit UI ===
st.set_page_config(page_title="📊 AI Stock Sentiment Dashboard", layout="wide")
tabs = st.tabs(["📊 Dashboard", "🔁 Backtest"])

with tabs[0]:
    st.title("📊 AI Stock Sentiment Dashboard")
    sector = st.sidebar.selectbox("Select Sector", options=list(ETF_SECTORS.keys()))
    st.sidebar.markdown("Data powered by PyTorch FinBERT + HuggingFace + Yahoo Finance")

    tickers = ETF_SECTORS[sector]
    with st.spinner("Processing data (this may take up to 30 seconds on first run)..."):
        data = process_ticker_data(tickers)

    df = pd.DataFrame(data)
    if df.empty:
        st.warning("No data available for this sector.")
        st.stop()

    col1, col2 = st.columns(2)
    col1.metric("🟢 Buy Recommendations", df[df["Buy Recommendation"] == "Yes"].shape[0])
    col2.metric("🔴 Not Buy", df[df["Buy Recommendation"] == "No"].shape[0])

    def highlight_buy(val):
        return "color: green; font-weight: bold" if val == "Yes" else "color: red; font-weight: bold"

    def highlight_signal(val):
        return "background-color: #c8f7c5; font-weight: bold" if val == "✅" else ""

    def highlight_volume(val):
        if val > 10_000_000:
            return "background-color: lightgreen"
        elif val > 1_000_000:
            return "background-color: lightyellow"
        return "background-color: lightcoral"

    styled_df = (
        df.style
        .applymap(highlight_buy, subset=["Buy Recommendation"])
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

    if st.button("Send Top 5 Buy Recommendations to Telegram"):
        top5 = df[df["Buy Recommendation"] == "Yes"].sort_values(by="Confidence", ascending=False).head(5)
        if top5.empty:
            st.warning("No buy recommendations to send.")
        else:
            lines = ["*Top 5 Buy Recommendations:*"]
            for _, row in top5.iterrows():
                lines.append(
                    f"{row['Ticker']} | {row['Sentiment']} ({row['Confidence']*100:.1f}%) | "
                    f"Price: ${row['Price']} | Predicted: {row['Predicted Price']}"
                )
            message = "\n".join(lines)
            if send_telegram_message(message):
                st.success("Telegram message sent.")
            else:
                st.error("Failed to send message.")

    st.download_button("📥 Download Signals CSV", df.to_csv(index=False), "signals.csv")

# --- Backtest tab ---
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

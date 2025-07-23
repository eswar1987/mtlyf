import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Ticker Fetcher with Debug", layout="wide")

@st.cache_data(ttl=86400)
def fetch_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    st.write(f"Found {len(tables)} tables on S&P 500 Wikipedia page")

    for i, table in enumerate(tables):
        st.write(f"Table {i} columns: {table.columns.tolist()}")
        if 'Symbol' in table.columns:
            st.write(f"Using table {i} for S&P 500 tickers")
            symbols = table['Symbol'].tolist()
            st.write(f"First 10 S&P 500 tickers: {symbols[:10]}")
            return symbols

    st.error("No table with 'Symbol' column found on S&P 500 Wikipedia")
    return []

@st.cache_data(ttl=86400)
def fetch_nasdaq100_tickers():
    url = "https://en.wikipedia.org/wiki/NASDAQ-100"
    tables = pd.read_html(url)
    st.write(f"Found {len(tables)} tables on Nasdaq-100 Wikipedia page")

    for i, table in enumerate(tables):
        st.write(f"Table {i} columns: {table.columns.tolist()}")
        if 'Ticker' in table.columns:
            st.write(f"Using table {i} for Nasdaq 100 tickers")
            symbols = table['Ticker'].tolist()
            # Replace '.' with '-' for yfinance compatibility (e.g. BRK.B -> BRK-B)
            symbols = [sym.replace('.', '-') for sym in symbols]
            st.write(f"First 10 Nasdaq 100 tickers: {symbols[:10]}")
            return symbols

    st.error("No table with 'Ticker' column found on Nasdaq-100 Wikipedia")
    return []

@st.cache_data(ttl=86400)
def fetch_nifty50_tickers():
    try:
        url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://www.nseindia.com/"
        }
        session = requests.Session()
        # Initial request to get cookies
        session.get("https://www.nseindia.com", headers=headers)
        response = session.get(url, headers=headers, timeout=10)
        data = response.json()
        symbols = [item['symbol'] + ".NS" for item in data['data']]
        st.write(f"Fetched {len(symbols)} Nifty 50 tickers")
        st.write(f"First 10 Nifty 50 tickers: {symbols[:10]}")
        return symbols
    except Exception as e:
        st.error(f"Failed to fetch Nifty 50: {e}")
        st.warning("Using static fallback list")
        fallback = [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
            "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "HDFC.NS"
        ]
        st.write(f"Fallback tickers: {fallback}")
        return fallback

def main():
    st.title("📈 Stock Ticker Fetcher with Debug Info")

    sp500_tickers = fetch_sp500_tickers()
    nasdaq_tickers = fetch_nasdaq100_tickers()
    nifty_tickers = fetch_nifty50_tickers()

    st.header("S&P 500 Tickers")
    st.write(f"Total tickers fetched: {len(sp500_tickers)}")
    st.dataframe(pd.DataFrame(sp500_tickers, columns=["Ticker"]))

    st.header("Nasdaq 100 Tickers")
    st.write(f"Total tickers fetched: {len(nasdaq_tickers)}")
    st.dataframe(pd.DataFrame(nasdaq_tickers, columns=["Ticker"]))

    st.header("Nifty 50 Tickers")
    st.write(f"Total tickers fetched: {len(nifty_tickers)}")
    st.dataframe(pd.DataFrame(nifty_tickers, columns=["Ticker"]))

if __name__ == "__main__":
    main()

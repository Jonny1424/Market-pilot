import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone

st.set_page_config(page_title="MarketPilot AI", page_icon="📈", layout="wide")

UNIVERSE = {
    "Forex": ["GBPUSD=X", "EURUSD=X", "USDJPY=X"],
    "Crypto": ["BTC-USD", "ETH-USD"],
    "Stocks & ETFs": ["AAPL", "MSFT", "NVDA", "SPY", "VUAG.L"]
}

st.title("📈 MarketPilot AI")
st.caption("Version 0.1 • Multi-market research scanner • No trade execution")
st.warning(
    "Research prototype only. Signals are rule-based, not AI predictions or financial advice. "
    "No broker connection, live orders, leverage or real-money execution is enabled."
)

with st.sidebar:
    st.header("Scanner settings")
    period = st.selectbox("Historical data", ["6mo", "1y", "2y", "5y"], index=1)
    interval = st.selectbox("Chart interval", ["1d", "1h"], index=0)
    starting_balance = st.number_input("Paper account balance (£)", min_value=50.0, max_value=100000.0,
                                       value=500.0, step=50.0)
    risk_pct = st.slider("Illustrative risk per trade (%)", 0.1, 1.0, 0.5, 0.1)
    min_score = st.slider("Minimum signal score", 1, 5, 3)
    st.caption("Daily candles are recommended for the first tests. Intraday data availability varies.")

@st.cache_data(ttl=300, show_spinner=False)
def load_data(symbol, period, interval):
    data = yf.download(symbol, period=period, interval=interval, auto_adjust=True,
                       progress=False, threads=False)
    if data is None or data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.rename(columns=str.title)
    if "Close" not in data.columns:
        return pd.DataFrame()
    return data.dropna(subset=["Close"])

def indicators(df):
    x = df.copy()
    close = x["Close"].astype(float)
    x["EMA20"] = close.ewm(span=20, adjust=False).mean()
    x["EMA50"] = close.ewm(span=50, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    x["RSI14"] = 100 - (100 / (1 + rs))
    prev = close.shift(1)
    high = x["High"].astype(float) if "High" in x else close
    low = x["Low"].astype(float) if "Low" in x else close
    tr = pd.concat([(high-low), (high-prev).abs(), (low-prev).abs()], axis=1).max(axis=1)
    x["ATR14"] = tr.rolling(14).mean()
    return x

def score_row(symbol, market, df):
    if len(df) < 55:
        return None
    x = indicators(df).dropna()
    if len(x) < 2:
        return None
    r = x.iloc[-1]
    score = 0
    reasons = []
    if r["EMA20"] > r["EMA50"]:
        score += 1; reasons.append("EMA20 above EMA50")
    if r["Close"] > r["EMA20"]:
        score += 1; reasons.append("Price above EMA20")
    if 50 <= r["RSI14"] <= 68:
        score += 1; reasons.append("RSI momentum in range")
    if r["RSI14"] > 72:
        reasons.append("RSI elevated")
    if r["RSI14"] < 30:
        reasons.append("RSI oversold")
    if len(x) > 1 and r["Close"] > x["Close"].iloc[-2]:
        score += 1; reasons.append("Latest close rising")
    if r["ATR14"] > 0:
        score += 1; reasons.append("ATR available")
    direction = "WATCH"
    if score >= min_score and r["EMA20"] > r["EMA50"] and r["Close"] > r["EMA20"]:
        direction = "BULLISH SETUP"
    elif r["EMA20"] < r["EMA50"] and r["Close"] < r["EMA20"]:
        direction = "BEARISH SETUP"
    stop = float(r["Close"] - 2*r["ATR14"]) if direction == "BULLISH SETUP" else (
        float(r["Close"] + 2*r["ATR14"]) if direction == "BEARISH SETUP" else np.nan)
    target = float(r["Close"] + 3*r["ATR14"]) if direction == "BULLISH SETUP" else (
        float(r["Close"] - 3*r["ATR14"]) if direction == "BEARISH SETUP" else np.nan)
    return {
        "Market": market, "Symbol": symbol, "Signal": direction, "Score": score,
        "Last price": float(r["Close"]), "RSI (14)": float(r["RSI14"]),
        "EMA20": float(r["EMA20"]), "EMA50": float(r["EMA50"]),
        "ATR (14)": float(r["ATR14"]), "Illustrative stop": stop,
        "Illustrative target": target, "Reasons": "; ".join(reasons),
        "Data time": str(x.index[-1])
    }

st.subheader("Market scan")
if st.button("Run market scan", type="primary"):
    rows = []
    failures = []
    progress = st.progress(0)
    total = sum(len(v) for v in UNIVERSE.values())
    done = 0
    for market, symbols in UNIVERSE.items():
        for symbol in symbols:
            try:
                raw = load_data(symbol, period, interval)
                result = score_row(symbol, market, raw)
                if result:
                    rows.append(result)
                else:
                    failures.append(symbol)
            except Exception:
                failures.append(symbol)
            done += 1
            progress.progress(done / total)
    progress.empty()
    if rows:
        results = pd.DataFrame(rows).sort_values(["Score", "Symbol"], ascending=[False, True])
        st.session_state["scan_results"] = results
        st.session_state["scan_time"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if failures:
        st.info("No usable data returned for: " + ", ".join(failures) +
                ". Data availability can vary by symbol, interval and provider.")

if "scan_results" in st.session_state:
    results = st.session_state["scan_results"]
    st.caption("Last scan: " + st.session_state.get("scan_time", ""))
    c1, c2, c3 = st.columns(3)
    c1.metric("Assets scanned", len(results))
    c2.metric("Bullish setups", int((results["Signal"] == "BULLISH SETUP").sum()))
    c3.metric("Bearish setups", int((results["Signal"] == "BEARISH SETUP").sum()))
    display = results.copy()
    for col in ["Last price", "RSI (14)", "EMA20", "EMA50", "ATR (14)", "Illustrative stop", "Illustrative target"]:
        display[col] = display[col].round(4)
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.download_button("Download scan as CSV", results.to_csv(index=False).encode("utf-8"),
                       file_name="marketpilot_scan.csv", mime="text/csv")

    st.subheader("Inspect an asset")
    selected = st.selectbox("Choose symbol", results["Symbol"].tolist())
    market = results.loc[results["Symbol"] == selected, "Market"].iloc[0]
    chart = indicators(load_data(selected, period, interval)).dropna()
    if not chart.empty:
        st.line_chart(chart[["Close", "EMA20", "EMA50"]], use_container_width=True)
        st.caption(f"{selected} • {market} • Data supplied by Yahoo Finance via yfinance.")

st.divider()
st.subheader("Paper-trading risk calculator")
st.write("Illustrative position sizing only. This does not place an order.")
a, b, c = st.columns(3)
entry = a.number_input("Hypothetical entry price", min_value=0.0001, value=100.0, step=1.0)
stop = b.number_input("Hypothetical stop price", min_value=0.0001, value=98.0, step=0.5)
risk_cash = starting_balance * (risk_pct / 100)
distance = abs(entry-stop)
units = risk_cash / distance if distance > 0 else 0
c.metric("Maximum theoretical units", f"{units:.4f}")
st.caption(f"At {risk_pct:.1f}% risk, the illustrative cash risk is £{risk_cash:.2f}. "
           "Real position sizing must account for currency conversion, contract sizes, spreads, slippage and broker minimums.")

st.divider()
st.subheader("Next development modules")
st.markdown("""
- **Paper ledger:** persist hypothetical entries, exits, fees and equity curve.
- **Backtesting:** walk-forward testing, out-of-sample periods and realistic transaction costs.
- **AI agents:** news/fundamental analyst, strategy challenger and performance reviewer.
- **Broker adapters:** OANDA practice, IBKR paper and crypto sandbox/public-data integrations.
""")
st.caption("Data note: yfinance is an unofficial research interface to Yahoo Finance. Check Yahoo's terms and data rights; do not redistribute data or use this prototype as a commercial data feed.")

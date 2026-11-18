import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone
from supabase import create_client
from postgrest.exceptions import APIError

st.set_page_config(page_title="MarketPilot", page_icon="📈", layout="wide")

UNIVERSE = {
    "Forex": ["GBPUSD=X", "EURUSD=X", "USDJPY=X"],
    "Crypto": ["BTC-USD", "ETH-USD"],
    "Stocks & ETFs": ["AAPL", "MSFT", "NVDA", "SPY", "VUAG.L"],
}

st.title("📈 MarketPilot")
st.caption("Version 0.2 • Market scanner and paper-trading journal • No live trading")
st.warning(
    "Paper trading only. Signals are rule-based research indicators, not AI predictions "
    "or financial advice. No broker connection or real-money orders are enabled."
)

def get_supabase():
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except Exception:
        return None

supabase = get_supabase()
if supabase is None:
    st.error(
        "Database connection settings are missing. In Streamlit app settings → Secrets, "
        "add SUPABASE_URL and SUPABASE_KEY. Do not put credentials in GitHub or share them."
    )
    st.stop()

def db_error_message(exc):
    msg = str(exc)
    if "permission" in msg.lower() or "rls" in msg.lower() or "row-level" in msg.lower():
        return (
            "Supabase rejected the request. Check that Streamlit Secrets contains the "
            "server-side secret/service_role key, not the publishable/anon key. "
            "Never expose that key in the browser or GitHub."
        )
    return f"Database request failed: {msg[:500]}"

@st.cache_data(ttl=300, show_spinner=False)
def load_data(symbol, period, interval):
    data = yf.download(
        symbol, period=period, interval=interval, auto_adjust=True,
        progress=False, threads=False
    )
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

def score_row(symbol, market, df, min_score):
    if len(df) < 55:
        return None
    x = indicators(df).dropna()
    if len(x) < 2:
        return None
    r = x.iloc[-1]
    score, reasons = 0, []
    if r["EMA20"] > r["EMA50"]:
        score += 1
        reasons.append("EMA20 above EMA50")
    if r["Close"] > r["EMA20"]:
        score += 1
        reasons.append("Price above EMA20")
    if 50 <= r["RSI14"] <= 68:
        score += 1
        reasons.append("RSI momentum in range")
    if r["RSI14"] > 72:
        reasons.append("RSI elevated")
    if r["RSI14"] < 30:
        reasons.append("RSI oversold")
    if r["Close"] > x["Close"].iloc[-2]:
        score += 1
        reasons.append("Latest close rising")
    if r["ATR14"] > 0:
        score += 1
        reasons.append("ATR available")
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

def fetch_trades(status=None):
    query = supabase.table("trades").select("*").order("opened_at", desc=True)
    if status:
        query = query.eq("status", status)
    return query.execute().data or []

def fetch_journal():
    return supabase.table("trade_journal").select("*").order("created_at", desc=True).execute().data or []

def fmt_money(value):
    try:
        return f"£{float(value):,.2f}"
    except (TypeError, ValueError):
        return "—"

tabs = st.tabs(["Market scanner", "Open paper trade", "Open positions", "Trade history", "Journal"])

with tabs[0]:
    st.subheader("Multi-market scanner")
    with st.sidebar:
        st.header("Scanner settings")
        period = st.selectbox("Historical data", ["6mo", "1y", "2y", "5y"], index=1)
        interval = st.selectbox("Chart interval", ["1d", "1h"], index=0)
        starting_balance = st.number_input(
            "Reference paper balance (£)", min_value=50.0, max_value=100000.0,
            value=500.0, step=50.0
        )
        risk_pct = st.slider("Illustrative risk per trade (%)", 0.1, 1.0, 0.5, 0.1)
        min_score = st.slider("Minimum signal score", 1, 5, 3)
        st.caption("Daily candles are recommended for initial testing. Intraday availability varies.")

    if st.button("Run market scan", type="primary"):
        rows, failures = [], []
        progress = st.progress(0)
        total = sum(len(v) for v in UNIVERSE.values())
        done = 0
        for market, symbols in UNIVERSE.items():
            for symbol in symbols:
                try:
                    raw = load_data(symbol, period, interval)
                    result = score_row(symbol, market, raw, min_score)
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
            st.info("No usable data returned for: " + ", ".join(failures))

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
        st.download_button(
            "Download scan as CSV", results.to_csv(index=False).encode("utf-8"),
            file_name="marketpilot_scan.csv", mime="text/csv"
        )
        st.subheader("Inspect an asset")
        selected = st.selectbox("Choose symbol", results["Symbol"].tolist())
        market = results.loc[results["Symbol"] == selected, "Market"].iloc[0]
        chart = indicators(load_data(selected, period, interval)).dropna()
        if not chart.empty:
            st.line_chart(chart[["Close", "EMA20", "EMA50"]], use_container_width=True)
            st.caption(f"{selected} • {market} • Yahoo Finance via yfinance.")

with tabs[1]:
    st.subheader("Record a simulated trade")
    st.caption("This records a hypothetical position only. It does not place an order.")
    with st.form("new_trade_form"):
        col1, col2 = st.columns(2)
        asset = col1.text_input("Asset / ticker", placeholder="e.g. GBPUSD=X or BTC-USD")
        asset_class = col2.selectbox("Market", ["Forex", "Crypto", "Stocks & ETFs", "Other"])
        direction = col1.selectbox("Direction", ["Long", "Short"])
        entry_price = col2.number_input("Entry price", min_value=0.00000001, value=100.0, format="%.8f")
        quantity = col1.number_input("Quantity / units", min_value=0.00000001, value=1.0, format="%.8f")
        stop_loss = col2.number_input("Stop loss (optional; 0 = none)", min_value=0.0, value=0.0, format="%.8f")
        take_profit = col1.number_input("Take profit (optional; 0 = none)", min_value=0.0, value=0.0, format="%.8f")
        fees = col2.number_input("Estimated fees (£)", min_value=0.0, value=0.0, step=0.01)
        slippage = col1.number_input("Estimated slippage (£)", min_value=0.0, value=0.0, step=0.01)
        notes = st.text_area("Trade notes / reason")
        submitted = st.form_submit_button("Save paper trade", type="primary")
    if submitted:
        if not asset.strip():
            st.error("Enter an asset or ticker.")
        elif (stop_loss > 0 and ((direction == "Long" and stop_loss >= entry_price) or
                                (direction == "Short" and stop_loss <= entry_price))):
            st.error("For a Long trade, stop loss must be below entry. For a Short trade, it must be above entry.")
        elif (take_profit > 0 and ((direction == "Long" and take_profit <= entry_price) or
                                  (direction == "Short" and take_profit >= entry_price))):
            st.error("For a Long trade, take profit must be above entry. For a Short trade, it must be below entry.")
        else:
            row = {
                "asset": asset.strip().upper(), "asset_class": asset_class, "direction": direction,
                "entry_price": float(entry_price), "quantity": float(quantity),
                "stop_loss": float(stop_loss) if stop_loss > 0 else None,
                "take_profit": float(take_profit) if take_profit > 0 else None,
                "fees": float(fees), "slippage": float(slippage), "status": "Open",
                "notes": notes.strip() or None, "opened_at": datetime.now(timezone.utc).isoformat()
            }
            try:
                supabase.table("trades").insert(row).execute()
                st.success("Paper trade saved to Supabase.")
                st.rerun()
            except Exception as exc:
                st.error(db_error_message(exc))

with tabs[2]:
    st.subheader("Open paper positions")
    try:
        open_trades = fetch_trades("Open")
        if not open_trades:
            st.info("No open paper trades yet. Use the Open paper trade tab to record one.")
        else:
            view = pd.DataFrame(open_trades)
            cols = [c for c in ["asset", "asset_class", "direction", "entry_price", "quantity",
                                "stop_loss", "take_profit", "fees", "opened_at", "notes"] if c in view]
            st.dataframe(view[cols], use_container_width=True, hide_index=True)
            options = {f"#{t['id'][:8]} • {t['asset']} • {t['direction']}": t for t in open_trades}
            chosen_label = st.selectbox("Choose a position to close", list(options.keys()))
            chosen = options[chosen_label]
            with st.form("close_trade_form"):
                exit_price = st.number_input("Exit price", min_value=0.00000001,
                                             value=float(chosen["entry_price"]), format="%.8f")
                closing_fees = st.number_input("Additional closing fees (£)", min_value=0.0, value=0.0, step=0.01)
                exit_reason = st.text_area("Exit reason")
                lessons = st.text_area("Lessons learned")
                close_submitted = st.form_submit_button("Close simulated position")
            if close_submitted:
                total_fees = float(chosen.get("fees") or 0) + float(closing_fees)
                update = {
                    "exit_price": float(exit_price), "fees": total_fees,
                    "status": "Closed", "closed_at": datetime.now(timezone.utc).isoformat()
                }
                try:
                    supabase.table("trades").update(update).eq("id", chosen["id"]).execute()
                    journal = {
                        "trade_id": chosen["id"],
                        "entry_reason": chosen.get("notes") or None,
                        "exit_reason": exit_reason.strip() or None,
                        "lessons_learned": lessons.strip() or None
                    }
                    supabase.table("trade_journal").insert(journal).execute()
                    st.success("Position closed and journal entry saved.")
                    st.rerun()
                except Exception as exc:
                    st.error(db_error_message(exc))
    except Exception as exc:
        st.error(db_error_message(exc))

with tabs[3]:
    st.subheader("Trade history and performance")
    try:
        all_trades = fetch_trades()
        if not all_trades:
            st.info("Your paper-trading history will appear here.")
        else:
            history = pd.DataFrame(all_trades)
            if "exit_price" in history.columns:
                def calc_pnl(row):
                    if row.get("status") != "Closed" or pd.isna(row.get("exit_price")):
                        return np.nan
                    delta = float(row["exit_price"]) - float(row["entry_price"])
                    if row.get("direction") == "Short":
                        delta = -delta
                    return delta * float(row["quantity"]) - float(row.get("fees") or 0) - float(row.get("slippage") or 0)
                history["Realised P/L (£, indicative)"] = history.apply(calc_pnl, axis=1)
            closed = history[history["status"] == "Closed"] if "status" in history else pd.DataFrame()
            if not closed.empty and "Realised P/L (£, indicative)" in history:
                realised = closed["Realised P/L (£, indicative)"].fillna(0)
                wins = int((realised > 0).sum())
                a, b, c = st.columns(3)
                a.metric("Closed trades", len(closed))
                b.metric("Indicative realised P/L", fmt_money(realised.sum()))
                c.metric("Winning trades", f"{wins}/{len(closed)}")
            st.dataframe(history, use_container_width=True, hide_index=True)
            st.download_button(
                "Download trade history CSV", history.to_csv(index=False).encode("utf-8"),
                file_name="marketpilot_trade_history.csv", mime="text/csv"
            )
            st.caption("Indicative P/L excludes currency conversion, contract multipliers, taxes and other market-specific costs.")
    except Exception as exc:
        st.error(db_error_message(exc))

with tabs[4]:
    st.subheader("Trade journal")
    try:
        journal_rows = fetch_journal()
        if not journal_rows:
            st.info("Journal notes will appear here when you close a simulated position.")
        else:
            journal = pd.DataFrame(journal_rows)
            st.dataframe(journal, use_container_width=True, hide_index=True)
            st.download_button(
                "Download journal CSV", journal.to_csv(index=False).encode("utf-8"),
                file_name="marketpilot_journal.csv", mime="text/csv"
            )
    except Exception as exc:
        st.error(db_error_message(exc))

st.divider()
st.caption(
    "Market data is supplied by Yahoo Finance via yfinance and may be delayed, incomplete or unavailable. "
    "This prototype is for personal research and simulated practice only; it is not financial advice."
)

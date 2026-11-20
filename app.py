"""MarketPilot v0.4 dashboard (Streamlit).

The bots run in the scheduled worker (GitHub Actions); this app only reads the
results and lets you flip the kill switch or trigger a cycle manually.
"""
import hmac
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="MarketPilot v0.4", page_icon="📈", layout="wide")

BLUE, ORANGE, GOOD, CRITICAL, MUTED, GRID = "#2a78d6", "#eb6834", "#0ca30c", "#d03b3b", "#898781", "#e1e0d9"


def _secret(name):
    try:
        return st.secrets.get(name)
    except Exception:
        return None


def authenticate():
    password = _secret("APP_PASSWORD") or ""
    if not password:
        st.warning("APP_PASSWORD is not set in Streamlit Secrets — keep this app private.")
        return
    if st.session_state.get("authenticated"):
        return
    st.title("🔒 MarketPilot sign in")
    with st.form("login"):
        attempt = st.text_input("App password", type="password")
        ok = st.form_submit_button("Unlock")
    if ok:
        if hmac.compare_digest(attempt, str(password)):
            st.session_state["authenticated"] = True
            st.rerun()
        st.error("Incorrect password")
    st.stop()


@st.cache_resource
def get_store():
    from marketpilot.storage import make_store
    url = _secret("SUPABASE_URL") or os.environ.get("SUPABASE_URL")
    key = _secret("SUPABASE_SERVICE_KEY") or _secret("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_KEY")
    if url and key:
        os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"] = url, key
        return make_store("supabase"), "Supabase"
    return make_store(os.environ.get("MARKETPILOT_DB")), "local SQLite (demo)"


def style(fig, height=320, y_title=None, pct=False, legend=False):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)", hovermode="closest", showlegend=legend,
                      legend=dict(orientation="h", y=1.12, x=0))
    fig.update_xaxes(showgrid=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=MUTED, title=y_title, tickformat=".0%" if pct else None)
    return fig


authenticate()
from marketpilot.bots.performance import report  # noqa: E402
from marketpilot.config import load_settings  # noqa: E402

settings = load_settings()
store, backend = get_store()
start_bal = settings.account["starting_balance"]
target_wr = settings["goals"]["target_win_rate"]

st.title("📈 MarketPilot v0.4 — multi-agent paper trading")
kill = store.get_control("kill_switch", "off") == "on"
c1, c2 = st.columns([3, 1])
c1.caption(f"Storage: {backend} • Paper trading only — no live orders, no broker connection. Not financial advice.")
if kill:
    c2.error("⛔ Kill switch ON")
else:
    c2.success("✅ Trading active (paper)")

trades = store.trades()
rep = report(trades, start_bal)
o = rep["overall"]
closed = rep["closed"]

tabs = st.tabs(["Overview", "Bot decisions", "Strategies", "Positions", "Trade log", "Bot activity", "Controls"])

# ---------------- Overview ----------------
with tabs[0]:
    m = st.columns(7)
    m[0].metric("Closed trades", o["trades"], help="Wins / losses")
    m[1].metric("Win rate", f"{o['win_rate']:.1%}", f"{(o['win_rate'] - target_wr) * 100:+.1f} pts vs {target_wr:.0%} goal",
                delta_color="normal")
    m[2].metric("Win : loss", f"{o['wins']} : {o['losses']}", help=f"Win/loss ratio {o['win_loss_ratio']:.2f}")
    m[3].metric("Expectancy", f"{o['expectancy_r']:+.3f}R", help="Average result per trade in R (1R = £ risked)")
    m[4].metric("Profit factor", f"{o['profit_factor']:.2f}" if o["trades"] else "—")
    m[5].metric("P&L", f"£{o['pnl_gbp']:+,.2f}", f"{o['return_pct']:+.2f}%")
    m[6].metric("Max drawdown", f"{o['max_drawdown_r']:.1f}R")

    if o["trades"]:
        st.caption(f"Win rate 95% range with {o['trades']} trades: {o['win_rate_lo']:.0%} – {o['win_rate_hi']:.0%}. "
                   "Narrow this with more trades before trusting it.")
    with st.expander("Why win rate isn't the whole story", expanded=o["trades"] == 0):
        st.markdown(
            "Any system can hit a 70–80% win rate by using a tiny take-profit and a wide stop — and still lose money, "
            "because the rare losses are big. MarketPilot **tracks win rate against your goal** but only approves "
            "strategies with positive **expectancy after costs** on unseen data. "
            "Expectancy = win rate × average win − loss rate × average loss (in R, where 1R = the £ risked).")

    eq = pd.DataFrame(store.equity_curve())
    if not eq.empty:
        eq["ts"] = pd.to_datetime(eq["ts"], utc=True, format="mixed")
        fig = go.Figure(go.Scatter(x=eq["ts"], y=eq["equity_gbp"], name="Equity (£)",
                                   line=dict(color=BLUE, width=2), hovertemplate="£%{y:,.2f}<extra></extra>"))
        fig.add_hline(y=start_bal, line=dict(color=MUTED, dash="dot", width=1),
                      annotation_text=f"start £{start_bal:,.0f}", annotation_font_color=MUTED)
        st.subheader("Paper account equity")
        st.plotly_chart(style(fig, y_title="£"), width="stretch")

    if not closed.empty:
        cl = closed.reset_index(drop=True)
        cl["n"] = cl.index + 1
        w = min(20, len(cl))
        cl["rolling_wr"] = (cl["r_multiple"] > 0).rolling(w, min_periods=1).mean()
        cl["cum_wr"] = (cl["r_multiple"] > 0).expanding().mean()
        a, b = st.columns(2)
        with a:
            st.subheader("Win rate vs goal")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cl["n"], y=cl["cum_wr"], name="All trades", line=dict(color=BLUE, width=2)))
            fig.add_trace(go.Scatter(x=cl["n"], y=cl["rolling_wr"], name=f"Last {w} trades",
                                     line=dict(color=ORANGE, width=2)))
            fig.add_hline(y=target_wr, line=dict(color=MUTED, dash="dash", width=1),
                          annotation_text=f"goal {target_wr:.0%}", annotation_font_color=MUTED)
            fig.update_xaxes(title="trade #")
            st.plotly_chart(style(fig, pct=True, legend=True), width="stretch")
        with b:
            st.subheader("Result per trade (R)")
            colors = [GOOD if r > 0 else CRITICAL for r in cl["r_multiple"]]
            fig = go.Figure(go.Bar(x=cl["n"], y=cl["r_multiple"], marker_color=colors, name="R per trade",
                                   customdata=cl[["symbol", "strategy", "exit_reason"]],
                                   hovertemplate="%{customdata[0]} %{y:+.2f}R<br>%{customdata[1]}<br>%{customdata[2]}<extra></extra>"))
            fig.update_xaxes(title="trade #")
            st.plotly_chart(style(fig, y_title="R"), width="stretch")

        st.subheader("Breakdown")
        dim = st.radio("By", ["strategy", "asset_class", "symbol", "regime"], horizontal=True)
        bd = rep["by"][dim]
        if not bd.empty:
            bd = bd.sort_values("expectancy_r")
            show = bd[[dim, "trades", "win_rate", "expectancy_r", "profit_factor", "total_r", "pnl_gbp", "max_drawdown_r"]]
            st.dataframe(show.style.format({"win_rate": "{:.0%}", "expectancy_r": "{:+.3f}", "profit_factor": "{:.2f}",
                                            "total_r": "{:+.2f}", "pnl_gbp": "£{:+.2f}", "max_drawdown_r": "{:.2f}"}),
                         hide_index=True, width="stretch")
    else:
        st.info("No closed paper trades yet. The worker trades only strategies that passed backtesting — "
                "check the Strategies tab, and give it time: daily-bar strategies may trade a few times a month.")

# ---------------- Bot decisions ----------------
with tabs[1]:
    st.subheader("Every candidate trade and why it was taken or rejected")
    cand = pd.DataFrame(store.candidates(1000))
    if cand.empty:
        st.info("No candidates yet.")
    else:
        f = st.selectbox("Show", ["all", "trade", "reject"])
        if f != "all":
            cand = cand[cand["decision"] == f]
        cand["why"] = cand["reasons"].apply(lambda r: "; ".join(r) if isinstance(r, list) and r else "all checks passed")
        cand["side"] = cand["direction"].map({1: "LONG", -1: "SHORT"})
        st.dataframe(cand[["bar_time", "symbol", "side", "strategy", "regime", "decision", "expected_r", "why"]],
                     hide_index=True, width="stretch",
                     column_config={"expected_r": st.column_config.NumberColumn("EV (R)", format="%+.3f")})

# ---------------- Strategies ----------------
with tabs[2]:
    st.subheader("Strategy research & backtesting results")
    st.caption("Tested walk-forward on unseen data with fees, spreads and slippage, pooled per market, then checked "
               "once on an untouched holdout. Only APPROVED strategies trade.")
    strat = pd.DataFrame(store.strategies())
    if strat.empty:
        st.info("No research yet — run `python -m marketpilot research` or the GitHub Action.")
    else:
        g = strat.drop_duplicates(["asset_class", "strategy"]).copy()
        g["OOS trades"] = g["oos"].apply(lambda x: x["trades"])
        g["OOS win rate"] = g["oos"].apply(lambda x: x["win_rate"])
        g["OOS expectancy R"] = g["oos"].apply(lambda x: x["expectancy_r"])
        g["OOS profit factor"] = g["oos"].apply(lambda x: x["profit_factor"])
        g["Holdout expectancy R"] = g["holdout"].apply(lambda x: x["expectancy_r"])
        g["Why"] = g["reasons"].apply(lambda r: "; ".join(r) if r else "passed all gates")
        st.dataframe(g[["asset_class", "strategy", "status", "OOS trades", "OOS win rate", "OOS expectancy R",
                        "OOS profit factor", "Holdout expectancy R", "Why"]].style.format(
            {"OOS win rate": "{:.0%}", "OOS expectancy R": "{:+.3f}", "OOS profit factor": "{:.2f}",
             "Holdout expectancy R": "{:+.3f}"}), hide_index=True, width="stretch")
        fig = go.Figure(go.Scatter(
            x=g["OOS win rate"], y=g["OOS expectancy R"], mode="markers+text", text=g["strategy"] + " · " + g["asset_class"],
            textposition="top center", textfont=dict(size=10, color=MUTED),
            marker=dict(size=11, color=[BLUE if s == "approved" else MUTED for s in g["status"]],
                        line=dict(color="#fcfcfb", width=2)),
            hovertemplate="%{text}<br>win %{x:.0%} · %{y:+.3f}R<extra></extra>", name="strategy"))
        fig.add_hline(y=0, line=dict(color=MUTED, width=1))
        fig.add_vline(x=target_wr, line=dict(color=MUTED, dash="dash", width=1), annotation_text="win-rate goal")
        fig.update_xaxes(title="out-of-sample win rate", tickformat=".0%")
        st.markdown("**Win rate vs. expectancy** — blue = approved, grey = rejected. High win rate ≠ profitable.")
        st.plotly_chart(style(fig, 380, "expectancy (R per trade)"), width="stretch")

# ---------------- Positions ----------------
with tabs[3]:
    st.subheader("Open & pending paper positions")
    op = pd.DataFrame([t for t in trades if t["status"] in ("open", "pending")])
    if op.empty:
        st.info("No open positions.")
    else:
        op["side"] = op["direction"].map({1: "LONG", -1: "SHORT"})
        st.dataframe(op[["symbol", "side", "status", "strategy", "opened_at", "entry_price", "stop", "target",
                         "quantity", "risk_gbp", "notional_gbp", "bars_held"]], hide_index=True, width="stretch")

# ---------------- Trade log ----------------
with tabs[4]:
    st.subheader("Closed paper trades")
    if closed.empty:
        st.info("No closed trades yet.")
    else:
        log = closed.sort_values("closed_at", ascending=False).copy()
        log["result"] = log["r_multiple"].apply(lambda r: "WIN" if r > 0 else "LOSS")
        log["side"] = log["direction"].map({1: "LONG", -1: "SHORT"})
        cols = ["closed_at", "symbol", "side", "strategy", "result", "r_multiple", "pnl_gbp", "fees_gbp",
                "entry_price", "exit_price", "exit_reason", "regime"]
        st.dataframe(log[cols], hide_index=True, width="stretch",
                     column_config={"r_multiple": st.column_config.NumberColumn("R", format="%+.2f"),
                                    "pnl_gbp": st.column_config.NumberColumn("P&L £", format="%+.2f")})
        st.download_button("Download trade log CSV", log[cols].to_csv(index=False), "marketpilot_trades.csv")

# ---------------- Bot activity ----------------
with tabs[5]:
    ev = pd.DataFrame(store.events(500))
    if ev.empty:
        st.info("No activity yet.")
    else:
        st.dataframe(ev[["ts", "level", "bot", "message"]], hide_index=True, width="stretch")

# ---------------- Controls ----------------
with tabs[6]:
    from marketpilot.bots.safety import SafetyBot
    safety = SafetyBot(store, settings)
    st.subheader("Kill switch")
    st.caption("ON = no new trades. Open positions keep their stops and targets managed.")
    if kill:
        st.write(f"Reason: {store.get_control('pause_reason', '') or 'manual'}")
        if st.button("Release kill switch", type="primary"):
            safety.release()
            st.rerun()
    elif st.button("⛔ Engage kill switch", type="primary"):
        safety.engage("manual kill switch from dashboard")
        st.rerun()
    st.divider()
    st.subheader("Run the bots now")
    st.caption("Normally the GitHub Actions worker does this every hour. Use this to test.")
    if st.button("Run one trading cycle"):
        from datetime import datetime, timezone
        from marketpilot.data import make_provider
        from marketpilot.engine import Engine
        with st.spinner("Scanning markets…"):
            s = Engine(settings, make_provider("yfinance"), store).cycle(datetime.now(timezone.utc))
        st.success(f"Cycle done: {s}")
    st.divider()
    st.subheader("Risk settings (from config/settings.yaml)")
    a = settings.account
    st.write(f"Balance £{a['starting_balance']:,.0f} • risk {a['risk_per_trade_pct']}% per trade "
             f"(£{a['starting_balance'] * a['risk_per_trade_pct'] / 100:.2f}) • daily loss limit {a['daily_loss_limit_pct']}% • "
             f"max {a['max_open_positions']} positions • leverage {a['max_leverage']}x")

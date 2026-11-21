"""MarketPilot v0.5 dashboard (Streamlit).

The bots run in the scheduled worker (GitHub Actions); this app only reads the
results and lets you flip the kill switch or trigger a cycle manually.
"""
import hmac
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="MarketPilot v0.8", page_icon="📈", layout="wide")

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

st.title("📈 MarketPilot v0.8 — multi-agent paper trading")
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

tabs = st.tabs(["Overview", "Bot decisions", "Strategies", "Positions", "Trade log", "Bot activity", "Controls",
                "Learning lab", "Learn", "Markets"])

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
        g["OOS t-stat"] = g["oos"].apply(lambda x: x.get("t_stat", float("nan")))
        g["Holdout expectancy R"] = g["holdout"].apply(lambda x: x["expectancy_r"])
        g["Why"] = g["reasons"].apply(lambda r: "; ".join(r) if r else "passed all gates")
        st.dataframe(g[["asset_class", "strategy", "status", "OOS trades", "OOS win rate", "OOS expectancy R",
                        "OOS profit factor", "OOS t-stat", "Holdout expectancy R", "Why"]].style.format(
            {"OOS win rate": "{:.0%}", "OOS expectancy R": "{:+.3f}", "OOS profit factor": "{:.2f}",
             "OOS t-stat": "{:.1f}", "Holdout expectancy R": "{:+.3f}"}), hide_index=True, width="stretch")
        st.caption("t-stat = how many standard errors the average result is above zero. Below 2 means the "
                   "result could easily be luck, so the strategy is not traded.")
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
                "entry_price", "exit_price", "exit_reason", "regime"] + (["note"] if "note" in log else [])
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
    st.subheader("Phone notifications")
    st.caption("Trade alerts and a daily summary are sent by the GitHub worker through the free ntfy app. "
               "To test: GitHub → Actions → MarketPilot worker → Run workflow → command `notify-test`.")
    st.divider()
    st.subheader("Risk settings (from config/settings.yaml)")
    a = settings.account
    st.write(f"Balance £{a['starting_balance']:,.0f} • risk {a['risk_per_trade_pct']}% per trade "
             f"(£{a['starting_balance'] * a['risk_per_trade_pct'] / 100:.2f}) • daily loss limit {a['daily_loss_limit_pct']}% • "
             f"max {a['max_open_positions']} positions • leverage {a['max_leverage']}x")


# ---------------- Learning lab ----------------
with tabs[7]:
    from marketpilot.bots.learning import forward_stats
    st.subheader("Learning lab — shadow bots practising every strategy on every market")
    st.caption("Shadow trades follow exactly the same entry and exit rules as the real paper account but are "
               "practice only: no position limits, no capital cap, no phone alerts, results in R. "
               f"Strategies with {settings['learning']['promote_min_trades']}+ strong shadow trades "
               f"(expectancy ≥ {settings['learning']['promote_min_expectancy_r']}R, t-stat ≥ "
               f"{settings['learning']['promote_min_tstat']}) are promoted to the real paper account; approved "
               f"ones that go negative over {settings['learning']['demote_min_trades']}+ are demoted.")
    try:
        shadow_all = pd.DataFrame(store.shadow_trades())
    except Exception:
        shadow_all = pd.DataFrame()
    if shadow_all.empty:
        st.info("No shadow trades yet. The shadow bots open practice trades whenever any strategy signals on "
                "any of the markets — expect the first ones within a day or two.")
    else:
        n_open = int(shadow_all["status"].isin(["open", "pending"]).sum())
        n_closed = int((shadow_all["status"] == "closed").sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Shadow trades open", n_open)
        c2.metric("Shadow trades closed", n_closed)
        fs = forward_stats(store)
        if not fs.empty:
            closed_r = shadow_all.loc[shadow_all["status"] == "closed", "r_multiple"].astype(float)
            c3.metric("Shadow win rate (all)", f"{(closed_r > 0).mean():.0%}")
            st.markdown("**Leaderboard** — strongest forward evidence first (ranked by t-stat, not by luckiest result)")
            fs = fs.sort_values(["t_stat", "trades"], ascending=False)  # strongest evidence first
            st.dataframe(fs[["asset_class", "strategy", "trades", "win_rate", "expectancy_r", "profit_factor",
                             "t_stat", "max_drawdown_r"]].style.format(
                {"win_rate": "{:.0%}", "expectancy_r": "{:+.3f}", "profit_factor": "{:.2f}", "t_stat": "{:.1f}",
                 "max_drawdown_r": "{:.1f}"}), hide_index=True, width="stretch")
            fig = go.Figure(go.Scatter(
                x=fs["win_rate"], y=fs["expectancy_r"], mode="markers",
                marker=dict(size=[max(8, min(30, n / 3)) for n in fs["trades"]], color=BLUE,
                            line=dict(color="#fcfcfb", width=2)),
                text=fs["strategy"] + " · " + fs["asset_class"] + " (" + fs["trades"].astype(str) + " trades)",
                hovertemplate="%{text}<br>win %{x:.0%} · %{y:+.3f}R<extra></extra>"))
            fig.add_hline(y=0, line=dict(color=MUTED, width=1))
            fig.add_vline(x=target_wr, line=dict(color=MUTED, dash="dash", width=1), annotation_text="win-rate goal")
            fig.update_xaxes(title="shadow win rate", tickformat=".0%")
            st.plotly_chart(style(fig, 360, "expectancy (R per trade)"), width="stretch")
            st.caption("Bubble size = number of shadow trades. Top-right is the goal: high win rate AND positive "
                       "expectancy. Small bubbles are not yet evidence.")
        from marketpilot.bots.learning import confluence_stats
        cst = confluence_stats(store)
        st.markdown("**Does confluence work?** Shadow results grouped by confluence score (0-8 factors agreeing)")
        if cst.empty:
            st.caption("Not enough closed shadow trades with a confluence score yet.")
        else:
            st.dataframe(cst[["confluence", "trades", "win_rate", "expectancy_r", "t_stat"]].style.format(
                {"win_rate": "{:.0%}", "expectancy_r": "{:+.3f}", "t_stat": "{:.1f}"}), hide_index=True,
                width="stretch")
            st.caption("If high-confluence trades clearly win more (with enough trades), raise "
                       "`confluence.min_score` in settings so the real paper account only takes them.")
        with st.expander("Open shadow positions"):
            op = shadow_all[shadow_all["status"].isin(["open", "pending"])]
            st.dataframe(op[["symbol", "asset_class", "strategy", "status", "opened_at", "entry_price", "stop"]],
                         hide_index=True, width="stretch")

# ---------------- Learn ----------------
with tabs[8]:
    st.subheader("Learn: the strategies and technical indicators MarketPilot uses")
    st.markdown("### Key ideas")
    st.markdown("""
- **R (risk unit)** — every trade risks a fixed amount, here 0.5% of the account (£2.50 on £500). A win of
  +2R made £5; a loss of −1R lost £2.50. Measuring in R makes different markets comparable.
- **Win rate vs expectancy** — win rate is how often you win; **expectancy** is the average result per trade:
  *win rate × average win − loss rate × average loss*. A 40% win rate with 2R wins and 1R losses makes
  +0.2R per trade; an 80% win rate with 0.25R wins and 1.5R losses loses money.
- **Stop-loss** — the price where a losing trade is closed. MarketPilot sets it a multiple of ATR away so it
  adapts to how volatile a market is. A **trailing stop** moves up behind the best price to lock in profit.
- **Walk-forward testing** — choose a strategy's settings on older data, then test on later data it has never
  seen. A final untouched **holdout** period is checked once at the end.
- **t-statistic** — how many standard errors the average result is above zero. Below ~2, the result could
  easily be luck.
- **Survivorship bias** — testing only on today's winners (e.g. NVDA) makes any strategy look better than it
  would have been. That's why MarketPilot uses broad ETFs.
- **Costs** — spreads, slippage and fees are charged on every simulated trade. Many strategies that look good
  before costs lose money after them.""")
    st.markdown("### How professional traders plan a trade")
    st.markdown("""
- **Build a case, not a hunch.** Chart Champions' approach is to stack *layers of confluence* — trend,
  key support/resistance, Fibonacci zones, market structure — and use indicators only as the last
  confirmation. MarketPilot's **Confluence bot** scores every signal on 8 such factors (shown in each
  trade's note) and the Learning lab measures whether higher scores really win more often.
- **Bull and bear scenario.** Write down what would make you right *and* what would prove you wrong before
  entering. The stop goes where the idea is proven wrong — not at a random amount.
- **Structure stops.** Discretionary traders put stops just beyond the last swing low/high, with a small
  buffer. MarketPilot now tests this ("swing" variants: 0.5 ATR beyond the 10-bar swing, kept between 1 and
  4 ATR) against plain ATR stops and lets the data choose.
- **Risk first.** Peter Brandt, a 40-year professional, says chart analysis matters less than risk
  management: small fixed risk per trade, take partial profits, tighten stops as a trade works, and get out
  of losers quickly. MarketPilot risks 0.5% per trade with a 2% daily loss limit.
- **Capitulation buys.** Crypto Crew University's "3 checkmarks" buys Bitcoin when price is below its
  200 MA, RSI(14) is under 26 and both Stochastic RSI lines are under 20 — MarketPilot tests it on every
  market as the `ccu_3checkmarks` strategy.
- **Reality check.** A study of Brazilian day traders found only about 1% earned more than the minimum
  wage, and found no evidence they improved with practice. Slow, rules-based, tested strategies with
  small risk are how MarketPilot tries to avoid that fate.""")
    st.markdown("### Technical indicators")
    INDICATORS = {
        "Moving average (SMA / EMA)": "The average price over the last N bars. SMA weights all bars equally; EMA "
            "weights recent bars more. Price above its 200-day average is the classic sign of a long-term uptrend.",
        "RSI (Relative Strength Index)": "0–100 scale of recent gains vs losses. RSI(2) below 10 means two sharp "
            "down days — very oversold. RSI(14) is the common slower version.",
        "ATR (Average True Range)": "The average size of a bar's move, including gaps. Used to set stops and size "
            "positions so a volatile market gets a wider stop and a smaller position.",
        "Bollinger Bands": "A 20-day average with bands 2 standard deviations above and below. A close outside "
            "the bands is unusually stretched.",
        "MACD": "The 12-day EMA minus the 26-day EMA, plus a 9-day 'signal' average of that. MACD crossing above "
            "its signal line means short-term momentum has turned up.",
        "Donchian channel": "The highest high and lowest low of the last N bars (20 or 55). Breaking out of it "
            "is the basis of the famous Turtle trading rules.",
        "52-week high": "The highest price of the past year. Research (George & Hwang 2004) found stocks near "
            "52-week highs tend to keep outperforming.",
        "Stochastic RSI": "RSI of the RSI, scaled 0–100 and smoothed. Both lines under 20 = deeply oversold "
            "momentum; one of the '3 checkmarks'.",
        "Fibonacci retracement": "After a swing, pullbacks often pause around 38.2%–61.8% of the move "
            "(the 'golden pocket' near 61.8%). The Confluence bot checks if price is in that zone.",
        "Market structure / swing points": "Swing highs and lows mark where buyers or sellers took control. A "
            "break of the last swing changes the structure; stops are often placed just beyond it.",
        "Efficiency ratio / regime": "How straight the path of price has been (1 = straight line, 0 = noise). "
            "MarketPilot labels each market trending, ranging or volatile from this and ATR.",
    }
    for k, v in INDICATORS.items():
        with st.expander(k):
            st.write(v)
    st.markdown("### Strategies")
    from marketpilot.strategies import ALL_STRATEGIES
    for cls in ALL_STRATEGIES:
        with st.expander(f"{cls.name}  ·  {cls.family}"):
            st.write((cls.__doc__ or "").strip())
            st.caption("Settings tested by research: " + ", ".join(f"{k} ∈ {v}" for k, v in cls.grid.items()))
    st.info("Educational information about how the bots work — not financial advice.")


# ---------------- Markets ----------------
with tabs[9]:
    import json
    st.subheader("Markets — all 66 markets the bots watch")
    fg = store.get_control("fear_greed", "")
    if fg:
        try:
            fgd = json.loads(fg)
            c1, c2 = st.columns([1, 3])
            c1.metric("Crypto Fear & Greed", f"{fgd['value']} / 100", fgd["label"], delta_color="off")
            c2.caption("0 = extreme fear, 100 = extreme greed. Recorded daily as learning data, not a trade "
                       "signal. Data: Alternative.me Crypto Fear & Greed Index (alternative.me).")
        except Exception:
            pass
    mk = pd.DataFrame(store.market())
    if mk.empty:
        st.info("The worker fills this in on its next hourly run.")
    else:
        groups = ["all"] + sorted(mk["asset_class"].unique())
        g = st.radio("Group", groups, horizontal=True, key="mkt_group")
        view = mk if g == "all" else mk[mk["asset_class"] == g]
        view = view.assign(trend=view["above_200d"].map({1: "up", 0: "down"}).fillna("–"))
        view = view.sort_values("conf_long", ascending=False)
        st.dataframe(
            view[["symbol", "asset_class", "price", "chg_1d", "chg_5d", "chg_20d", "trend", "rsi14", "regime",
                  "atr_pct", "conf_long", "conf_short", "bar_time"]].style.format(
                {"price": "{:,.4f}", "chg_1d": "{:+.2%}", "chg_5d": "{:+.2%}", "chg_20d": "{:+.2%}",
                 "rsi14": "{:.0f}", "atr_pct": "{:.2%}"}, na_rep="–"),
            hide_index=True, width="stretch",
            column_config={"conf_long": st.column_config.ProgressColumn("Long confluence", min_value=0,
                                                                         max_value=8, format="%d/8"),
                           "conf_short": st.column_config.ProgressColumn("Short confluence", min_value=0,
                                                                          max_value=8, format="%d/8")})
        st.caption("Confluence = how many of 8 independent factors currently agree with a long or short. "
                   "It describes the chart; trades still need an approved strategy's signal.")
        chg = view.dropna(subset=["chg_20d"]).sort_values("chg_20d")
        fig = go.Figure(go.Bar(x=chg["chg_20d"], y=chg["symbol"], orientation="h",
                               marker_color=[GOOD if v > 0 else CRITICAL for v in chg["chg_20d"]],
                               hovertemplate="%{y}: %{x:+.1%}<extra></extra>"))
        fig.update_xaxes(tickformat=".0%", title="20-day change")
        st.plotly_chart(style(fig, max(300, 18 * len(chg))), width="stretch")

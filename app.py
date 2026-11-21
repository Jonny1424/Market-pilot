"""MarketPilot v0.9 dashboard (Streamlit).

The bots run in the scheduled worker (GitHub Actions); this app reads the
results, plans trades with the Time Machine and lets you flip the kill switch.
"""
import hmac
import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

VERSION = "v0.12"
st.set_page_config(page_title=f"MarketPilot {VERSION}", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")

# ---------------- design tokens ----------------
BG, PANEL, PANEL2, BORDER = "#0b0e14", "#131722", "#1a1f2b", "#232937"
TEXT, MUTED, UP, DOWN = "#e6e8ee", "#8a93a6", "#16c784", "#ea3943"
ACCENT, WARN, VIOLET = "#4f7cff", "#f0a020", "#a78bfa"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, select, textarea {{
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }}
.block-container {{ padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1480px; }}
[data-testid="stSidebar"] {{ background: {PANEL}; border-right: 1px solid {BORDER}; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{ padding: 7px 10px; border-radius: 8px; margin: 1px 0;
  width: 100%; }}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background: {PANEL2}; }}
[data-testid="stSidebar"] [role="radiogroup"] label[data-selected="true"] {{ background: rgba(79,124,255,.16);
  box-shadow: inset 3px 0 0 {ACCENT}; }}
[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child:not([data-testid]) {{ display: none; }}
[data-testid="stSidebar"] [data-testid="stRadioOption"] p {{ font-size: .92rem; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
h3 {{ font-size: 1.02rem !important; font-weight: 600 !important; color: {TEXT}; margin-top: .4rem; }}
.mp-top {{ display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap;
  padding: 2px 0 14px 0; border-bottom: 1px solid {BORDER}; margin-bottom: 18px; }}
.mp-title {{ font-size: 1.45rem; font-weight: 700; color:{TEXT}; }}
.mp-sub {{ color:{MUTED}; font-size:.82rem; margin-top:2px; }}
.mp-pill {{ display:inline-flex; align-items:center; gap:7px; padding:5px 12px; border-radius:999px;
  font-size:.78rem; font-weight:600; border:1px solid {BORDER}; background:{PANEL}; }}
.mp-dot {{ width:8px; height:8px; border-radius:50%; display:inline-block; }}
.mp-grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap:12px; margin: 4px 0 18px 0; }}
.mp-card {{ background:{PANEL}; border:1px solid {BORDER}; border-radius:12px; padding:14px 16px; }}
.mp-label {{ color:{MUTED}; font-size:.74rem; font-weight:500; text-transform:uppercase; letter-spacing:.04em; }}
.mp-value {{ color:{TEXT}; font-size:1.42rem; font-weight:700; margin-top:4px;
  font-family:'JetBrains Mono', ui-monospace, monospace; }}
.mp-delta {{ font-size:.78rem; margin-top:3px; font-weight:500; }}
.up {{ color:{UP}; }} .down {{ color:{DOWN}; }} .muted {{ color:{MUTED}; }} .warn {{ color:{WARN}; }}
.mp-panel {{ background:{PANEL}; border:1px solid {BORDER}; border-radius:12px; padding:14px 16px; margin-bottom:12px; }}
.mp-news a {{ color:{TEXT}; text-decoration:none; font-weight:500; }}
.mp-news a:hover {{ color:{ACCENT}; }}
.mp-news .item {{ padding:9px 0; border-bottom:1px solid {BORDER}; }}
.mp-news .item:last-child {{ border-bottom:none; }}
.mp-tag {{ display:inline-block; font-size:.68rem; font-weight:600; padding:2px 7px; border-radius:6px;
  background:{PANEL2}; color:{MUTED}; margin-right:6px; border:1px solid {BORDER}; }}
.mp-odds {{ display:flex; height:14px; border-radius:7px; overflow:hidden; margin:8px 0 4px 0; }}
.mp-note {{ color:{MUTED}; font-size:.8rem; }}
[data-testid="stMetric"] {{ background:{PANEL}; border:1px solid {BORDER}; border-radius:12px; padding:12px 14px; }}
[data-testid="stDataFrame"] {{ border:1px solid {BORDER}; border-radius:10px; }}
div[data-testid="stExpander"] details {{ background:{PANEL}; border:1px solid {BORDER}; border-radius:10px; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
.stTabs [data-baseweb="tab"] {{ background:{PANEL}; border-radius:8px 8px 0 0; padding: 6px 14px; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


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
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        st.markdown(f"<div style='height:12vh'></div><div class='mp-title'>📈 MarketPilot</div>"
                    f"<div class='mp-sub'>Multi-agent paper trading · {VERSION}</div><br>", unsafe_allow_html=True)
        with st.form("login"):
            attempt = st.text_input("App password", type="password")
            ok = st.form_submit_button("Sign in", type="primary", width="stretch")
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


@st.cache_data(ttl=900, show_spinner=False)
def price_history(symbol: str, days: int = 1200) -> pd.DataFrame:
    """Daily bars for charts and the trade planner (cached 15 min)."""
    from marketpilot.data import make_provider
    now = datetime.now(timezone.utc)
    try:
        df = make_provider(os.environ.get("MARKETPILOT_PROVIDER", "yfinance")).history(
            symbol, "1d", now - timedelta(days=days), now)
    except Exception:
        df = pd.DataFrame()
    if df is None or df.empty:
        from marketpilot.data import SyntheticProvider
        df = SyntheticProvider().history(symbol, "1d", now - timedelta(days=days), now)
        df.attrs["synthetic"] = True
    return df


def style(fig, height=320, y_title=None, pct=False, legend=False):
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=24, b=8), plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified" if legend else "closest",
                      showlegend=legend, font=dict(family="Inter, sans-serif", color=MUTED, size=12),
                      legend=dict(orientation="h", y=1.1, x=0, font=dict(color=TEXT)),
                      hoverlabel=dict(bgcolor=PANEL2, bordercolor=BORDER, font=dict(color=TEXT)))
    fig.update_xaxes(showgrid=False, linecolor=BORDER, tickcolor=BORDER, zeroline=False)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER, title=y_title, tickformat=".0%" if pct else None,
                     side="right")
    return fig


def cards(items):
    """items: list of (label, value, delta_text, delta_class)."""
    html = "<div class='mp-grid'>"
    for label, value, delta, cls in items:
        html += (f"<div class='mp-card'><div class='mp-label'>{label}</div><div class='mp-value'>{value}</div>"
                 f"<div class='mp-delta {cls}'>{delta}</div></div>")
    st.markdown(html + "</div>", unsafe_allow_html=True)


def odds_bar(p_target, p_stop, p_neither):
    st.markdown(
        f"<div class='mp-odds'><div style='width:{p_target*100:.1f}%;background:{UP}'></div>"
        f"<div style='width:{p_neither*100:.1f}%;background:{BORDER}'></div>"
        f"<div style='width:{p_stop*100:.1f}%;background:{DOWN}'></div></div>"
        f"<div class='mp-note'><span class='up'>■</span> target first {p_target:.0%} &nbsp; "
        f"<span class='muted'>■</span> neither {p_neither:.0%} &nbsp; <span class='down'>■</span> stop first "
        f"{p_stop:.0%}</div>", unsafe_allow_html=True)


def tv_chart(df, symbol, levels=None, trades=None, height=540):
    """TradingView-style chart: drag the price axis to zoom prices, the date axis to zoom time."""
    from marketpilot.tvchart import chart_html
    components.html(chart_html(df, symbol, levels, trades, height), height=height + 18)


authenticate()
from marketpilot.bots import conviction as cv  # noqa: E402
from marketpilot.bots import timemachine as tm  # noqa: E402
from marketpilot.bots.performance import report  # noqa: E402
from marketpilot.config import load_settings  # noqa: E402

settings = load_settings()
store, backend = get_store()
start_bal = settings.account["starting_balance"]
target_wr = settings["goals"]["target_win_rate"]
kill = store.get_control("kill_switch", "off") == "on"
trades = store.trades()
rep = report(trades, start_bal)
o = rep["overall"]
closed = rep["closed"]
eq_curve = pd.DataFrame(store.equity_curve())
equity_now = float(eq_curve["equity_gbp"].iloc[-1]) if not eq_curve.empty else start_bal + o["pnl_gbp"]
open_pos = [t for t in trades if t["status"] in ("open", "pending")]
ALL_SYMBOLS = {s: cls for cls, u in settings["universe"].items() for s in u["symbols"]}

# ---------------- sidebar navigation ----------------
PAGES = ["Dashboard", "Markets", "Trade planner", "Positions", "Trade history", "Bot decisions", "Strategies",
         "Learning lab", "Bot activity", "Learn", "Settings"]
ICONS = {"Dashboard": "space_dashboard", "Markets": "candlestick_chart", "Trade planner": "model_training",
         "Positions": "account_balance_wallet", "Trade history": "receipt_long", "Bot decisions": "rule",
         "Strategies": "science", "Learning lab": "psychology", "Bot activity": "monitor_heart",
         "Learn": "school", "Settings": "tune"}
with st.sidebar:
    st.markdown(f"<div class='mp-title' style='font-size:1.2rem'>📈 MarketPilot</div>"
                f"<div class='mp-sub'>Paper trading · {VERSION}</div><br>", unsafe_allow_html=True)
    page = st.radio("Navigate", PAGES, format_func=lambda p: f":material/{ICONS[p]}:  {p}", label_visibility="collapsed")
    st.markdown("<br>", unsafe_allow_html=True)
    dot = DOWN if kill else UP
    st.markdown(f"<span class='mp-pill'><span class='mp-dot' style='background:{dot}'></span>"
                f"{'Kill switch ON' if kill else 'Bots active'}</span>", unsafe_allow_html=True)
    st.caption(f"Data: {backend}. Paper only — no real money, not financial advice.")

pnl = equity_now - start_bal
st.markdown(
    f"<div class='mp-top'><div><div class='mp-title'>{page}</div>"
    f"<div class='mp-sub'>{len(ALL_SYMBOLS)} markets · {len(open_pos)} open positions · "
    f"updated {(eq_curve['ts'].iloc[-1][:16].replace('T', ' ') + ' UTC') if not eq_curve.empty else '—'}</div></div>"
    f"<div style='display:flex;gap:10px;flex-wrap:wrap'>"
    f"<span class='mp-pill'>Equity <b style='color:{TEXT}'>£{equity_now:,.2f}</b></span>"
    f"<span class='mp-pill'>P&amp;L <b class='{'up' if pnl >= 0 else 'down'}'>£{pnl:+,.2f}</b></span>"
    f"<span class='mp-pill'><span class='mp-dot' style='background:{dot}'></span>"
    f"{'Paused' if kill else 'Live (paper)'}</span></div></div>", unsafe_allow_html=True)


def sign_cls(v):
    return "up" if v > 0 else ("down" if v < 0 else "muted")


# ================= Dashboard =================
if page == "Dashboard":
    goals = settings["goals"]
    try:
        rs = json.loads(store.get_control("risk_state", "") or "{}")
    except Exception:
        rs = {}
    eq_vals = eq_curve["equity_gbp"].tolist() if not eq_curve.empty else [start_bal]
    peak = max(eq_vals + [equity_now])
    dd_now = (peak - equity_now) / peak * 100 if peak else 0.0
    dd_max = 0.0
    run_peak = eq_vals[0] if eq_vals else start_bal
    for v in eq_vals + [equity_now]:
        run_peak = max(run_peak, v)
        dd_max = max(dd_max, (run_peak - v) / run_peak * 100 if run_peak else 0)
    mult = float(rs.get("account_mult", 1.0))
    cards([
        ("Equity", f"£{equity_now:,.2f}", f"{o['return_pct']:+.2f}% since start", sign_cls(o["return_pct"])),
        ("Expectancy", f"{o['expectancy_r']:+.2f}R" if o["trades"] else "—",
         f"goal above {goals.get('min_expectancy_r', 0.1):+.2f}R per trade",
         sign_cls(o["expectancy_r"] - goals.get("min_expectancy_r", 0.1)) if o["trades"] else "muted"),
        ("Drawdown now", f"{dd_now:.1f}%", f"worst {dd_max:.1f}% · limit {goals.get('max_drawdown_pct', 10):.0f}%",
         "down" if dd_max >= goals.get("max_drawdown_pct", 10) else ("warn" if dd_now >= 3 else "up")),
        ("Average loss", f"{o['avg_loss_r']:+.2f}R" if o["losses"] else "—",
         f"{o['losses']} losing trades", "up" if o["losses"] and o["avg_loss_r"] > -1.0 else "muted"),
        ("Risk per trade now", f"{mult:.2g}×", ", ".join(rs.get("notes", [])) or "normal size",
         "warn" if mult < 1 else "up"),
        ("Win rate", f"{o['win_rate']:.0%}" if o["trades"] else "—",
         f"{o['wins']} won · {o['losses']} lost", "muted"),
    ])
    left, right = st.columns([2.1, 1])
    with left:
        st.markdown("### Account equity")
        if not eq_curve.empty:
            e = eq_curve.copy()
            e["ts"] = pd.to_datetime(e["ts"], utc=True, format="mixed")
            col = UP if e["equity_gbp"].iloc[-1] >= start_bal else DOWN
            fig = go.Figure(go.Scatter(x=e["ts"], y=e["equity_gbp"], line=dict(color=col, width=2),
                                       fill="tozeroy", fillcolor="rgba(22,199,132,.08)" if col == UP else
                                       "rgba(234,57,67,.08)", hovertemplate="£%{y:,.2f}<extra></extra>"))
            fig.add_hline(y=start_bal, line=dict(color=MUTED, dash="dot", width=1))
            lo, hi = e["equity_gbp"].min(), e["equity_gbp"].max()
            pad = max((hi - lo) * 0.3, start_bal * 0.01)
            fig.update_yaxes(range=[min(lo, start_bal) - pad, max(hi, start_bal) + pad], tickprefix="£")
            st.plotly_chart(style(fig, 300), width="stretch")
        else:
            st.info("The equity curve appears after the first worker run.")
        if not closed.empty:
            cl = closed.reset_index(drop=True)
            cl["n"] = cl.index + 1
            st.markdown("### Result per trade")
            fig = go.Figure(go.Bar(x=cl["n"], y=cl["r_multiple"],
                                   marker_color=[UP if r > 0 else DOWN for r in cl["r_multiple"]],
                                   customdata=cl[["symbol", "strategy", "exit_reason"]],
                                   hovertemplate="%{customdata[0]} %{y:+.2f}R<br>%{customdata[1]}<br>"
                                                 "%{customdata[2]}<extra></extra>"))
            fig.update_xaxes(title="trade #")
            st.plotly_chart(style(fig, 240, "R"), width="stretch")
    with right:
        st.markdown("### Open positions")
        if open_pos:
            mk = {m["symbol"]: m for m in store.market()}
            html = "<div class='mp-panel'>"
            for t in open_pos[:8]:
                side = "LONG" if int(t["direction"]) == 1 else "SHORT"
                px = (mk.get(t["symbol"]) or {}).get("price")
                ent = t.get("entry_price")
                upnl = ""
                if px and ent and t.get("risk_per_unit"):
                    r_now = int(t["direction"]) * (float(px) - float(ent)) / float(t["risk_per_unit"])
                    upnl = f"<span class='{sign_cls(r_now)}' style='float:right'>{r_now:+.2f}R</span>"
                odds = (f"<div class='mp-note'>odds: target first {float(t['tm_p_target']):.0%}</div>"
                        if t.get("tm_p_target") is not None else "")
                html += (f"<div class='item' style='padding:8px 0;border-bottom:1px solid {BORDER}'>"
                         f"<b>{t['symbol']}</b> <span class='mp-tag'>{side}</span>"
                         f"<span class='mp-tag'>{t['status']}</span>{upnl}"
                         f"<div class='mp-note'>{t['strategy'].split('(')[0]}</div>{odds}</div>")
            st.markdown(html + "</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='mp-panel mp-note'>No open positions. The bots only trade strategies that "
                        "passed research, so quiet days are normal.</div>", unsafe_allow_html=True)
        fg = store.get_control("fear_greed", "")
        if fg:
            try:
                f = json.loads(fg)
                v = int(f["value"])
                col = DOWN if v < 25 else (WARN if v < 45 else (MUTED if v < 55 else UP))
                st.markdown(f"<div class='mp-panel'><div class='mp-label'>Crypto Fear &amp; Greed</div>"
                            f"<div class='mp-value' style='color:{col}'>{v}<span class='muted' "
                            f"style='font-size:.9rem'> / 100</span></div><div class='mp-delta' style='color:{col}'>"
                            f"{f['label']}</div><div style='height:6px;background:{BORDER};border-radius:3px;"
                            f"margin-top:8px'><div style='width:{v}%;height:6px;background:{col};border-radius:3px'>"
                            f"</div></div><div class='mp-note' style='margin-top:6px'>Source: alternative.me</div>"
                            f"</div>", unsafe_allow_html=True)
            except Exception:
                pass
        st.markdown("### Headlines")
        hl = store.get_control("headlines", "")
        items = []
        if hl:
            try:
                items = json.loads(hl).get("items", [])
            except Exception:
                items = []
        if items:
            html = "<div class='mp-panel mp-news'>"
            for it in items[:9]:
                title = it["title"].replace("<", "&lt;")
                link = f"<a href='{it['url']}' target='_blank'>{title}</a>" if it.get("url") else title
                html += (f"<div class='item'><span class='mp-tag'>{it['symbol']}</span>{link}"
                         f"<div class='mp-note'>{it.get('publisher', '')}</div></div>")
            st.markdown(html + "</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='mp-panel mp-note'>Headlines appear after the next hourly run.</div>",
                        unsafe_allow_html=True)
    st.caption("The goal is small losses and a positive average result, not a high win rate. Loss Control "
               "sizes new trades down in wild markets, after a drawdown or a losing streak, and caps how much "
               "can be at risk at once.")

# ================= Markets =================
elif page == "Markets":
    mk = pd.DataFrame(store.market())
    if mk.empty:
        st.info("The worker fills this in on its next hourly run.")
    else:
        groups = ["All"] + sorted(mk["asset_class"].unique())
        g = st.segmented_control("Group", groups, default="All", label_visibility="collapsed")
        view = mk if g in (None, "All") else mk[mk["asset_class"] == g]
        up_n = int((view["chg_1d"] > 0).sum())
        best = view.loc[view["chg_20d"].idxmax()] if view["chg_20d"].notna().any() else None
        worst = view.loc[view["chg_20d"].idxmin()] if view["chg_20d"].notna().any() else None
        cards([
            ("Markets", f"{len(view)}", f"{up_n} up today · {len(view) - up_n} down", "muted"),
            ("In uptrend", f"{int((view['above_200d'] == 1).sum())}", "above 200-day average", "up"),
            ("Best 20 days", best["symbol"] if best is not None else "—",
             f"{best['chg_20d']:+.1%}" if best is not None else "", "up"),
            ("Worst 20 days", worst["symbol"] if worst is not None else "—",
             f"{worst['chg_20d']:+.1%}" if worst is not None else "", "down"),
        ])
        left, right = st.columns([1.15, 1.6])
        with left:
            st.markdown("### Watchlist")
            view = view.sort_values("conf_long", ascending=False)
            wl = view[["symbol", "price", "chg_1d", "chg_20d", "conf_long"]].reset_index(drop=True)
            tint = lambda v: f"color: {UP}" if pd.notna(v) and v > 0 else (f"color: {DOWN}" if pd.notna(v) and v < 0 else "")
            styled = wl.style.map(tint, subset=["chg_1d", "chg_20d"]).format(
                {"price": "{:,.4g}", "chg_1d": "{:+.2%}", "chg_20d": "{:+.2%}"}, na_rep="–")
            sel = st.dataframe(
                styled, hide_index=True, width="stretch", height=min(600, 35 * (len(wl) + 1) + 3),
                on_select="rerun", selection_mode="single-row", key="watch",
                column_config={
                    "symbol": "Market", "price": "Price", "chg_1d": "24h", "chg_20d": "20d",
                    "conf_long": st.column_config.ProgressColumn("Long conf.", min_value=0, max_value=8,
                                                                 format="%d/8", width="small")})
            rows = sel.selection.rows if sel and sel.selection else []
        with right:
            choice = view.iloc[rows[0]]["symbol"] if rows else view.iloc[0]["symbol"]
            row = mk[mk["symbol"] == choice].iloc[0]
            st.markdown(f"### {choice} <span class='mp-tag'>{row['asset_class']}</span>"
                        f"<span class='mp-tag'>{row['regime']}</span>", unsafe_allow_html=True)
            df = price_history(choice)
            pos = [t for t in open_pos if t["symbol"] == choice]
            levels = []
            if pos:
                t = pos[0]
                levels = [("entry", t.get("entry_price"), ACCENT), ("stop", t.get("stop"), DOWN),
                          ("target", t.get("target"), UP)]
            tv_chart(df, choice, levels, [t for t in trades if t["symbol"] == choice])
            c = st.columns(4)
            c[0].metric("RSI 14", f"{row['rsi14']:.0f}")
            c[1].metric("Daily range (ATR)", f"{row['atr_pct']:.2%}" if pd.notna(row["atr_pct"]) else "–")
            c[2].metric("Long confluence", f"{int(row['conf_long'])}/8")
            c[3].metric("Short confluence", f"{int(row['conf_short'])}/8")
            if df.attrs.get("synthetic"):
                st.caption("Live prices unavailable right now — showing synthetic demo data.")
            st.caption("Click a row in the watchlist to chart it. Arrows mark where the bots bought or sold; "
                       "circles mark exits with the result in R. Dashed lines show an open position's entry, stop "
                       "and target.")

# ================= Trade planner (Time Machine) =================
elif page == "Trade planner":
    st.markdown("<div class='mp-note'>Plan a trade and see the honest odds first. The Time Machine rebuilds the "
                "next days 10,000 times from this market's own past daily moves (drawn in 5-day blocks) and "
                "counts how often the target or the stop is hit first. It's a simulation, not a forecast: it "
                "assumes the future behaves like the past ~3 years and knows nothing about upcoming news."
                "</div><br>", unsafe_allow_html=True)
    f1, f2, f3, f4, f5 = st.columns([1.4, 1, 1, 1, 1])
    syms = sorted(ALL_SYMBOLS)
    symbol = f1.selectbox("Market", syms, index=syms.index("SPY") if "SPY" in syms else 0)
    side = f2.segmented_control("Side", ["Long", "Short"], default="Long")
    stop_atr = f3.number_input("Stop (× ATR)", 0.5, 6.0, 2.0, 0.25)
    rr = f4.number_input("Target (× risk)", 0.5, 6.0, 2.0, 0.25)
    hold = f5.number_input("Max days", 2, 40, 20, 1)
    df = price_history(symbol)
    from marketpilot.indicators import add_indicators
    ind = add_indicators(df)
    last = ind.iloc[-1]
    d = 1 if side != "Short" else -1
    entry = float(last["close"])
    atr = float(last["atr14"])
    stop = entry - d * stop_atr * atr
    target = entry + d * rr * stop_atr * atr
    odds, det = tm.simulate(ind, d, entry, stop, target, int(hold), detail=True)
    if odds is None:
        st.warning("Not enough history for this market.")
    else:
        risk_gbp = start_bal * settings.account["risk_per_trade_pct"] / 100
        cards([
            ("Target first", f"{odds.p_target:.0%}", f"+{rr:.2g}R = £{rr * risk_gbp:,.2f}", "up"),
            ("Stop first", f"{odds.p_stop:.0%}", f"−1R = −£{risk_gbp:,.2f}", "down"),
            ("Neither by day " + str(int(hold)), f"{odds.p_neither:.0%}", "closed at time limit", "muted"),
            ("Average result", f"{odds.exp_r:+.2f}R", f"£{odds.exp_r * risk_gbp:+,.2f} per trade",
             sign_cls(odds.exp_r)),
            ("Typical length", f"{odds.median_bars:.0f} days", "median, all replays", "muted"),
        ])
        odds_bar(odds.p_target, odds.p_stop, odds.p_neither)
        from marketpilot.bots import confluence as conf
        mf_cfg = (settings.raw.get("market_filter") or {}).get(ALL_SYMBOLS.get(symbol, ""), {})
        bench = mf_cfg.get("symbol") if isinstance(mf_cfg, dict) else None
        market_ok = None
        if bench and symbol not in (mf_cfg.get("exclude") or []):
            bc = price_history(bench)["close"]
            if len(bc) > 200:
                market_ok = bool(bc.iloc[-1] > bc.rolling(200).mean().iloc[-1])
        c_score, c_hits = conf.score(last, d, market_ok)
        g = cv.grade(c_score, odds, settings.raw.get("conviction", {}))
        g_col = {"A+": UP, "A": ACCENT, "B": WARN, "C": MUTED}[g]
        names = {"trend": "Long-term trend", "swing_trend": "Medium-term trend", "momentum": "Momentum",
                 "level": "Key level", "fib": "Fibonacci zone", "volume": "Volume", "calm": "Calm volatility",
                 "market": "Market trend"}
        ticks = "".join(f"<span class='mp-tag' style='color:{UP if k in c_hits else MUTED}'>"
                        f"{'✓' if k in c_hits else '·'} {names.get(k, k)}</span>" for k, _ in conf.FACTORS)
        g_text = {"A+": "High confluence and the Time Machine confirms.",
                  "A": "High confluence, but the Time Machine does not confirm.",
                  "B": "Medium confluence: the Time Machine's odds don't count toward the grade below 6/8.",
                  "C": "Low confluence: few factors agree with this side."}[g]
        st.markdown(f"<div class='mp-panel' style='display:flex;gap:18px;align-items:center;flex-wrap:wrap'>"
                    f"<div style='font-size:2.1rem;font-weight:800;color:{g_col};font-family:JetBrains Mono,"
                    f"monospace;min-width:70px;text-align:center'>{g}</div><div style='flex:1;min-width:260px'>"
                    f"<b>Conviction grade · confluence {c_score}/8</b><div class='mp-note'>{g_text}</div>"
                    f"<div style='margin-top:6px;line-height:2'>{ticks}</div></div></div>",
                    unsafe_allow_html=True)
        verdict = ("This plan has a positive simulated edge." if odds.exp_r > 0.05 else
                   "This is close to a coin flip after the stop and target are accounted for."
                   if odds.exp_r > -0.05 else "The odds lean against this plan.")
        st.markdown(f"<div class='mp-panel' style='margin-top:10px'><b>{verdict}</b> "
                    f"<span class='mp-note'>Entry {entry:,.4g} · stop {stop:,.4g} · target {target:,.4g} · "
                    f"ATR {atr:,.4g}. Break-even win rate for {rr:.2g}:1 is {1 / (1 + rr):.0%}.</span></div>",
                    unsafe_allow_html=True)
        st.markdown("### Chart with your plan")
        tv_chart(df, symbol, [("entry", entry, ACCENT), ("stop", stop, DOWN), ("target", target, UP)],
                 [t for t in trades if t["symbol"] == symbol], height=460)
        a, b = st.columns([1.6, 1])
        with a:
            st.markdown("### 10,000 possible futures")
            closes = det["closes"]
            hist = ind["close"].tail(int(hold) + 10)
            x_hist = list(range(-len(hist) + 1, 1))
            x_fut = list(range(0, closes.shape[1] + 1))
            pct = np.percentile(np.concatenate([np.full((len(closes), 1), entry), closes], axis=1),
                                [5, 25, 50, 75, 95], axis=0)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_fut + x_fut[::-1], y=list(pct[4]) + list(pct[0][::-1]), fill="toself",
                                     fillcolor="rgba(79,124,255,.16)", line=dict(width=0), name="90% of paths",
                                     hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=x_fut + x_fut[::-1], y=list(pct[3]) + list(pct[1][::-1]), fill="toself",
                                     fillcolor="rgba(79,124,255,.32)", line=dict(width=0), name="50% of paths",
                                     hoverinfo="skip"))
            rng = np.random.default_rng(3)
            for i in rng.choice(len(closes), 25, replace=False):
                col = {"target": UP, "stop": DOWN}.get(det["outcome"][i], MUTED)
                fig.add_trace(go.Scatter(x=x_fut, y=[entry, *closes[i]], line=dict(color=col, width=0.8),
                                         opacity=0.45, showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=x_fut, y=pct[2], line=dict(color=ACCENT, width=2), name="median"))
            fig.add_trace(go.Scatter(x=x_hist, y=hist.values, line=dict(color=TEXT, width=1.6), name="actual"))
            fig.add_hline(y=target, line=dict(color=UP, dash="dash", width=1.2), annotation_text="target",
                          annotation_font_color=UP)
            fig.add_hline(y=stop, line=dict(color=DOWN, dash="dash", width=1.2), annotation_text="stop",
                          annotation_font_color=DOWN)
            fig.update_xaxes(title="days from today")
            st.plotly_chart(style(fig, 380, legend=True), width="stretch")
            st.caption("Thin lines are 25 sample replays: green reached the target first, red the stop. "
                       "Paths keep moving after an exit; the bands show where price could go.")
        with b:
            st.markdown("### Spread of results")
            r = det["r"]
            fig = go.Figure(go.Histogram(x=r, nbinsx=40, marker_color=ACCENT, opacity=0.85,
                                         hovertemplate="%{x:.2f}R: %{y} replays<extra></extra>"))
            fig.add_vline(x=0, line=dict(color=MUTED, width=1))
            fig.add_vline(x=odds.exp_r, line=dict(color=WARN, width=1.5, dash="dot"),
                          annotation_text=f"avg {odds.exp_r:+.2f}R", annotation_font_color=WARN,
                          annotation_position="top right")
            fig.update_xaxes(title="result (R)")
            st.plotly_chart(style(fig, 380), width="stretch")
        st.markdown("### What the bots planned recently")
        cand = [c for c in store.candidates(300) if (c.get("evidence") or {}).get("time_machine")]
        if cand:
            rows = []
            for c in cand[:40]:
                t_ = c["evidence"]["time_machine"]
                rows.append({"time": c["bar_time"][:10], "market": c["symbol"],
                             "side": "LONG" if c["direction"] == 1 else "SHORT",
                             "strategy": c["strategy"].split("(")[0], "decision": c["decision"],
                             "target first": t_["p_target"], "stop first": t_["p_stop"], "avg R": t_["exp_r"],
                             "grade": c["evidence"].get("grade", ""),
                             "setup history": t_.get("hist_trades", 0)})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
                "target first": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
                "stop first": st.column_config.NumberColumn(format="percent"),
                "avg R": st.column_config.NumberColumn(format="%+.2f")})
        else:
            st.caption("Every new bot candidate gets Time Machine odds from v0.9 on; they'll list here.")

# ================= Positions =================
elif page == "Positions":
    if not open_pos:
        st.info("No open positions.")
    else:
        op = pd.DataFrame(open_pos)
        op["side"] = op["direction"].map({1: "LONG", -1: "SHORT"})
        for c_ in ("tm_p_target", "tm_exp_r", "grade"):
            if c_ not in op:
                op[c_] = None
        st.dataframe(op[["symbol", "side", "status", "strategy", "opened_at", "entry_price", "stop", "target",
                         "quantity", "risk_gbp", "notional_gbp", "bars_held", "grade", "confluence", "tm_p_target",
                         "tm_exp_r"]], hide_index=True, width="stretch", column_config={
            "tm_p_target": st.column_config.NumberColumn("Odds: target first", format="percent"),
            "tm_exp_r": st.column_config.NumberColumn("Odds: avg R", format="%+.2f"),
            "risk_gbp": st.column_config.NumberColumn("Risk £", format="£%.2f"),
            "notional_gbp": st.column_config.NumberColumn("Size £", format="£%.2f")})
        pick = st.selectbox("Chart a position", [t["symbol"] for t in open_pos])
        t = next(x for x in open_pos if x["symbol"] == pick)
        tv_chart(price_history(pick), pick, [
            ("entry", t.get("entry_price"), ACCENT), ("stop", t.get("stop"), DOWN),
            ("target", t.get("target"), UP)], [x for x in trades if x["symbol"] == pick])
        if t.get("note"):
            st.caption(t["note"])

# ================= Trade history =================
elif page == "Trade history":
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
                                    "pnl_gbp": st.column_config.NumberColumn("P&L", format="£%+.2f")})
        st.download_button("Download CSV", log[cols].to_csv(index=False), "marketpilot_trades.csv")
        st.markdown("### Breakdown")
        dim = st.segmented_control("By", ["strategy", "asset_class", "symbol", "regime"], default="strategy")
        bd = rep["by"][dim or "strategy"]
        if not bd.empty:
            st.dataframe(bd.sort_values("expectancy_r")[[dim or "strategy", "trades", "win_rate", "expectancy_r",
                                                         "profit_factor", "total_r", "pnl_gbp"]],
                         hide_index=True, width="stretch", column_config={
                "win_rate": st.column_config.NumberColumn(format="percent"),
                "expectancy_r": st.column_config.NumberColumn(format="%+.3f"),
                "pnl_gbp": st.column_config.NumberColumn(format="£%+.2f")})

# ================= Bot decisions =================
elif page == "Bot decisions":
    cand = pd.DataFrame(store.candidates(1000))
    if cand.empty:
        st.info("No candidates yet.")
    else:
        f = st.segmented_control("Show", ["all", "trade", "reject"], default="all")
        if f and f != "all":
            cand = cand[cand["decision"] == f]
        cand["why"] = cand["reasons"].apply(lambda r: "; ".join(r) if isinstance(r, list) and r else "all checks passed")
        cand["side"] = cand["direction"].map({1: "LONG", -1: "SHORT"})
        cand["odds"] = cand["evidence"].apply(lambda e: ((e or {}).get("time_machine") or {}).get("p_target")
                                              if isinstance(e, dict) else None)
        cand["confluence"] = cand["evidence"].apply(lambda e: (e or {}).get("confluence") if isinstance(e, dict) else None)
        cand["grade"] = cand["evidence"].apply(lambda e: (e or {}).get("grade") if isinstance(e, dict) else None)
        st.dataframe(cand[["bar_time", "symbol", "side", "strategy", "regime", "decision", "expected_r",
                           "grade", "confluence", "odds", "why"]], hide_index=True, width="stretch",
                     column_config={"expected_r": st.column_config.NumberColumn("EV (R)", format="%+.3f"),
                                    "odds": st.column_config.NumberColumn("Target first", format="percent")})

# ================= Strategies =================
elif page == "Strategies":
    st.caption("Tested walk-forward on unseen data with fees, spreads and slippage, pooled per market group, "
               "then checked once on an untouched holdout. Only APPROVED strategies trade.")
    strat = pd.DataFrame(store.strategies())
    if strat.empty:
        st.info("No research yet.")
    else:
        g = strat.drop_duplicates(["asset_class", "strategy"]).copy()
        g["trades"] = g["oos"].apply(lambda x: x["trades"])
        g["win_rate"] = g["oos"].apply(lambda x: x["win_rate"])
        g["expectancy"] = g["oos"].apply(lambda x: x["expectancy_r"])
        g["pf"] = g["oos"].apply(lambda x: x["profit_factor"])
        g["t"] = g["oos"].apply(lambda x: x.get("t_stat", float("nan")))
        g["holdout"] = g["holdout"].apply(lambda x: x["expectancy_r"])
        g["why"] = g["reasons"].apply(lambda r: "; ".join(r) if r else "passed all gates")
        g["max_dd"] = g["oos"].apply(lambda x: x.get("max_drawdown_r"))
        g["dd_cut"] = g["oos"].apply(lambda x: (x.get("loss_control") or {}).get("dd_cut_pct"))
        n_app = int((g["status"] == "approved").sum())
        cards([("Approved", f"{n_app}", f"of {len(g)} tested", "up"),
               ("Rejected", f"{len(g) - n_app}", "failed a gate", "muted")])
        st.dataframe(g.sort_values(["status", "t"], ascending=[True, False])[
            ["asset_class", "strategy", "status", "trades", "expectancy", "pf", "max_dd", "dd_cut", "win_rate", "t",
             "holdout", "why"]],
            hide_index=True, width="stretch", column_config={
                "win_rate": st.column_config.NumberColumn("Win rate", format="percent"),
                "expectancy": st.column_config.NumberColumn("Expectancy R", format="%+.3f"),
                "pf": st.column_config.NumberColumn("Profit factor", format="%.2f"),
                "t": st.column_config.NumberColumn("t-stat", format="%.1f"),
                "holdout": st.column_config.NumberColumn("Holdout R", format="%+.3f"),
                "max_dd": st.column_config.NumberColumn("Max drawdown R", format="%.1f"),
                "dd_cut": st.column_config.NumberColumn("Drawdown cut by loss control", format="percent")})
        st.caption("'Drawdown cut' = how much smaller the worst losing run would have been with volatility "
                   "scaling, on the same trades. Filled in by the next Saturday research run.")
        fig = go.Figure(go.Scatter(
            x=g["win_rate"], y=g["expectancy"], mode="markers",
            text=g["strategy"] + " · " + g["asset_class"],
            marker=dict(size=11, color=[UP if s == "approved" else MUTED for s in g["status"]],
                        line=dict(color=BG, width=1.5)),
            hovertemplate="%{text}<br>win %{x:.0%} · %{y:+.3f}R<extra></extra>"))
        fig.add_hline(y=0, line=dict(color=MUTED, width=1))
        fig.add_vline(x=target_wr, line=dict(color=MUTED, dash="dash", width=1), annotation_text="win-rate goal")
        fig.update_xaxes(title="out-of-sample win rate", tickformat=".0%")
        st.markdown("### Win rate vs expectancy")
        st.plotly_chart(style(fig, 380, "R per trade"), width="stretch")
        st.caption("Green = approved. A high win rate is not the same as profitable; t-stat below 2 could be luck.")
        st.markdown("### Trade management: same trades, three ways to handle them")
        mode_now = (settings.raw.get("trade_management") or {}).get("mode", "base")
        st.caption(f"Base = each strategy's own exits · Breakeven = stop to breakeven once a day closes +1R · "
                   f"Partial = bank half at +1R, rest to breakeven. In use now: **{mode_now}**. "
                   "Filled in by each Saturday research run.")
        rows_m = []
        for _, r in g.iterrows():
            for mode, v in ((r["oos"] or {}).get("management") or {}).items():
                rows_m.append({"group": r["asset_class"], "strategy": r["strategy"], "status": r["status"],
                               "mode": mode, "expectancy": v["expectancy_r"], "full_losses": v["full_loss_rate"],
                               "scratches": v.get("scratch_rate"), "avg_loss": v["avg_loss_r"],
                               "max_dd": v["max_drawdown_r"], "holdout": v["holdout_expectancy_r"]})
        if rows_m:
            mdf = pd.DataFrame(rows_m)
            appr = mdf[mdf["status"] == "approved"]
            if not appr.empty:
                summ = appr.groupby("mode").agg(expectancy=("expectancy", "mean"), full_losses=("full_losses", "mean"),
                                                avg_loss=("avg_loss", "mean"), max_dd=("max_dd", "mean"),
                                                holdout=("holdout", "mean")).reindex(["base", "breakeven", "partial"]).dropna(how="all")
                cards([(m.title(), f"{v['expectancy']:+.3f}R", f"full losses {v['full_losses']:.0%} · worst run "
                        f"{v['max_dd']:.1f}R", "up" if m == mode_now else "muted") for m, v in summ.iterrows()])
            st.dataframe(mdf.sort_values(["status", "group", "strategy", "mode"]), hide_index=True, width="stretch",
                         column_config={
                             "expectancy": st.column_config.NumberColumn("Expectancy R", format="%+.3f"),
                             "full_losses": st.column_config.NumberColumn("Full −1R losses", format="percent"),
                             "scratches": st.column_config.NumberColumn("Breakeven exits", format="percent"),
                             "avg_loss": st.column_config.NumberColumn("Avg loss R", format="%+.2f"),
                             "max_dd": st.column_config.NumberColumn("Worst run R", format="%.1f"),
                             "holdout": st.column_config.NumberColumn("Holdout R", format="%+.3f")})
        else:
            st.info("Appears after the next research run.")

# ================= Learning lab =================
elif page == "Learning lab":
    from marketpilot.bots.learning import confluence_stats, forward_stats
    st.caption("Shadow bots practise every strategy on every market with the same rules as the paper account, "
               "in R, with no position limits. Strong forward evidence promotes a strategy; weak evidence demotes it.")
    try:
        shadow_all = pd.DataFrame(store.shadow_trades())
    except Exception:
        shadow_all = pd.DataFrame()
    if shadow_all.empty:
        st.info("No shadow trades yet.")
    else:
        closed_sh = shadow_all[shadow_all["status"] == "closed"]
        r_sh = closed_sh["r_multiple"].astype(float) if not closed_sh.empty else pd.Series(dtype=float)
        cards([("Shadow open", f"{int(shadow_all['status'].isin(['open', 'pending']).sum())}", "practice trades", "muted"),
               ("Shadow closed", f"{len(closed_sh)}", "evidence collected", "muted"),
               ("Shadow win rate", f"{(r_sh > 0).mean():.0%}" if len(r_sh) else "—", "all strategies", "muted"),
               ("Shadow expectancy", f"{r_sh.mean():+.2f}R" if len(r_sh) else "—", "all strategies",
                sign_cls(r_sh.mean()) if len(r_sh) else "muted")])
        t1, t2, t4, t3 = st.tabs(["Leaderboard", "Does confluence work?", "Do A+ trades win more?",
                                  "Is the Time Machine right?"])
        with t4:
            gs = cv.grade_stats(shadow_all.to_dict("records") + trades)
            st.caption("A+ = 6-8 confluence factors agree AND the Time Machine confirms. A = 6-8 without "
                       "confirmation, B = 4-5, C = 0-3. If A+ clearly beats the rest over 30+ trades, set "
                       "`conviction.min_grade: \"A+\"` so the paper account only takes A+ setups.")
            if gs.empty:
                st.info("Grades are recorded from v0.10 on; this fills in as graded trades close.")
            else:
                fig = go.Figure(go.Bar(x=gs["grade"], y=gs["expectancy_r"],
                                       marker_color=[UP if v > 0 else DOWN for v in gs["expectancy_r"]],
                                       text=[f"{n} trades" for n in gs["trades"]], textposition="outside",
                                       hovertemplate="%{x}: %{y:+.3f}R<extra></extra>"))
                fig.add_hline(y=0, line=dict(color=MUTED, width=1))
                st.plotly_chart(style(fig, 280, "R per trade"), width="stretch")
                st.dataframe(gs, hide_index=True, width="stretch", column_config={
                    "grade": "Grade", "trades": "Trades",
                    "win_rate": st.column_config.NumberColumn("Win rate", format="percent"),
                    "expectancy_r": st.column_config.NumberColumn("Expectancy R", format="%+.3f"),
                    "profit_factor": st.column_config.NumberColumn("Profit factor", format="%.2f"),
                    "t_stat": st.column_config.NumberColumn("t-stat", format="%.1f")})
        with t1:
            fs = forward_stats(store)
            if fs.empty:
                st.caption("Leaderboard fills as shadow trades close.")
            else:
                st.dataframe(fs.sort_values(["t_stat", "trades"], ascending=False)[
                    ["asset_class", "strategy", "trades", "win_rate", "expectancy_r", "profit_factor", "t_stat"]],
                    hide_index=True, width="stretch", column_config={
                        "asset_class": "Group", "strategy": "Strategy", "trades": "Trades",
                        "win_rate": st.column_config.NumberColumn("Win rate", format="percent"),
                        "expectancy_r": st.column_config.NumberColumn("Expectancy R", format="%+.3f"),
                        "profit_factor": st.column_config.NumberColumn("Profit factor", format="%.2f"),
                        "t_stat": st.column_config.NumberColumn("t-stat", format="%.1f")})
        with t2:
            cst = confluence_stats(store)
            if cst.empty:
                st.caption("Not enough closed shadow trades with a confluence score yet.")
            else:
                st.dataframe(cst, hide_index=True, width="stretch", column_config={
                    "win_rate": st.column_config.NumberColumn(format="percent"),
                    "expectancy_r": st.column_config.NumberColumn(format="%+.3f")})
                st.caption("If high scores clearly win more with 30+ trades per group, raise `confluence.min_score`.")
        with t3:
            cal = tm.calibration(shadow_all.to_dict("records") + trades)
            if cal.empty:
                st.caption("Shadow and paper trades get Time Machine odds from v0.9 on. Once they close, this "
                           "compares the forecast with what actually happened.")
            else:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=cal["forecast"], y=cal["forecast_target_pct"], name="forecast: target first",
                                     marker_color=ACCENT))
                fig.add_trace(go.Bar(x=cal["forecast"], y=cal["actual_win_pct"], name="actual: won",
                                     marker_color=UP))
                fig.update_layout(barmode="group")
                st.plotly_chart(style(fig, 320, pct=True, legend=True), width="stretch")
                st.dataframe(cal, hide_index=True, width="stretch", column_config={
                    "forecast_target_pct": st.column_config.NumberColumn("Forecast target first", format="percent"),
                    "actual_win_pct": st.column_config.NumberColumn("Actually won", format="percent"),
                    "avg_forecast_r": st.column_config.NumberColumn("Forecast avg R", format="%+.2f"),
                    "actual_avg_r": st.column_config.NumberColumn("Actual avg R", format="%+.2f")})
                st.caption("The Time Machine's vote only counts on 6-8 confluence setups (the A+ grade). "
                           "This table checks whether its forecasts match reality overall.")

# ================= Bot activity =================
elif page == "Bot activity":
    ev = pd.DataFrame(store.events(500))
    if ev.empty:
        st.info("No activity yet.")
    else:
        lv = st.segmented_control("Level", ["all", "info", "warning"], default="all")
        if lv and lv != "all":
            ev = ev[ev["level"] == lv]
        st.dataframe(ev[["ts", "level", "bot", "message"]], hide_index=True, width="stretch", height=620)

# ================= Learn =================
elif page == "Learn":
    t1, t2, t3, t4 = st.tabs(["Key ideas", "How pros plan a trade", "Indicators", "Strategies"])
    with t1:
        st.markdown("""
- **R (risk unit)**: every trade risks a fixed amount, here 0.5% of the account (£2.50 on £500). +2R made £5; −1R lost £2.50.
- **Win rate vs expectancy**: expectancy = win rate × average win − loss rate × average loss. A 40% win rate with
  2R wins makes +0.2R per trade; an 80% win rate with tiny wins and big losses loses money.
- **Stop-loss**: where a losing trade is closed. A **trailing stop** follows the best price to lock in profit.
- **Walk-forward testing**: choose settings on older data, test on newer data never seen, then one final **holdout**.
- **t-statistic**: how many standard errors the average result is above zero. Below ~2 it could be luck.
- **Monte Carlo (Time Machine)**: replay thousands of possible futures from a market's own history to see the odds
  of a plan. It shows risk honestly but can't predict news; its forecasts are checked in the Learning lab.
- **Costs**: spreads, slippage and fees are charged on every simulated trade.
- **Loss control over win rate**: what decides survival is how big and how clustered the losses are.
  Halving size when markets are wild, after a drawdown, or after a losing streak keeps bad runs shallow,
  and a shallow drawdown is much easier to recover from (−10% needs +11%; −50% needs +100%).""")
    with t2:
        st.markdown("""
- **Build a case, not a hunch.** Stack layers of confluence (trend, key level, Fibonacci zone, structure) and use
  indicators as the last check. The Confluence bot scores every signal on 8 factors.
- **Bull and bear scenario.** Decide what proves you wrong before entering; the stop goes there.
- **Structure stops.** Stops just beyond the last swing low/high with a small buffer ("swing" variants).
- **Check the odds before you risk money.** Use the Trade planner to see how often a plan's target beats its stop.
- **Risk first.** Peter Brandt: small fixed risk, partial profits, tighten stops as a trade works, cut losers fast.
- **Capitulation buys.** Crypto Crew University's "3 checkmarks": below the 200 MA, RSI(14) < 26, Stoch RSI < 20.
- **Reality check.** Only ~1% of Brazilian day traders studied earned more than the minimum wage.""")
    with t3:
        INDICATORS = {
            "Moving average (SMA / EMA)": "Average price over N bars; EMA weights recent bars more. Above the 200-day = long-term uptrend.",
            "RSI": "0–100 scale of recent gains vs losses. RSI(2) < 10 is very oversold; RSI(14) is the slower version.",
            "ATR": "Average size of a bar's move including gaps. Sets stop distance and position size.",
            "Bollinger Bands": "20-day average ± 2 standard deviations. A close outside is unusually stretched.",
            "MACD": "12-day EMA minus 26-day EMA, with a 9-day signal line. Crossing up = momentum turned up.",
            "Donchian channel": "Highest high / lowest low of N bars. Breakouts are the basis of the Turtle rules.",
            "52-week high": "Highest price of the past year; stocks near it tend to keep outperforming.",
            "Stochastic RSI": "RSI of the RSI, 0–100. Both lines under 20 = deeply oversold momentum.",
            "Fibonacci retracement": "Pullbacks often pause around 38.2%–61.8% of the prior swing.",
            "Market structure": "Swing highs/lows mark where buyers or sellers took control; stops sit just beyond them.",
        }
        for k, v in INDICATORS.items():
            with st.expander(k):
                st.write(v)
    with t4:
        from marketpilot.strategies import ALL_STRATEGIES
        for cls in ALL_STRATEGIES:
            with st.expander(f"{cls.name} · {cls.family}"):
                st.write((cls.__doc__ or "").strip())
                st.caption("Settings tested: " + ", ".join(f"{k} ∈ {v}" for k, v in cls.grid.items()))
    st.caption("Educational information about how the bots work — not financial advice.")

# ================= Settings =================
elif page == "Settings":
    from marketpilot.bots.safety import SafetyBot
    safety = SafetyBot(store, settings)
    a, b = st.columns(2)
    with a:
        st.markdown("### Kill switch")
        st.caption("ON = no new trades. Open positions keep their stops and targets managed.")
        if kill:
            st.write(f"Reason: {store.get_control('pause_reason', '') or 'manual'}")
            if st.button("Release kill switch", type="primary"):
                safety.release()
                st.rerun()
        elif st.button("⛔ Engage kill switch", type="primary"):
            safety.engage("manual kill switch from dashboard")
            st.rerun()
        st.markdown("### Run the bots now")
        st.caption("The GitHub worker runs every hour. Use this to test.")
        if st.button("Run one trading cycle"):
            from marketpilot.data import make_provider
            from marketpilot.engine import Engine
            with st.spinner("Scanning markets…"):
                s = Engine(settings, make_provider("yfinance"), store).cycle(datetime.now(timezone.utc))
            st.success(f"Cycle done: {s}")
    with b:
        acc = settings.account
        st.markdown("### Risk rules")
        cards([("Balance", f"£{acc['starting_balance']:,.0f}", "paper", "muted"),
               ("Risk / trade", f"{acc['risk_per_trade_pct']}%",
                f"£{acc['starting_balance'] * acc['risk_per_trade_pct'] / 100:.2f}", "muted"),
               ("Daily loss limit", f"{acc['daily_loss_limit_pct']}%", "then pause", "muted"),
               ("Max positions", f"{acc['max_open_positions']}", f"leverage {acc['max_leverage']}x", "muted")])
        lcc = settings.raw.get("loss_control", {})
        st.markdown("### Loss control")
        st.markdown(
            f"- Volatility in the top {1 - lcc.get('vol_high_rank', 0.8):.0%} of the year → "
            f"**{lcc.get('vol_high_mult', 0.5):g}×** risk (top {1 - lcc.get('vol_extreme_rank', 0.95):.0%} → "
            f"**{lcc.get('vol_extreme_mult', 0.25):g}×**)\n"
            f"- Account {lcc.get('dd_soft_pct', 3):g}% below peak → **{lcc.get('dd_soft_mult', 0.5):g}×**; "
            f"{lcc.get('dd_hard_pct', 6):g}% → **{lcc.get('dd_hard_mult', 0.25):g}×**\n"
            f"- {lcc.get('streak_losses', 4)} losses in a row → **{lcc.get('streak_mult', 0.5):g}×** until a win\n"
            f"- Max **{lcc.get('max_positions_per_group', 3)}** open positions per market group; total open risk "
            f"max **{lcc.get('max_open_risk_pct', 3):g}%**\n"
            f"- Never below **{lcc.get('min_mult', 0.25):g}×** normal risk")
        st.markdown(f"### Trade management: **{(settings.raw.get('trade_management') or {}).get('mode', 'base')}**")
        st.caption("base = strategy's own exits · breakeven = stop to breakeven at +1R · partial = bank half at "
                   "+1R, rest to breakeven. Compared on the same trades every Saturday (Strategies page).")
        st.markdown("### Filters (record-only until proven)")
        cvc = settings.raw.get("conviction", {})
        st.write(f"Confluence minimum: **{settings.raw.get('confluence', {}).get('min_score', 0)}** / 8 · "
                 f"minimum grade: **{cvc.get('min_grade') or 'off'}** · Time Machine votes at "
                 f"**{cvc.get('high_confluence', 6)}+/8** confluence")
        st.caption("Change these in config/settings.yaml once the Learning lab shows they help.")
        st.markdown("### Phone notifications")
        st.caption("Trade alerts and a daily summary go to the free ntfy app. Test: GitHub → Actions → "
                   "MarketPilot worker → Run workflow → `notify-test`.")

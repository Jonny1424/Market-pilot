"""Performance Analyst Bot — win rate, win/loss ratio, profit factor, expectancy,
drawdown, broken down by strategy, asset class, symbol and regime."""
from __future__ import annotations

import pandas as pd

from ..metrics import summarize


def closed_frame(trades: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame([t for t in trades if t["status"] == "closed"])
    if not df.empty:
        df["closed_at"] = pd.to_datetime(df["closed_at"], utc=True, format="mixed")
        df = df.sort_values("closed_at")
    return df


def report(trades: list[dict], starting_balance: float) -> dict:
    df = closed_frame(trades)
    overall = summarize(df["r_multiple"] if not df.empty else [])
    overall["pnl_gbp"] = float(df["pnl_gbp"].sum()) if not df.empty else 0.0
    overall["fees_gbp"] = float(df["fees_gbp"].sum()) if not df.empty else 0.0
    overall["return_pct"] = overall["pnl_gbp"] / starting_balance * 100
    breakdowns = {}
    for col in ("strategy", "asset_class", "symbol", "regime"):
        rows = []
        if not df.empty:
            for k, g in df.groupby(col):
                s = summarize(g["r_multiple"])
                s[col] = k
                s["pnl_gbp"] = float(g["pnl_gbp"].sum())
                rows.append(s)
        breakdowns[col] = pd.DataFrame(rows)
    return {"overall": overall, "by": breakdowns, "closed": df}


def realized_today(trades: list[dict], now: pd.Timestamp) -> tuple[float, int]:
    day = now.normalize()
    pnl = sum(float(t["pnl_gbp"] or 0) for t in trades
              if t["status"] == "closed" and pd.Timestamp(t["closed_at"]).tz_convert("UTC") >= day)
    since = now - pd.Timedelta(hours=24)
    opened = sum(1 for t in trades if t.get("signal_bar_time")
                 and pd.Timestamp(t["signal_bar_time"]) >= since)
    return pnl, opened


def retirement_review(store, settings) -> list[str]:
    """Retire approved strategies whose forward paper results are negative."""
    cfg, msgs = settings["retirement"], []
    df = closed_frame(store.trades())
    if df.empty:
        return msgs
    for s in store.strategies("approved"):
        g = df[(df["symbol"] == s["symbol"]) & (df["strategy"] == s["params"])]
        if len(g) >= cfg["min_live_trades"]:
            e = g["r_multiple"].mean()
            if e < cfg["retire_if_expectancy_below_r"]:
                store.save_strategy({**s, "status": "retired",
                                     "reasons": [f"forward paper expectancy {e:+.3f}R over {len(g)} trades"]})
                msgs.append(f"retired {s['params']} on {s['symbol']} ({e:+.3f}R, n={len(g)})")
    return msgs

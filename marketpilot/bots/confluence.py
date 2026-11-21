"""Confluence bot — scores how many independent factors agree with a trade.

Educators such as Chart Champions build a trade case from several layers of
confluence (trend, key level, Fibonacci zone, momentum, volume...) and treat
indicators as the last confirmation. This bot turns that checklist into an
objective 0-8 score for every signal, records it with every candidate, backtest
trade and shadow trade, and lets the data answer the real question:
**do high-confluence trades actually win more often?**

The score is NOT used as a trade filter until the evidence says it helps
(settings: confluence.min_score, default 0 = record only).
"""
from __future__ import annotations

import math

FACTORS = [
    ("trend", "Long-term trend agrees (price vs 200-day average)"),
    ("swing_trend", "Medium-term trend agrees (20 vs 50 EMA)"),
    ("momentum", "Momentum agrees (RSI14 on the right side of 50)"),
    ("level", "At a key level (within 1 ATR of a 20-bar high/low, 52-week high/low or 200-day average)"),
    ("fib", "In the 38.2%-61.8% Fibonacci retracement zone of the last 60-bar swing"),
    ("volume", "Volume confirms (above 1.2x its 20-day average)"),
    ("calm", "Volatility not extreme (ATR below its 80th percentile)"),
    ("market", "Benchmark market trend agrees (e.g. SPY / BTC above 200-day average)"),
]
MAX_SCORE = len(FACTORS)


def _ok(x) -> bool:
    try:
        return x is not None and not math.isnan(float(x))
    except (TypeError, ValueError):
        return False


def score(row, direction: int, market_ok: bool | None = None) -> tuple[int, list[str]]:
    """Return (score 0-8, list of factor keys that agreed) for a long (+1) or short (-1)."""
    d, hits = direction, []
    c = row["close"]
    if _ok(row.get("sma200")) and d * (c - row["sma200"]) > 0:
        hits.append("trend")
    if _ok(row.get("ema50")) and d * (row["ema20"] - row["ema50"]) > 0:
        hits.append("swing_trend")
    if _ok(row.get("rsi14")) and d * (row["rsi14"] - 50) > 0:
        hits.append("momentum")
    atr = row.get("atr14")
    if _ok(atr) and atr > 0:
        levels = [row.get(k) for k in ("dc_high", "dc_low", "hi252", "lo252", "sma200")]
        if any(_ok(lv) and abs(c - lv) <= atr for lv in levels):
            hits.append("level")
    hi, lo = row.get("range_hi60"), row.get("range_lo60")
    if _ok(hi) and _ok(lo) and hi > lo:
        pos = (c - lo) / (hi - lo)                 # 0 = swing low, 1 = swing high
        retrace = 1 - pos if d == 1 else pos       # how far price has pulled back
        if 0.382 <= retrace <= 0.618:
            hits.append("fib")
    vol, vavg = row.get("volume"), row.get("vol_avg20")
    if _ok(vol) and _ok(vavg) and vavg > 0 and vol > 1.2 * vavg:
        hits.append("volume")
    rank = row.get("atr_pct_rank")
    if _ok(rank) and rank < 0.8:
        hits.append("calm")
    if market_ok is True:          # None = this market group has no benchmark filter
        hits.append("market")
    return len(hits), hits


def bucket(s: int) -> str:
    return "low (0-3)" if s <= 3 else ("medium (4-5)" if s <= 5 else "high (6-8)")


def describe(hits: list[str]) -> str:
    names = dict(FACTORS)
    return "; ".join(names[h].split(" (")[0] for h in hits) or "none"

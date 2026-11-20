"""Market Scanner Bot: trend, volatility, momentum and liquidity snapshot per symbol."""
from __future__ import annotations

import numpy as np

MIN_BARS = 250
MIN_DOLLAR_VOL = {"stocks": 5_000_000, "international": 2_000_000, "bonds": 5_000_000,
                  "commodities": 2_000_000, "crypto": 1_000_000}


def scan(symbol: str, asset_class: str, ind) -> dict:
    last = ind.iloc[-1]
    reasons = []
    if len(ind) < MIN_BARS:
        reasons.append(f"only {len(ind)} bars of history (< {MIN_BARS})")
    atr_pct = float(last["atr_pct"]) if np.isfinite(last["atr_pct"]) else np.nan
    if not np.isfinite(atr_pct) or atr_pct <= 0:
        reasons.append("ATR unavailable")
    elif atr_pct > 0.15:
        reasons.append(f"volatility too extreme (ATR {atr_pct:.1%} of price)")
    liq = float(last["dollar_vol"]) if np.isfinite(last["dollar_vol"]) else 0.0
    floor = MIN_DOLLAR_VOL.get(asset_class)  # forex volume on Yahoo is not meaningful
    if floor and symbol.endswith(".L"):
        floor *= 100                          # London prices are in pence
    if floor and liq < floor:
        reasons.append(f"illiquid (avg traded value {liq:,.0f} < {floor:,.0f})")
    trend = "up" if last["ema20"] > last["ema50"] else "down"
    return {
        "symbol": symbol, "asset_class": asset_class, "ok": not reasons, "reasons": reasons,
        "price": float(last["close"]), "trend": trend,
        "atr_pct": atr_pct, "momentum_20": float(last["roc20"]) if np.isfinite(last["roc20"]) else None,
        "rsi14": float(last["rsi14"]), "efficiency": float(last["er20"]) if np.isfinite(last["er20"]) else None,
        "bar_time": str(ind.index[-1]),
    }

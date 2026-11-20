"""Causal indicators: every value at row i uses only rows <= i (no look-ahead)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(),
                    (df["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100)


def efficiency_ratio(s: pd.Series, n: int = 20) -> pd.Series:
    """Kaufman efficiency ratio: 1 = straight-line trend, 0 = pure noise."""
    change = (s - s.shift(n)).abs()
    path = s.diff().abs().rolling(n, min_periods=n).sum()
    return change / path.replace(0, np.nan)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c = out["close"]
    out["ema20"], out["ema50"] = ema(c, 20), ema(c, 50)
    for n in (5, 10, 20, 50, 200):
        out[f"sma{n}"] = sma(c, n)
    out["roc252"] = c.pct_change(252)
    out["atr14"] = atr(out, 14)
    out["atr_pct"] = out["atr14"] / c
    out["atr_pct_rank"] = out["atr_pct"].rolling(250, min_periods=50).rank(pct=True)
    out["rsi2"], out["rsi14"] = rsi(c, 2), rsi(c, 14)
    out["er20"] = efficiency_ratio(c, 20)
    out["roc20"] = c.pct_change(20)
    # Donchian channel of the PREVIOUS 20 bars (shifted so today's bar can break it)
    out["dc_high"] = out["high"].rolling(20).max().shift(1)
    out["dc_low"] = out["low"].rolling(20).min().shift(1)
    mid = sma(c, 20)
    sd = c.rolling(20, min_periods=20).std()
    out["bb_mid"], out["bb_up"], out["bb_dn"] = mid, mid + 2 * sd, mid - 2 * sd
    out["dollar_vol"] = (c * out["volume"]).rolling(20, min_periods=5).mean()
    return out

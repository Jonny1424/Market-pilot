"""Strategy rules. The same `signal()` is used by the backtester and by live
paper trading, so what was tested is exactly what trades."""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Signal:
    direction: int          # +1 long, -1 short
    stop_atr: float         # stop distance in ATR multiples
    target_atr: float       # target distance in ATR multiples
    max_hold: int           # bars
    note: str = ""


class Strategy:
    name = "base"
    family = "base"
    regimes: frozenset[str] = frozenset()
    grid: dict[str, list] = {}

    def __init__(self, **params):
        defaults = {k: v[0] for k, v in self.grid.items()}
        defaults.update(params)
        self.params = {k: float(v) for k, v in defaults.items()}

    @property
    def key(self) -> str:
        return self.name + "(" + ",".join(f"{k}={v:g}" for k, v in sorted(self.params.items())) + ")"

    @classmethod
    def variants(cls) -> list["Strategy"]:
        keys = list(cls.grid)
        return [cls(**dict(zip(keys, combo))) for combo in itertools.product(*cls.grid.values())]

    def signal(self, prev: pd.Series, row: pd.Series) -> Signal | None:
        raise NotImplementedError


class TrendFollowing(Strategy):
    """EMA 20/50 crossover, only when price action is efficient (trending)."""
    name, family = "trend_ema", "trend-following"
    regimes = frozenset({"trending", "volatile", "ranging"})
    grid = {"stop": [2.0, 3.0], "target": [4.0, 6.0], "er_min": [0.0, 0.25]}

    def signal(self, prev, row):
        if pd.isna(row["ema50"]) or pd.isna(row["er20"]) or row["er20"] < self.params["er_min"]:
            return None
        p = self.params
        if prev["ema20"] <= prev["ema50"] and row["ema20"] > row["ema50"]:
            return Signal(1, p["stop"], p["target"], 60, "EMA20 crossed above EMA50")
        if prev["ema20"] >= prev["ema50"] and row["ema20"] < row["ema50"]:
            return Signal(-1, p["stop"], p["target"], 60, "EMA20 crossed below EMA50")
        return None


class Breakout(Strategy):
    """20-bar Donchian channel breakout."""
    name, family = "breakout_donchian", "breakout"
    regimes = frozenset({"trending", "volatile"})
    grid = {"stop": [1.5, 2.5], "target": [3.0, 5.0]}

    def signal(self, prev, row):
        if pd.isna(row["dc_high"]) or pd.isna(row["atr14"]):
            return None
        p = self.params
        if row["close"] > row["dc_high"] and prev["close"] <= prev["dc_high"]:
            return Signal(1, p["stop"], p["target"], 40, "close broke 20-bar high")
        if row["close"] < row["dc_low"] and prev["close"] >= prev["dc_low"]:
            return Signal(-1, p["stop"], p["target"], 40, "close broke 20-bar low")
        return None


class MeanReversion(Strategy):
    """RSI(2) pullback in the direction of the 200-bar trend.
    Naturally high win rate (small target, wider stop) — watch expectancy."""
    name, family = "mean_reversion_rsi2", "mean-reversion"
    regimes = frozenset({"ranging", "trending"})
    grid = {"stop": [2.5, 3.5], "target": [1.0, 1.5], "rsi_lo": [10]}

    def signal(self, prev, row):
        if pd.isna(row["sma200"]):
            return None
        p = self.params
        if row["close"] > row["sma200"] and row["rsi2"] < p["rsi_lo"]:
            return Signal(1, p["stop"], p["target"], 10, f"RSI2 {row['rsi2']:.0f} above SMA200")
        if row["close"] < row["sma200"] and row["rsi2"] > 100 - p["rsi_lo"]:
            return Signal(-1, p["stop"], p["target"], 10, f"RSI2 {row['rsi2']:.0f} below SMA200")
        return None


class MomentumPullback(Strategy):
    """Buy a pullback inside an established uptrend (EMA20 > EMA50 and 20-bar
    momentum positive) when RSI14 recovers back through the threshold."""
    name, family = "momentum_pullback", "momentum"
    regimes = frozenset({"trending", "ranging"})
    grid = {"stop": [2.0, 3.0], "target": [3.0, 4.5], "rsi": [45]}

    def signal(self, prev, row):
        if pd.isna(row["ema50"]) or pd.isna(row["roc20"]):
            return None
        p, lvl = self.params, self.params["rsi"]
        if row["ema20"] > row["ema50"] and row["roc20"] > 0 and prev["rsi14"] < lvl <= row["rsi14"]:
            return Signal(1, p["stop"], p["target"], 30, "uptrend pullback, RSI14 recovered")
        if row["ema20"] < row["ema50"] and row["roc20"] < 0 and prev["rsi14"] > 100 - lvl >= row["rsi14"]:
            return Signal(-1, p["stop"], p["target"], 30, "downtrend bounce, RSI14 rolled over")
        return None


ALL_STRATEGIES: list[type[Strategy]] = [TrendFollowing, Breakout, MeanReversion, MomentumPullback]


def strategy_from_key(key: str) -> Strategy:
    name, args = key.split("(", 1)
    cls = next(c for c in ALL_STRATEGIES if c.name == name)
    params = {}
    for part in filter(None, args.rstrip(")").split(",")):
        k, v = part.split("=")
        params[k] = float(v)
    return cls(**params)


# Asset classes where shorting is allowed in the simulation (spot stocks/crypto
# with no leverage are long-only; forex pairs are naturally two-way).
SHORTABLE = {"forex"}

"""Strategy rules. The same `signal()` is used by the backtester and by live
paper trading, so what was tested is exactly what trades.

v0.5 changes (see docs/RESEARCH.md):
  * new TrendSMA200 — the 200-day / 12-month trend rule (Faber 2007; Moskowitz,
    Ooi & Pedersen 2012; Hurst, Ooi & Pedersen 2017)
  * MeanReversion now follows Connors' published rules: buy RSI(2) dips below the
    5-day average in an uptrend, EXIT on a close back above the 5-day average
  * trend/breakout strategies can use a chandelier trailing stop instead of a
    fixed target — offered as variants, and walk-forward testing picks
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import pandas as pd

NO_TARGET = 99.0   # target this many ATRs away = effectively no fixed target


@dataclass(frozen=True)
class Signal:
    direction: int            # +1 long, -1 short
    stop_atr: float           # initial stop distance in ATR multiples
    target_atr: float         # target distance in ATR multiples (NO_TARGET = none)
    max_hold: int             # bars
    note: str = ""
    trail_atr: float = 0.0    # >0: chandelier trailing stop, ATRs behind best close
    exit_ma: int = 0          # >0: close-based exit vs SMA(exit_ma)
    exit_on: str = ""         # "revert": exit when close crosses MA in our favour
                              # "break":  exit when close crosses MA against us


class Strategy:
    name = "base"
    family = "base"
    regimes: frozenset[str] = frozenset()
    grid: dict[str, list] = {}

    def __init__(self, **params):
        defaults = {k: v[0] for k, v in self.grid.items()}
        defaults.update({k: v for k, v in params.items() if k in self.grid})
        self.params = {k: float(v) for k, v in defaults.items()}

    @property
    def key(self) -> str:
        return self.name + "(" + ",".join(f"{k}={v:g}" for k, v in sorted(self.params.items())) + ")"

    @classmethod
    def variants(cls) -> list["Strategy"]:
        keys = list(cls.grid)
        out = [cls(**dict(zip(keys, combo))) for combo in itertools.product(*cls.grid.values())]
        # a fixed-target strategy with no target and no trail would never exit early
        return [v for v in out if not (v.params.get("target", 0) >= NO_TARGET and v.params.get("trail", 1) == 0)]

    def signal(self, prev, row) -> Signal | None:
        raise NotImplementedError


def _na(*vals) -> bool:
    return any(v is None or pd.isna(v) for v in vals)


class TrendFollowing(Strategy):
    """EMA 20/50 crossover."""
    name, family = "trend_ema", "trend-following"
    regimes = frozenset({"trending", "volatile", "ranging"})
    grid = {"stop": [2.0, 3.0], "target": [5.0, NO_TARGET], "trail": [0.0, 3.0]}

    def signal(self, prev, row):
        if _na(row["ema50"], prev["ema50"], row["atr14"]):
            return None
        p = self.params
        if p["target"] >= NO_TARGET and p["trail"] == 0:
            return None  # no target and no trail = no exit other than time: skip variant
        if prev["ema20"] <= prev["ema50"] and row["ema20"] > row["ema50"]:
            return Signal(1, p["stop"], p["target"], 60, "EMA20 crossed above EMA50", trail_atr=p["trail"])
        if prev["ema20"] >= prev["ema50"] and row["ema20"] < row["ema50"]:
            return Signal(-1, p["stop"], p["target"], 60, "EMA20 crossed below EMA50", trail_atr=p["trail"])
        return None


class Breakout(Strategy):
    """20-bar Donchian channel breakout."""
    name, family = "breakout_donchian", "breakout"
    regimes = frozenset({"trending", "volatile"})
    grid = {"stop": [1.5, 2.5], "target": [4.0, NO_TARGET], "trail": [0.0, 3.0]}

    def signal(self, prev, row):
        if _na(row["dc_high"], prev["dc_high"], row["atr14"]):
            return None
        p = self.params
        if p["target"] >= NO_TARGET and p["trail"] == 0:
            return None
        if row["close"] > row["dc_high"] and prev["close"] <= prev["dc_high"]:
            return Signal(1, p["stop"], p["target"], 40, "close broke 20-bar high", trail_atr=p["trail"])
        if row["close"] < row["dc_low"] and prev["close"] >= prev["dc_low"]:
            return Signal(-1, p["stop"], p["target"], 40, "close broke 20-bar low", trail_atr=p["trail"])
        return None


class MeanReversion(Strategy):
    """Connors RSI(2): in a long-term uptrend (close > SMA200), buy when RSI(2) is
    oversold and price is below its 5-day average; exit on a close back above the
    5-day average, after 10 bars, or at a wide protective stop."""
    name, family = "mean_reversion_rsi2", "mean-reversion"
    regimes = frozenset({"ranging", "trending", "volatile"})
    grid = {"rsi_lo": [5.0, 10.0], "stop": [3.0, 4.0]}

    def signal(self, prev, row):
        if _na(row["sma200"], row["sma5"], row["atr14"]):
            return None
        p = self.params
        if row["close"] > row["sma200"] and row["close"] < row["sma5"] and row["rsi2"] < p["rsi_lo"]:
            return Signal(1, p["stop"], NO_TARGET, 10, f"RSI2 {row['rsi2']:.0f} dip in uptrend",
                          exit_ma=5, exit_on="revert")
        if row["close"] < row["sma200"] and row["close"] > row["sma5"] and row["rsi2"] > 100 - p["rsi_lo"]:
            return Signal(-1, p["stop"], NO_TARGET, 10, f"RSI2 {row['rsi2']:.0f} pop in downtrend",
                          exit_ma=5, exit_on="revert")
        return None


class MomentumPullback(Strategy):
    """Buy a pullback inside an established uptrend (EMA20 > EMA50 and 20-bar
    momentum positive) when RSI14 recovers back through the threshold."""
    name, family = "momentum_pullback", "momentum"
    regimes = frozenset({"trending", "ranging"})
    grid = {"stop": [2.0, 3.0], "target": [3.0, 4.5], "rsi": [45]}

    def signal(self, prev, row):
        if _na(row["ema50"], row["roc20"], prev["rsi14"]):
            return None
        p, lvl = self.params, self.params["rsi"]
        if row["ema20"] > row["ema50"] and row["roc20"] > 0 and prev["rsi14"] < lvl <= row["rsi14"]:
            return Signal(1, p["stop"], p["target"], 30, "uptrend pullback, RSI14 recovered")
        if row["ema20"] < row["ema50"] and row["roc20"] < 0 and prev["rsi14"] > 100 - lvl >= row["rsi14"]:
            return Signal(-1, p["stop"], p["target"], 30, "downtrend bounce, RSI14 rolled over")
        return None


class TrendSMA200(Strategy):
    """Long-term trend (time-series momentum): enter when price closes above its
    200-day average AND the 12-month return is positive; exit on a close back
    below the 200-day average (or a wide protective / trailing stop)."""
    name, family = "trend_sma200", "time-series momentum"
    regimes = frozenset({"trending", "ranging", "volatile"})
    grid = {"stop": [3.0, 5.0], "trail": [0.0, 4.0]}

    def signal(self, prev, row):
        if _na(row["sma200"], prev["sma200"], row["roc252"], row["atr14"]):
            return None
        p = self.params
        if prev["close"] <= prev["sma200"] and row["close"] > row["sma200"] and row["roc252"] > 0:
            return Signal(1, p["stop"], NO_TARGET, 250, "closed above 200-day avg, 12m return > 0",
                          trail_atr=p["trail"], exit_ma=200, exit_on="break")
        if prev["close"] >= prev["sma200"] and row["close"] < row["sma200"] and row["roc252"] < 0:
            return Signal(-1, p["stop"], NO_TARGET, 250, "closed below 200-day avg, 12m return < 0",
                          trail_atr=p["trail"], exit_ma=200, exit_on="break")
        return None


class BollingerReversion(Strategy):
    """Bollinger Band pullback: in an uptrend (close > SMA200), buy a close below
    the lower band (2 standard deviations under the 20-day average); exit when
    price closes back above the 20-day average (the middle band)."""
    name, family = "bollinger_reversion", "mean-reversion"
    regimes = frozenset({"ranging", "trending", "volatile"})
    grid = {"stop": [2.5, 3.5]}

    def signal(self, prev, row):
        if _na(row["bb_dn"], row["sma200"], row["atr14"]):
            return None
        p = self.params
        if row["close"] < row["bb_dn"] and row["close"] > row["sma200"]:
            return Signal(1, p["stop"], NO_TARGET, 15, "close below lower Bollinger band in uptrend",
                          exit_ma=20, exit_on="revert")
        if row["close"] > row["bb_up"] and row["close"] < row["sma200"]:
            return Signal(-1, p["stop"], NO_TARGET, 15, "close above upper Bollinger band in downtrend",
                          exit_ma=20, exit_on="revert")
        return None


class Turtle55(Strategy):
    """Turtle-style breakout: buy a close above the highest high of the last 55
    bars; protect with a 2-ATR stop that trails behind the best close."""
    name, family = "turtle_55", "breakout"
    regimes = frozenset({"trending", "volatile", "ranging"})
    grid = {"stop": [2.0, 3.0], "trail": [3.0, 4.5]}

    def signal(self, prev, row):
        if _na(row["dc55_high"], prev["dc55_high"], row["atr14"]):
            return None
        p = self.params
        if row["close"] > row["dc55_high"] and prev["close"] <= prev["dc55_high"]:
            return Signal(1, p["stop"], NO_TARGET, 120, "close broke 55-bar high", trail_atr=p["trail"])
        if row["close"] < row["dc55_low"] and prev["close"] >= prev["dc55_low"]:
            return Signal(-1, p["stop"], NO_TARGET, 120, "close broke 55-bar low", trail_atr=p["trail"])
        return None


class MACDTrend(Strategy):
    """MACD crossover in the direction of the 200-day trend; exit on a close back
    through the 50-day average or a trailing stop."""
    name, family = "macd_trend", "trend-following"
    regimes = frozenset({"trending", "ranging", "volatile"})
    grid = {"stop": [2.0, 3.0], "trail": [0.0, 3.0]}

    def signal(self, prev, row):
        if _na(row["macd_sig"], prev["macd_sig"], row["sma200"], row["sma50"], row["atr14"]):
            return None
        p = self.params
        if prev["macd"] <= prev["macd_sig"] and row["macd"] > row["macd_sig"] and row["close"] > row["sma200"]:
            return Signal(1, p["stop"], NO_TARGET, 80, "MACD crossed above signal, above 200-day avg",
                          trail_atr=p["trail"], exit_ma=50, exit_on="break")
        if prev["macd"] >= prev["macd_sig"] and row["macd"] < row["macd_sig"] and row["close"] < row["sma200"]:
            return Signal(-1, p["stop"], NO_TARGET, 80, "MACD crossed below signal, below 200-day avg",
                          trail_atr=p["trail"], exit_ma=50, exit_on="break")
        return None


class High52Week(Strategy):
    """52-week-high momentum (George & Hwang 2004): buy when price closes at a new
    52-week high while the 50-day average is above the 200-day; exit on a close
    below the 50-day average or a trailing stop."""
    name, family = "high_52week", "momentum"
    regimes = frozenset({"trending", "ranging", "volatile"})
    grid = {"stop": [2.5, 3.5], "trail": [0.0, 4.0]}

    def signal(self, prev, row):
        if _na(row["hi252"], prev["hi252"], row["sma200"], row["sma50"], row["atr14"]):
            return None
        p = self.params
        if row["close"] >= prev["hi252"] and prev["close"] < prev["hi252"] and row["sma50"] > row["sma200"]:
            return Signal(1, p["stop"], NO_TARGET, 120, "new 52-week high in an uptrend",
                          trail_atr=p["trail"], exit_ma=50, exit_on="break")
        if row["close"] <= prev["lo252"] and prev["close"] > prev["lo252"] and row["sma50"] < row["sma200"]:
            return Signal(-1, p["stop"], NO_TARGET, 120, "new 52-week low in a downtrend",
                          trail_atr=p["trail"], exit_ma=50, exit_on="break")
        return None


ALL_STRATEGIES: list[type[Strategy]] = [TrendSMA200, TrendFollowing, Breakout, MeanReversion, MomentumPullback,
                                        BollingerReversion, Turtle55, MACDTrend, High52Week]


def strategy_from_key(key: str) -> Strategy:
    name, args = key.split("(", 1)
    cls = next(c for c in ALL_STRATEGIES if c.name == name)
    params = {}
    for part in filter(None, args.rstrip(")").split(",")):
        k, v = part.split("=")
        params[k] = float(v)
    return cls(**params)


# Asset classes where shorting is allowed in the simulation (spot ETFs/crypto
# with no leverage are long-only; forex pairs are naturally two-way).
SHORTABLE = {"forex"}

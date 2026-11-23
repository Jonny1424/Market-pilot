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
    stop_swing: bool = False  # True: stop goes beyond the recent 10-bar swing low/high
                              # (market-structure stop, as discretionary traders place it)


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
    grid = {"stop": [1.5, 2.5], "target": [4.0, NO_TARGET], "trail": [0.0, 3.0], "swing": [0, 1]}

    def signal(self, prev, row):
        if _na(row["dc_high"], prev["dc_high"], row["atr14"]):
            return None
        p = self.params
        if p["target"] >= NO_TARGET and p["trail"] == 0:
            return None
        if row["close"] > row["dc_high"] and prev["close"] <= prev["dc_high"]:
            return Signal(1, p["stop"], p["target"], 40, "close broke 20-bar high", trail_atr=p["trail"],
                          stop_swing=bool(p.get("swing", 0)))
        if row["close"] < row["dc_low"] and prev["close"] >= prev["dc_low"]:
            return Signal(-1, p["stop"], p["target"], 40, "close broke 20-bar low", trail_atr=p["trail"],
                          stop_swing=bool(p.get("swing", 0)))
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
    grid = {"stop": [2.0, 3.0], "target": [3.0, 4.5], "rsi": [45], "swing": [0, 1]}

    def signal(self, prev, row):
        if _na(row["ema50"], row["roc20"], prev["rsi14"]):
            return None
        p, lvl = self.params, self.params["rsi"]
        if row["ema20"] > row["ema50"] and row["roc20"] > 0 and prev["rsi14"] < lvl <= row["rsi14"]:
            return Signal(1, p["stop"], p["target"], 30, "uptrend pullback, RSI14 recovered",
                          stop_swing=bool(p.get("swing", 0)))
        if row["ema20"] < row["ema50"] and row["roc20"] < 0 and prev["rsi14"] > 100 - lvl >= row["rsi14"]:
            return Signal(-1, p["stop"], p["target"], 30, "downtrend bounce, RSI14 rolled over",
                          stop_swing=bool(p.get("swing", 0)))
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
    grid = {"stop": [2.0, 3.0], "trail": [3.0, 4.5], "swing": [0, 1]}

    def signal(self, prev, row):
        if _na(row["dc55_high"], prev["dc55_high"], row["atr14"]):
            return None
        p = self.params
        if row["close"] > row["dc55_high"] and prev["close"] <= prev["dc55_high"]:
            return Signal(1, p["stop"], NO_TARGET, 120, "close broke 55-bar high", trail_atr=p["trail"],
                          stop_swing=bool(p.get("swing", 0)))
        if row["close"] < row["dc55_low"] and prev["close"] >= prev["dc55_low"]:
            return Signal(-1, p["stop"], NO_TARGET, 120, "close broke 55-bar low", trail_atr=p["trail"],
                          stop_swing=bool(p.get("swing", 0)))
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


class CCUBottom(Strategy):
    """"3 checkmarks" capitulation buy, as published by Steve Courtney (Crypto Crew
    University) for Bitcoin on the 2-day chart: price below its 200 MA, RSI(14)
    below 26 and both Stochastic RSI lines below 20. Tested here on daily bars for
    every market; exit on a close back above the 20-day average, a wide stop or
    after 60 bars. Long only (it is a bottom-fishing signal)."""
    name, family = "ccu_3checkmarks", "capitulation reversal"
    regimes = frozenset({"ranging", "trending", "volatile"})
    grid = {"rsi": [26.0, 30.0], "stop": [3.0, 4.5]}

    def signal(self, prev, row):
        if _na(row["sma200"], row["rsi14"], row["stoch_k"], row["stoch_d"], row["atr14"]):
            return None
        p = self.params
        fresh = not (prev["rsi14"] < p["rsi"] and prev["stoch_k"] < 20 and prev["stoch_d"] < 20)
        if (row["close"] < row["sma200"] and row["rsi14"] < p["rsi"] and row["stoch_k"] < 20
                and row["stoch_d"] < 20 and fresh):
            return Signal(1, p["stop"], NO_TARGET, 60, "3 checkmarks: below 200 MA, RSI & StochRSI oversold",
                          exit_ma=20, exit_on="revert")
        return None


# ---- v0.13: published strategies added to the library -----------------------
# Each one still has to pass the same walk-forward approval gate before it trades.

class GoldenCross(Strategy):
    """50/200-day moving-average cross (the "golden cross" / "death cross"), the
    best-known long-term trend rule in technical-analysis texts. Enter when the
    50-day average crosses the 200-day; exit on a trailing stop or a close back
    through the 200-day average."""
    name, family = "golden_cross", "trend-following"
    regimes = frozenset({"trending", "ranging", "volatile"})
    grid = {"stop": [3.0, 4.0], "trail": [0.0, 4.0]}

    def signal(self, prev, row):
        if _na(row["sma50"], prev["sma50"], row["sma200"], prev["sma200"], row["atr14"]):
            return None
        p = self.params
        if prev["sma50"] <= prev["sma200"] and row["sma50"] > row["sma200"]:
            return Signal(1, p["stop"], NO_TARGET, 250, "50-day avg crossed above 200-day (golden cross)",
                          trail_atr=p["trail"], exit_ma=200, exit_on="break")
        if prev["sma50"] >= prev["sma200"] and row["sma50"] < row["sma200"]:
            return Signal(-1, p["stop"], NO_TARGET, 250, "50-day avg crossed below 200-day (death cross)",
                          trail_atr=p["trail"], exit_ma=200, exit_on="break")
        return None


class NR7Breakout(Strategy):
    """Toby Crabel's NR7 (Day Trading with Short Term Price Patterns and Opening
    Range Breakout, 1990): after the narrowest-range bar of the last 7, trade the
    close that breaks out of that bar's range, in the direction of the 50-day trend.
    Quiet bars tend to come before larger moves."""
    name, family = "nr7_breakout", "volatility breakout"
    regimes = frozenset({"trending", "ranging", "volatile"})
    grid = {"stop": [1.5, 2.5], "target": [3.0, NO_TARGET], "trail": [0.0, 2.5]}

    def signal(self, prev, row):
        if _na(prev["nr7"], row["sma50"], row["atr14"]):
            return None
        p = self.params
        if p["target"] >= NO_TARGET and p["trail"] == 0:
            return None
        if prev["nr7"] < 1:
            return None
        if row["close"] > prev["high"] and row["close"] > row["sma50"]:
            return Signal(1, p["stop"], p["target"], 20, "broke out above an NR7 bar, above 50-day avg",
                          trail_atr=p["trail"])
        if row["close"] < prev["low"] and row["close"] < row["sma50"]:
            return Signal(-1, p["stop"], p["target"], 20, "broke down below an NR7 bar, below 50-day avg",
                          trail_atr=p["trail"])
        return None


class KeltnerBreakout(Strategy):
    """Keltner channel breakout (Chester Keltner, 1960, in its modern 20-day EMA
    +/- 2 ATR form popularised by Linda Raschke): a close outside the channel
    starts a trend trade; exit on a trailing stop or a close back through the
    20-day average."""
    name, family = "keltner_breakout", "trend-following"
    regimes = frozenset({"trending", "volatile"})
    grid = {"stop": [2.0, 3.0], "trail": [2.5, 4.0]}

    def signal(self, prev, row):
        if _na(row["kc_up"], prev["kc_up"], row["atr14"]):
            return None
        p = self.params
        if row["close"] > row["kc_up"] and prev["close"] <= prev["kc_up"]:
            return Signal(1, p["stop"], NO_TARGET, 60, "close broke above the Keltner channel",
                          trail_atr=p["trail"], exit_ma=20, exit_on="break")
        if row["close"] < row["kc_dn"] and prev["close"] >= prev["kc_dn"]:
            return Signal(-1, p["stop"], NO_TARGET, 60, "close broke below the Keltner channel",
                          trail_atr=p["trail"], exit_ma=20, exit_on="break")
        return None


LONG_ONLY = {"ccu_3checkmarks"}   # bottom-fishing signals have no short version


def _short_version(base: type[Strategy]) -> type[Strategy]:
    """The same rules, short trades only. Tested and approved separately from the
    long side, so a strategy's buys are never approved on the strength of its sells
    (or the other way round)."""
    def signal(self, prev, row):
        sig = base.signal(self, prev, row)
        return sig if sig is not None and sig.direction == -1 else None
    return type(base.__name__ + "Short", (base,), {
        "name": base.name + "_short", "family": base.family + " (short)", "short_only": True,
        "signal": signal, "__doc__": f"Short side only of {base.name}."})


LONG_STRATEGIES: list[type[Strategy]] = [TrendSMA200, TrendFollowing, Breakout, MeanReversion, MomentumPullback,
                                         BollingerReversion, Turtle55, MACDTrend, High52Week, CCUBottom,
                                         GoldenCross, NR7Breakout, KeltnerBreakout]
SHORT_STRATEGIES: list[type[Strategy]] = [_short_version(c) for c in LONG_STRATEGIES if c.name not in LONG_ONLY]
ALL_STRATEGIES: list[type[Strategy]] = LONG_STRATEGIES + SHORT_STRATEGIES


def strategy_from_key(key: str) -> Strategy:
    name, args = key.split("(", 1)
    cls = next(c for c in ALL_STRATEGIES if c.name == name)
    params = {}
    for part in filter(None, args.rstrip(")").split(",")):
        k, v = part.split("=")
        params[k] = float(v)
    return cls(**params)


# Asset classes where the ordinary strategies trade both ways (forex pairs are
# naturally two-way). Other classes are long-only unless listed under
# `shorting.classes` in settings, in which case the separate "_short" strategies
# are researched for them (v0.13).
SHORTABLE = {"forex"}


def short_classes(settings) -> set[str]:
    raw = getattr(settings, "raw", settings) or {}
    cfg = raw.get("shorting") or {}
    return set(cfg.get("classes") or []) if cfg.get("enabled", True) else set()


def strategies_for(asset_class: str, settings) -> list[type[Strategy]]:
    """Strategies researched for an asset class: the normal (long, or two-way for
    forex) set, plus the short-only versions where shorting is switched on."""
    out = list(LONG_STRATEGIES)
    if asset_class in short_classes(settings) and asset_class not in SHORTABLE:
        out += SHORT_STRATEGIES
    return out


def allows_short(asset_class: str, strategy, settings=None) -> bool:
    """May this strategy open a short in this asset class? Short-only strategies
    may, where shorting is switched on for the class (`settings` None = assume on)."""
    if asset_class in SHORTABLE:
        return True
    if getattr(strategy, "short_only", False):
        return settings is None or asset_class in short_classes(settings)
    return False

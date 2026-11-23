"""Market data providers.

All providers return a DataFrame indexed by UTC timestamps (bar OPEN time) with
columns open, high, low, close, volume — containing ONLY fully closed bars.
"""
from __future__ import annotations

import zlib
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

INTERVALS = {"15m": timedelta(minutes=15), "1h": timedelta(hours=1), "4h": timedelta(hours=4),
             "1d": timedelta(days=1)}

# v0.13: when is a DAILY bar finished? At its market's close (plus a few minutes
# for the data to settle), not 24 hours after the bar's date. US-listed ETFs close
# 16:00 New York, London-listed ETFs 16:30 London. Crypto and forex bars run to
# midnight UTC, so for them the old rule (bar date + 1 day) still applies.
SESSION_CLOSE = {"us": ("America/New_York", 16, 15), "lse": ("Europe/London", 16, 45)}


def session_of(symbol: str | None) -> str | None:
    if not symbol or symbol.endswith("=X") or "-USD" in symbol or "/" in symbol:
        return None          # forex / crypto: 24-hour markets, bar ends at midnight UTC
    if symbol.endswith(".L"):
        return "lse"
    if "." in symbol or symbol.startswith("^"):
        return None          # other venues / indices: keep the conservative rule
    return "us"


def bar_close_time(bar_ts: pd.Timestamp, interval: str, symbol: str | None = None) -> pd.Timestamp:
    """UTC time at which the bar stamped `bar_ts` is finished."""
    sess = session_of(symbol) if interval == "1d" else None
    if sess is None:
        return pd.Timestamp(bar_ts) + pd.Timedelta(interval_td(interval))
    tz, hh, mm = SESSION_CLOSE[sess]
    day = pd.Timestamp(bar_ts).tz_convert("UTC").date()
    return pd.Timestamp(datetime(day.year, day.month, day.day, hh, mm), tz=tz).tz_convert("UTC")


def interval_td(interval: str) -> timedelta:
    return INTERVALS[interval]


SESSION_OPEN = {"us": ("America/New_York", 9, 30), "lse": ("Europe/London", 8, 0)}
LATE_GRACE_24H = pd.Timedelta(hours=3)   # 24-hour markets: how long after the bar's close a signal stays fresh


def signal_is_late(bar_ts: pd.Timestamp, interval: str, symbol: str | None, now) -> bool:
    """v0.13.1: a signal is only tradable if the entry the backtest assumes (the next
    session's opening price) is still in the future. Exchange markets: the next
    weekday's open must not have happened yet. 24-hour markets: the decision must
    come within a few hours of the bar's close."""
    now = pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    sess = session_of(symbol) if interval == "1d" else None
    if sess is None:
        return now > bar_close_time(bar_ts, interval, symbol) + LATE_GRACE_24H
    tz, hh, mm = SESSION_OPEN[sess]
    day = pd.Timestamp(bar_ts).tz_convert("UTC").date() + timedelta(days=1)
    while day.weekday() >= 5:            # skip the weekend
        day += timedelta(days=1)
    nxt = pd.Timestamp(datetime(day.year, day.month, day.day, hh, mm), tz=tz).tz_convert("UTC")
    return now >= nxt


def drop_incomplete(df: pd.DataFrame, interval: str, now: datetime, symbol: str | None = None) -> pd.DataFrame:
    """Remove the in-progress bar so signals never use unfinished data. Daily bars
    of exchange-traded markets count as finished at that exchange's close."""
    if df.empty:
        return df
    now = pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    if interval != "1d" or session_of(symbol) is None:
        cutoff = now - pd.Timedelta(interval_td(interval))
        return df[df.index <= cutoff]
    # only the last couple of bars can be unfinished; check those individually
    keep = np.ones(len(df), dtype=bool)
    for k in range(max(0, len(df) - 3), len(df)):
        keep[k] = bar_close_time(df.index[k], interval, symbol) <= now
    return df[keep]


def quote_currency(symbol: str, asset_class: str) -> str:
    """Currency a symbol is priced in. London-listed (.L) ETFs are quoted in pence (GBp)."""
    if asset_class == "forex":
        return symbol.replace("=X", "")[3:6]
    if asset_class == "crypto":
        return symbol.split("-")[-1]
    if symbol.endswith(".L"):
        return "GBp"
    return "USD"


FX_FALLBACK = {"GBP": 1.0, "GBp": 0.01, "USD": 0.79, "EUR": 0.85, "JPY": 0.0052, "CHF": 0.90,
               "CAD": 0.56, "AUD": 0.51, "NZD": 0.46}


class DataProvider:
    name = "base"

    def history(self, symbol: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
        raise NotImplementedError

    def fx_to_gbp(self, currency: str, now: datetime) -> float:
        """Multiply an amount in `currency` by this to get GBP."""
        if currency in ("GBP", "GBp"):
            return FX_FALLBACK[currency]
        fallback = FX_FALLBACK.get(currency)
        pair = f"GBP{currency}=X"          # GBP per 1 unit of `currency` = 1 / (GBPxxx rate)
        try:
            df = self.history(pair, "1d", now - timedelta(days=10), now)
            px = float(df["close"].iloc[-1])
            if px > 0:
                return 1.0 / px
        except Exception:
            pass
        if fallback is None:
            raise RuntimeError(f"no GBP conversion for {currency}")
        return fallback


class YFinanceProvider(DataProvider):
    """Yahoo Finance via yfinance. Free and unofficial — fine for research and
    paper trading, not a substitute for a broker's bid/ask feed."""
    name = "yfinance"

    def __init__(self):
        # (symbol, interval) -> (start, end, frame). Lives for one worker run, so the
        # same market is never downloaded twice (e.g. SPY as a market AND as the filter).
        self._cache: dict[tuple[str, str], tuple] = {}

    @staticmethod
    def _clean(df: pd.DataFrame, interval: str, end, symbol: str | None = None) -> pd.DataFrame:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        idx = pd.DatetimeIndex(df.index)
        df.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        return drop_incomplete(df.dropna(), interval, end, symbol)

    def prefetch(self, symbols, interval, start, end) -> int:
        """Download many markets in ONE batched request (much faster than one by one).
        Anything missing or malformed simply falls back to a normal per-symbol fetch."""
        import yfinance as yf
        symbols = [s for s in dict.fromkeys(symbols) if (s, interval) not in self._cache]
        if len(symbols) < 2:
            return 0
        if interval != "1d":
            start = max(start, end - timedelta(days=729))
        try:
            raw = yf.download(symbols, start=start, end=end, interval=interval, auto_adjust=True,
                              progress=False, threads=True, group_by="ticker")
        except Exception:
            return 0
        if raw is None or raw.empty or not isinstance(raw.columns, pd.MultiIndex):
            return 0
        got = 0
        for sym in symbols:
            try:
                df = self._clean(raw[sym].copy(), interval, end, sym)
            except Exception:
                continue
            if len(df) > 0:
                self._cache[(sym, interval)] = (start, end, df)
                got += 1
        return got

    def history(self, symbol, interval, start, end):
        import yfinance as yf
        # Yahoo caps intraday history at ~730 days.
        if interval != "1d":
            start = max(start, end - timedelta(days=729))
        hit = self._cache.get((symbol, interval))
        if hit and hit[0] <= start + timedelta(days=3) and abs((hit[1] - end).total_seconds()) < 6 * 3600:
            df = hit[2]
            return df[df.index >= pd.Timestamp(start).tz_convert("UTC")].copy()
        df = yf.download(symbol, start=start, end=end, interval=interval,
                         auto_adjust=True, progress=False, threads=False)
        if df is None or df.empty:
            raise RuntimeError(f"no data returned for {symbol}")
        df = self._clean(df, interval, end, symbol)
        self._cache[(symbol, interval)] = (start, end, df)
        return df


class CCXTProvider(DataProvider):
    """Crypto OHLCV from an exchange's public API via ccxt (no keys needed).
    Map Yahoo-style symbols like BTC-USD to exchange pairs like BTC/USD."""
    name = "ccxt"

    def __init__(self, exchange: str = "kraken"):
        import ccxt
        self.ex = getattr(ccxt, exchange)({"enableRateLimit": True})

    def history(self, symbol, interval, start, end):
        pair = symbol.replace("-", "/")
        rows = self.ex.fetch_ohlcv(pair, timeframe=interval,
                                   since=int(start.timestamp() * 1000), limit=1000)
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df.index = pd.to_datetime(df.pop("ts"), unit="ms", utc=True)
        return drop_incomplete(df, interval, end, symbol)


class SyntheticProvider(DataProvider):
    """Deterministic regime-switching random walk for tests and the demo.
    Trend regimes persist for ~200 bars, so trend strategies can show an edge
    here that real markets may not offer. Demo results say nothing about live
    profitability — they only prove the pipeline works."""
    name = "synthetic"

    def __init__(self, seed: int = 7, start: datetime | None = None):
        self.seed = seed
        self.origin = start or datetime(2018, 1, 1, tzinfo=timezone.utc)
        self._cache: dict[tuple[str, str], pd.DataFrame] = {}

    def _series(self, symbol: str, interval: str) -> pd.DataFrame:
        key = (symbol, interval)
        if key in self._cache:
            return self._cache[key]
        step = interval_td(interval)
        n = int((datetime(2027, 6, 1, tzinfo=timezone.utc) - self.origin) / step)
        rng = np.random.default_rng(zlib.crc32(f"{self.seed}|{symbol}|{interval}".encode()))
        scale = {"1d": 0.015, "1h": 0.004, "4h": 0.008}[interval]
        # regimes: 0 ranging, 1 up-trend, 2 down-trend, 3 volatile
        regimes = np.repeat(rng.integers(0, 4, n // 200 + 1), 200)[:n]
        drift = np.select([regimes == 1, regimes == 2], [scale * 0.12, -scale * 0.12], 0.0)
        vol = np.where(regimes == 3, scale * 2.0, scale)
        rets = drift + vol * rng.standard_t(5, n) / np.sqrt(5 / 3)
        # mild mean reversion in ranging regimes
        base = 100 * (1 + rng.random()) * np.exp(np.cumsum(rets))
        close = base
        open_ = np.concatenate([[close[0]], close[:-1]]) * (1 + rng.normal(0, scale * 0.1, n))
        hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, scale * 0.5, n)))
        lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, scale * 0.5, n)))
        idx = pd.date_range(self.origin, periods=n, freq=step, tz="UTC")
        df = pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close,
                           "volume": rng.integers(1_000, 100_000, n).astype(float)}, index=idx)
        self._cache[key] = df
        return df

    def history(self, symbol, interval, start, end):
        df = self._series(symbol, interval)
        df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
        return drop_incomplete(df, interval, end)

    def fx_to_gbp(self, currency, now):
        return FX_FALLBACK[currency]


def make_provider(name: str) -> DataProvider:
    if name == "yfinance":
        return YFinanceProvider()
    if name == "synthetic":
        return SyntheticProvider()
    if name.startswith("ccxt"):
        return CCXTProvider(name.split(":", 1)[1] if ":" in name else "kraken")
    raise ValueError(name)

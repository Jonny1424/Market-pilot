"""Time Machine bot — the honest odds of a trade before it is taken.

Two views, both built only from the market's own past price action:

1. Monte Carlo replay. The next `max_hold` bars are rebuilt 10,000 times by
   drawing real past bars of this market at random (each bar's open, high, low
   and close are taken RELATIVE to the previous close, so they rescale to
   today's price). Every replay walks the plan's stop and target exactly the
   way the paper broker does: stop checked before target inside a bar, gaps
   fill at the open. Result: how often the target is hit first, how often the
   stop is hit first, how often neither by the time limit, and the average R.
   Draws are taken in short blocks of consecutive days so volatility clusters
   (calm weeks, wild weeks) are kept rather than shuffled away.

2. Setup history. The same strategy is replayed over this market's history
   with the real backtester, so you see what happened every past time it fired.

This is a simulation, not a forecast. It ignores news and assumes the future
looks like the recent past. Its odds are saved with every trade so the Learning
Lab can check them against what really happened (calibration), and the odds
are only allowed to filter trades once they have proven reliable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

N_PATHS = 10_000
HISTORY_BARS = 750      # ~3 years of daily bars
BLOCK = 5               # consecutive bars per draw (keeps volatility clusters)
MAX_HORIZON = 40        # bars simulated at most (long trend trades are judged on the first 40)
NO_TARGET_ATR = 50.0    # a target this many ATRs away means "no fixed target"
PROXY_TARGET_R = 2.0    # ...then the time machine asks: +2R before the stop?


@dataclass
class Odds:
    p_target: float
    p_stop: float
    p_neither: float
    exp_r: float
    median_bars: float
    paths: int
    reward_risk: float
    hist_trades: int = 0
    hist_win_rate: float | None = None
    hist_exp_r: float | None = None

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}

    def describe(self) -> str:
        s = (f"time machine: target first {self.p_target:.0%}, stop first {self.p_stop:.0%}, "
             f"neither {self.p_neither:.0%}, avg {self.exp_r:+.2f}R")
        if self.hist_trades:
            s += f" | setup history {self.hist_trades} trades, win {self.hist_win_rate:.0%}"
        return s


def _relative_bars(df: pd.DataFrame) -> np.ndarray:
    """Past bars as multiples of the previous close: columns open, high, low, close."""
    d = df[["open", "high", "low", "close"]].astype(float).dropna().tail(HISTORY_BARS + 1)
    prev = d["close"].shift(1)
    rel = d[["open", "high", "low", "close"]].div(prev, axis=0).dropna()
    rel = rel[(rel > 0).all(axis=1)]
    # guard against bad ticks: high >= max(open, close), low <= min(open, close)
    arr = rel.to_numpy(copy=True)
    arr[:, 1] = np.maximum(arr[:, 1], arr[:, [0, 3]].max(axis=1))
    arr[:, 2] = np.minimum(arr[:, 2], arr[:, [0, 3]].min(axis=1))
    return arr


def simulate(df: pd.DataFrame, direction: int, entry: float, stop: float, target: float,
             max_hold: int, n_paths: int = N_PATHS, seed: int | None = 19, detail: bool = False):
    """Monte Carlo odds for one trade plan. `df` = this market's past OHLC bars.
    With detail=True returns (odds, {"closes": paths x bars, "r": R per path, "outcome": ...})."""
    rel = _relative_bars(df)
    risk = abs(entry - stop)
    if len(rel) < 60 or risk <= 0 or not np.isfinite(entry):
        return (None, {}) if detail else None
    h = int(max(1, min(max_hold, MAX_HORIZON)))
    rng = np.random.default_rng(seed)
    n_blocks = -(-h // BLOCK)
    starts = rng.integers(0, len(rel) - BLOCK, size=(n_paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(BLOCK)[None, None, :]).reshape(n_paths, -1)[:, :h]
    bars = rel[idx]                                   # (paths, h, 4)
    closes = entry * np.cumprod(bars[:, :, 3], axis=1)
    prev_close = np.concatenate([np.full((n_paths, 1), entry), closes[:, :-1]], axis=1)
    o, hi, lo = (prev_close * bars[:, :, k] for k in range(3))
    d = direction
    if d == 1:
        stop_hit, tgt_hit = lo <= stop, hi >= target
        stop_px = np.where(o <= stop, o, stop)        # gap through the stop fills at the open
        tgt_px = np.where(o >= target, o, target)
    else:
        stop_hit, tgt_hit = hi >= stop, lo <= target
        stop_px = np.where(o >= stop, o, stop)
        tgt_px = np.where(o <= target, o, target)
    big = h + 1
    first_stop = np.where(stop_hit.any(1), stop_hit.argmax(1), big)
    first_tgt = np.where(tgt_hit.any(1), tgt_hit.argmax(1), big)
    stop_first = (first_stop <= first_tgt) & (first_stop < big)   # same bar -> stop (conservative)
    tgt_first = (first_tgt < first_stop)
    neither = ~(stop_first | tgt_first)
    rows = np.arange(n_paths)
    exit_px = np.where(stop_first, stop_px[rows, np.minimum(first_stop, h - 1)],
                       np.where(tgt_first, tgt_px[rows, np.minimum(first_tgt, h - 1)], closes[:, -1]))
    r = d * (exit_px - entry) / risk
    bars_held = np.where(stop_first, first_stop + 1, np.where(tgt_first, first_tgt + 1, h))
    odds = Odds(p_target=float(tgt_first.mean()), p_stop=float(stop_first.mean()),
                p_neither=float(neither.mean()), exp_r=float(r.mean()),
                median_bars=float(np.median(bars_held)), paths=n_paths,
                reward_risk=float(abs(target - entry) / risk))
    if detail:
        outcome = np.where(tgt_first, "target", np.where(stop_first, "stop", "neither"))
        return odds, {"closes": closes, "r": r, "outcome": outcome, "bars_held": bars_held}
    return odds


def setup_history(ind: pd.DataFrame, strategy, cost, allow_short: bool) -> dict:
    """Replay the same strategy over this market's history with the real backtester."""
    from ..backtest import run_backtest
    try:
        trades = run_backtest(ind, strategy, cost, allow_short)
    except Exception:
        return {"hist_trades": 0}
    if not trades:
        return {"hist_trades": 0}
    r = np.array([t.r_multiple for t in trades], dtype=float)
    return {"hist_trades": int(len(r)), "hist_win_rate": float((r > 0).mean()), "hist_exp_r": float(r.mean())}


def odds_for(ind: pd.DataFrame, sig, cost, swing=None, strategy=None, allow_short: bool = True,
             n_paths: int = N_PATHS) -> Odds | None:
    """Odds for a fresh signal on the latest bar, using the same levels the broker will use."""
    from ..backtest import plan_levels
    row = ind.iloc[-1]
    atr = float(row["atr14"])
    if not np.isfinite(atr) or atr <= 0:
        return None
    entry = cost.market_fill(float(row["close"]), sig.direction)
    stop, target, _ = plan_levels(entry, atr, sig, cost, swing)
    if not getattr(sig, "target_atr", 0) or sig.target_atr >= NO_TARGET_ATR:
        # Trend strategies with no fixed target (they trail): the odds question
        # becomes "does it reach +2R before the stop?"
        target = entry + sig.direction * PROXY_TARGET_R * abs(entry - stop)
    odds = simulate(ind, sig.direction, entry, stop, target, sig.max_hold, n_paths=n_paths)
    if odds is not None and strategy is not None:
        for k, v in setup_history(ind, strategy, cost, allow_short).items():
            setattr(odds, k, v)
    return odds


def trade_fields(odds: Odds | None) -> dict:
    """Columns saved on mp_trades / mp_shadow_trades."""
    if odds is None:
        return {}
    return {"tm_p_target": round(odds.p_target, 4), "tm_p_stop": round(odds.p_stop, 4),
            "tm_exp_r": round(odds.exp_r, 4)}


def calibration(trades: list[dict]) -> pd.DataFrame:
    """Closed trades grouped by the time machine's forecast: did reality match the odds?"""
    rows = [t for t in trades if t.get("tm_p_target") is not None and t.get("r_multiple") is not None]
    cols = ["forecast", "trades", "forecast_target_pct", "actual_win_pct", "avg_forecast_r", "actual_avg_r"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    for c in ("tm_p_target", "tm_exp_r", "r_multiple"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["tm_p_target", "r_multiple"])
    bins = [0, 0.2, 0.35, 0.5, 0.65, 1.01]
    labels = ["under 20%", "20-35%", "35-50%", "50-65%", "over 65%"]
    df["forecast"] = pd.cut(df["tm_p_target"], bins=bins, labels=labels, right=False)
    out = []
    for lab, g in df.groupby("forecast", observed=True):
        out.append({"forecast": str(lab), "trades": int(len(g)),
                    "forecast_target_pct": float(g["tm_p_target"].mean()),
                    "actual_win_pct": float((g["r_multiple"] > 0).mean()),
                    "avg_forecast_r": float(g["tm_exp_r"].mean()),
                    "actual_avg_r": float(g["r_multiple"].mean())})
    return pd.DataFrame(out, columns=cols)

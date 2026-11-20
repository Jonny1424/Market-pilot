"""Backtesting Bot core: event-driven, cost-aware, no look-ahead.

Fill rules (identical to the paper broker):
  * signal on the CLOSE of bar i  -> market entry at the OPEN of bar i+1 (+ half spread + slippage)
  * stop hit if the bar trades through it; if the bar OPENS beyond the stop, fill at the open (gap)
  * if stop and target are both inside one bar, assume the STOP filled first (conservative)
  * take-profit fills as a limit at the target minus half spread
  * time exit at the close after `max_hold` bars
R = net P&L / planned risk, where planned risk includes estimated exit costs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .costs import CostModel
from .indicators import add_indicators
from .metrics import summarize
from .strategies import Signal, Strategy


@dataclass
class SimTrade:
    strategy: str
    direction: int
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    entry_price: float
    stop: float
    target: float
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: str = ""
    r_multiple: float = 0.0
    cost_r: float = 0.0


def plan_levels(entry_fill: float, atr: float, sig: Signal, cost: CostModel) -> tuple[float, float, float]:
    """Return (stop, target, risk_per_unit) for an entry."""
    d = sig.direction
    stop = entry_fill - d * sig.stop_atr * atr
    target = entry_fill + d * sig.target_atr * atr
    exit_cost = stop * (cost.half_spread + cost.slip) + cost.commission_per_unit(stop)
    risk_per_unit = abs(entry_fill - stop) + exit_cost + cost.commission_per_unit(entry_fill)
    return stop, target, risk_per_unit


def check_exit(bar: pd.Series, direction: int, stop: float, target: float,
               cost: CostModel) -> tuple[float, str] | None:
    """Evaluate one bar for a stop/target exit. Returns (fill_price, reason) or None."""
    o, h, l = bar["open"], bar["high"], bar["low"]
    side = -direction  # closing trade side
    if direction == 1:
        if o <= stop:
            return cost.market_fill(o, side), "stop (gap)"
        if l <= stop:
            return cost.market_fill(stop, side), "stop"
        if o >= target:
            return cost.limit_fill(o, side), "target (gap)"
        if h >= target:
            return cost.limit_fill(target, side), "target"
    else:
        if o >= stop:
            return cost.market_fill(o, side), "stop (gap)"
        if h >= stop:
            return cost.market_fill(stop, side), "stop"
        if o <= target:
            return cost.limit_fill(o, side), "target (gap)"
        if l <= target:
            return cost.limit_fill(target, side), "target"
    return None


def realized_r(direction: int, entry: float, exit_: float, risk_per_unit: float,
               cost: CostModel) -> tuple[float, float]:
    """Return (net R, cost in R) per unit."""
    commissions = cost.commission_per_unit(entry) + cost.commission_per_unit(exit_)
    gross = direction * (exit_ - entry)
    net = gross - commissions
    return net / risk_per_unit, 0.0 if risk_per_unit == 0 else commissions / risk_per_unit


def run_backtest(ind: pd.DataFrame, strategy: Strategy, cost: CostModel, allow_short: bool,
                 start: pd.Timestamp | None = None, end: pd.Timestamp | None = None,
                 regime: pd.Series | None = None) -> list[SimTrade]:
    """Simulate one strategy on one symbol. `ind` must come from add_indicators().
    Only signals with signal_time in [start, end) are taken; one position at a time."""
    idx = ind.index
    n = len(ind)
    rows = ind.to_dict("records")
    trades: list[SimTrade] = []
    lo = idx.searchsorted(start) if start is not None else 1
    hi = idx.searchsorted(end) if end is not None else n
    i = max(lo, 1)
    while i < min(hi, n - 1):
        row, prev = rows[i], rows[i - 1]
        sig = strategy.signal(prev, row)
        if sig is None or (sig.direction == -1 and not allow_short) or not np.isfinite(row["atr14"]):
            i += 1
            continue
        if regime is not None and regime.iloc[i] not in strategy.regimes:
            i += 1
            continue
        e = i + 1
        entry = cost.market_fill(rows[e]["open"], sig.direction)
        stop, target, rpu = plan_levels(entry, row["atr14"], sig, cost)
        t = SimTrade(strategy.key, sig.direction, idx[i], idx[e], entry, stop, target)
        j = e
        while j < n:
            bar = rows[j]
            hit = check_exit(bar, sig.direction, stop, target, cost) if j > e else \
                _entry_bar_exit(bar, sig.direction, stop, target, cost)
            if hit:
                t.exit_price, t.exit_reason = hit
                break
            if j - e >= sig.max_hold:
                t.exit_price, t.exit_reason = cost.market_fill(bar["close"], -sig.direction), "time"
                break
            j += 1
        if t.exit_price is None:  # still open at end of data: ignore
            break
        t.exit_time = idx[j]
        t.r_multiple, _ = realized_r(sig.direction, entry, t.exit_price, rpu, cost)
        t.cost_r = cost.round_trip_cost_frac() * entry / rpu  # estimated friction in R
        trades.append(t)
        i = j + 1  # next signal after exit bar
    return trades


def _entry_bar_exit(bar, direction, stop, target, cost):
    """On the entry bar we filled at the open, so only intrabar high/low matter."""
    h, l = bar["high"], bar["low"]
    if direction == 1:
        if l <= stop:
            return cost.market_fill(stop, -1), "stop"
        if h >= target:
            return cost.limit_fill(target, -1), "target"
    else:
        if h >= stop:
            return cost.market_fill(stop, 1), "stop"
        if l <= target:
            return cost.limit_fill(target, 1), "target"
    return None


def trades_frame(trades: list[SimTrade]) -> pd.DataFrame:
    return pd.DataFrame([asdict(t) for t in trades])


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    from .bots.regime import classify_series
    ind = add_indicators(df)
    return ind, classify_series(ind)


def walk_forward(df: pd.DataFrame, strategy_cls: type[Strategy], cost: CostModel,
                 allow_short: bool, folds: int = 4, holdout_fraction: float = 0.2,
                 min_is_trades: int = 5) -> dict:
    """Anchored walk-forward with a final untouched holdout.

    For each fold: pick the parameter variant with the best in-sample expectancy,
    then record its trades on the next (unseen) segment. Aggregated OOS trades
    are the evidence. The holdout is only evaluated once, with params chosen on
    all pre-holdout data."""
    ind, regime = prepare(df)
    n = len(ind)
    warm = min(250, n // 5)
    hold_start = int(n * (1 - holdout_fraction))
    edges = np.linspace(warm, hold_start, folds + 2).astype(int)
    variants = strategy_cls.variants()
    oos: list[SimTrade] = []
    chosen: list[str] = []
    ts = ind.index

    def best_on(a, b):
        best, best_e = None, -np.inf
        for v in variants:
            tr = run_backtest(ind, v, cost, allow_short, ts[a], ts[b], regime)
            if len(tr) < min_is_trades:
                continue
            e = np.mean([t.r_multiple for t in tr])
            if e > best_e:
                best, best_e = v, e
        return best

    for k in range(1, folds + 1):
        pick = best_on(edges[0], edges[k])
        if pick is None:
            continue
        chosen.append(pick.key)
        oos += run_backtest(ind, pick, cost, allow_short, ts[edges[k]], ts[edges[k + 1]], regime)

    final = best_on(edges[0], hold_start) or variants[0]
    holdout = run_backtest(ind, final, cost, allow_short, ts[hold_start], None, regime)
    return {
        "strategy": strategy_cls.name,
        "family": strategy_cls.family,
        "final_params": final.key,
        "fold_choices": chosen,
        "oos": summarize(t.r_multiple for t in oos),
        "holdout": summarize(t.r_multiple for t in holdout),
        "oos_trades": oos,
        "holdout_trades": holdout,
        "avg_cost_r": float(np.mean([t.cost_r for t in oos])) if oos else 0.0,
        "data_start": str(ts[0]), "data_end": str(ts[-1]), "bars": n,
    }


def passes_gate(result: dict, gate: dict) -> tuple[bool, list[str]]:
    o, h = result["oos"], result["holdout"]
    reasons = []
    if o["trades"] < gate["min_oos_trades"]:
        reasons.append(f"only {o['trades']} out-of-sample trades (< {gate['min_oos_trades']})")
    if o["expectancy_r"] < gate["min_oos_expectancy_r"]:
        reasons.append(f"OOS expectancy {o['expectancy_r']:+.3f}R below {gate['min_oos_expectancy_r']}R")
    if o["profit_factor"] < gate["min_oos_profit_factor"]:
        reasons.append(f"OOS profit factor {o['profit_factor']:.2f} below {gate['min_oos_profit_factor']}")
    if h["trades"] < gate["min_holdout_trades"]:
        reasons.append(f"only {h['trades']} holdout trades (< {gate['min_holdout_trades']})")
    elif h["expectancy_r"] < gate["min_holdout_expectancy_r"]:
        reasons.append(f"holdout expectancy {h['expectancy_r']:+.3f}R failed")
    return (not reasons), reasons

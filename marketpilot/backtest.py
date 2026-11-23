"""Backtesting Bot core: event-driven, cost-aware, no look-ahead.

Fill rules (identical to the paper broker — both call `step_position`):
  * signal on the CLOSE of bar i  -> market entry at the OPEN of bar i+1 (+ half spread + slippage)
  * stop hit if the bar trades through it; if the bar OPENS beyond the stop, fill at the open (gap)
  * if stop and target are both inside one bar, assume the STOP filled first (conservative)
  * take-profit fills as a limit at the target minus half spread
  * close-based exits (e.g. close back above the 5-day average) fill at that close
  * time exit at the close after `max_hold` bars
  * trailing stops are updated at each close and apply from the NEXT bar
R = net P&L / planned risk, where planned risk includes estimated exit costs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .costs import CostModel
from .indicators import add_indicators
from .metrics import summarize
from .bots.confluence import bucket as conf_bucket
from .bots.confluence import score as conf_score
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
    confluence: int = 0
    vol_rank: float | None = None
    alt_r: dict = field(default_factory=dict)   # R under other trade-management modes


def swing_level(row, direction: int):
    """Recent 10-bar swing low (longs) / high (shorts) at the signal bar."""
    v = row.get("swing_lo10") if direction == 1 else row.get("swing_hi10")
    try:
        return None if v is None or np.isnan(float(v)) else float(v)
    except (TypeError, ValueError):
        return None


def plan_levels(entry_fill: float, atr: float, sig: Signal, cost: CostModel,
                swing: float | None = None) -> tuple[float, float, float]:
    """Return (stop, target, risk_per_unit) for an entry.

    ATR stop by default. With sig.stop_swing, the stop goes 0.5 ATR beyond the
    recent swing low/high (market structure), clamped to 1-4 ATR from entry so it
    is never absurdly tight or wide."""
    d = sig.direction
    stop = entry_fill - d * sig.stop_atr * atr
    if getattr(sig, "stop_swing", False) and swing is not None and atr > 0:
        dist = d * (entry_fill - (swing - d * 0.5 * atr))
        dist = min(max(dist, 1.0 * atr), 4.0 * atr)
        stop = entry_fill - d * dist
    target = entry_fill + d * sig.target_atr * atr
    exit_cost = stop * (cost.half_spread + cost.slip) + cost.commission_per_unit(stop)
    risk_per_unit = abs(entry_fill - stop) + exit_cost + cost.commission_per_unit(entry_fill)
    return stop, target, risk_per_unit


def check_exit(bar, direction: int, stop: float, target: float,
               cost: CostModel) -> tuple[float, str] | None:
    """Evaluate one bar (after the entry bar) for a stop/target exit."""
    o, h, l = bar["open"], bar["high"], bar["low"]
    side = -direction
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


# Trade management (v0.12). How an open trade is handled once it moves in our favour.
#   base       stops/targets/trailing exactly as each strategy defines them
#   breakeven  once a bar closes 1R in profit, the stop moves to breakeven (entry + costs)
#   partial    bank half the position at +1R, then move the rest's stop to breakeven
MANAGEMENT = {
    "base": {},
    "breakeven": {"be_at_r": 1.0},
    "partial": {"partial_at_r": 1.0, "partial_frac": 0.5, "be_after_partial": True},
}


def _be_level(st: dict, cost: CostModel) -> float:
    """Stop price that exits at about 0R after spreads, slippage and commission."""
    e, d = st["entry"], int(st["direction"])
    return e + d * cost.round_trip_cost_frac() * e


def _raise_stop(st: dict, level: float) -> None:
    d = int(st["direction"])
    st["stop"] = max(st["stop"], level) if d == 1 else min(st["stop"], level)


def step_position(st: dict, row, cost: CostModel, entry_bar: bool) -> tuple[float, str] | None:
    """Advance an open position by one bar. Mutates `st` (stop, best, held).

    st keys: direction, stop, target, held, max_hold, trail_atr, exit_ma, exit_on, best.
    `row` must carry open/high/low/close, atr14 and sma<N> for the exit_ma used.
    This single function is used by BOTH the backtester and the paper broker."""
    d = int(st["direction"])
    hit = (_entry_bar_exit if entry_bar else check_exit)(row, d, st["stop"], st["target"], cost)
    if hit:
        return hit
    m = st.get("mgmt") or {}
    rpu, entry = st.get("rpu"), st.get("entry")
    if m.get("partial_at_r") and rpu and entry and st.get("partial_price") is None:
        level = entry + d * m["partial_at_r"] * rpu
        o, h, lo_ = row["open"], row["high"], row["low"]
        if (d == 1 and h >= level) or (d == -1 and lo_ <= level):
            px = o if d * (o - level) >= 0 and not entry_bar else level   # gap through the level: fill at open
            st["partial_price"] = cost.market_fill(px, -d)
            if m.get("be_after_partial"):
                _raise_stop(st, _be_level(st, cost))
    close = row["close"]
    ma_n = int(st.get("exit_ma") or 0)
    if ma_n:
        ma = row.get(f"sma{ma_n}")
        if ma is not None and np.isfinite(ma):
            rule = st.get("exit_on")
            if (rule == "revert" and d * (close - ma) > 0) or (rule == "break" and d * (close - ma) < 0):
                return cost.market_fill(close, -d), f"close {'above' if close > ma else 'below'} {ma_n}-bar avg"
    if st["held"] >= st["max_hold"]:
        return cost.market_fill(close, -d), "time"
    # update trailing stop at the close (applies from the next bar)
    best = st.get("best")
    best = close if best is None else (max(best, close) if d == 1 else min(best, close))
    st["best"] = best
    trail = float(st.get("trail_atr") or 0)
    atr = row.get("atr14")
    if trail > 0 and atr is not None and np.isfinite(atr):
        new_stop = best - d * trail * atr
        st["stop"] = max(st["stop"], new_stop) if d == 1 else min(st["stop"], new_stop)
    if m.get("be_at_r") and rpu and entry and d * (close - entry) >= m["be_at_r"] * rpu:
        _raise_stop(st, _be_level(st, cost))
    st["held"] += 1
    return None


def trade_r(st: dict, direction: int, entry: float, exit_px: float, rpu: float, cost: CostModel) -> float:
    """Net R of a closed trade, blending in any partial profit banked earlier.
    Shorts also pay a daily holding cost (borrow / overnight funding, v0.13)."""
    r_exit, _ = realized_r(direction, entry, exit_px, rpu, cost)
    carry = 0.0
    if direction == -1 and cost.short_carry_bps_day and rpu:
        carry = cost.short_carry_bps_day / 10_000 * entry * max(int(st.get("held") or 0), 1) / rpu
    pp = st.get("partial_price")
    if pp is None:
        return r_exit - carry
    f = float((st.get("mgmt") or {}).get("partial_frac", 0.5))
    r_part, _ = realized_r(direction, entry, pp, rpu, cost)
    return (1 - f) * r_exit + f * r_part - carry


def realized_r(direction: int, entry: float, exit_: float, risk_per_unit: float,
               cost: CostModel) -> tuple[float, float]:
    """Return (net R, commission in R) per unit."""
    commissions = cost.commission_per_unit(entry) + cost.commission_per_unit(exit_)
    net = direction * (exit_ - entry) - commissions
    return net / risk_per_unit, 0.0 if risk_per_unit == 0 else commissions / risk_per_unit


def new_state(sig: Signal, stop: float, target: float, entry: float | None = None,
              rpu: float | None = None, mgmt: dict | None = None) -> dict:
    return {"direction": sig.direction, "stop": stop, "target": target, "held": 0,
            "max_hold": sig.max_hold, "trail_atr": sig.trail_atr, "exit_ma": sig.exit_ma,
            "exit_on": sig.exit_on, "best": None, "entry": entry, "rpu": rpu,
            "mgmt": mgmt or {}, "partial_price": None}


def _simulate_exit(rows, e: int, sig: Signal, stop: float, target: float, entry: float, rpu: float,
                   cost: CostModel, mgmt: dict | None):
    st = new_state(sig, stop, target, entry, rpu, mgmt)
    j, n = e, len(rows)
    while j < n:
        hit = step_position(st, rows[j], cost, entry_bar=(j == e))
        if hit:
            return j, hit, st
        j += 1
    return None, None, st


def run_backtest(ind: pd.DataFrame, strategy: Strategy, cost: CostModel, allow_short: bool,
                 start: pd.Timestamp | None = None, end: pd.Timestamp | None = None,
                 regime: pd.Series | None = None, market_ok: pd.Series | None = None,
                 mgmt: str = "base", compare_modes: tuple[str, ...] = ()) -> list[SimTrade]:
    """Simulate one strategy on one symbol. `ind` must come from add_indicators().
    Only signals with signal_time in [start, end) are taken; one position at a time.
    `market_ok` (bool series on the same index) blocks LONG entries when False and
    SHORT entries when True (v0.13: only short a group whose benchmark is below
    its 200-day average)."""
    idx = ind.index
    n = len(ind)
    rows = ind.to_dict("records")
    mk = market_ok.to_numpy() if market_ok is not None else None
    trades: list[SimTrade] = []
    lo = idx.searchsorted(start) if start is not None else 1
    hi = idx.searchsorted(end) if end is not None else n
    i = max(lo, 1)
    while i < min(hi, n - 1):
        row, prev = rows[i], rows[i - 1]
        sig = strategy.signal(prev, row)
        if (sig is None or (sig.direction == -1 and not allow_short) or not np.isfinite(row["atr14"])
                or (regime is not None and regime.iloc[i] not in strategy.regimes)
                or (mk is not None and sig.direction == 1 and not mk[i])
                or (mk is not None and sig.direction == -1 and mk[i])):
            i += 1
            continue
        e = i + 1
        entry = cost.market_fill(rows[e]["open"], sig.direction)
        stop, target, rpu = plan_levels(entry, row["atr14"], sig, cost, swing_level(row, sig.direction))
        t = SimTrade(strategy.key, sig.direction, idx[i], idx[e], entry, stop, target)
        t.confluence = conf_score(row, sig.direction, None if mk is None else bool(mk[i]))[0]
        vr = row.get("atr_pct_rank")
        t.vol_rank = float(vr) if vr is not None and np.isfinite(vr) else None
        j, hit, st = _simulate_exit(rows, e, sig, stop, target, entry, rpu, cost, MANAGEMENT.get(mgmt, {}))
        if hit is None:  # still open at end of data: ignore
            break
        t.exit_price, t.exit_reason = hit
        t.exit_time = idx[j]
        t.r_multiple = trade_r(st, sig.direction, entry, t.exit_price, rpu, cost)
        for mode in compare_modes:   # same entry, different management: for the research comparison
            if mode == mgmt:
                t.alt_r[mode] = t.r_multiple
                continue
            _, h2, st2 = _simulate_exit(rows, e, sig, stop, target, entry, rpu, cost, MANAGEMENT.get(mode, {}))
            if h2 is not None:
                t.alt_r[mode] = trade_r(st2, sig.direction, entry, h2[0], rpu, cost)
        t.cost_r = cost.round_trip_cost_frac() * entry / rpu
        trades.append(t)
        i = j + 1
    return trades


def trades_frame(trades: list[SimTrade]) -> pd.DataFrame:
    return pd.DataFrame([asdict(t) for t in trades])


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    from .bots.regime import classify_series
    ind = add_indicators(df)
    return ind, classify_series(ind)


def market_filter_series(filter_df: pd.DataFrame | None, index: pd.DatetimeIndex) -> pd.Series | None:
    """True where the filter market (e.g. SPY) closed above its 200-day average,
    aligned to `index` using only information available at each bar."""
    if filter_df is None or filter_df.empty:
        return None
    c = filter_df["close"]
    ok = (c > c.rolling(200, min_periods=200).mean())
    return ok.reindex(index.union(ok.index)).ffill().reindex(index).fillna(False).astype(bool)


def walk_forward(df: pd.DataFrame, strategy_cls: type[Strategy], cost: CostModel,
                 allow_short: bool, folds: int = 4, holdout_fraction: float = 0.2,
                 min_is_trades: int = 5, market_ok: pd.Series | None = None,
                 mgmt: str = "base", compare_modes: tuple[str, ...] = ()) -> dict:
    """Anchored walk-forward with a final untouched holdout.

    For each fold: pick the parameter variant with the best in-sample expectancy,
    then record its trades on the next (unseen) segment. Aggregated OOS trades
    are the evidence. The holdout is only evaluated once, with params chosen on
    all pre-holdout data."""
    ind, regime = prepare(df)
    n = len(ind)
    warm = min(260, n // 5)
    hold_start = int(n * (1 - holdout_fraction))
    edges = np.linspace(warm, hold_start, folds + 2).astype(int)
    variants = strategy_cls.variants()
    oos: list[SimTrade] = []
    chosen: list[str] = []
    ts = ind.index

    def best_on(a, b):
        best, best_e = None, -np.inf
        for v in variants:
            tr = run_backtest(ind, v, cost, allow_short, ts[a], ts[b], regime, market_ok, mgmt)
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
        oos += run_backtest(ind, pick, cost, allow_short, ts[edges[k]], ts[edges[k + 1]], regime, market_ok,
                            mgmt, compare_modes)

    final = best_on(edges[0], hold_start) or variants[0]
    holdout = run_backtest(ind, final, cost, allow_short, ts[hold_start], None, regime, market_ok,
                           mgmt, compare_modes)
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
        "by_confluence": confluence_breakdown(oos),
        "management": management_breakdown(oos, holdout, compare_modes),
        "data_start": str(ts[0]), "data_end": str(ts[-1]), "bars": n,
    }


def management_breakdown(oos: list[SimTrade], holdout: list[SimTrade], modes) -> dict:
    """Same trades under each management mode: expectancy, loss rate, drawdown."""
    out = {}
    for mode in modes:
        o = [t.alt_r[mode] for t in oos if mode in t.alt_r]
        h = [t.alt_r[mode] for t in holdout if mode in t.alt_r]
        if not o:
            continue
        so, sh = summarize(o), summarize(h)
        full_losses = sum(1 for r in o if r <= -0.9)
        scratches = sum(1 for r in o if abs(r) < 0.15)
        out[mode] = {"trades": so["trades"], "expectancy_r": so["expectancy_r"], "win_rate": so["win_rate"],
                     "loss_rate": 1 - so["win_rate"], "full_loss_rate": full_losses / len(o), "scratch_rate": scratches / len(o),
                     "avg_loss_r": so["avg_loss_r"], "max_drawdown_r": so["max_drawdown_r"],
                     "profit_factor": so["profit_factor"], "t_stat": so["t_stat"],
                     "holdout_expectancy_r": sh["expectancy_r"]}
    return out


def confluence_breakdown(trades: list[SimTrade]) -> dict:
    """Win rate / expectancy of trades grouped by confluence score bucket."""
    groups: dict[str, list[float]] = {}
    for t in trades:
        groups.setdefault(conf_bucket(t.confluence), []).append(t.r_multiple)
    out = {}
    for b, rs in groups.items():
        s = summarize(rs)
        out[b] = {"trades": s["trades"], "win_rate": s["win_rate"], "expectancy_r": s["expectancy_r"]}
    return out


def passes_gate(result: dict, gate: dict) -> tuple[bool, list[str]]:
    o, h = result["oos"], result["holdout"]
    reasons = []
    if o["trades"] < gate["min_oos_trades"]:
        reasons.append(f"only {o['trades']} out-of-sample trades (< {gate['min_oos_trades']})")
    if o["expectancy_r"] < gate["min_oos_expectancy_r"]:
        reasons.append(f"OOS expectancy {o['expectancy_r']:+.3f}R below {gate['min_oos_expectancy_r']}R")
    if o["profit_factor"] < gate["min_oos_profit_factor"]:
        reasons.append(f"OOS profit factor {o['profit_factor']:.2f} below {gate['min_oos_profit_factor']}")
    t_min = gate.get("min_oos_tstat")
    if t_min is not None and o["trades"] >= gate["min_oos_trades"] and o.get("t_stat", 0) < t_min:
        reasons.append(f"OOS t-stat {o.get('t_stat', 0):.2f} below {t_min} (could be luck)")
    if h["trades"] < gate["min_holdout_trades"]:
        reasons.append(f"only {h['trades']} holdout trades (< {gate['min_holdout_trades']})")
    elif h["expectancy_r"] < gate["min_holdout_expectancy_r"]:
        reasons.append(f"holdout expectancy {h['expectancy_r']:+.3f}R failed")
    return (not reasons), reasons

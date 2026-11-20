"""Auto-Trading Bot — paper execution only.

Lifecycle: pending (signal on bar close) -> open (filled at next bar's open)
-> closed (stop / target / exit rule / time). Uses exactly the same
`step_position` function as the backtester, so paper results are comparable to
the backtest evidence.

There is intentionally no live broker adapter in this module.
"""
from __future__ import annotations

import uuid

import pandas as pd

from ..backtest import plan_levels, realized_r, step_position
from ..costs import CostModel
from ..data import interval_td
from ..strategies import Signal


def _f(x, default=0.0) -> float:
    try:
        return default if x is None or pd.isna(x) else float(x)
    except (TypeError, ValueError):
        return default


class PaperBroker:
    def __init__(self, store, on_open=None, on_close=None):
        self.store = store
        self.on_open = on_open or (lambda t: None)
        self.on_close = on_close or (lambda t: None)

    def submit(self, *, candidate_id: str, symbol: str, asset_class: str, strategy_key: str,
               signal: Signal, regime: str, bar_time: pd.Timestamp, atr: float, qty: float,
               risk_gbp: float, risk_per_unit: float, fx: float, notional: float, cost_r: float) -> bool:
        return self.store.open_trade({
            "trade_id": str(uuid.uuid4()), "candidate_id": candidate_id, "symbol": symbol,
            "asset_class": asset_class, "strategy": strategy_key, "direction": signal.direction,
            "regime": regime, "status": "pending", "opened_at": str(bar_time),
            "signal_bar_time": str(bar_time), "atr": atr, "stop_atr": signal.stop_atr,
            "target_atr": signal.target_atr, "max_hold_bars": signal.max_hold, "quantity": qty,
            "risk_gbp": risk_gbp, "risk_per_unit": risk_per_unit, "fx_to_gbp": fx,
            "notional_gbp": notional, "cost_r": cost_r, "note": signal.note,
            "trail_atr": signal.trail_atr, "exit_ma": signal.exit_ma, "exit_on": signal.exit_on,
            "last_bar_time": str(bar_time)})

    def process(self, symbol: str, ind: pd.DataFrame, interval: str, cost: CostModel,
                now: pd.Timestamp, log) -> list[dict]:
        """Fill pending orders and manage open positions bar by bar.
        `ind` = price bars with indicators (add_indicators). Returns closed trades."""
        closed = []
        step = pd.Timedelta(interval_td(interval))
        for t in [x for x in self.store.trades() if x["symbol"] == symbol and x["status"] in ("pending", "open")]:
            last = pd.Timestamp(t["last_bar_time"])
            new_bars = ind[ind.index > last]
            if new_bars.empty:
                continue
            if t["status"] == "pending":
                first = new_bars.iloc[0]
                fill_bar_time = new_bars.index[0]
                if now - (fill_bar_time + step) > 2 * step:
                    self.store.update_trade(t["trade_id"], {"status": "cancelled", "exit_reason":
                                            "missed fill: worker ran too late", "closed_at": str(now)})
                    log("auto_trader", f"{symbol}: cancelled stale pending order", "warning")
                    continue
                d = int(t["direction"])
                entry = cost.market_fill(float(first["open"]), d)
                sig = Signal(d, _f(t["stop_atr"]), _f(t["target_atr"]), int(t["max_hold_bars"]))
                stop, target, rpu = plan_levels(entry, _f(t["atr"]), sig, cost)
                qty = _f(t["risk_gbp"]) / (rpu * _f(t["fx_to_gbp"], 1.0))  # keep £ risk at plan
                t.update({"status": "open", "entry_price": entry, "stop": stop, "target": target,
                          "risk_per_unit": rpu, "quantity": qty, "opened_at": str(fill_bar_time),
                          "notional_gbp": qty * entry * _f(t["fx_to_gbp"], 1.0), "bars_held": 0,
                          "best_price": None})
                self.store.update_trade(t["trade_id"], {k: t[k] for k in (
                    "status", "entry_price", "stop", "target", "risk_per_unit", "quantity",
                    "opened_at", "notional_gbp")})
                log("auto_trader", f"FILLED {symbol} {'LONG' if d > 0 else 'SHORT'} {qty:.4f} @ {entry:.5f} "
                    f"stop {stop:.5f} [{t['strategy']}]")
                self.on_open(t)
            result = self._walk(t, new_bars, cost)
            if result:
                closed.append(result)
                log("auto_trader", f"CLOSED {symbol} {result['exit_reason']} {result['r_multiple']:+.2f}R "
                    f"(£{result['pnl_gbp']:+.2f})")
                self.on_close(result)
        return closed

    def _walk(self, t: dict, bars: pd.DataFrame, cost: CostModel) -> dict | None:
        opened = pd.Timestamp(t["opened_at"])
        st = {"direction": int(t["direction"]), "stop": _f(t["stop"]), "target": _f(t["target"]),
              "held": int(_f(t.get("bars_held"))), "max_hold": int(t["max_hold_bars"]),
              "trail_atr": _f(t.get("trail_atr")), "exit_ma": int(_f(t.get("exit_ma"))),
              "exit_on": t.get("exit_on") or "",
              "best": None if t.get("best_price") is None or pd.isna(t.get("best_price")) else float(t["best_price"])}
        for ts, bar in bars.iterrows():
            if ts < opened:
                continue
            hit = step_position(st, bar.to_dict(), cost, entry_bar=(ts == opened))
            if hit:
                exit_px, reason = hit
                d = st["direction"]
                rpu, qty, fx = _f(t["risk_per_unit"]), _f(t["quantity"]), _f(t["fx_to_gbp"], 1.0)
                r, _ = realized_r(d, _f(t["entry_price"]), exit_px, rpu, cost)
                fees = (cost.commission_per_unit(_f(t["entry_price"])) + cost.commission_per_unit(exit_px)) * qty * fx \
                    + 2 * cost.commission_fixed
                pnl = r * rpu * qty * fx - 2 * cost.commission_fixed
                changes = {"status": "closed", "closed_at": str(ts), "exit_price": exit_px,
                           "exit_reason": reason, "r_multiple": pnl / _f(t["risk_gbp"], 1.0),
                           "pnl_gbp": pnl, "fees_gbp": fees, "bars_held": st["held"],
                           "stop": st["stop"], "last_bar_time": str(ts)}
                self.store.update_trade(t["trade_id"], changes)
                return {**t, **changes}
        self.store.update_trade(t["trade_id"], {"bars_held": st["held"], "stop": st["stop"],
                                                "best_price": st["best"], "last_bar_time": str(bars.index[-1])})
        return None


def mark_to_market(t: dict, price: float) -> float:
    if t["status"] != "open":
        return 0.0
    return int(t["direction"]) * (price - _f(t["entry_price"])) * _f(t["quantity"]) * _f(t["fx_to_gbp"], 1.0)

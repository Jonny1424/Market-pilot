"""Auto-Trading Bot — paper execution only.

Lifecycle: pending (signal on bar close) -> open (filled at next bar's open)
-> closed (stop / target / time). Uses exactly the same fill functions as the
backtester, so paper results are comparable to backtest evidence.

There is intentionally no live broker adapter in this module.
"""
from __future__ import annotations

import uuid

import pandas as pd

from ..backtest import _entry_bar_exit, check_exit, plan_levels, realized_r
from ..costs import CostModel
from ..data import interval_td
from ..strategies import Signal


class PaperBroker:
    def __init__(self, store):
        self.store = store

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
            "last_bar_time": str(bar_time)})

    def process(self, symbol: str, df: pd.DataFrame, interval: str, cost: CostModel,
                now: pd.Timestamp, log) -> list[dict]:
        """Fill pending orders and manage open positions bar by bar. Returns closed trades."""
        closed = []
        step = pd.Timedelta(interval_td(interval))
        for t in [x for x in self.store.trades() if x["symbol"] == symbol and x["status"] in ("pending", "open")]:
            last = pd.Timestamp(t["last_bar_time"])
            new_bars = df[df.index > last]
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
                sig = Signal(d, float(t["stop_atr"]), float(t["target_atr"]), int(t["max_hold_bars"]))
                stop, target, rpu = plan_levels(entry, float(t["atr"]), sig, cost)
                # keep £ risk at plan: resize for actual fill
                qty = float(t["risk_gbp"]) / (rpu * float(t["fx_to_gbp"]))
                t.update({"status": "open", "entry_price": entry, "stop": stop, "target": target,
                          "risk_per_unit": rpu, "quantity": qty, "opened_at": str(fill_bar_time),
                          "notional_gbp": qty * entry * float(t["fx_to_gbp"])})
                self.store.update_trade(t["trade_id"], {k: t[k] for k in (
                    "status", "entry_price", "stop", "target", "risk_per_unit", "quantity",
                    "opened_at", "notional_gbp")})
                log("auto_trader", f"FILLED {symbol} {'LONG' if d > 0 else 'SHORT'} {qty:.4f} @ {entry:.5f} "
                    f"stop {stop:.5f} target {target:.5f} [{t['strategy']}]")
            result = self._walk(t, new_bars, cost)
            if result:
                closed.append(result)
                log("auto_trader", f"CLOSED {symbol} {result['exit_reason']} {result['r_multiple']:+.2f}R "
                    f"(£{result['pnl_gbp']:+.2f})")
        return closed

    def _walk(self, t: dict, bars: pd.DataFrame, cost: CostModel) -> dict | None:
        d, stop, target = int(t["direction"]), float(t["stop"]), float(t["target"])
        held = int(t.get("bars_held") or 0)
        opened = pd.Timestamp(t["opened_at"])
        for ts, bar in bars.iterrows():
            if ts < opened:
                continue
            b = bar.to_dict()
            hit = _entry_bar_exit(b, d, stop, target, cost) if ts == opened else check_exit(b, d, stop, target, cost)
            if not hit and held >= int(t["max_hold_bars"]):
                hit = (cost.market_fill(b["close"], -d), "time")
            if hit:
                exit_px, reason = hit
                r, _ = realized_r(d, float(t["entry_price"]), exit_px, float(t["risk_per_unit"]), cost)
                qty, fx = float(t["quantity"]), float(t["fx_to_gbp"])
                fees = (cost.commission_per_unit(float(t["entry_price"])) + cost.commission_per_unit(exit_px)) * qty * fx \
                    + 2 * cost.commission_fixed
                pnl = r * float(t["risk_per_unit"]) * qty * fx - 2 * cost.commission_fixed
                changes = {"status": "closed", "closed_at": str(ts), "exit_price": exit_px,
                           "exit_reason": reason, "r_multiple": pnl / float(t["risk_gbp"]),
                           "pnl_gbp": pnl, "fees_gbp": fees, "bars_held": held, "last_bar_time": str(ts)}
                self.store.update_trade(t["trade_id"], changes)
                return {**t, **changes}
            held += 1
        self.store.update_trade(t["trade_id"], {"bars_held": held, "last_bar_time": str(bars.index[-1])})
        return None


def mark_to_market(t: dict, price: float) -> float:
    if t["status"] != "open":
        return 0.0
    return int(t["direction"]) * (price - float(t["entry_price"])) * float(t["quantity"]) * float(t["fx_to_gbp"])

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

from ..backtest import MANAGEMENT, _entry_bar_exit, check_exit, plan_levels, step_position, trade_r
from ..costs import CostModel
from ..data import interval_td
from ..strategies import Signal


def _f(x, default=0.0) -> float:
    try:
        return default if x is None or pd.isna(x) else float(x)
    except (TypeError, ValueError):
        return default


class PaperBroker:
    def __init__(self, store, on_open=None, on_close=None, on_partial=None):
        self.store = store
        self.on_open = on_open or (lambda t: None)
        self.on_close = on_close or (lambda t: None)
        self.on_partial = on_partial or (lambda t, st: None)

    def log_partial(self, t: dict, st: dict) -> None:
        try:
            self.on_partial(t, st)
        except Exception:
            pass

    def submit(self, *, candidate_id: str, symbol: str, asset_class: str, strategy_key: str,
               signal: Signal, regime: str, bar_time: pd.Timestamp, atr: float, qty: float,
               risk_gbp: float, risk_per_unit: float, fx: float, notional: float, cost_r: float,
               swing: float | None = None, confluence: int | None = None, conf_text: str = "",
               extra: dict | None = None, extra_note: str = "") -> bool:
        return self.store.open_trade({
            "trade_id": str(uuid.uuid4()), "candidate_id": candidate_id, "symbol": symbol,
            "asset_class": asset_class, "strategy": strategy_key, "direction": signal.direction,
            "regime": regime, "status": "pending", "opened_at": str(bar_time),
            "signal_bar_time": str(bar_time), "atr": atr, "stop_atr": signal.stop_atr,
            "target_atr": signal.target_atr, "max_hold_bars": signal.max_hold, "quantity": qty,
            "risk_gbp": risk_gbp, "risk_per_unit": risk_per_unit, "fx_to_gbp": fx,
            "notional_gbp": notional, "cost_r": cost_r,
            "note": signal.note + (f" | confluence {confluence}/8: {conf_text}" if confluence is not None else "")
                    + (f" | {extra_note}" if extra_note else ""),
            "trail_atr": signal.trail_atr, "exit_ma": signal.exit_ma, "exit_on": signal.exit_on,
            "stop_swing": int(signal.stop_swing), "swing_level": swing, "confluence": confluence,
            "last_bar_time": str(bar_time), **(extra or {})})

    def _fill(self, t: dict, open_px: float, fill_bar_time, cost: CostModel, log, how: str = "") -> None:
        """Fill a pending order at the session's opening price (same rule as the backtest)."""
        symbol, d = t["symbol"], int(t["direction"])
        entry = cost.market_fill(open_px, d)
        sig = Signal(d, _f(t["stop_atr"]), _f(t["target_atr"]), int(t["max_hold_bars"]),
                     stop_swing=bool(_f(t.get("stop_swing"))))
        sw = t.get("swing_level")
        sw = None if sw is None or pd.isna(sw) else float(sw)
        stop, target, rpu = plan_levels(entry, _f(t["atr"]), sig, cost, sw)
        qty = _f(t["risk_gbp"]) / (rpu * _f(t["fx_to_gbp"], 1.0))  # keep £ risk at plan
        t.update({"status": "open", "entry_price": entry, "stop": stop, "target": target,
                  "risk_per_unit": rpu, "quantity": qty, "opened_at": str(fill_bar_time),
                  "notional_gbp": qty * entry * _f(t["fx_to_gbp"], 1.0), "bars_held": 0,
                  "best_price": None})
        self.store.update_trade(t["trade_id"], {k: t[k] for k in (
            "status", "entry_price", "stop", "target", "risk_per_unit", "quantity",
            "opened_at", "notional_gbp")})
        log("auto_trader", f"FILLED {symbol} {'LONG' if d > 0 else 'SHORT'} {qty:.4f} @ {entry:.5f} "
            f"stop {stop:.5f} [{t['strategy']}]{how}")
        self.on_open(t)

    def _state(self, t: dict) -> dict:
        return {"direction": int(t["direction"]), "stop": _f(t["stop"]), "target": _f(t["target"]),
                "held": int(_f(t.get("bars_held"))), "entry": _f(t.get("entry_price")) or None,
                "rpu": _f(t.get("risk_per_unit")) or None,
                "mgmt": MANAGEMENT.get(t.get("mgmt") or "base", {}),
                "partial_price": None if t.get("partial_price") is None or pd.isna(t.get("partial_price"))
                else float(t["partial_price"])}

    def _close(self, t: dict, st: dict, exit_px: float, reason: str, when, cost: CostModel) -> dict:
        d = int(t["direction"])
        rpu, qty, fx = _f(t["risk_per_unit"]), _f(t["quantity"]), _f(t["fx_to_gbp"], 1.0)
        r = trade_r(st, d, _f(t["entry_price"]), exit_px, rpu, cost)
        fees = (cost.commission_per_unit(_f(t["entry_price"])) + cost.commission_per_unit(exit_px)) * qty * fx \
            + 2 * cost.commission_fixed
        pnl = r * rpu * qty * fx - 2 * cost.commission_fixed
        changes = {"status": "closed", "closed_at": str(when), "exit_price": exit_px,
                   "exit_reason": reason, "r_multiple": pnl / _f(t["risk_gbp"], 1.0),
                   "pnl_gbp": pnl, "fees_gbp": fees, "bars_held": st["held"],
                   "stop": st["stop"], "partial_price": st["partial_price"]}
        self.store.update_trade(t["trade_id"], changes)
        return {**t, **changes}

    def watch_intraday(self, symbol: str, intra: pd.DataFrame, cost: CostModel, now: pd.Timestamp,
                       log) -> list[dict]:
        """v0.13 intraday watch. `intra` = today's finished 15-minute bars. Books stop and
        target hits within the hour (stop checked first, as in the backtest), and fills
        pending orders at the session's real open. Trailing stops, breakeven moves and
        close-based exits still happen once per day at the close, exactly as tested."""
        closed = []
        if intra is None or intra.empty:
            return closed
        for t in [x for x in self.store.trades() if x["symbol"] == symbol and x["status"] in ("pending", "open")]:
            last = pd.Timestamp(t["last_bar_time"])
            session_start = (last + pd.Timedelta(days=1)).normalize()
            bars = intra[intra.index >= session_start]
            if bars.empty:
                continue
            session_day = bars.index[0].normalize()
            bars = bars[bars.index.normalize() == session_day]     # one session only
            entry_session = False
            if t["status"] == "pending":
                self._fill(t, float(bars["open"].iloc[0]), session_day, cost, log, " (at the open, intraday)")
                entry_session = True
            elif pd.Timestamp(t["opened_at"]).normalize() == session_day:
                entry_session = True
            bar = {"open": float(bars["open"].iloc[0]), "high": float(bars["high"].max()),
                   "low": float(bars["low"].min()), "close": float(bars["close"].iloc[-1])}
            st = self._state(t)
            hit = (_entry_bar_exit if entry_session else check_exit)(bar, st["direction"], st["stop"],
                                                                     st["target"], cost)
            if hit:
                result = self._close(t, st, hit[0], hit[1] + " (intraday)", now, cost)
                closed.append(result)
                log("auto_trader", f"CLOSED {symbol} {result['exit_reason']} {result['r_multiple']:+.2f}R "
                    f"(£{result['pnl_gbp']:+.2f})")
                self.on_close(result)
        return closed

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
                self._fill(t, float(first["open"]), fill_bar_time, cost, log)
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
              "best": None if t.get("best_price") is None or pd.isna(t.get("best_price")) else float(t["best_price"]),
              "entry": _f(t.get("entry_price")) or None, "rpu": _f(t.get("risk_per_unit")) or None,
              "mgmt": MANAGEMENT.get(t.get("mgmt") or "base", {}),
              "partial_price": None if t.get("partial_price") is None or pd.isna(t.get("partial_price"))
              else float(t["partial_price"])}
        had_partial = st["partial_price"] is not None
        for ts, bar in bars.iterrows():
            if ts < opened:
                continue
            hit = step_position(st, bar.to_dict(), cost, entry_bar=(ts == opened))
            if hit:
                exit_px, reason = hit
                d = st["direction"]
                rpu, qty, fx = _f(t["risk_per_unit"]), _f(t["quantity"]), _f(t["fx_to_gbp"], 1.0)
                r = trade_r(st, d, _f(t["entry_price"]), exit_px, rpu, cost)
                fees = (cost.commission_per_unit(_f(t["entry_price"])) + cost.commission_per_unit(exit_px)) * qty * fx \
                    + 2 * cost.commission_fixed
                pnl = r * rpu * qty * fx - 2 * cost.commission_fixed
                changes = {"status": "closed", "closed_at": str(ts), "exit_price": exit_px,
                           "exit_reason": reason, "r_multiple": pnl / _f(t["risk_gbp"], 1.0),
                           "pnl_gbp": pnl, "fees_gbp": fees, "bars_held": st["held"],
                           "stop": st["stop"], "last_bar_time": str(ts), "partial_price": st["partial_price"]}
                self.store.update_trade(t["trade_id"], changes)
                return {**t, **changes}
        upd = {"bars_held": st["held"], "stop": st["stop"], "best_price": st["best"],
               "last_bar_time": str(bars.index[-1])}
        if st["partial_price"] is not None and not had_partial:
            upd["partial_price"] = st["partial_price"]
            self.log_partial(t, st)
        self.store.update_trade(t["trade_id"], upd)
        return None


def _noop(*_a, **_k):
    return None


def mark_to_market(t: dict, price: float) -> float:
    if t["status"] != "open":
        return 0.0
    return int(t["direction"]) * (price - _f(t["entry_price"])) * _f(t["quantity"]) * _f(t["fx_to_gbp"], 1.0)

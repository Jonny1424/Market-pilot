"""Orchestrator: one cycle of the 10-bot pipeline.

  Safety -> (per symbol) data -> stale check -> Auto-Trader manages positions
  -> Scanner -> Regime -> approved strategies' signals -> News veto
  -> Trade Decision (EV, may abstain) -> Risk Manager -> paper order
  -> Performance snapshot

Kill switch ON: open positions are still managed (stops/targets honoured), no new trades.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from .backtest import plan_levels
from .bots.autotrader import PaperBroker, mark_to_market
from .bots.decision import DecisionBot
from .bots.news import NewsBot
from .bots.performance import realized_today, report, retirement_review
from .bots.regime import classify_row
from .bots.risk import RiskManager
from .bots.safety import SafetyBot, is_stale
from .bots.scanner import scan
from .costs import cost_model_for
from .data import interval_td, quote_currency
from .indicators import add_indicators
from .strategies import SHORTABLE, strategy_from_key

HISTORY_BARS = 700


class Engine:
    def __init__(self, settings, provider, store, live: bool = True):
        self.s, self.provider, self.store, self.live = settings, provider, store, live
        self.safety = SafetyBot(store, settings)
        self.news = NewsBot(settings, live)
        self.risk = RiskManager(settings)
        self.decider = DecisionBot(settings)
        self.broker = PaperBroker(store)

    def log(self, bot, msg, level="info"):
        self.store.log(bot, msg, level, self._ts)

    def equity(self, trades, last_prices) -> tuple[float, float]:
        realized = sum(float(t["pnl_gbp"] or 0) for t in trades if t["status"] == "closed")
        unreal = sum(mark_to_market(t, last_prices[t["symbol"]]) for t in trades
                     if t["status"] == "open" and t["symbol"] in last_prices)
        return self.s.account["starting_balance"] + realized + unreal, realized

    def cycle(self, now: datetime) -> dict:
        now_ts = pd.Timestamp(now)
        self._ts = now_ts.isoformat()
        summary = {"new_orders": 0, "closed": 0, "candidates": 0, "rejected": 0, "failed_feeds": 0}
        approved = {}
        for a in self.store.strategies("approved"):
            approved.setdefault(a["symbol"], []).append(a)
        last_prices: dict[str, float] = {}
        symbols = self.s.all_symbols()
        fx_cache: dict[str, float] = {}

        for symbol in symbols:
            cls = self.s.asset_class_of(symbol)
            interval = self.s.interval_of(symbol)
            cost = cost_model_for(self.s, cls)
            step = interval_td(interval)
            lookback = step * HISTORY_BARS * (1.6 if interval == "1d" else 1.0)
            try:
                df = self.provider.history(symbol, interval, now - lookback, now)
                if df.empty:
                    raise RuntimeError("empty")
                self.safety.feed_ok(symbol)
            except Exception as exc:
                summary["failed_feeds"] += 1
                self.safety.feed_failure(symbol, self._ts)
                self.log("monitoring", f"{symbol}: data feed failed ({type(exc).__name__})", "warning")
                continue
            if is_stale(cls, interval, df.index[-1], now_ts, self.s["safety"]["stale_multiplier"]):
                self.log("monitoring", f"{symbol}: stale data (last bar {df.index[-1]}) — skipped", "warning")
                continue
            last_prices[symbol] = float(df["close"].iloc[-1])

            closed = self.broker.process(symbol, df, interval, cost, now_ts, self.log)
            summary["closed"] += len(closed)

            if symbol not in approved:
                continue
            ind = add_indicators(df)
            row, prev = ind.iloc[-1], ind.iloc[-2]
            bar_time = ind.index[-1]
            regime = classify_row(row)
            snap = scan(symbol, cls, ind)
            qc = quote_currency(symbol, cls)
            if qc not in fx_cache:
                fx_cache[qc] = self.provider.fx_to_gbp(qc, now)
            fx = fx_cache[qc]

            for appr in approved[symbol]:
                strat = strategy_from_key(appr["params"])
                sig = strat.signal(prev, row)
                if sig is None or (sig.direction == -1 and cls not in SHORTABLE):
                    continue
                cid = f"{symbol}|{strat.key}|{bar_time.isoformat()}"
                if self.store.candidates_exists(cid):
                    continue
                summary["candidates"] += 1
                news = self.news.check(symbol, cls, now)
                live_r = [float(t["r_multiple"]) for t in self.store.trades("closed")
                          if t["symbol"] == symbol and t["strategy"] == strat.key]
                dec = self.decider.decide(approval=appr, regime=regime, strategy_regimes=strat.regimes,
                                          scan=snap, news_vetoes=news, live_r=live_r)
                reasons = list(dec.reasons)
                risk_res = None
                if dec.trade:
                    trades = self.store.trades()
                    eq, _ = self.equity(trades, last_prices)
                    pnl_today, n_today = realized_today(trades, now_ts)
                    entry_est = cost.market_fill(float(row["close"]), sig.direction)
                    _, _, rpu = plan_levels(entry_est, float(row["atr14"]), sig, cost)
                    risk_res = self.risk.evaluate(
                        symbol=symbol, asset_class=cls, equity=eq, realized_today=pnl_today,
                        trades_today=n_today, open_trades=[t for t in trades if t["status"] in ("open", "pending")],
                        entry_est=entry_est, risk_per_unit=rpu, fx_to_gbp=fx,
                        kill_switch=self.safety.kill_switch_on())
                    reasons += risk_res.reasons
                go = dec.trade and risk_res is not None and risk_res.ok
                evidence = {**dec.evidence, "signal": sig.note}
                if risk_res is not None:
                    evidence.update({"qty": risk_res.quantity, "risk_gbp": risk_res.risk_gbp,
                                     "notional_gbp": risk_res.notional_gbp, "cost_r": risk_res.cost_r})
                self.store.record_candidate({
                    "candidate_id": cid, "bar_time": bar_time.isoformat(), "symbol": symbol,
                    "asset_class": cls, "strategy": strat.key, "direction": sig.direction,
                    "regime": regime, "price": float(row["close"]), "decision": "trade" if go else "reject",
                    "expected_r": dec.expected_r, "reasons": reasons, "evidence": evidence})
                if go:
                    self.broker.submit(candidate_id=cid, symbol=symbol, asset_class=cls,
                                       strategy_key=strat.key, signal=sig, regime=regime, bar_time=bar_time,
                                       atr=float(row["atr14"]), qty=risk_res.quantity,
                                       risk_gbp=risk_res.risk_gbp, risk_per_unit=rpu, fx=fx,
                                       notional=risk_res.notional_gbp, cost_r=risk_res.cost_r)
                    summary["new_orders"] += 1
                    self.log("trade_decision", f"ORDER {symbol} {'LONG' if sig.direction > 0 else 'SHORT'} "
                             f"[{strat.key}] EV {dec.expected_r:+.3f}R, risk £{risk_res.risk_gbp:.2f}")
                else:
                    summary["rejected"] += 1

        self.safety.end_of_cycle(summary["failed_feeds"], len(symbols), self._ts)
        trades = self.store.trades()
        eq, realized = self.equity(trades, last_prices)
        self.store.snapshot_equity(self._ts, eq, realized,
                                   sum(t["status"] == "open" for t in trades))
        if eq < self.s.account["starting_balance"] * 0.85:
            self.safety.engage(f"equity £{eq:.2f} is >15% below starting balance", self._ts)
        for msg in retirement_review(self.store, self.s):
            self.log("performance", msg, "warning")
        summary["equity"] = eq
        return summary


def run_demo(settings, store, start: datetime, days: int, relaxed: bool, log=print) -> dict:
    """Offline end-to-end run on synthetic data: research as of `start`, then
    step the clock hour by hour, exactly like the scheduled worker would."""
    from .bots.research import research
    from .data import SyntheticProvider
    provider = SyntheticProvider()
    gate = {"min_oos_trades": 10, "min_oos_expectancy_r": -1.0, "min_oos_profit_factor": 0.0,
            "min_holdout_trades": 1, "min_holdout_expectancy_r": -1.0} if relaxed else None
    research(settings, provider, store, start, gate_override=gate, log=log)
    eng = Engine(settings, provider, store, live=False)
    t = start
    end = start + timedelta(days=days)
    totals = {"new_orders": 0, "closed": 0}
    while t <= end:
        s = eng.cycle(t)
        totals["new_orders"] += s["new_orders"]
        totals["closed"] += s["closed"]
        t += timedelta(hours=1)
    rep = report(store.trades(), settings.account["starting_balance"])
    return {**totals, "overall": rep["overall"]}

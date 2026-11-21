"""Orchestrator: one cycle of the 10-bot pipeline.

  Safety -> market filters -> (per symbol) data -> stale check -> Auto-Trader manages
  positions -> Scanner -> Regime -> approved strategies' signals -> News veto
  -> Trade Decision (EV, may abstain) -> Risk Manager -> paper order
  -> Performance snapshot -> phone notifications

Kill switch ON: open positions are still managed (stops/targets honoured), no new trades.
Positions in symbols removed from the universe keep being managed until they close.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from . import notify
from .backtest import plan_levels, swing_level
from .bots import confluence as conf
from .bots.autotrader import PaperBroker, mark_to_market
from .bots.decision import DecisionBot
from .bots.learning import ShadowView, apply_learning, researched_params, shadow_signals
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


def market_filter_status(settings, provider, now: datetime) -> dict[str, dict]:
    """Per asset class: {'symbol': 'SPY', 'ok': bool, 'exclude': [...]}. ok=None if unknown."""
    out = {}
    for cls, spec in (settings.raw.get("market_filter") or {}).items():
        sym = spec["symbol"]
        try:
            df = provider.history(sym, "1d", now - timedelta(days=500), now)
            c = df["close"]
            ok = bool(c.iloc[-1] > c.rolling(200).mean().iloc[-1]) if len(c) >= 200 else None
        except Exception:
            ok = None
        out[cls] = {"symbol": sym, "ok": ok, "exclude": set(spec.get("exclude", []))}
    return out


class Engine:
    def __init__(self, settings, provider, store, live: bool = True):
        self.s, self.provider, self.store, self.live = settings, provider, store, live
        self.safety = SafetyBot(store, settings)
        self.news = NewsBot(settings, live)
        self.risk = RiskManager(settings)
        self.decider = DecisionBot(settings)
        self.broker = PaperBroker(store, on_open=notify.trade_opened, on_close=notify.trade_closed)
        self.shadow_view = ShadowView(store)
        self.shadow = PaperBroker(self.shadow_view)          # Learning Lab: no alerts
        self.learning_on = bool(settings.raw.get("learning", {}).get("enabled", True))

    def log(self, bot, msg, level="info"):
        self.store.log(bot, msg, level, self._ts)

    def equity(self, trades, last_prices) -> tuple[float, float]:
        realized = sum(float(t["pnl_gbp"] or 0) for t in trades if t["status"] == "closed")
        unreal = sum(mark_to_market(t, last_prices[t["symbol"]]) for t in trades
                     if t["status"] == "open" and t["symbol"] in last_prices)
        return self.s.account["starting_balance"] + realized + unreal, realized

    def _symbols(self) -> list[tuple[str, str, bool]]:
        """(symbol, asset_class, tradable). Non-tradable = only managing an old position."""
        out = [(s, self.s.asset_class_of(s), True) for s in self.s.all_symbols()]
        known = {s for s, _, _ in out}
        for t in self.store.trades():
            if t["status"] in ("open", "pending") and t["symbol"] not in known and t.get("asset_class") in self.s["universe"]:
                out.append((t["symbol"], t["asset_class"], False))
                known.add(t["symbol"])
        return out

    def cycle(self, now: datetime) -> dict:
        now_ts = pd.Timestamp(now)
        self._ts = now_ts.isoformat()
        summary = {"new_orders": 0, "closed": 0, "candidates": 0, "rejected": 0, "failed_feeds": 0,
                   "shadow_opened": 0, "shadow_closed": 0}
        if self.learning_on:
            self.shadow_view.refresh()
            shadow_params = researched_params(self.store)
        approved = {}
        for a in self.store.strategies("approved"):
            approved.setdefault(a["symbol"], []).append(a)
        last_prices: dict[str, float] = {}
        symbols = self._symbols()
        fx_cache: dict[str, float] = {}
        filters = market_filter_status(self.s, self.provider, now)
        was_killed = self.safety.kill_switch_on()

        for symbol, cls, tradable in symbols:
            interval = self.s["universe"][cls]["interval"]
            cost = cost_model_for(self.s, cls)
            step = interval_td(interval)
            lookback = step * HISTORY_BARS * (1.6 if interval == "1d" and cls != "crypto" else 1.0)
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
            ind = add_indicators(df)
            mf_state = (filters.get(cls) or {}).get("ok")
            if tradable:
                self._snapshot(symbol, cls, ind, mf_state)

            closed = self.broker.process(symbol, ind, interval, cost, now_ts, self.log)
            summary["closed"] += len(closed)

            if self.learning_on:  # Learning Lab: every strategy, every market, practice ledger
                summary["shadow_closed"] += len(self.shadow.process(symbol, ind, interval, cost, now_ts,
                                                                    lambda *a, **k: None))
                if tradable:
                    summary["shadow_opened"] += shadow_signals(self.shadow_view, self.shadow, symbol, cls, ind,
                                                               classify_row(ind.iloc[-1]), shadow_params, mf_state)

            if not tradable or symbol not in approved:
                continue
            row, prev = ind.iloc[-1], ind.iloc[-2]
            bar_time = ind.index[-1]
            regime = classify_row(row)
            snap = scan(symbol, cls, ind)
            qc = quote_currency(symbol, cls)
            if qc not in fx_cache:
                fx_cache[qc] = self.provider.fx_to_gbp(qc, now)
            fx = fx_cache[qc]
            mf = filters.get(cls)

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
                if mf and sig.direction == 1 and symbol not in mf["exclude"] and mf["ok"] is False:
                    news = news + [f"market filter: {mf['symbol']} is below its 200-day average"]
                live_r = [float(t["r_multiple"]) for t in self.store.trades("closed")
                          if t["symbol"] == symbol and t["strategy"] == strat.key]
                c_score, c_hits = conf.score(row, sig.direction, None if not mf else mf["ok"])
                dec = self.decider.decide(approval=appr, regime=regime, strategy_regimes=strat.regimes,
                                          scan=snap, news_vetoes=news, live_r=live_r)
                reasons = list(dec.reasons)
                min_conf = int(self.s.raw.get("confluence", {}).get("min_score", 0))
                if min_conf and c_score < min_conf:
                    reasons.append(f"confluence {c_score}/{conf.MAX_SCORE} below minimum {min_conf}")
                    dec.trade = False
                risk_res = None
                rpu = None
                if dec.trade:
                    trades = self.store.trades()
                    eq, _ = self.equity(trades, last_prices)
                    pnl_today, n_today = realized_today(trades, now_ts)
                    entry_est = cost.market_fill(float(row["close"]), sig.direction)
                    _, _, rpu = plan_levels(entry_est, float(row["atr14"]), sig, cost,
                                            swing_level(row, sig.direction))
                    risk_res = self.risk.evaluate(
                        symbol=symbol, asset_class=cls, equity=eq, realized_today=pnl_today,
                        trades_today=n_today, open_trades=[t for t in trades if t["status"] in ("open", "pending")],
                        entry_est=entry_est, risk_per_unit=rpu, fx_to_gbp=fx,
                        kill_switch=self.safety.kill_switch_on())
                    reasons += risk_res.reasons
                go = dec.trade and risk_res is not None and risk_res.ok
                evidence = {**dec.evidence, "signal": sig.note, "confluence": c_score,
                            "confluence_factors": c_hits}
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
                                       notional=risk_res.notional_gbp, cost_r=risk_res.cost_r,
                                       swing=swing_level(row, sig.direction), confluence=c_score,
                                       conf_text=conf.describe(c_hits))
                    summary["new_orders"] += 1
                    self.log("trade_decision", f"ORDER {symbol} {'LONG' if sig.direction > 0 else 'SHORT'} "
                             f"[{strat.key}] EV {dec.expected_r:+.3f}R, risk £{risk_res.risk_gbp:.2f}")
                else:
                    summary["rejected"] += 1

        if self.live:
            self._fear_greed()
        self.safety.end_of_cycle(summary["failed_feeds"], len(symbols), self._ts)
        trades = self.store.trades()
        eq, realized = self.equity(trades, last_prices)
        n_open = sum(t["status"] == "open" for t in trades)
        self.store.snapshot_equity(self._ts, eq, realized, n_open)
        if eq < self.s.account["starting_balance"] * 0.85:
            self.safety.engage(f"equity £{eq:.2f} is >15% below starting balance", self._ts)
        for msg in retirement_review(self.store, self.s):
            self.log("performance", msg, "warning")
        if self.learning_on:
            for msg in apply_learning(self.store, self.s, self.log):
                notify.send("MarketPilot learned something", msg, tags="brain", priority=3)
            if summary["shadow_opened"] or summary["shadow_closed"]:
                self.log("learning_lab", f"shadow trades opened {summary['shadow_opened']}, "
                         f"closed {summary['shadow_closed']}")
        if not was_killed and self.safety.kill_switch_on():
            notify.kill_switch(self.store.get_control("pause_reason", "") or "safety rule triggered")
        self._daily_summary(now_ts, trades, eq, n_open)
        summary["equity"] = eq
        return summary

    def _snapshot(self, symbol: str, cls: str, ind, market_ok) -> None:
        """Market overview row for the dashboard's Markets tab."""
        try:
            row = ind.iloc[-1]
            c = ind["close"]

            def chg(n):
                return float(c.iloc[-1] / c.iloc[-1 - n] - 1) if len(c) > n else None
            cl, _ = conf.score(row, 1, market_ok)
            cs, _ = conf.score(row, -1, market_ok)
            self.store.save_market({
                "symbol": symbol, "asset_class": cls, "ts": self._ts, "bar_time": str(ind.index[-1]),
                "price": float(row["close"]), "chg_1d": chg(1), "chg_5d": chg(5), "chg_20d": chg(20),
                "above_200d": None if pd.isna(row["sma200"]) else int(row["close"] > row["sma200"]),
                "rsi14": float(row["rsi14"]), "regime": classify_row(row),
                "atr_pct": None if pd.isna(row["atr_pct"]) else float(row["atr_pct"]),
                "conf_long": cl, "conf_short": cs})
        except Exception as exc:  # the overview must never break trading
            self.log("monitoring", f"{symbol}: market snapshot failed ({type(exc).__name__})", "warning")

    def _fear_greed(self) -> None:
        """Crypto Fear & Greed Index (alternative.me) — recorded as learning data, not a signal."""
        try:
            import json

            import requests
            r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10).json()["data"][0]
            self.store.set_control("fear_greed", json.dumps(
                {"value": int(r["value"]), "label": r["value_classification"], "ts": self._ts}))
        except Exception:
            pass

    def _daily_summary(self, now_ts: pd.Timestamp, trades, eq: float, n_open: int) -> None:
        hour = int(self.s.raw.get("notifications", {}).get("daily_summary_hour_utc", 21))
        today = now_ts.strftime("%Y-%m-%d")
        if now_ts.hour >= hour and self.store.get_control("summary_sent", "") != today:
            rep = report(trades, self.s.account["starting_balance"])
            notify.daily_summary(rep["overall"], eq, n_open, self.s["goals"]["target_win_rate"])
            self.store.set_control("summary_sent", today)


def run_demo(settings, store, start: datetime, days: int, relaxed: bool, log=print) -> dict:
    """Offline end-to-end run on synthetic data: research as of `start`, then
    step the clock hour by hour, exactly like the scheduled worker would."""
    from .bots.research import research
    from .data import SyntheticProvider
    provider = SyntheticProvider()
    gate = {"min_oos_trades": 10, "min_oos_expectancy_r": -1.0, "min_oos_profit_factor": 0.0,
            "min_oos_tstat": None, "min_holdout_trades": 1, "min_holdout_expectancy_r": -1.0} if relaxed else None
    research(settings, provider, store, start, gate_override=gate, log=log)
    eng = Engine(settings, provider, store, live=False)
    t = start
    end = start + timedelta(days=days)
    totals = {"new_orders": 0, "closed": 0}
    step = timedelta(hours=1) if any(u["interval"] != "1d" for u in settings["universe"].values()) else timedelta(hours=6)
    while t <= end:
        s = eng.cycle(t)
        totals["new_orders"] += s["new_orders"]
        totals["closed"] += s["closed"]
        t += step
    rep = report(store.trades(), settings.account["starting_balance"])
    return {**totals, "overall": rep["overall"]}

"""Learning Lab — shadow bots + the Learning bot.

Shadow bots paper-trade EVERY strategy on EVERY market (approved or not) in a
separate practice ledger, `mp_shadow_trades`. They use the same entry rules and
the same fill/exit engine as the real paper account, but no position limits, no
capital cap and no phone alerts, and results are counted in R only (1R = 1 unit
of risk). That collects forward-test evidence many times faster than the real
paper account can, without loosening its rules.

The Learning bot reads that evidence per market group x strategy:
  * PROMOTE a strategy that failed the backtest gate if its shadow forward results
    are strong (enough trades, positive expectancy, t-stat >= threshold) and its
    backtest was at least positive;
  * DEMOTE an approved strategy whose shadow forward results are negative.
Research uses the same verdict, so the weekly re-test cannot flip it back.
"""
from __future__ import annotations

import pandas as pd

from ..metrics import summarize
from ..strategies import ALL_STRATEGIES, allows_short, strategies_for


class ShadowView:
    """Adapts the store so PaperBroker reads/writes the shadow ledger.
    Caches the ledger for one cycle so 66 markets don't mean 66 database reads."""

    def __init__(self, store):
        self.store = store
        self._cache: list[dict] | None = None

    def refresh(self) -> None:
        self._cache = [t for t in self.store.shadow_trades() if t["status"] in ("pending", "open")]

    def trades(self, status: str | None = None) -> list[dict]:
        if self._cache is None:
            self.refresh()
        return [t for t in self._cache if status is None or t["status"] == status]

    def open_trade(self, row: dict) -> bool:
        ok = self.store.open_shadow(row)
        if ok and self._cache is not None:
            self._cache.append({"status": "open", "bars_held": 0, **row})
        return ok

    def update_trade(self, trade_id: str, changes: dict) -> None:
        self.store.update_shadow(trade_id, changes)
        for t in self._cache or []:
            if t["trade_id"] == trade_id:
                t.update(changes)
        if self._cache is not None:
            self._cache = [t for t in self._cache if t["status"] in ("pending", "open")]


def researched_params(store) -> dict[str, str]:
    """symbol|strategy -> the parameter set research chose (any status)."""
    return {s["strategy_id"]: s["params"] for s in store.strategies() if s.get("params")}


def shadow_signals(view: ShadowView, broker, symbol: str, cls: str, ind: pd.DataFrame,
                   regime: str, params: dict[str, str], market_ok: bool | None = None,
                   cost=None, tm_paths: int = 0, cv_cfg: dict | None = None, mgmt: str = "base",
                   settings=None, now=None, late_check: str | None = None,
                   found: list | None = None) -> int:
    """Open a shadow trade for every strategy that signals on the latest bar and
    has no open shadow position on this symbol. Returns the number opened."""
    from ..backtest import swing_level
    from ..strategies import strategy_from_key
    from . import confluence as conf
    from . import conviction as cv
    from . import timemachine as tm

    row, prev = ind.iloc[-1], ind.iloc[-2]
    bar_time = ind.index[-1]
    from ..data import signal_is_late
    if late_check is not None and signal_is_late(bar_time, late_check, symbol, now):
        return 0          # v0.13.1: never open a practice trade whose entry price is already in the past
    open_keys = {(t["symbol"], t["strategy"].split("(")[0]) for t in view.trades()}
    opened = 0
    for cls_ in (strategies_for(cls, settings) if settings is not None else ALL_STRATEGIES):
        key = params.get(f"{symbol}|{cls_.name}")
        strat = strategy_from_key(key) if key else cls_()
        if (symbol, strat.name) in open_keys:
            continue
        sig = strat.signal(prev, row)
        if sig is None or (sig.direction == -1 and not allows_short(cls, strat, settings)) or pd.isna(row["atr14"]):
            continue
        odds = None
        if tm_paths and cost is not None:
            try:
                odds = tm.odds_for(ind, sig, cost, swing_level(row, sig.direction), n_paths=tm_paths)
            except Exception:
                odds = None
        ok = broker.submit(
            candidate_id=f"shadow|{symbol}|{strat.key}|{bar_time.isoformat()}", symbol=symbol,
            asset_class=cls, strategy_key=strat.key, signal=sig, regime=regime, bar_time=bar_time,
            atr=float(row["atr14"]), qty=0.0, risk_gbp=1.0, risk_per_unit=1.0, fx=1.0,
            notional=0.0, cost_r=0.0, swing=swing_level(row, sig.direction),
            confluence=(cs := conf.score(row, sig.direction, market_ok))[0], conf_text=conf.describe(cs[1]),
            extra={**tm.trade_fields(odds), "grade": (g := cv.grade(cs[0], odds, cv_cfg)), "mgmt": mgmt},
            extra_note=cv.describe(g, cs[0], odds) + (f" | {odds.describe()}" if odds else ""))
        opened += int(bool(ok))
        if ok and found is not None:   # v0.14 daily scan
            found.append({"bar": bar_time.isoformat(), "symbol": symbol, "asset_class": cls,
                          "strategy": strat.name, "side": "LONG" if sig.direction > 0 else "SHORT",
                          "grade": g, "confluence": cs[0], "p_target": None if odds is None else odds.p_target,
                          "status": "practice", "why": "practice trade in the Learning lab"})
    return opened


PROMOTED = "promoted"   # reasons starting with this mark a strategy that earned approval live

PIPELINE_DEFAULTS = {"enabled": True, "new_families": [], "new_min_tstat": 3.0, "probation_min_trades": 20,
                     "probation_min_days": 28, "probation_min_expectancy_r": 0.0, "probation_min_pf": 1.0,
                     "probation_fail_below_r": -0.10}


def pipeline_cfg(settings) -> dict:
    raw = getattr(settings, "raw", settings) or {}
    return {**PIPELINE_DEFAULTS, **(raw.get("pipeline") or {})}


def is_new_strategy(name: str, cfg: dict) -> bool:
    """New strategies go through the v0.14 pipeline: higher backtest bar, then probation."""
    if not cfg.get("enabled", True):
        return False
    base = name[:-6] if name.endswith("_short") else name
    return name.endswith("_short") or base in set(cfg.get("new_families") or [])


def practice_start(store) -> dict[tuple, pd.Timestamp]:
    """First practice trade date per (asset_class, strategy family), from all shadow trades."""
    out: dict[tuple, pd.Timestamp] = {}
    for t in store.shadow_trades():
        if t.get("status") == "cancelled" or not t.get("opened_at"):
            continue
        k = (t["asset_class"], str(t["strategy"]).split("(")[0])
        ts = pd.Timestamp(t["opened_at"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        if k not in out or ts < out[k]:
            out[k] = ts
    return out


def probation_verdict(stats: pd.DataFrame, starts: dict, cls: str, strategy: str, cfg: dict,
                      now: pd.Timestamp) -> tuple[str | None, str]:
    """('promote' | 'fail' | None, reason) for a strategy on probation."""
    if stats is None or stats.empty:
        return None, ""
    m = stats[(stats["asset_class"] == cls) & (stats["strategy"] == strategy)]
    if m.empty:
        return None, ""
    r = m.iloc[0]
    n, exp_r, pf = int(r["trades"]), float(r["expectancy_r"]), float(r.get("profit_factor", 0) or 0)
    if n < cfg["probation_min_trades"]:
        return None, ""
    if exp_r < cfg["probation_fail_below_r"]:
        return "fail", f"failed probation: {n} practice trades averaging {exp_r:+.3f}R"
    start = starts.get((cls, strategy))
    days = (now - start).days if start is not None else 0
    if days >= cfg["probation_min_days"] and exp_r >= cfg["probation_min_expectancy_r"] \
            and pf >= cfg["probation_min_pf"]:
        return "promote", (f"{PROMOTED} after probation: {n} practice trades over {days} days, "
                           f"{exp_r:+.3f}R per trade, profit factor {pf:.2f}")
    return None, ""


def forward_stats(store) -> pd.DataFrame:
    """Closed shadow trades summarised per (asset_class, strategy family name)."""
    rows = [t for t in store.shadow_trades("closed")]
    if not rows:
        return pd.DataFrame(columns=["asset_class", "strategy", "trades", "win_rate", "expectancy_r", "t_stat"])
    df = pd.DataFrame(rows)
    df["strategy"] = df["strategy"].str.split("(").str[0]
    out = []
    for (cls, strat), g in df.groupby(["asset_class", "strategy"]):
        s = summarize(g["r_multiple"].astype(float))
        out.append({"asset_class": cls, "strategy": strat, **s})
    return pd.DataFrame(out)


def confluence_stats(store) -> pd.DataFrame:
    """Closed shadow trades by confluence bucket: does more confluence win more often?"""
    from .confluence import bucket
    rows = [t for t in store.shadow_trades("closed") if t.get("confluence") is not None]
    if not rows:
        return pd.DataFrame(columns=["confluence", "trades", "win_rate", "expectancy_r", "t_stat"])
    df = pd.DataFrame(rows)
    df["bucket"] = df["confluence"].astype(int).map(bucket)
    out = []
    for b, g in df.groupby("bucket"):
        s = summarize(g["r_multiple"].astype(float))
        out.append({"confluence": b, **s})
    order = {"low (0-3)": 0, "medium (4-5)": 1, "high (6-8)": 2}
    return pd.DataFrame(sorted(out, key=lambda r: order.get(r["confluence"], 9)))


def forward_verdict(stats: pd.DataFrame, cls: str, strategy: str, cfg: dict) -> tuple[str | None, str]:
    """('promote' | 'demote' | None, reason) for one market group x strategy."""
    if stats is None or stats.empty:
        return None, ""
    m = stats[(stats["asset_class"] == cls) & (stats["strategy"] == strategy)]
    if m.empty:
        return None, ""
    r = m.iloc[0]
    if (r["trades"] >= cfg["promote_min_trades"] and r["expectancy_r"] >= cfg["promote_min_expectancy_r"]
            and r["t_stat"] >= cfg["promote_min_tstat"]):
        return "promote", (f"{PROMOTED} by forward test: {int(r['trades'])} shadow trades, "
                           f"{r['expectancy_r']:+.3f}R, t {r['t_stat']:.1f}, win {r['win_rate']:.0%}")
    if r["trades"] >= cfg["demote_min_trades"] and r["expectancy_r"] < cfg["demote_below_expectancy_r"]:
        return "demote", (f"demoted by forward test: {int(r['trades'])} shadow trades, "
                          f"{r['expectancy_r']:+.3f}R")
    return None, ""


def apply_learning(store, settings, log=lambda *a: None) -> list[str]:
    """Promote / demote strategies from forward (shadow) evidence. Returns messages."""
    cfg = settings["learning"]
    stats = forward_stats(store)
    if stats.empty:
        return []
    msgs: dict[tuple, str] = {}
    pipe = pipeline_cfg(settings)
    starts = practice_start(store) if any(s["status"] == "probation" for s in store.strategies()) else {}
    now = pd.Timestamp.now(tz="UTC")
    for s in store.strategies():
        if s["status"] in ("retired", "removed"):
            continue
        if s["status"] == "probation":
            pv, why = probation_verdict(stats, starts, s["asset_class"], s["strategy"], pipe, now)
            if pv:
                store.save_strategy({**s, "status": "approved" if pv == "promote" else "rejected",
                                     "reasons": [why]})
                msgs[(s["asset_class"], s["strategy"])] = f"{s['asset_class']} {s['strategy']}: {why}"
            continue
        verdict, why = forward_verdict(stats, s["asset_class"], s["strategy"], cfg)
        oos = s.get("oos") or {}
        if verdict == "promote" and s["status"] != "approved" and oos.get("expectancy_r", -1) > 0:
            store.save_strategy({**s, "status": "approved", "reasons": [why]})
            msgs[(s["asset_class"], s["strategy"])] = f"{s['asset_class']} {s['strategy']}: {why}"
        elif verdict == "demote" and s["status"] == "approved":
            store.save_strategy({**s, "status": "rejected", "reasons": [why]})
            msgs[(s["asset_class"], s["strategy"])] = f"{s['asset_class']} {s['strategy']}: {why}"
    for m in msgs.values():
        log("learning", m, "warning")
    return list(msgs.values())


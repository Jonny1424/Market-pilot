"""Strategy Research Bot + Backtesting Bot runner.

For each asset class x strategy family:
  1. walk-forward test every symbol separately (parameters chosen in-sample only),
  2. POOL the out-of-sample and holdout trades across the class — one symbol on
     daily bars rarely produces enough trades to judge anything,
  3. apply the approval gate to the pooled evidence.
Only approved strategies can be traded. Retired strategies stay retired until
you reset them, so the system can't flip back into a strategy that just failed
forward testing.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from ..backtest import (confluence_breakdown, management_breakdown, market_filter_series, passes_gate,
                        walk_forward)
from ..costs import cost_model_for
from ..metrics import summarize
from ..storage import now_iso
from ..strategies import allows_short, strategies_for
from .learning import PROMOTED, is_new_strategy, pipeline_cfg
from .learning import forward_stats, forward_verdict
from .losscontrol import compare as lc_compare
from .losscontrol import config as lc_config


def research(settings, provider, store, now: datetime, symbols: list[str] | None = None,
             gate_override: dict | None = None, log=print) -> list[dict]:
    gate = {**settings["approval"], **(gate_override or {})}
    retired = {s["strategy_id"] for s in store.strategies("retired")}
    wanted = set(symbols or settings.all_symbols())
    universe = set(settings.all_symbols())
    for old in store.strategies("approved"):  # symbols dropped from the universe stop trading
        if old["symbol"] not in universe:
            store.save_strategy({**old, "status": "removed", "reasons": ["symbol no longer in universe"]})
    filters = settings.raw.get("market_filter") or {}
    learn_cfg = settings.raw.get("learning")
    lc_cfg = lc_config(settings.raw)
    tm_cfg = settings.raw.get("trade_management") or {}
    mgmt = tm_cfg.get("mode", "base")
    compare_modes = tuple(tm_cfg.get("compare", ["base", "breakeven", "partial"]))
    fstats = forward_stats(store) if learn_cfg else None
    pipe = pipeline_cfg(settings)
    earned = {(x["asset_class"], x["strategy"]) for x in store.strategies("approved")
              if any(str(r).startswith(PROMOTED) for r in (x.get("reasons") or []))}
    rows_out = []
    for cls_name, spec in settings["universe"].items():
        data = {}
        for symbol in spec["symbols"]:
            if symbol not in wanted:
                continue
            try:
                df = provider.history(symbol, spec["interval"], now - timedelta(days=spec["lookback_days"]), now)
            except Exception as exc:
                log(f"research: {symbol} data failed ({type(exc).__name__})")
                continue
            if len(df) < 600:
                log(f"research: {symbol} only {len(df)} bars, skipped")
                continue
            data[symbol] = df
        if not data:
            continue
        cost = cost_model_for(settings, cls_name)
        filt_spec = filters.get(cls_name)
        filt_df = None
        if filt_spec:
            filt_df = data.get(filt_spec["symbol"])
            if filt_df is None:
                try:
                    filt_df = provider.history(filt_spec["symbol"], "1d",
                                               now - timedelta(days=spec["lookback_days"]), now)
                except Exception:
                    filt_df = None
        for strat in strategies_for(cls_name, settings):
            per_symbol, oos_r, hold_r, costs, oos_all = {}, [], [], [], []
            for symbol, df in data.items():
                mk = None
                if filt_spec and symbol not in set(filt_spec.get("exclude", [])):
                    mk = market_filter_series(filt_df, df.index)
                r = walk_forward(df, strat, cost, allows_short(cls_name, strat, settings),
                                 folds=gate["walk_forward_folds"], holdout_fraction=gate["holdout_fraction"],
                                 market_ok=mk, mgmt=mgmt, compare_modes=compare_modes)
                per_symbol[symbol] = r
                oos_r += [t.r_multiple for t in r["oos_trades"]]
                oos_all += r["oos_trades"]
                hold_r += [t.r_multiple for t in r["holdout_trades"]]
                costs += [t.cost_r for t in r["oos_trades"]]
            pooled = {"oos": summarize(oos_r), "holdout": summarize(hold_r)}
            by_conf = confluence_breakdown(oos_all)
            lc = lc_compare(sorted(oos_all, key=lambda t: t.entry_time), lc_cfg)
            hold_all = [t for r in per_symbol.values() for t in r["holdout_trades"]]
            mg = management_breakdown(sorted(oos_all, key=lambda t: t.entry_time), hold_all, compare_modes)
            new = is_new_strategy(strat.name, pipe)
            gate_s = {**gate, "min_oos_tstat": max(gate.get("min_oos_tstat") or 0, pipe["new_min_tstat"])} \
                if new else gate
            ok, reasons = passes_gate(pooled, gate_s)
            if learn_cfg:  # forward (shadow) evidence can promote or demote
                verdict, why = forward_verdict(fstats, cls_name, strat.name, learn_cfg)
                if verdict == "promote" and not ok and pooled["oos"]["expectancy_r"] > 0:
                    ok, reasons = True, [why]
                elif verdict == "demote" and ok:
                    ok, reasons = False, [why]
            status = "approved" if ok else "rejected"
            if ok and new and (cls_name, strat.name) not in earned:
                status = "probation"
                reasons = [f"passed the backtest gate (new strategy, t-stat >= {pipe['new_min_tstat']}); on "
                           f"probation until it proves itself in the Learning lab"]
            o = pooled["oos"]
            log(f"research: {cls_name:<7} {strat.name:<20} {status.upper() if ok else 'rejected':<9} "
                f"OOS n={o['trades']:<4} win {o['win_rate']:.0%} exp {o['expectancy_r']:+.3f}R "
                f"PF {o['profit_factor']:.2f} t {o['t_stat']:.1f} | holdout n={pooled['holdout']['trades']} "
                f"exp {pooled['holdout']['expectancy_r']:+.3f}R" + ("" if ok else f"  — {reasons[0]}"))
            if lc:
                log(f"research: {cls_name:<7} {strat.name:<20} loss control: max drawdown "
                    f"{lc['flat']['max_drawdown_r']:.1f}R -> {lc['scaled']['max_drawdown_r']:.1f}R "
                    f"({lc['dd_cut_pct']:.0%} smaller), total {lc['flat']['total_r']:+.1f}R -> "
                    f"{lc['scaled']['total_r']:+.1f}R, {lc['share_reduced']:.0%} of trades sized down")
            if mg:
                log(f"research: {cls_name:<7} {strat.name:<20} management: " + " | ".join(
                    f"{m} exp {v['expectancy_r']:+.3f}R full-loss {v['full_loss_rate']:.0%} "
                    f"DD {v['max_drawdown_r']:.1f}R" for m, v in mg.items()))
            for symbol, r in per_symbol.items():
                sid = f"{symbol}|{strat.name}"
                if sid in retired:
                    continue
                row = {"strategy_id": sid, "symbol": symbol, "asset_class": cls_name,
                       "strategy": strat.name, "family": strat.family, "params": r["final_params"],
                       "status": status, "reasons": reasons,
                       "oos": {**pooled["oos"], "symbol_trades": r["oos"]["trades"],
                               "symbol_expectancy_r": r["oos"]["expectancy_r"], "by_confluence": by_conf,
                               "loss_control": lc, "management": mg, "mgmt_mode": mgmt},
                       "holdout": pooled["holdout"],
                       "avg_cost_r": float(sum(costs) / len(costs)) if costs else 0.0,
                       "researched_at": now_iso(), "data_start": r["data_start"], "data_end": r["data_end"]}
                store.save_strategy(row)
                rows_out.append(row)
    n_ok = len({(r["asset_class"], r["strategy"]) for r in rows_out if r["status"] == "approved"})
    n_prob = len({(r["asset_class"], r["strategy"]) for r in rows_out if r["status"] == "probation"})
    store.log("strategy_research", f"research complete: {n_ok} strategy/market combinations approved, "
                                   f"{n_prob} on probation")
    try:
        store.prune(int((settings.raw.get("storage") or {}).get("keep_log_days", 60)))
    except Exception:
        pass
    return rows_out

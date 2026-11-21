from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from marketpilot.backtest import (check_exit, market_filter_series, passes_gate, plan_levels, realized_r,
                                  run_backtest, step_position)
from marketpilot.bots.autotrader import PaperBroker
from marketpilot.bots.decision import DecisionBot
from marketpilot.bots.risk import RiskManager
from marketpilot.bots.safety import SafetyBot, is_stale
from marketpilot.config import load_settings
from marketpilot.costs import CostModel
from marketpilot.data import SyntheticProvider
from marketpilot.engine import Engine
from marketpilot.indicators import add_indicators
from marketpilot.metrics import summarize
from marketpilot.storage import SQLiteStore
from marketpilot.strategies import (NO_TARGET, Breakout, MeanReversion, Signal, TrendFollowing, TrendSMA200,
                                    strategy_from_key)

UTC = timezone.utc


@pytest.fixture
def settings():
    return load_settings()


@pytest.fixture
def small_settings():
    s = load_settings()
    s.raw["universe"] = {"forex": {"interval": "1d", "lookback_days": 3000, "symbols": ["EURUSD=X"]},
                         "stocks": {"interval": "1d", "lookback_days": 3000, "symbols": ["SPY"]}}
    s.raw["market_filter"] = {}
    return s


@pytest.fixture
def store(tmp_path):
    return SQLiteStore(str(tmp_path / "t.db"))


def test_expectancy_matches_v04_doc_example():
    # 55% win rate, 1.5:1 average win vs loss -> +0.375R per trade, +37.5R over 100
    r = [1.5] * 55 + [-1.0] * 45
    s = summarize(r)
    assert s["win_rate"] == pytest.approx(0.55)
    assert s["expectancy_r"] == pytest.approx(0.375)
    assert s["total_r"] == pytest.approx(37.5)


def test_high_win_rate_can_still_lose():
    s = summarize([0.3] * 75 + [-1.2] * 25)
    assert s["win_rate"] == 0.75 and s["expectancy_r"] < 0


def test_costs_are_always_adverse():
    c = CostModel(spread_bps=10, slippage_bps=5)
    assert c.market_fill(100, +1) > 100 > c.market_fill(100, -1)
    assert c.limit_fill(100, -1) < 100


def test_indicators_have_no_lookahead():
    df = SyntheticProvider().history("SPY", "1d", datetime(2019, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC))
    full = add_indicators(df)
    cut = add_indicators(df.iloc[:400])
    cols = ["ema20", "ema50", "atr14", "rsi2", "er20", "dc_high", "sma200"]
    pd.testing.assert_frame_equal(full[cols].iloc[:400], cut[cols])


def test_gap_through_stop_fills_at_open():
    c = CostModel()
    bar = {"open": 90.0, "high": 95.0, "low": 89.0, "close": 94.0}
    px, reason = check_exit(bar, +1, stop=95.0, target=110.0, cost=c)
    assert reason == "stop (gap)" and px == 90.0


def test_stop_and_target_same_bar_assumes_stop():
    c = CostModel()
    bar = {"open": 100.0, "high": 111.0, "low": 94.0, "close": 100.0}
    _, reason = check_exit(bar, +1, stop=95.0, target=110.0, cost=c)
    assert reason == "stop"


def test_stop_out_is_about_minus_one_r_after_costs():
    c = CostModel(spread_bps=4, slippage_bps=4, commission_pct=0.1)
    sig = Signal(1, 2.0, 3.0, 10)
    entry = c.market_fill(100, 1)
    stop, target, rpu = plan_levels(entry, 1.0, sig, c)
    r, _ = realized_r(1, entry, c.market_fill(stop, -1), rpu, c)
    assert r == pytest.approx(-1.0, abs=0.01)


def test_strategy_key_roundtrip():
    for s in TrendFollowing.variants() + MeanReversion.variants():
        assert strategy_from_key(s.key).key == s.key


def test_risk_manager_sizes_to_two_pounds_fifty(settings):
    rm = RiskManager(settings)
    res = rm.evaluate(symbol="AAPL", asset_class="stocks", equity=500, realized_today=0, trades_today=0,
                      open_trades=[], entry_est=10.0, risk_per_unit=0.5, fx_to_gbp=1.0, kill_switch=False)
    assert res.ok and res.risk_gbp == pytest.approx(2.50) and res.quantity == pytest.approx(5.0)


def test_risk_manager_enforces_no_leverage(settings):
    rm = RiskManager(settings)
    # tight stop -> huge position -> must be scaled to fit £500 notional or rejected
    res = rm.evaluate(symbol="SPY", asset_class="stocks", equity=500, realized_today=0, trades_today=0,
                      open_trades=[], entry_est=500.0, risk_per_unit=1.0, fx_to_gbp=1.0, kill_switch=False)
    assert (not res.ok) or res.notional_gbp <= 500 + 1e-9


def test_risk_manager_blocks_on_kill_switch_and_daily_loss(settings):
    rm = RiskManager(settings)
    kw = dict(symbol="AAPL", asset_class="stocks", equity=500, trades_today=0, open_trades=[],
              entry_est=10.0, risk_per_unit=0.5, fx_to_gbp=1.0)
    assert not rm.evaluate(realized_today=0, kill_switch=True, **kw).ok
    assert not rm.evaluate(realized_today=-10.0, kill_switch=False, **kw).ok  # -2% of £500


def test_risk_manager_rejects_when_costs_eat_the_risk(settings):
    rm = RiskManager(settings)
    # crypto ~0.6% round trip on a very wide stop relative to size -> cost > 35% of R
    res = rm.evaluate(symbol="BTC-USD", asset_class="crypto", equity=500, realized_today=0, trades_today=0,
                      open_trades=[], entry_est=100.0, risk_per_unit=0.3, fx_to_gbp=1.0, kill_switch=False)
    assert not res.ok and "costs" in res.reasons[0]


def test_decision_bot_abstains_on_news_and_thin_edge(settings):
    d = DecisionBot(settings)
    appr = {"oos": {"trades": 200, "expectancy_r": 0.3, "win_rate": 0.4}}
    scan = {"ok": True, "reasons": []}
    assert d.decide(approval=appr, regime="trending", strategy_regimes={"trending"}, scan=scan,
                    news_vetoes=[], live_r=[]).trade
    assert not d.decide(approval=appr, regime="trending", strategy_regimes={"trending"}, scan=scan,
                        news_vetoes=["event blackout: CPI"], live_r=[]).trade
    weak = {"oos": {"trades": 30, "expectancy_r": 0.06, "win_rate": 0.7}}
    assert not d.decide(approval=weak, regime="trending", strategy_regimes={"trending"}, scan=scan,
                        news_vetoes=[], live_r=[]).trade  # shrunk EV 0.03R < 0.05R


def test_gate_rejects_thin_or_negative_evidence(settings):
    gate = settings["approval"]
    good = {"oos": summarize([1.0, -0.5] * 30), "holdout": summarize([1.0, -0.5] * 6)}
    assert passes_gate(good, gate)[0]
    thin = {"oos": summarize([1.0] * 5), "holdout": summarize([1.0] * 10)}
    assert not passes_gate(thin, gate)[0]
    bad_hold = {"oos": summarize([1.0, -0.5] * 30), "holdout": summarize([-1.0] * 10)}
    assert not passes_gate(bad_hold, gate)[0]


def test_paper_broker_matches_backtest(store):
    """The same signal must give the same R in the backtester and the paper broker."""
    df = SyntheticProvider().history("EURUSD=X", "1h", datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 6, 1, tzinfo=UTC))
    ind = add_indicators(df)
    cost = CostModel(spread_bps=1.5, slippage_bps=1.0)
    strat = Breakout(stop=1.5, target=3.0)
    bt = run_backtest(ind, strat, cost, allow_short=True)
    assert len(bt) > 5
    t0 = bt[0]
    i = ind.index.get_loc(t0.signal_time)
    sig = strat.signal(ind.iloc[i - 1], ind.iloc[i])
    broker = PaperBroker(store)
    store.record_candidate({"candidate_id": "c1"})
    broker.submit(candidate_id="c1", symbol="EURUSD=X", asset_class="forex", strategy_key=strat.key, signal=sig,
                  regime="trending", bar_time=t0.signal_time, atr=float(ind["atr14"].iloc[i]), qty=1.0,
                  risk_gbp=2.5, risk_per_unit=1.0, fx=1.0, notional=1.0, cost_r=0.0)
    # feed bars one at a time, like hourly worker runs
    for k in range(i + 1, len(df)):
        now = df.index[k] + pd.Timedelta(hours=1)
        broker.process("EURUSD=X", ind.iloc[: k + 1], "1h", cost, now, lambda *a, **kw: None)
        tr = store.trades()[0]
        if tr["status"] == "closed":
            break
    assert tr["status"] == "closed"
    assert tr["entry_price"] == pytest.approx(t0.entry_price)
    assert tr["exit_price"] == pytest.approx(t0.exit_price)
    assert tr["r_multiple"] == pytest.approx(t0.r_multiple, abs=1e-9)
    assert tr["pnl_gbp"] == pytest.approx(2.5 * t0.r_multiple, rel=1e-6)


def test_late_worker_cancels_instead_of_backfilling(store):
    df = SyntheticProvider().history("EURUSD=X", "1h", datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 20, tzinfo=UTC))
    broker = PaperBroker(store)
    store.record_candidate({"candidate_id": "c1"})
    broker.submit(candidate_id="c1", symbol="EURUSD=X", asset_class="forex", strategy_key="x",
                  signal=Signal(1, 2, 3, 10), regime="ranging", bar_time=df.index[100], atr=0.001, qty=1,
                  risk_gbp=2.5, risk_per_unit=0.002, fx=1, notional=1, cost_r=0)
    broker.process("EURUSD=X", add_indicators(df), "1h", CostModel(), df.index[-1] + pd.Timedelta(hours=1),
                   lambda *a, **k: None)
    assert store.trades()[0]["status"] == "cancelled"


def test_candidates_are_idempotent(store):
    assert store.record_candidate({"candidate_id": "AAPL|x|t1"})
    assert not store.record_candidate({"candidate_id": "AAPL|x|t1"})


def test_engine_cycle_twice_does_not_duplicate_orders(small_settings, store):
    settings = small_settings
    provider = SyntheticProvider()
    store.save_strategy({"strategy_id": "EURUSD=X|breakout_donchian", "symbol": "EURUSD=X", "asset_class": "forex",
                         "strategy": "breakout_donchian", "family": "breakout",
                         "params": Breakout(stop=1.5, target=3.0).key, "status": "approved", "reasons": [],
                         "oos": {"trades": 500, "expectancy_r": 0.3, "win_rate": 0.4}, "holdout": {}})
    eng = Engine(settings, provider, store, live=False)
    t = datetime(2025, 9, 1, 0, 7, tzinfo=UTC)
    seen = 0
    for d in range(200):
        now = t + timedelta(days=d)
        eng.cycle(now)
        eng.cycle(now)  # duplicate run at the same moment (e.g. overlapping cron)
        seen = len(store.trades())
    ids = [x["candidate_id"] for x in store.trades()]
    assert len(ids) == len(set(ids))
    assert seen >= 1


def test_kill_switch_stops_new_trades_but_manages_positions(small_settings, store):
    settings = small_settings
    SafetyBot(store, settings).engage("test")
    assert store.get_control("kill_switch") == "on"
    store.save_strategy({"strategy_id": "EURUSD=X|b", "symbol": "EURUSD=X", "asset_class": "forex",
                         "strategy": "breakout_donchian", "family": "breakout",
                         "params": Breakout(stop=1.5, target=3.0).key, "status": "approved", "reasons": [],
                         "oos": {"trades": 500, "expectancy_r": 0.3, "win_rate": 0.4}, "holdout": {}})
    eng = Engine(settings, SyntheticProvider(), store, live=False)
    t = datetime(2025, 9, 1, 0, 7, tzinfo=UTC)
    for d in range(200):
        eng.cycle(t + timedelta(days=d))
    assert not store.trades()
    rejected = [c for c in store.candidates() if c["decision"] == "reject"]
    assert rejected and any("kill switch" in r for c in rejected for r in c["reasons"])


def test_stale_data_detection():
    now = pd.Timestamp("2026-11-18 15:00", tz="UTC")  # Wednesday
    assert is_stale("crypto", "1h", now - pd.Timedelta(hours=6), now, 3.0)
    assert not is_stale("crypto", "1h", now - pd.Timedelta(hours=1), now, 3.0)
    sat = pd.Timestamp("2026-11-21 12:00", tz="UTC")
    assert not is_stale("forex", "1h", pd.Timestamp("2026-11-20 20:00", tz="UTC"), sat, 3.0)


# ---------------- v0.5 ----------------

def _st(**kw):
    base = {"direction": 1, "stop": 90.0, "target": 1e9, "held": 1, "max_hold": 10,
            "trail_atr": 0.0, "exit_ma": 0, "exit_on": "", "best": None}
    base.update(kw)
    return base


def test_connors_exit_on_close_above_5day_average():
    st = _st(exit_ma=5, exit_on="revert")
    bar = {"open": 100, "high": 103, "low": 99, "close": 102, "atr14": 2.0, "sma5": 101.0}
    px, reason = step_position(st, bar, CostModel(), entry_bar=False)
    assert reason.startswith("close above 5") and px == 102


def test_trend_exit_on_close_below_200day_average():
    st = _st(exit_ma=200, exit_on="break")
    bar = {"open": 100, "high": 101, "low": 97, "close": 98, "atr14": 2.0, "sma200": 99.0}
    _, reason = step_position(st, bar, CostModel(), entry_bar=False)
    assert reason.startswith("close below 200")


def test_trailing_stop_only_moves_in_our_favour():
    st = _st(trail_atr=3.0)
    step_position(st, {"open": 100, "high": 111, "low": 99, "close": 110, "atr14": 2.0}, CostModel(), False)
    assert st["stop"] == pytest.approx(104.0)   # 110 - 3*2
    step_position(st, {"open": 109, "high": 109, "low": 105, "close": 106, "atr14": 2.0}, CostModel(), False)
    assert st["stop"] == pytest.approx(104.0)   # never loosens
    _, reason = step_position(st, {"open": 105, "high": 105, "low": 103, "close": 104, "atr14": 2.0},
                              CostModel(), False)
    assert reason == "stop"


def test_no_target_variants_without_exit_are_dropped():
    for cls in (TrendFollowing, Breakout):
        for v in cls.variants():
            assert not (v.params["target"] >= NO_TARGET and v.params["trail"] == 0)


def test_trend_sma200_signal_requires_positive_12m_return():
    s = TrendSMA200()
    prev = {"close": 99, "sma200": 100, "roc252": 0.1, "atr14": 1}
    row = {"close": 101, "sma200": 100, "roc252": 0.1, "atr14": 1}
    assert s.signal(prev, row).direction == 1
    assert s.signal(prev, {**row, "roc252": -0.05}) is None


def test_market_filter_blocks_long_entries():
    df = SyntheticProvider().history("SPY", "1d", datetime(2018, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC))
    ind = add_indicators(df)
    strat = Breakout(stop=1.5, target=4.0)
    free = run_backtest(ind, strat, CostModel(), allow_short=False)
    blocked = run_backtest(ind, strat, CostModel(), allow_short=False,
                           market_ok=pd.Series(False, index=ind.index))
    assert free and not blocked


def test_market_filter_series_has_no_lookahead():
    df = SyntheticProvider().history("SPY", "1d", datetime(2018, 1, 1, tzinfo=UTC), datetime(2024, 1, 1, tzinfo=UTC))
    full = market_filter_series(df, df.index)
    part = market_filter_series(df.iloc[:900], df.index[:900])
    assert (full.iloc[:900].values == part.values).all()


def test_gate_requires_t_stat(settings):
    gate = settings["approval"]
    noisy = [3.0, -1.0, -1.0, 2.5, -1.0, -1.0, -1.0, 2.0] * 5   # positive mean, high variance
    res = {"oos": summarize(noisy), "holdout": summarize([1.0, -0.5] * 6)}
    ok, reasons = passes_gate(res, gate)
    assert summarize(noisy)["expectancy_r"] > 0.05
    assert not ok and any("t-stat" in r for r in reasons)


def test_notify_is_silent_without_topic(monkeypatch):
    from marketpilot import notify
    monkeypatch.delenv("NTFY_TOPIC", raising=False)
    assert notify.send("x", "y") is False


def test_notify_posts_to_topic(monkeypatch):
    import sys
    import types
    from marketpilot import notify
    sent = {}

    def fake_post(url, data=None, headers=None, timeout=None):
        sent.update(url=url, data=data, headers=headers)
        return types.SimpleNamespace(ok=True)
    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(post=fake_post))
    monkeypatch.setenv("NTFY_TOPIC", "mp-test-topic")
    notify.trade_closed({"symbol": "SPY", "r_multiple": 1.8, "pnl_gbp": 4.5, "exit_reason": "target",
                         "exit_price": 501.2, "strategy": "trend_sma200(stop=3,trail=4)"})
    assert sent["url"].endswith("/mp-test-topic")
    assert sent["headers"]["Title"].startswith("WIN: SPY +1.80R")


def test_removed_symbol_position_still_managed(small_settings, store):
    settings = small_settings
    """A position in a symbol dropped from the universe (e.g. XOM) keeps its stops managed."""
    provider = SyntheticProvider()
    df = provider.history("XOM", "1d", datetime(2025, 1, 1, tzinfo=UTC), datetime(2026, 3, 1, tzinfo=UTC))
    store.record_candidate({"candidate_id": "old"})
    PaperBroker(store).submit(candidate_id="old", symbol="XOM", asset_class="stocks", strategy_key="x",
                              signal=Signal(1, 2.5, 1.5, 10), regime="ranging", bar_time=df.index[-15],
                              atr=float(add_indicators(df)["atr14"].iloc[-15]), qty=1, risk_gbp=2.5,
                              risk_per_unit=1, fx=0.79, notional=40, cost_r=0)
    eng = Engine(settings, provider, store, live=False)
    eng.cycle(df.index[-14].to_pydatetime() + timedelta(days=1, minutes=7))
    assert store.trades()[0]["status"] in ("open", "closed")
    for k in range(13, 0, -1):
        eng.cycle(df.index[-k].to_pydatetime() + timedelta(days=1, minutes=7))
    assert store.trades()[0]["status"] == "closed"   # max hold 10 bars at most


# ---------------- v0.6 ----------------

def test_quote_currencies():
    from marketpilot.data import quote_currency
    assert quote_currency("USDCAD=X", "forex") == "CAD"
    assert quote_currency("EURCHF=X", "forex") == "CHF"
    assert quote_currency("ISF.L", "international") == "GBp"
    assert quote_currency("EWJ", "international") == "USD"
    assert quote_currency("LINK-USD", "crypto") == "USD"


def test_fx_conversion_is_generic_and_never_silently_one():
    from marketpilot.data import DataProvider

    class Fake(DataProvider):
        def history(self, symbol, interval, start, end):
            rates = {"GBPCAD=X": 1.80, "GBPUSD=X": 1.25, "GBPJPY=X": 190.0}
            return pd.DataFrame({"close": [rates[symbol]]})
    p, now = Fake(), datetime(2026, 11, 20, tzinfo=UTC)
    assert p.fx_to_gbp("CAD", now) == pytest.approx(1 / 1.80)
    assert p.fx_to_gbp("USD", now) == pytest.approx(0.8)
    assert p.fx_to_gbp("GBp", now) == 0.01
    with pytest.raises(RuntimeError):
        p.fx_to_gbp("XYZ", now)


def test_every_market_group_has_costs_and_valid_filter(settings):
    for cls in settings["universe"]:
        assert cls in settings["costs"]
    for cls, spec in settings.raw["market_filter"].items():
        assert spec["symbol"] in settings.all_symbols()
    assert len(settings.all_symbols()) == len(set(settings.all_symbols()))


# ---------------- v0.7 Learning Lab ----------------

def test_new_strategies_registered_and_roundtrip():
    from marketpilot.strategies import ALL_STRATEGIES
    names = {c.name for c in ALL_STRATEGIES}
    assert {"bollinger_reversion", "turtle_55", "macd_trend", "high_52week"} <= names
    for c in ALL_STRATEGIES:
        for v in c.variants():
            assert strategy_from_key(v.key).key == v.key


def test_new_strategies_trade_in_backtest():
    from marketpilot.strategies import BollingerReversion, High52Week, MACDTrend, Turtle55
    df = SyntheticProvider().history("SPY", "1d", datetime(2018, 1, 1, tzinfo=UTC), datetime(2026, 6, 1, tzinfo=UTC))
    ind = add_indicators(df)
    for cls in (BollingerReversion, Turtle55, MACDTrend, High52Week):
        trades = run_backtest(ind, cls(), CostModel(2, 3), allow_short=False)
        assert trades, cls.name
        assert all(np.isfinite(t.r_multiple) for t in trades)


def test_shadow_bots_trade_every_strategy_without_touching_real_account(small_settings, store):
    eng = Engine(small_settings, SyntheticProvider(), store, live=False)
    t = datetime(2025, 6, 2, 0, 7, tzinfo=UTC)
    for d in range(150):
        eng.cycle(t + timedelta(days=d))
    shadow = store.shadow_trades()
    assert len(shadow) > 10
    assert len({s["strategy"].split("(")[0] for s in shadow}) >= 4   # many strategies practising
    assert store.trades() == []                                       # nothing approved -> no real trades
    closed = [s for s in shadow if s["status"] == "closed"]
    assert closed and all(np.isfinite(float(s["r_multiple"])) for s in closed)
    # one open shadow position per symbol x strategy at a time
    active = [(s["symbol"], s["strategy"].split("(")[0]) for s in shadow if s["status"] in ("open", "pending")]
    assert len(active) == len(set(active))


def test_forward_verdict_promotes_and_demotes(settings):
    from marketpilot.bots.learning import forward_verdict
    cfg = settings["learning"]
    good = pd.DataFrame([{"asset_class": "forex", "strategy": "turtle_55", "trades": 80,
                          "expectancy_r": 0.3, "t_stat": 2.6, "win_rate": 0.4}])
    bad = pd.DataFrame([{"asset_class": "stocks", "strategy": "trend_ema", "trades": 40,
                         "expectancy_r": -0.2, "t_stat": -1.5, "win_rate": 0.3}])
    thin = pd.DataFrame([{"asset_class": "forex", "strategy": "turtle_55", "trades": 12,
                          "expectancy_r": 0.9, "t_stat": 3.0, "win_rate": 0.7}])
    assert forward_verdict(good, "forex", "turtle_55", cfg)[0] == "promote"
    assert forward_verdict(bad, "stocks", "trend_ema", cfg)[0] == "demote"
    assert forward_verdict(thin, "forex", "turtle_55", cfg)[0] is None


def test_apply_learning_updates_strategy_status(settings, store):
    from marketpilot.bots.learning import apply_learning
    store.save_strategy({"strategy_id": "EURUSD=X|turtle_55", "symbol": "EURUSD=X", "asset_class": "forex",
                         "strategy": "turtle_55", "family": "breakout", "params": "turtle_55(stop=2,trail=3)",
                         "status": "rejected", "reasons": ["t-stat"], "oos": {"expectancy_r": 0.1}, "holdout": {}})
    for i in range(60):
        store.open_shadow({"trade_id": f"s{i}", "candidate_id": f"c{i}", "symbol": "EURUSD=X",
                           "asset_class": "forex", "strategy": "turtle_55(stop=2,trail=3)", "status": "closed",
                           "opened_at": f"2026-01-{i % 28 + 1:02d}", "r_multiple": 2.0 if i % 2 else -0.6})
    msgs = apply_learning(store, settings)
    assert msgs and store.strategies("approved")[0]["strategy"] == "turtle_55"


# ---------------- v0.8 confluence, structure stops, CCU ----------------

def test_confluence_score_counts_agreeing_factors():
    from marketpilot.bots.confluence import MAX_SCORE, score
    row = {"close": 105.0, "sma200": 100.0, "ema20": 104.0, "ema50": 102.0, "rsi14": 60.0, "atr14": 2.0,
           "dc_high": 106.0, "dc_low": 95.0, "hi252": 120.0, "lo252": 80.0, "range_hi60": 110.0,
           "range_lo60": 100.0, "volume": 150.0, "vol_avg20": 100.0, "atr_pct_rank": 0.5}
    s_long, hits = score(row, 1, market_ok=True)
    assert s_long == len(hits) and {"trend", "swing_trend", "momentum", "level", "fib", "volume", "calm",
                                    "market"} == set(hits) and s_long == MAX_SCORE
    s_short, _ = score(row, -1, market_ok=False)
    assert s_short < s_long


def test_structure_stop_is_beyond_swing_and_clamped():
    from marketpilot.backtest import plan_levels
    c = CostModel()
    sig = Signal(1, 2.0, NO_TARGET, 10, stop_swing=True)
    stop, _, _ = plan_levels(100.0, 2.0, sig, c, swing=97.0)      # 0.5 ATR below the 97 swing low
    assert stop == pytest.approx(96.0)
    stop, _, _ = plan_levels(100.0, 2.0, sig, c, swing=99.9)      # too tight -> at least 1 ATR
    assert stop == pytest.approx(98.0)
    stop, _, _ = plan_levels(100.0, 2.0, sig, c, swing=80.0)      # too wide -> at most 4 ATR
    assert stop == pytest.approx(92.0)
    stop, _, _ = plan_levels(100.0, 2.0, Signal(1, 2.0, NO_TARGET, 10), c, swing=80.0)
    assert stop == pytest.approx(96.0)                            # ATR stop when not a swing variant


def test_ccu_three_checkmarks_signal():
    from marketpilot.strategies import CCUBottom
    s = CCUBottom()
    prev = {"close": 90, "sma200": 100, "rsi14": 30, "stoch_k": 25, "stoch_d": 30, "atr14": 2}
    row = {"close": 88, "sma200": 100, "rsi14": 24, "stoch_k": 10, "stoch_d": 15, "atr14": 2}
    assert s.signal(prev, row).direction == 1
    assert s.signal(prev, {**row, "close": 101}) is None          # must be below the 200 MA
    assert s.signal(row, row) is None                             # only on the first day of the setup


def test_backtest_records_confluence_and_breakdown():
    from marketpilot.backtest import walk_forward
    from marketpilot.strategies import Breakout as B
    df = SyntheticProvider().history("SPY", "1d", datetime(2018, 1, 1, tzinfo=UTC), datetime(2026, 6, 1, tzinfo=UTC))
    r = walk_forward(df, B, CostModel(2, 3), False)
    assert r["oos_trades"] and all(0 <= t.confluence <= 8 for t in r["oos_trades"])
    assert sum(v["trades"] for v in r["by_confluence"].values()) == r["oos"]["trades"]


def test_cycle_writes_market_snapshot_and_shadow_confluence(small_settings, store):
    eng = Engine(small_settings, SyntheticProvider(), store, live=False)
    t = datetime(2025, 6, 2, 0, 7, tzinfo=UTC)
    for d in range(60):
        eng.cycle(t + timedelta(days=d))
    mk = {m["symbol"]: m for m in store.market()}
    assert set(mk) == {"EURUSD=X", "SPY"} and 0 <= mk["SPY"]["conf_long"] <= 8
    sh = store.shadow_trades()
    assert sh and all(s["confluence"] is not None and "confluence" in s["note"] for s in sh)


# ---------------- v0.9: Time Machine ----------------
from marketpilot.bots import timemachine as tm


def _walk(n=800, drift=0.0, vol=0.01, seed=1):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    o = np.concatenate([[100], c[:-1]])
    h = np.maximum(o, c) * (1 + vol / 2)
    l = np.minimum(o, c) * (1 - vol / 2)
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": 1e6}, index=idx)


def test_time_machine_symmetric_plan_is_near_coin_flip():
    df = _walk(seed=4)                                     # a sample whose realised drift is ~0
    p = float(df["close"].iloc[-1])
    odds = tm.simulate(df, 1, p, p * 0.97, p * 1.03, max_hold=40, n_paths=10_000)
    assert odds.paths == 10_000
    assert abs(odds.p_target + odds.p_stop + odds.p_neither - 1) < 1e-9
    assert abs(odds.p_target - odds.p_stop) < 0.08       # no edge in a driftless walk


def test_time_machine_sees_drift_and_is_repeatable():
    up = _walk(drift=0.003)
    p = float(up["close"].iloc[-1])
    long_odds = tm.simulate(up, 1, p, p * 0.97, p * 1.03, 40)
    short_odds = tm.simulate(up, -1, p, p * 1.03, p * 0.97, 40)
    assert long_odds.p_target > long_odds.p_stop
    assert short_odds.p_stop > short_odds.p_target
    assert long_odds.exp_r > short_odds.exp_r
    again = tm.simulate(up, 1, p, p * 0.97, p * 1.03, 40)
    assert again.p_target == long_odds.p_target           # fixed seed -> same answer


def test_time_machine_stop_hit_at_most_minus_one_r_unless_gap():
    df = _walk(vol=0.002)                                  # tiny moves: no gaps through a 3% stop
    p = float(df["close"].iloc[-1])
    odds = tm.simulate(df, 1, p, p * 0.97, p * 1.20, 40)
    assert odds.exp_r >= -1.0 - 1e-9


def test_time_machine_odds_for_signal_and_calibration():
    ind = add_indicators(_walk(drift=0.001))
    cost = CostModel(spread_bps=2, slippage_bps=1)
    sig = Signal(1, 2.0, NO_TARGET, 60, "test")          # no fixed target -> +2R proxy
    odds = tm.odds_for(ind, sig, cost, strategy=TrendFollowing(), allow_short=False, n_paths=2000)
    assert odds is not None and abs(odds.reward_risk - 2.0) < 1e-6
    assert odds.hist_trades >= 0 and "time machine" in odds.describe()
    f = tm.trade_fields(odds)
    assert set(f) == {"tm_p_target", "tm_p_stop", "tm_exp_r"}
    trades = [{"tm_p_target": 0.6, "tm_exp_r": 0.3, "r_multiple": 1.0},
              {"tm_p_target": 0.62, "tm_exp_r": 0.3, "r_multiple": -1.0},
              {"tm_p_target": 0.1, "tm_exp_r": -0.4, "r_multiple": -1.0},
              {"tm_p_target": None, "r_multiple": 2.0}]
    cal = tm.calibration(trades)
    assert list(cal["forecast"]) == ["under 20%", "50-65%"]
    assert cal.set_index("forecast").loc["50-65%", "actual_win_pct"] == 0.5


def test_shadow_trades_carry_time_machine_odds(small_settings, store):
    small_settings.raw["time_machine"] = {"enabled": True, "paths": 500, "shadow_paths": 300}
    eng = Engine(small_settings, SyntheticProvider(), store, live=False)
    t = datetime(2025, 6, 2, 0, 7, tzinfo=UTC)
    for d in range(40):
        eng.cycle(t + timedelta(days=d))
    sh = store.shadow_trades()
    assert sh and all(s["tm_p_target"] is not None and 0 <= s["tm_p_target"] <= 1 for s in sh)
    assert any("time machine" in s["note"] for s in sh)


# ---------------- v0.10: conviction grade ----------------
from marketpilot.bots import conviction as cv


def test_time_machine_only_votes_on_high_confluence():
    good = tm.Odds(p_target=0.5, p_stop=0.3, p_neither=0.2, exp_r=0.3, median_bars=8, paths=100, reward_risk=2)
    bad = tm.Odds(p_target=0.2, p_stop=0.7, p_neither=0.1, exp_r=-0.4, median_bars=8, paths=100, reward_risk=2)
    assert cv.grade(7, good) == "A+" and cv.grade(6, bad) == "A" and cv.grade(8, None) == "A"
    assert cv.grade(5, good) == "B" and cv.grade(5, bad) == "B"      # odds ignored below 6
    assert cv.grade(2, good) == "C"
    assert cv.passes("A+", "A+") and not cv.passes("A", "A+") and cv.passes("C", None)
    assert "time machine confirms" in cv.describe("A+", 7, good)


def test_min_grade_filter_blocks_lower_grades_and_grades_are_saved(small_settings, store):
    small_settings.raw["time_machine"] = {"enabled": True, "paths": 300, "shadow_paths": 200}
    eng = Engine(small_settings, SyntheticProvider(), store, live=False)
    t = datetime(2025, 6, 2, 0, 7, tzinfo=UTC)
    for d in range(30):
        eng.cycle(t + timedelta(days=d))
    sh = store.shadow_trades()
    assert sh and all(s["grade"] in cv.GRADES for s in sh)
    stats = cv.grade_stats([{**s, "status": "closed", "r_multiple": 1.0} for s in sh])
    assert set(stats["grade"]) <= set(cv.GRADES) and stats["trades"].sum() == len(sh)


def test_market_factor_matches_trade_direction():
    from marketpilot.bots.confluence import score
    row = pd.Series({"close": 100.0, "volume": 0.0})
    assert "market" in score(row, 1, True)[1] and "market" not in score(row, -1, True)[1]
    assert "market" in score(row, -1, False)[1] and "market" not in score(row, 1, False)[1]
    assert "market" not in score(row, 1, None)[1]


# ---------------- v0.11: loss control ----------------
from marketpilot.bots import losscontrol as lc


def test_loss_control_multipliers_combine_with_floor():
    cfg = lc.config({})
    assert lc.multiplier(cfg, vol_rank=0.5)[0] == 1.0
    assert lc.multiplier(cfg, vol_rank=0.85)[0] == 0.5
    assert lc.multiplier(cfg, vol_rank=0.97)[0] == 0.25
    assert lc.multiplier(cfg, drawdown_pct=3.5)[0] == 0.5
    assert lc.multiplier(cfg, drawdown_pct=7.0)[0] == 0.25
    assert lc.multiplier(cfg, streak=4)[0] == 0.5
    m, notes = lc.multiplier(cfg, vol_rank=0.97, drawdown_pct=7.0, streak=5)
    assert m == cfg["min_mult"] and len(notes) == 3
    assert lc.multiplier({**cfg, "enabled": False}, vol_rank=0.99)[0] == 1.0
    assert lc.multiplier(cfg, vol_rank=float("nan"))[0] == 1.0


def test_drawdown_streak_and_concentration():
    assert lc.drawdown_pct([500, 520, 510], 494) == pytest.approx(5.0)
    assert lc.drawdown_pct([500], 510) == 0.0
    closed = [{"closed_at": f"2026-01-0{i}", "r_multiple": r} for i, r in enumerate([1.0, -1, -1, -0.5], 1)]
    assert lc.loss_streak(closed) == 3
    cfg = lc.config({})
    open_t = [{"asset_class": "crypto", "risk_gbp": 2.5}] * 3
    reasons = lc.concentration_reasons("crypto", open_t, 2.5, 500, cfg)
    assert any("crypto" in r for r in reasons)
    assert lc.concentration_reasons("bonds", open_t[:1], 2.5, 500, cfg) == []
    big = [{"asset_class": "stocks", "risk_gbp": 7.0}, {"asset_class": "bonds", "risk_gbp": 7.0}]
    assert any("open risk" in r for r in lc.concentration_reasons("forex", big, 2.5, 500, cfg))


def test_risk_manager_sizes_down_in_high_volatility(small_settings):
    from marketpilot.bots.risk import RiskManager
    rm = RiskManager(small_settings)
    kw = dict(symbol="SPY", asset_class="stocks", equity=500.0, realized_today=0.0, trades_today=0,
              open_trades=[], entry_est=100.0, risk_per_unit=2.0, fx_to_gbp=1.0, kill_switch=False)
    normal, calm_high = rm.evaluate(**kw, vol_rank=0.3), rm.evaluate(**kw, vol_rank=0.9)
    assert normal.ok and calm_high.ok
    assert calm_high.risk_gbp == pytest.approx(normal.risk_gbp / 2)
    assert calm_high.risk_mult == 0.5 and calm_high.risk_notes


def test_backtest_comparison_shows_smaller_drawdown_when_wild_trades_lose():
    class T:
        def __init__(self, r, v):
            self.r_multiple, self.vol_rank = r, v
    trades = [T(1.0, 0.3), T(-1.0, 0.9), T(-1.0, 0.9), T(-1.0, 0.9), T(1.5, 0.4), T(-1.0, 0.3)]
    out = lc.compare(trades, lc.config({}))
    assert out["scaled"]["max_drawdown_r"] < out["flat"]["max_drawdown_r"]
    assert out["share_reduced"] == pytest.approx(0.5) and 0 < out["dd_cut_pct"] < 1


def test_engine_saves_account_risk_state(small_settings, store):
    import json
    eng = Engine(small_settings, SyntheticProvider(), store, live=False)
    eng.cycle(datetime(2025, 6, 2, 0, 7, tzinfo=UTC))
    st = json.loads(store.get_control("risk_state", "{}"))
    assert {"drawdown_pct", "loss_streak", "account_mult", "open_risk_gbp", "open_risk_cap_gbp"} <= set(st)


# ---------------- v0.12: trade management ----------------
from marketpilot.backtest import MANAGEMENT, new_state, step_position, trade_r

_ZERO = CostModel()


def _bar(o, h, l, c):
    return {"open": o, "high": h, "low": l, "close": c, "atr14": 2.0}


def _run_path(mode, bars, entry=100.0, stop=98.0, target=200.0):
    sig = Signal(1, 1.0, NO_TARGET, 50, "t")
    st = new_state(sig, stop, target, entry, entry - stop, MANAGEMENT[mode])
    for k, b in enumerate(bars):
        hit = step_position(st, b, _ZERO, entry_bar=(k == 0))
        if hit:
            return trade_r(st, 1, entry, hit[0], entry - stop, _ZERO), hit[1], st
    return None, None, st


# entry 100, risk 2: rallies to 102.5 (> +1R), then falls through the original stop
_PATH = [_bar(100, 101, 99.5, 100.5), _bar(100.5, 102.6, 100.4, 102.5), _bar(102.4, 102.5, 97.0, 97.5)]


def test_base_management_takes_full_loss():
    r, reason, _ = _run_path("base", _PATH)
    assert r == pytest.approx(-1.0) and reason == "stop"


def test_breakeven_turns_the_loss_into_a_scratch():
    r, reason, st = _run_path("breakeven", _PATH)
    assert r == pytest.approx(0.0, abs=1e-9) and st["stop"] == pytest.approx(100.0)


def test_partial_banks_half_at_one_r():
    r, _, st = _run_path("partial", _PATH)
    assert st["partial_price"] == pytest.approx(102.0)
    assert r == pytest.approx(0.5)            # half at +1R, half at breakeven


def test_management_never_moves_stop_backwards_and_handles_shorts():
    sig = Signal(-1, 1.0, NO_TARGET, 50, "t")
    st = new_state(sig, 102.0, 0.0, 100.0, 2.0, MANAGEMENT["breakeven"])
    step_position(st, _bar(100, 100.5, 97.5, 97.8), _ZERO, entry_bar=True)   # +1.1R for a short
    assert st["stop"] == pytest.approx(100.0)
    st["stop"] = 99.0                                                       # tighter (e.g. trailing)
    step_position(st, _bar(97.8, 98.0, 97.0, 97.2), _ZERO, entry_bar=False)
    assert st["stop"] == pytest.approx(99.0)


def test_backtest_records_all_management_modes_on_the_same_trades():
    ind = add_indicators(_walk(drift=0.001, seed=3))
    trades = run_backtest(ind, TrendFollowing(), _ZERO, False, compare_modes=("base", "breakeven", "partial"))
    assert trades
    for t in trades:
        assert set(t.alt_r) == {"base", "breakeven", "partial"}
        assert t.alt_r["base"] == pytest.approx(t.r_multiple)
    from marketpilot.backtest import management_breakdown
    mb = management_breakdown(trades, [], ("base", "breakeven", "partial"))
    assert mb["breakeven"]["full_loss_rate"] <= mb["base"]["full_loss_rate"] + 1e-9


def test_paper_broker_matches_backtest_partial_across_cycles(store):
    from marketpilot.bots.autotrader import PaperBroker
    idx = pd.date_range("2026-01-05", periods=4, freq="D", tz="UTC")
    bars = pd.DataFrame([_bar(100, 100.5, 99.8, 100.2), _bar(100.2, 101, 99.5, 100.5),
                         _bar(100.5, 102.6, 100.4, 102.5), _bar(102.4, 102.5, 97.0, 97.5)], index=idx)
    pb = PaperBroker(store)
    sig = Signal(1, 1.0, NO_TARGET, 50, "t")
    pb.submit(candidate_id="c1", symbol="XYZ", asset_class="stocks", strategy_key="s", signal=sig,
              regime="trending", bar_time=idx[0], atr=2.0, qty=1.0, risk_gbp=2.0, risk_per_unit=2.0, fx=1.0,
              notional=100.0, cost_r=0.0, extra={"mgmt": "partial"})
    pb.process("XYZ", bars.iloc[:3], "1d", _ZERO, idx[2], log=lambda *a, **k: None)   # cycle 1: partial taken
    t = store.trades()[0]
    assert t["status"] == "open" and t["partial_price"] == pytest.approx(102.2)   # entry 100.2 + 1R
    closed = pb.process("XYZ", bars, "1d", _ZERO, idx[3], log=lambda *a, **k: None)   # cycle 2: rest stopped
    assert closed and closed[0]["r_multiple"] == pytest.approx(0.5)


# ---------------- v0.12: storage robustness and speed ----------------
def test_json_fields_never_store_infinity_or_nan(store):
    import json
    store.save_strategy({"strategy_id": "X|s", "symbol": "X", "asset_class": "stocks", "strategy": "s",
                         "family": "f", "params": "p", "status": "rejected", "reasons": [],
                         "oos": {"profit_factor": float("inf"), "t": float("nan"), "nested": [float("-inf"), 1.0]},
                         "holdout": {}, "avg_cost_r": float("nan"), "researched_at": "now",
                         "data_start": "a", "data_end": "b"})
    raw = store._select("mp_strategies", {"strategy_id": "X|s"})[0]["oos"]
    assert "Infinity" not in raw and "NaN" not in raw
    assert json.loads(raw) == {"profit_factor": None, "t": None, "nested": [None, 1.0]}


def test_cycle_read_cache_serves_repeats_and_clears_on_write(store):
    calls = []
    orig = store._select
    store._select = lambda *a, **k: (calls.append(a[0]), orig(*a, **k))[1]
    store.begin_cycle()
    store.trades(); store.trades(); store.get_control("a"); store.get_control("b")
    assert calls.count("mp_trades") == 1 and calls.count("mp_control") == 1
    store.set_control("a", "1")
    assert store.get_control("a") == "1"                     # write cleared the cache
    store.open_trade({"trade_id": "t1", "candidate_id": "c1", "symbol": "S", "status": "open"})
    assert len(store.trades()) == 1
    store.end_cycle()
    store.trades(); store.trades()
    assert calls.count("mp_trades") == 4                     # no caching outside a cycle (the app)


def test_supabase_select_pages_past_1000_rows():
    from marketpilot.storage import SupabaseStore

    class Q:
        def __init__(self, data): self.data, self.r = data, None
        def select(self, *_): return self
        def eq(self, *_): return self
        def order(self, *_a, **_k): return self
        def limit(self, n): self.r = (0, n - 1); return self
        def range(self, a, b): self.r = (a, b); return self
        def execute(self):
            a, b = self.r or (0, 999)
            b = min(b, a + 999)          # server cap: 1000 rows per request
            return type("R", (), {"data": self.data[a:b + 1]})()

    rows = [{"i": i} for i in range(2500)]
    st = SupabaseStore.__new__(SupabaseStore)
    st.db = type("DB", (), {"table": lambda self, _t: Q(rows)})()
    assert len(st._select("mp_shadow_trades")) == 2500
    assert len(st._select("mp_events", limit=300)) == 300
    assert len(st._select("mp_candidates", limit=1500)) == 1500


def test_market_overview_is_saved_in_one_batch(small_settings, store):
    eng = Engine(small_settings, SyntheticProvider(), store, live=False)
    eng.cycle(datetime(2025, 6, 2, 0, 7, tzinfo=UTC))
    assert len(store.market()) == len(eng._symbols())
    assert any("cycle done" in e["message"] for e in store.events(50))


def test_yfinance_batch_prefetch_and_cache(monkeypatch):
    import sys, types
    from marketpilot.data import YFinanceProvider
    idx = pd.date_range("2026-01-01", periods=30, freq="D")
    cols = pd.MultiIndex.from_product([["AAA", "BBB"], ["Open", "High", "Low", "Close", "Volume"]])
    batch = pd.DataFrame(1.0, index=idx, columns=cols)
    batch[("BBB", "Close")] = float("nan")                 # broken ticker -> falls back
    single = pd.DataFrame(2.0, index=idx, columns=["Open", "High", "Low", "Close", "Volume"])
    calls = []

    def download(tickers, **kw):
        calls.append(tickers)
        return batch if isinstance(tickers, list) else single
    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=download))
    p = YFinanceProvider()
    end = datetime(2026, 2, 5, tzinfo=UTC)
    start = end - timedelta(days=40)
    assert p.prefetch(["AAA", "BBB"], "1d", start, end) == 1
    a = p.history("AAA", "1d", start + timedelta(days=5), end)   # served from the batch
    assert len(calls) == 1 and (a["close"] == 1.0).all()
    b = p.history("BBB", "1d", start, end)                       # not in batch -> own request
    assert len(calls) == 2 and (b["close"] == 2.0).all()
    p.history("BBB", "1d", start, end)                           # now cached
    assert len(calls) == 2

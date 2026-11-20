from datetime import datetime, timedelta, timezone

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

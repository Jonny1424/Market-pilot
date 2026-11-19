from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from marketpilot.backtest import check_exit, passes_gate, plan_levels, realized_r, run_backtest
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
from marketpilot.strategies import Breakout, MeanReversion, Signal, TrendFollowing, strategy_from_key

UTC = timezone.utc


@pytest.fixture
def settings():
    return load_settings()


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
        broker.process("EURUSD=X", df.iloc[: k + 1], "1h", cost, now, lambda *a, **kw: None)
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
    broker.process("EURUSD=X", df, "1h", CostModel(), df.index[-1] + pd.Timedelta(hours=1), lambda *a, **k: None)
    assert store.trades()[0]["status"] == "cancelled"


def test_candidates_are_idempotent(store):
    assert store.record_candidate({"candidate_id": "AAPL|x|t1"})
    assert not store.record_candidate({"candidate_id": "AAPL|x|t1"})


def test_engine_cycle_twice_does_not_duplicate_orders(settings, store):
    provider = SyntheticProvider()
    store.save_strategy({"strategy_id": "EURUSD=X|breakout_donchian", "symbol": "EURUSD=X", "asset_class": "forex",
                         "strategy": "breakout_donchian", "family": "breakout",
                         "params": Breakout(stop=1.5, target=3.0).key, "status": "approved", "reasons": [],
                         "oos": {"trades": 500, "expectancy_r": 0.3, "win_rate": 0.4}, "holdout": {}})
    eng = Engine(settings, provider, store, live=False)
    t = datetime(2026, 3, 2, 0, tzinfo=UTC)
    seen = 0
    for h in range(24 * 10):
        now = t + timedelta(hours=h)
        eng.cycle(now)
        eng.cycle(now)  # duplicate run at the same moment (e.g. overlapping cron)
        seen = len(store.trades())
    ids = [x["candidate_id"] for x in store.trades()]
    assert len(ids) == len(set(ids))
    assert seen >= 1


def test_kill_switch_stops_new_trades_but_manages_positions(settings, store):
    SafetyBot(store, settings).engage("test")
    assert store.get_control("kill_switch") == "on"
    store.save_strategy({"strategy_id": "EURUSD=X|b", "symbol": "EURUSD=X", "asset_class": "forex",
                         "strategy": "breakout_donchian", "family": "breakout",
                         "params": Breakout(stop=1.5, target=3.0).key, "status": "approved", "reasons": [],
                         "oos": {"trades": 500, "expectancy_r": 0.3, "win_rate": 0.4}, "holdout": {}})
    eng = Engine(settings, SyntheticProvider(), store, live=False)
    t = datetime(2026, 3, 2, tzinfo=UTC)
    for h in range(24 * 5):
        eng.cycle(t + timedelta(hours=h))
    assert not store.trades()
    rejected = [c for c in store.candidates() if c["decision"] == "reject"]
    assert rejected and any("kill switch" in r for c in rejected for r in c["reasons"])


def test_stale_data_detection():
    now = pd.Timestamp("2026-11-18 15:00", tz="UTC")  # Wednesday
    assert is_stale("crypto", "1h", now - pd.Timedelta(hours=6), now, 3.0)
    assert not is_stale("crypto", "1h", now - pd.Timedelta(hours=1), now, 3.0)
    sat = pd.Timestamp("2026-11-21 12:00", tz="UTC")
    assert not is_stale("forex", "1h", pd.Timestamp("2026-11-20 20:00", tz="UTC"), sat, 3.0)

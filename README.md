# MarketPilot v0.8 — Multi-Agent AI Paper Trading

A team of trading bots that research strategies, backtest them honestly, then
**automatically open and close paper trades** across US stock-market, bond and gold
ETFs, crypto and forex — logging every decision, win and loss, with push
notifications to your phone.

**v0.8:** ideas from professional trading educators and exchange apps — a
Confluence bot that scores every signal on 8 factors and measures whether high
scores really win more; market-structure (swing) stops tested against ATR stops;
the Crypto Crew University "3 checkmarks" strategy; a Markets tab with all 66
markets and the Crypto Fear & Greed index. Upgrade: run
`supabase/migrations/20261121_marketpilot_v08.sql`.

**v0.7:** Learning Lab — shadow bots practise all 9 strategies on all 66 markets to
collect forward evidence fast; the Learning bot promotes strategies that prove
themselves and demotes ones that fail. Four new technical strategies (Bollinger,
Turtle 55, MACD, 52-week high) and a **Learn** tab explaining every strategy and
indicator. Upgrade: run `supabase/migrations/20261120_marketpilot_v07.sql`.

**v0.6:** six market groups (US stocks, international, bonds, commodities, crypto,
forex), 66 markets.

**v0.5:** research-backed upgrade — see [docs/RESEARCH.md](docs/RESEARCH.md). Broad
ETF universe (no hindsight stock picks), daily bars for crypto and forex, a 200-day
trend strategy, a market trend filter, Connors' RSI(2) exit, trailing-stop variants,
realistic crypto fees, a t-statistic test against luck, and phone notifications.

> **Paper trading only.** There is no broker connection and no code path that
> places real orders. Live trading should only be considered after months of
> forward-test evidence and a separate review of broker integration, security
> and UK (FCA) rules. Not financial advice.

## The 10 bots

| # | Bot | File | AI or code? |
|---|-----|------|-------------|
| 1 | Market Scanner | `marketpilot/bots/scanner.py` | code |
| 2 | Strategy Research | `marketpilot/bots/research.py` | code |
| 3 | News & Economic Events | `marketpilot/bots/news.py` | calendar + optional Claude (veto only) |
| 4 | Market Regime | `marketpilot/bots/regime.py` | code |
| 5 | Risk Manager | `marketpilot/bots/risk.py` | code |
| 6 | Trade Decision (can abstain) | `marketpilot/bots/decision.py` | code |
| 7 | Auto-Trading (paper) | `marketpilot/bots/autotrader.py` | code |
| 8 | Performance Analyst | `marketpilot/bots/performance.py` | code |
| 9 | Backtesting | `marketpilot/backtest.py` | code |
| 10 | Monitoring & Safety | `marketpilot/bots/safety.py` | code |

The orchestrator is `marketpilot/engine.py`. Calculations, execution and risk
limits are deterministic code; Claude is only used to screen news headlines
(off by default).

```
Scanner ─► Regime ─► approved strategy signals ─► News veto ─► Trade Decision (EV after costs, may abstain)
                                                                  │
                                                         Risk Manager (size, limits, kill switch)
                                                                  │
                                                Paper order ─► Auto-Trader ─► Performance & Monitoring
```

## How strategies earn the right to trade

1. **Walk-forward backtest** per symbol: parameters are chosen on past data only,
   then tested on the next unseen period. Fees, spreads and slippage are charged;
   orders fill at the next bar's open; gaps through stops fill at the worse price.
2. Out-of-sample trades are **pooled across each market** (stocks, crypto, forex).
3. **Approval gate** (in `config/settings.yaml`): ≥30 OOS trades, expectancy
   ≥ +0.05R, profit factor ≥ 1.15, t-statistic ≥ 2, *and* non-negative on a final
   untouched holdout.
4. Live paper results are blended in; strategies that go negative over 30+
   forward trades are **retired** automatically.

### About the 70–80% win-rate goal

The dashboard tracks win rate against your goal (set in `goals.target_win_rate`).
But the bots are deliberately **not** told to maximise win rate: a tiny take-profit
with a wide stop wins 80% of the time and can still lose money. In testing, the
mean-reversion strategy reached a 62–66% win rate and was *rejected* because it
lost money on unseen data, while 36–41% win-rate trend strategies were approved
because their average win is ~2× their average loss. The "Strategies" tab plots
win rate against expectancy so you can see this for yourself.

## Risk controls (defaults)

£500 reference balance • 0.5% risk per trade (£2.50) • 2% daily loss limit •
max 4 positions • no leverage (total notional ≤ equity) • trades rejected if costs
exceed 35% of the risk • spot stocks/crypto long-only • auto kill switch if more
than half the data feeds fail or equity falls 15% • manual kill switch in the
dashboard and CLI. With the kill switch on, open positions keep their stops.

## Setup

### 1. Supabase
Open your project → **SQL Editor** → paste `supabase/migrations/20261119_marketpilot_v04.sql` → **Run**.
This adds `mp_*` tables and leaves your v0.3 `trades`/`trade_journal` tables alone.
Copy the **service_role** key (Project Settings → API). Keep it secret.

### 2. GitHub Actions (the always-on worker)
Repo → Settings → Secrets and variables → Actions → add:
- `SUPABASE_URL`
- `SUPABASE_SERVICE_KEY`
- `ANTHROPIC_API_KEY` (optional, only if you enable `news.llm_enabled`)

Then Actions → **MarketPilot worker** → *Run workflow* with `research` once.
After that it trades every hour at :07 and re-researches every Saturday.
(GitHub pauses scheduled workflows after 60 days with no commits — push a
commit or re-enable it if that happens.)

### 2b. Phone notifications (optional, free)
1. Install **ntfy** from the App Store / Google Play and subscribe to a private topic
   name (treat it like a password — anyone who knows it can read the alerts).
2. Add that name as the GitHub Actions secret `NTFY_TOPIC`.
3. Actions → MarketPilot worker → Run workflow → `notify-test`.

You'll get a message when a paper trade opens or closes, if the kill switch trips,
and a daily summary at 21:00 UTC.

### 2c. Upgrading from v0.4
Run `supabase/migrations/20261120_marketpilot_v05.sql` in the Supabase SQL Editor,
then run the worker with `research` once.

### 3. Streamlit dashboard
In Streamlit Community Cloud secrets add `APP_PASSWORD`, `SUPABASE_URL` and
`SUPABASE_SERVICE_KEY`. The app file is still `app.py`.

## Local use

```bash
pip install -r requirements.txt
python -m marketpilot demo --days 21          # offline demo on synthetic data
streamlit run app.py                          # view the demo results
python -m marketpilot research                # real data (Yahoo Finance)
python -m marketpilot run-once                # one trading cycle
python -m marketpilot report                  # win rate, win/loss, expectancy, P&L
python -m marketpilot kill   |  resume        # kill switch
pytest                                        # 44 tests
```

Add `--db supabase` (with `SUPABASE_URL`/`SUPABASE_SERVICE_KEY` in env) to use Supabase.

## Configuration
Everything is in `config/settings.yaml`: symbols, intervals, cost model, risk
limits, approval gate, win-rate goal. Keep `config/events.yaml` updated with
high-impact events (CPI, FOMC, NFP, BoE…) — new trades are blocked around them.

## Known limitations
- Yahoo Finance (`yfinance`) is free and unofficial: delayed, sometimes missing,
  no real bid/ask. Costs are modelled, not observed. Before any live use, switch
  to a broker or exchange data feed.
- Fills are simulated at the next bar's open; real orders can fill worse.
- Strategies are simple, well-known rules. Passing the gate is evidence, not proof;
  forward paper results are what count.
- The v0.3 manual app is kept in `legacy/app_v03.py`.

# MarketPilot v0.5 — research review and changes

Date: 20 November 2026. This summarises what published research and first live
results say about MarketPilot's design, and what was changed because of it.
Paper trading only; nothing here is financial advice.

## 1. What the first real-data run showed (v0.4, 20 Nov 2026)

| Market | Result |
|---|---|
| Stocks (10 hand-picked large caps) | 3 strategies approved: EMA trend (+0.69R/trade OOS), Donchian breakout (+0.59R), RSI(2) mean reversion (+0.06R, 66% win rate) |
| Crypto (hourly bars) | all 4 strategies rejected — every one lost money after costs |
| Forex (hourly bars) | all 4 strategies rejected — every one lost money after costs |

## 2. Findings

**Survivorship / hindsight bias.** The v0.4 stock list (NVDA, META, MSFT…) was
chosen knowing those companies became today's winners. Testing only on survivors
overstates returns and understates risk, and using today's index membership in the
past is a form of look-ahead bias. The stock results above were therefore almost
certainly flattered.
*Change:* the stock universe is now 20 broad, long-lived ETFs (US indices,
sectors, international, bonds, gold, property). Some selection effect remains
(these funds survived), but no single-company hindsight.

**Trend following has the deepest evidence.** Moskowitz, Ooi & Pedersen (2012)
found 12-month time-series momentum in all 58 futures markets they studied, with
persistence for 1–12 months that partly reverses afterwards. Hurst, Ooi & Pedersen
(2017) extend the evidence to a century of data across 67 markets. Faber (2007)
showed a simple 10-month / 200-day moving-average rule cut volatility and drawdowns
across asset classes. Counterpoint: Huang et al. (2020) find asset-by-asset evidence
much weaker than pooled results — so diversification across many markets matters.
*Change:* new `trend_sma200` strategy (enter above the 200-day average with a
positive 12-month return, exit on a close back below it), and a **market filter**:
no new long equity-ETF or crypto trades while SPY / BTC is below its 200-day average.
Bonds and gold are excluded from the filter because they often rise when stocks fall.

**Short-timeframe forex rules rarely survive costs.** Studies of intraday FX rules
find profits disappear once realistic transaction costs and data-snooping are
accounted for; evidence for technical rules is stronger on daily data, and even then
inconsistent over time. This matches our hourly forex results.
*Change:* forex moved from hourly to **daily** bars (6 pairs, 20 years of history).

**Crypto momentum is real but noisy, and fees are high.** Liu & Tsyvinski find a
strong time-series momentum effect in crypto; others find weaker or inconsistent
results depending on the sample, with momentum concentrated in large coins. Retail
fees are much higher than we modelled: entry-tier taker fees are about 0.40% on
Kraken Pro and 0.60% on Coinbase Advanced.
*Change:* crypto moved to **daily** bars; commission raised from 0.10% to **0.40% per side**.

**RSI(2) mean reversion: use the published exit.** Connors' rules buy RSI(2)
extremes above the 200-day average and exit on a close back above the 5-day
average, not at a fixed price target. It is a short, high-win-rate strategy — the
closest match to a 70%+ win-rate goal — but its edge per trade is small.
*Change:* entry now also requires price below its 5-day average; exit on a close
above the 5-day average (or 10 bars, or a wide protective stop).

**Exits: trailing stops vs fixed targets.** Evidence is mixed: trailing exits suit
trend-following entries, but several tests found fixed ATR brackets did as well or
better. *Change:* trend and breakout strategies now offer chandelier trailing-stop
variants alongside fixed targets, and walk-forward testing decides on unseen data.

**Backtest overfitting.** Trying many variants and keeping the best inflates results
(Bailey & López de Prado, deflated Sharpe ratio; probability of backtest overfitting).
*Change:* the approval gate now also requires an out-of-sample **t-statistic ≥ 2**
(average result at least two standard errors above zero), on top of the existing
expectancy, profit-factor and untouched-holdout tests.

**Calendar effects (turn of the month).** Evidence is mixed and several studies find
the effects weakened or disappeared in recent decades. *Not implemented.*

**About the 70–80% win-rate goal.** The FCA found that 82% of retail CFD clients
in its sample lost money. A high win rate is easy to engineer with tight targets and
wide stops and is not the same as making money. MarketPilot keeps tracking win rate
against the goal, but approves strategies only on after-cost expectancy and
statistical evidence.

## 3. Also added in v0.5
- Phone push notifications (ntfy): trade opened/closed, kill switch, daily summary.
- Positions in symbols removed from the universe (e.g. the XOM trade from v0.4) keep
  being managed until they close; strategies for removed symbols stop trading.
- One shared exit function (`step_position`) for backtests and paper trades, so the
  new exits behave identically in both.

## 4. Sources
- Moskowitz, Ooi & Pedersen (2012), *Time Series Momentum*, Journal of Financial Economics — https://papers.ssrn.com/abstract=2089463
- Hurst, Ooi & Pedersen (2017), *A Century of Evidence on Trend-Following Investing* — https://www.aqr.com/library/journal-articles/a-century-of-evidence-on-trend-following-investing
- Huang, Li, Wang & Zhou (2020), *Time-series momentum: Is it there?* — https://core.ac.uk/download/287750773.pdf
- Faber (2007), *A Quantitative Approach to Tactical Asset Allocation* (summary) — https://www.etf.com/sections/index-investor-corner/swedroe-why-financial-trends-persist
- Liu & Tsyvinski, *Risks and Returns of Cryptocurrency* — https://papers.ssrn.com/abstract=3226806
- Han, Kang & Ryu, *Time-Series and Cross-Sectional Momentum in the Cryptocurrency Market* — https://acfr.aut.ac.nz/__data/assets/pdf_file/0009/918729/Time_Series_and_Cross_Sectional_Momentum_in_the_Cryptocurrency_Market_with_IA.pdf
- Neely & Weller, *Intraday Technical Trading in the Foreign Exchange Market* — https://text.www2.warwick.ac.uk/fac/soc/wbs/subjects/finance/research/wpaperseries/1999/99-89.pdf
- Intraday FX trading rule and breakeven-cost study (UCT) — https://open.uct.ac.za/bitstreams/1ca4b008-7ebc-4b87-8804-c8674358c228/download
- Bailey & López de Prado (2014), *The Deflated Sharpe Ratio* — https://papers.ssrn.com/abstract=2460551
- Bailey et al., *The Probability of Backtest Overfitting* — https://escholarship.org/uc/item/4w1110bb
- Survivorship and look-ahead bias in backtests — https://arxiv.org/pdf/2505.07078
- Connors RSI(2) rules (StockCharts ChartSchool) — https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models/trading-strategies/rsi-2
- Kraken vs Coinbase fees 2026 — https://financer.com/crypto/coinbase-vs-kraken/
- FCA: 82% of CFD clients lost money — https://www.fca.org.uk/news/press-releases/fca-proposes-stricter-rules-contract-difference-products
- Calendar anomalies fading — https://www.up.ac.za/media/shared/61/WP/wp_2019_02.zp166894.pdf

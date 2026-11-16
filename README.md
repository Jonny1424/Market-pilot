# MarketPilot AI — Version 0.1

A Streamlit multi-market research scanner for Forex, cryptocurrency and stocks/ETFs.

## Features
- Public historical market data via `yfinance`.
- EMA20, EMA50, RSI14 and ATR14 indicators.
- Transparent rule-based bullish, bearish and watch signals.
- Illustrative ATR-based stop and target levels.
- CSV export and a position-risk calculator.

## Deploy on Streamlit Community Cloud
1. Upload `app.py`, `requirements.txt` and `README.md` to the root of your GitHub repository.
2. In Streamlit Community Cloud, choose **Create app**.
3. Select the GitHub repository and branch `main`.
4. Set the app file path to `app.py`.
5. Deploy.

No Supabase credentials are needed for v0.1. This version does not yet save trades to Supabase.

## Important limitations
- Research prototype only; signals are rules-based, not AI predictions or financial advice.
- No broker connection, order placement, leverage or real-money execution.
- No persistent paper-trade ledger or trade journal yet.
- `yfinance` is an unofficial interface to Yahoo Finance. Data can be delayed, incomplete, rate-limited or unavailable. Check applicable data terms and rights.
- No strategy profitability is promised or established.

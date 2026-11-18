import hmac
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from supabase import create_client

st.set_page_config(page_title='MarketPilot v0.3', page_icon='📈', layout='wide')

UNIVERSE = {
    'Forex': ['GBPUSD=X', 'EURUSD=X', 'USDJPY=X'],
    'Crypto': ['BTC-USD', 'ETH-USD'],
    'Stocks & ETFs': ['AAPL', 'MSFT', 'NVDA', 'SPY', 'VUAG.L'],
}


def authenticate():
    password = st.secrets.get('APP_PASSWORD', '')
    if not password:
        st.warning('Access password is not configured. Keep the app private and add APP_PASSWORD in Streamlit Secrets before sharing its URL.')
        return
    if st.session_state.get('authenticated'):
        return
    st.title('🔒 MarketPilot sign in')
    with st.form('login'):
        attempt = st.text_input('App password', type='password')
        submitted = st.form_submit_button('Unlock')
    if submitted:
        if hmac.compare_digest(attempt, str(password)):
            st.session_state['authenticated'] = True
            st.rerun()
        else:
            st.error('Incorrect password')
    st.stop()


authenticate()
st.title('📈 MarketPilot')
st.caption('Version 0.3 • Research scanner, risk controls and Supabase paper trading')
st.warning('Paper trading only. Scores are heuristic rankings, not success probabilities. No live orders or broker integration.')

try:
    db = create_client(st.secrets['SUPABASE_URL'], st.secrets['SUPABASE_KEY'])
except Exception:
    st.error('Supabase connection could not be configured. Check the existing SUPABASE_URL and SUPABASE_KEY in Streamlit Secrets. Never share them.')
    st.stop()


def err(exc):
    # Avoid printing raw provider errors that might contain sensitive information.
    st.error(f'Database operation failed ({type(exc).__name__}). Check Supabase table permissions and Streamlit logs. Do not share credentials.')


@st.cache_data(ttl=300, show_spinner=False)
def load(symbol, period='1y', interval='1d'):
    frame = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False, threads=False)
    if frame is None or frame.empty:
        return pd.DataFrame()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame.columns = [str(x).title() for x in frame.columns]
    return frame.dropna(subset=['Close']) if 'Close' in frame else pd.DataFrame()


def calculate(frame):
    x = frame.copy()
    close = x['Close'].astype(float)
    x['EMA20'] = close.ewm(span=20, adjust=False).mean()
    x['EMA50'] = close.ewm(span=50, adjust=False).mean()
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    x['RSI14'] = (100 - 100 / (1 + rs)).fillna(50)
    high = x['High'].astype(float) if 'High' in x else close
    low = x['Low'].astype(float) if 'Low' in x else close
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    x['ATR14'] = tr.rolling(14).mean()
    return x


def setup(symbol, market, frame):
    if len(frame) < 55:
        return None
    x = calculate(frame)
    row = x.iloc[-1]
    if not np.isfinite(row['ATR14']) or row['ATR14'] <= 0:
        return None
    price, ema20, ema50, rsi, atr = (float(row[k]) for k in ['Close', 'EMA20', 'EMA50', 'RSI14', 'ATR14'])
    if ema20 > ema50 and price > ema20:
        direction = 'Long'
    elif ema20 < ema50 and price < ema20:
        direction = 'Short'
    else:
        direction = 'Watch'
    # Transparent 0–100 *ranking*, not an estimated win probability.
    trend = 30 if direction != 'Watch' else 0
    momentum = 25 if (direction == 'Long' and 50 <= rsi <= 68) or (direction == 'Short' and 32 <= rsi <= 50) else 10
    slope = 20 if (direction == 'Long' and price > float(x['Close'].iloc[-2])) or (direction == 'Short' and price < float(x['Close'].iloc[-2])) else 0
    volatility = 15 if 0.002 <= atr / price <= 0.08 else 5
    data_quality = 10 if len(frame) >= 100 else 5
    score = trend + momentum + slope + volatility + data_quality
    stop = price - 2 * atr if direction == 'Long' else price + 2 * atr if direction == 'Short' else np.nan
    target = price + 3 * atr if direction == 'Long' else price - 3 * atr if direction == 'Short' else np.nan
    return {'Market': market, 'Symbol': symbol, 'Setup': direction, 'Ranking / 100': score,
            'Last price': price, 'RSI14': rsi, 'ATR14': atr, 'Stop (illustrative)': stop,
            'Target (illustrative)': target, 'Data time': str(x.index[-1])}


def trades(status=None):
    q = db.table('trades').select('*').order('opened_at', desc=True)
    if status:
        q = q.eq('status', status)
    return q.execute().data or []


def pnl(trade):
    if trade.get('status') != 'Closed' or trade.get('exit_price') is None:
        return np.nan
    difference = float(trade['exit_price']) - float(trade['entry_price'])
    if trade['direction'] == 'Short':
        difference *= -1
    return difference * float(trade['quantity']) - float(trade.get('fees') or 0) - float(trade.get('slippage') or 0)


with st.sidebar:
    st.header('Research settings')
    period = st.selectbox('History', ['6mo', '1y', '2y', '5y'], index=1)
    interval = st.selectbox('Interval', ['1d', '1h'])
    balance = st.number_input('Reference balance (£)', min_value=50.0, value=500.0, step=50.0)
    risk_pct = st.slider('Risk budget per trade (%)', 0.1, 2.0, 0.5, 0.1)
    st.caption('Daily candles are recommended. Intraday history is limited by the data provider.')
    if st.session_state.get('authenticated') and st.button('Sign out'):
        st.session_state['authenticated'] = False
        st.rerun()

t1, t2, t3, t4, t5, t6 = st.tabs(['Scanner', 'Risk calculator', 'Open paper trade', 'Positions', 'Performance', 'Journal'])

with t1:
    st.subheader('Ranked market opportunities')
    if st.button('Scan markets', type='primary'):
        rows, missed = [], []
        with st.spinner('Downloading market data…'):
            for market, symbols in UNIVERSE.items():
                for symbol in symbols:
                    try:
                        result = setup(symbol, market, load(symbol, period, interval))
                        if result:
                            rows.append(result)
                        else:
                            missed.append(symbol)
                    except Exception:
                        missed.append(symbol)
        st.session_state['scan'] = pd.DataFrame(rows).sort_values('Ranking / 100', ascending=False) if rows else pd.DataFrame()
        st.session_state['scan_at'] = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
        if missed:
            st.info('No usable data for: ' + ', '.join(missed))
    if 'scan' in st.session_state:
        data = st.session_state['scan']
        st.caption('Last scan: ' + st.session_state.get('scan_at', ''))
        if data.empty:
            st.info('No usable market data returned. Try daily interval or another history range.')
        else:
            st.dataframe(data, width='stretch', hide_index=True)
            st.download_button('Download scan CSV', data.to_csv(index=False), 'marketpilot_scan.csv', 'text/csv')
            selected = st.selectbox('Chart symbol', data['Symbol'].tolist())
            chart = calculate(load(selected, period, interval))
            if not chart.empty:
                st.line_chart(chart[['Close', 'EMA20', 'EMA50']], width='stretch')
    st.caption('Ranking: trend 30, momentum up to 25, recent price direction 20, volatility up to 15, data length up to 10. Rankings are not calibrated probabilities.')

with t2:
    st.subheader('Illustrative risk calculator')
    st.caption('Assumes price and account currency are the same and one unit has a multiplier of 1. Not suitable for forex lots, derivatives or non-GBP instruments without conversion.')
    entry = st.number_input('Entry price', min_value=0.00000001, value=100.0, format='%.8f', key='risk_entry')
    stop = st.number_input('Stop price', min_value=0.00000001, value=95.0, format='%.8f', key='risk_stop')
    target = st.number_input('Target price', min_value=0.00000001, value=110.0, format='%.8f', key='risk_target')
    distance = abs(entry - stop)
    budget = balance * risk_pct / 100
    units = budget / distance if distance else 0
    reward = abs(target - entry)
    st.metric('Risk budget (£)', f'£{budget:,.2f}')
    st.metric('Illustrative quantity', f'{units:,.4f}' if distance else '—')
    st.metric('Reward / risk', f'{reward / distance:.2f}:1' if distance else '—')
    if distance == 0:
        st.error('Entry and stop must be different.')
    if not ((stop < entry < target) or (target < entry < stop)):
        st.warning('Stop and target should be on opposite sides of entry.')
    st.info('For USD stocks and FX, do not interpret this quantity or £ amount as executable until currency conversion, lot size and contract multiplier are handled.')

with t3:
    st.subheader('Record a simulated trade')
    with st.form('new_trade'):
        c1, c2 = st.columns(2)
        asset = c1.text_input('Ticker', placeholder='AAPL')
        asset_class = c2.selectbox('Market', ['Forex', 'Crypto', 'Stocks & ETFs', 'Other'])
        direction = c1.selectbox('Direction', ['Long', 'Short'])
        entry_price = c2.number_input('Entry price', min_value=0.00000001, value=100.0, format='%.8f')
        quantity = c1.number_input('Units', min_value=0.00000001, value=1.0, format='%.8f')
        stop_loss = c2.number_input('Stop (0 = none)', min_value=0.0, value=0.0, format='%.8f')
        take_profit = c1.number_input('Target (0 = none)', min_value=0.0, value=0.0, format='%.8f')
        fees = c2.number_input('Fees (illustrative £)', min_value=0.0, value=0.0)
        slippage = c1.number_input('Slippage (illustrative £)', min_value=0.0, value=0.0)
        notes = st.text_area('Reason / notes')
        submit = st.form_submit_button('Save paper trade')
    if submit:
        invalid_stop = stop_loss > 0 and ((direction == 'Long' and stop_loss >= entry_price) or (direction == 'Short' and stop_loss <= entry_price))
        invalid_target = take_profit > 0 and ((direction == 'Long' and take_profit <= entry_price) or (direction == 'Short' and take_profit >= entry_price))
        if not asset.strip() or invalid_stop or invalid_target:
            st.error('Check ticker and stop/target direction before saving.')
        else:
            row = {'asset': asset.strip().upper(), 'asset_class': asset_class, 'direction': direction,
                   'entry_price': float(entry_price), 'quantity': float(quantity),
                   'stop_loss': float(stop_loss) if stop_loss else None,
                   'take_profit': float(take_profit) if take_profit else None,
                   'fees': float(fees), 'slippage': float(slippage), 'status': 'Open',
                   'notes': notes.strip() or None, 'opened_at': datetime.now(timezone.utc).isoformat()}
            try:
                db.table('trades').insert(row).execute()
                st.success('Saved simulated trade to Supabase.')
            except Exception as exc:
                err(exc)

with t4:
    st.subheader('Open paper positions')
    try:
        active = trades('Open')
        if not active:
            st.info('No open positions.')
        else:
            st.dataframe(pd.DataFrame(active), width='stretch', hide_index=True)
            options = {f"{str(x['id'])[:8]} • {x['asset']} • {x['direction']}": x for x in active}
            selected = st.selectbox('Position to close', list(options))
            chosen = options[selected]
            with st.form('close_position'):
                exit_price = st.number_input('Exit price', min_value=0.00000001, value=float(chosen['entry_price']), format='%.8f')
                closing_fees = st.number_input('Additional fees (£)', min_value=0.0, value=0.0)
                exit_reason = st.text_area('Exit reason')
                lessons = st.text_area('Lessons learned')
                closing = st.form_submit_button('Close simulated position')
            if closing:
                try:
                    # Existing schema uses two separate writes; a failed journal insert may leave a closed trade.
                    db.table('trades').update({'exit_price': float(exit_price), 'fees': float(chosen.get('fees') or 0) + float(closing_fees), 'status': 'Closed', 'closed_at': datetime.now(timezone.utc).isoformat()}).eq('id', chosen['id']).execute()
                    try:
                        db.table('trade_journal').insert({'trade_id': chosen['id'], 'entry_reason': chosen.get('notes'), 'exit_reason': exit_reason.strip() or None, 'lessons_learned': lessons.strip() or None}).execute()
                        st.success('Position closed and journal saved.')
                    except Exception as exc:
                        st.warning('Position closed, but journal could not be saved. Check the database before retrying.')
                        err(exc)
                except Exception as exc:
                    err(exc)
    except Exception as exc:
        err(exc)

with t5:
    st.subheader('Paper-trading performance')
    try:
        history = pd.DataFrame(trades())
        if history.empty:
            st.info('No trades recorded.')
        else:
            history['Indicative P/L'] = history.apply(pnl, axis=1)
            closed = history[history['status'] == 'Closed'].copy()
            if not closed.empty:
                values = closed['Indicative P/L'].fillna(0)
                equity = values.iloc[::-1].cumsum()
                drawdown = (equity - equity.cummax()).min()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric('Closed trades', len(closed))
                c2.metric('Indicative P/L', f'{values.sum():,.2f} (native quote units*)')
                c3.metric('Win rate', f'{(values > 0).mean() * 100:.1f}%')
                c4.metric('Max realised drawdown*', f'{abs(drawdown):,.2f}')
                st.line_chart(pd.DataFrame({'Cumulative indicative P/L': equity.to_numpy()}), width='stretch')
            st.dataframe(history, width='stretch', hide_index=True)
            st.download_button('Download trades CSV', history.to_csv(index=False), 'marketpilot_trades.csv', 'text/csv')
            st.caption('* P/L is not GBP-normalised. For USD assets it is in USD before fees; mixed-currency totals and drawdown are NOT financially meaningful. No unrealised P/L is included.')
    except Exception as exc:
        err(exc)

with t6:
    st.subheader('Trade journal')
    try:
        journal = db.table('trade_journal').select('*').order('created_at', desc=True).execute().data or []
        if journal:
            data = pd.DataFrame(journal)
            st.dataframe(data, width='stretch', hide_index=True)
            st.download_button('Download journal CSV', data.to_csv(index=False), 'marketpilot_journal.csv', 'text/csv')
        else:
            st.info('No journal entries yet.')
    except Exception as exc:
        err(exc)

st.divider()
st.caption('Yahoo Finance data may be delayed or unavailable. This is a single-user research prototype; protect your Streamlit app and do not share the URL or API credentials. Not financial advice.')

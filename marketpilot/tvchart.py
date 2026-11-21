"""TradingView-style interactive price chart for the dashboard.

Built on TradingView's open-source Lightweight Charts library (Apache 2.0,
bundled in marketpilot/vendor). It behaves like TradingView:

* drag the price axis on the right up/down to zoom prices in and out
* drag the date axis along the bottom left/right to zoom time in and out
* drag the chart itself to slide back through history (with momentum on touch)
* mouse wheel / trackpad zooms; double-click an axis to reset it
* range buttons (1M ... All), live OHLC legend, entry/stop/target lines and
  markers where the bots entered and exited trades.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

_VENDOR = Path(__file__).parent / "vendor" / "lightweight-charts.standalone.production.js"


@lru_cache(maxsize=1)
def _library() -> str:
    return _VENDOR.read_text(encoding="utf-8")


def _day(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def chart_payload(df: pd.DataFrame, levels=None, trades=None) -> dict:
    """Plain JSON for the chart: candles, volume, averages, price lines, trade markers."""
    d = df.dropna(subset=["open", "high", "low", "close"]).copy()
    d = d[~d.index.duplicated(keep="last")]
    times = [_day(t) for t in d.index]
    candles = [{"time": t, "open": float(o), "high": float(h), "low": float(l), "close": float(c)}
               for t, o, h, l, c in zip(times, d["open"], d["high"], d["low"], d["close"])]
    vol = []
    if "volume" in d and d["volume"].fillna(0).sum() > 0:
        for t, v, o, c in zip(times, d["volume"].fillna(0), d["open"], d["close"]):
            vol.append({"time": t, "value": float(v),
                        "color": "rgba(22,199,132,.35)" if c >= o else "rgba(234,57,67,.35)"})
    close = d["close"]
    lines = {}
    for name, series in (("EMA 20", close.ewm(span=20, adjust=False).mean()),
                         ("EMA 50", close.ewm(span=50, adjust=False).mean()),
                         ("SMA 200", close.rolling(200).mean())):
        lines[name] = [{"time": t, "value": float(v)} for t, v in zip(times, series) if np.isfinite(v)]
    price_lines = [{"title": n, "price": float(p), "color": c} for n, p, c in (levels or [])
                   if p is not None and np.isfinite(float(p))]
    markers = []
    valid = set(times)
    for t in trades or []:
        side = int(t.get("direction") or 1)
        op = t.get("opened_at") or t.get("signal_bar_time")
        if op and t.get("entry_price") is not None:
            day = _day(op)
            if day in valid:
                markers.append({"time": day, "position": "belowBar" if side == 1 else "aboveBar",
                                "color": "#4f7cff", "shape": "arrowUp" if side == 1 else "arrowDown",
                                "text": ("BUY" if side == 1 else "SELL") + (f" {t['grade']}" if t.get("grade") else "")})
        if t.get("closed_at") and t.get("r_multiple") is not None:
            day = _day(t["closed_at"])
            if day in valid:
                r = float(t["r_multiple"])
                markers.append({"time": day, "position": "aboveBar" if side == 1 else "belowBar",
                                "color": "#16c784" if r > 0 else "#ea3943", "shape": "circle",
                                "text": f"{r:+.1f}R"})
    markers.sort(key=lambda m: m["time"])
    return {"candles": candles, "volume": vol, "lines": lines, "priceLines": price_lines, "markers": markers}


def chart_html(df: pd.DataFrame, symbol: str, levels=None, trades=None, height: int = 520) -> str:
    payload = json.dumps(chart_payload(df, levels, trades))
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  html, body {{ margin:0; padding:0; background:#0b0e14; font-family: Inter, -apple-system, sans-serif; }}
  #wrap {{ position:relative; border:1px solid #232937; border-radius:12px; overflow:hidden; background:#0b0e14; }}
  #bar {{ display:flex; gap:6px; align-items:center; padding:8px 10px; border-bottom:1px solid #232937;
          flex-wrap:wrap; }}
  #bar button {{ background:#131722; color:#8a93a6; border:1px solid #232937; border-radius:6px;
                 padding:4px 9px; font-size:12px; cursor:pointer; font-weight:600; }}
  #bar button:hover, #bar button.on {{ color:#e6e8ee; border-color:#4f7cff; }}
  #sym {{ color:#e6e8ee; font-weight:700; font-size:14px; margin-right:8px; }}
  #chart {{ width:100%; height:{height - 70}px; touch-action:none; }}
  #legend {{ position:absolute; left:12px; top:50px; z-index:3; font-size:12px; color:#8a93a6;
             pointer-events:none; }}
  #legend b {{ color:#e6e8ee; font-weight:600; }}
  .up {{ color:#16c784 !important; }} .down {{ color:#ea3943 !important; }}
  #hint {{ color:#5d6578; font-size:11px; padding:5px 10px; border-top:1px solid #232937; }}
</style></head><body>
<div id="wrap">
  <div id="bar"><span id="sym">{symbol}</span>
    <button data-r="22">1M</button><button data-r="66">3M</button><button data-r="130">6M</button>
    <button data-r="260" class="on">1Y</button><button data-r="780">3Y</button><button data-r="0">All</button>
    <span style="flex:1"></span><button id="reset" title="Reset zoom and price scale">Reset</button></div>
  <div id="legend"></div>
  <div id="chart"></div>
  <div id="hint">Drag the price axis (right) to zoom prices · drag the date axis (bottom) to zoom time ·
    drag the chart to slide back in time · double-click an axis to reset</div>
</div>
<script>{_library()}</script>
<script>
const D = {payload};
const LW = LightweightCharts;
const el = document.getElementById('chart');
const chart = LW.createChart(el, {{
  autoSize: true,
  layout: {{ background: {{ type: 'solid', color: '#0b0e14' }}, textColor: '#8a93a6', fontSize: 11,
            fontFamily: 'Inter, -apple-system, sans-serif' }},
  grid: {{ vertLines: {{ color: 'rgba(35,41,55,.6)' }}, horzLines: {{ color: 'rgba(35,41,55,.6)' }} }},
  crosshair: {{ mode: LW.CrosshairMode.Normal,
               vertLine: {{ color: '#4f7cff', labelBackgroundColor: '#4f7cff', style: 3 }},
               horzLine: {{ color: '#4f7cff', labelBackgroundColor: '#4f7cff', style: 3 }} }},
  rightPriceScale: {{ borderColor: '#232937', scaleMargins: {{ top: 0.08, bottom: 0.22 }} }},
  timeScale: {{ borderColor: '#232937', rightOffset: 6, barSpacing: 7, minBarSpacing: 0.5,
               fixLeftEdge: false, lockVisibleTimeRangeOnResize: true }},
  handleScale: {{ axisPressedMouseMove: {{ time: true, price: true }},
                 axisDoubleClickReset: {{ time: true, price: true }}, mouseWheel: true, pinch: true }},
  handleScroll: {{ mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false }},
  kineticScroll: {{ touch: true, mouse: false }},
}});
const candles = chart.addSeries(LW.CandlestickSeries, {{
  upColor: '#16c784', downColor: '#ea3943', borderUpColor: '#16c784', borderDownColor: '#ea3943',
  wickUpColor: '#16c784', wickDownColor: '#ea3943', priceLineColor: '#4f7cff' }});
candles.setData(D.candles);
if (D.volume.length) {{
  const vol = chart.addSeries(LW.HistogramSeries, {{ priceScaleId: 'vol', priceFormat: {{ type: 'volume' }},
                                                    lastValueVisible: false, priceLineVisible: false }});
  chart.priceScale('vol').applyOptions({{ scaleMargins: {{ top: 0.82, bottom: 0 }}, visible: false }});
  vol.setData(D.volume);
}}
const colors = {{ 'EMA 20': '#f5c451', 'EMA 50': '#4f7cff', 'SMA 200': '#a78bfa' }};
const lineSeries = {{}};
for (const [name, pts] of Object.entries(D.lines)) {{
  if (!pts.length) continue;
  lineSeries[name] = chart.addSeries(LW.LineSeries, {{ color: colors[name], lineWidth: 1,
    lastValueVisible: false, priceLineVisible: false, crosshairMarkerVisible: false }});
  lineSeries[name].setData(pts);
}}
for (const p of D.priceLines) {{
  candles.createPriceLine({{ price: p.price, color: p.color, lineWidth: 1, lineStyle: 2,
                            axisLabelVisible: true, title: p.title }});
}}
if (D.markers.length && LW.createSeriesMarkers) LW.createSeriesMarkers(candles, D.markers);

const fmt = v => v >= 1000 ? v.toLocaleString(undefined, {{maximumFractionDigits: 2}})
                : v.toPrecision(v >= 1 ? 6 : 4).replace(/\\.?0+$/, '');
const legend = document.getElementById('legend');
const last = D.candles[D.candles.length - 1];
function showLegend(bar, prev) {{
  if (!bar) return;
  const chg = prev ? (bar.close / prev.close - 1) * 100 : 0;
  const cls = bar.close >= bar.open ? 'up' : 'down';
  legend.innerHTML = `O <b class="${{cls}}">${{fmt(bar.open)}}</b> H <b class="${{cls}}">${{fmt(bar.high)}}</b> ` +
    `L <b class="${{cls}}">${{fmt(bar.low)}}</b> C <b class="${{cls}}">${{fmt(bar.close)}}</b> ` +
    `<b class="${{chg >= 0 ? 'up' : 'down'}}">${{chg >= 0 ? '+' : ''}}${{chg.toFixed(2)}}%</b>` +
    `<br><span style="color:#f5c451">EMA 20</span> · <span style="color:#4f7cff">EMA 50</span> · ` +
    `<span style="color:#a78bfa">SMA 200</span>`;
}}
const byTime = new Map(D.candles.map((c, i) => [c.time, i]));
showLegend(last, D.candles[D.candles.length - 2]);
chart.subscribeCrosshairMove(param => {{
  const bar = param.time ? param.seriesData.get(candles) : null;
  if (!bar) {{ showLegend(last, D.candles[D.candles.length - 2]); return; }}
  const i = byTime.get(typeof param.time === 'string' ? param.time :
    `${{param.time.year}}-${{String(param.time.month).padStart(2,'0')}}-${{String(param.time.day).padStart(2,'0')}}`);
  showLegend(bar, i ? D.candles[i - 1] : null);
}});

function setRange(n) {{
  const total = D.candles.length;
  if (!n || n >= total) chart.timeScale().fitContent();
  else chart.timeScale().setVisibleLogicalRange({{ from: total - n, to: total + 4 }});
  chart.priceScale('right').applyOptions({{ autoScale: true }});
}}
document.querySelectorAll('#bar button[data-r]').forEach(b => b.addEventListener('click', () => {{
  document.querySelectorAll('#bar button[data-r]').forEach(x => x.classList.remove('on'));
  b.classList.add('on'); setRange(parseInt(b.dataset.r));
}}));
document.getElementById('reset').addEventListener('click', () => {{
  document.querySelectorAll('#bar button[data-r]').forEach(x => x.classList.toggle('on', x.dataset.r === '260'));
  setRange(260);
}});
setRange(260);
</script></body></html>"""

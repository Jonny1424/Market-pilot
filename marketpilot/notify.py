"""Phone push notifications via ntfy (https://ntfy.sh) — free, no account.

Set the NTFY_TOPIC secret to your private channel name, install the ntfy app on
your phone and subscribe to the same name. If NTFY_TOPIC is not set, nothing is
sent. A failed notification never stops trading.
"""
from __future__ import annotations

import os

NO_TARGET = 50.0  # target_atr at/above this means "no fixed target"
NTFY_SERVER = os.environ.get("NTFY_SERVER", "https://ntfy.sh")


def send(title: str, message: str, tags: str = "chart_with_upwards_trend", priority: int = 3) -> bool:
    topic = os.environ.get("NTFY_TOPIC", "").strip()
    if not topic:
        return False
    try:
        import requests
        r = requests.post(f"{NTFY_SERVER}/{topic}", data=message.encode("utf-8"), timeout=10,
                          headers={"Title": title.encode("ascii", "ignore").decode(),
                                   "Tags": tags, "Priority": str(priority)})
        return r.ok
    except Exception:
        return False


def trade_opened(t: dict) -> None:
    side = "LONG" if int(t["direction"]) > 0 else "SHORT"
    g = f" [{t['grade']}]" if t.get("grade") else ""
    odds = (f"\nOdds: target first {float(t['tm_p_target']):.0%}, stop first {float(t['tm_p_stop']):.0%}"
            if t.get("tm_p_target") is not None and t.get("tm_p_stop") is not None else "")
    send(f"Paper trade opened: {t['symbol']} {side}{g}",
         f"Entry {float(t['entry_price']):.4f} | stop {float(t['stop']):.4f} | target "
         f"{'exit rule / trailing stop' if float(t.get('target_atr') or 0) >= NO_TARGET else _lvl(t['target'])}\nRisk £{float(t['risk_gbp']):.2f} | {t['strategy'].split('(')[0]}{odds}",
         tags="arrow_forward")


def partial_taken(t: dict, st: dict) -> None:
    side = "LONG" if int(t["direction"]) > 0 else "SHORT"
    f = float((st.get("mgmt") or {}).get("partial_frac", 0.5))
    send(f"Partial profit: {t['symbol']} {side}",
         f"Banked {f:.0%} at {float(st['partial_price']):.4f} (+1R). Stop on the rest moved to breakeven "
         f"({float(st['stop']):.4f}).", tags="moneybag")


def trade_closed(t: dict) -> None:
    win = float(t["r_multiple"]) > 0
    send(f"{'WIN' if win else 'LOSS'}: {t['symbol']} {float(t['r_multiple']):+.2f}R (£{float(t['pnl_gbp']):+.2f})",
         f"Exit: {t['exit_reason']} @ {float(t['exit_price']):.4f}\nStrategy: {t['strategy'].split('(')[0]}",
         tags="white_check_mark" if win else "x")


def kill_switch(reason: str) -> None:
    send("MarketPilot paused (kill switch)", reason, tags="rotating_light", priority=5)


def daily_summary(overall: dict, equity: float, open_positions: int, target_wr: float,
                  scan_text: str = "") -> None:
    o = overall
    send("MarketPilot daily summary",
         f"Equity £{equity:,.2f} | open positions {open_positions}\n"
         f"Closed trades {o['trades']} | avg loss {o['avg_loss_r']:+.2f}R | max drawdown {o['max_drawdown_r']:.1f}R\n"
         f"Expectancy {o['expectancy_r']:+.2f}R | win rate {o['win_rate']:.0%} | P&L £{o['pnl_gbp']:+.2f}"
         + (f"\n\n{scan_text}" if scan_text else ""),
         tags="bar_chart", priority=2)


def _lvl(x) -> str:
    try:
        v = float(x)
        return "none (trailing/exit rule)" if v > 1e8 or v <= 0 else f"{v:.4f}"
    except (TypeError, ValueError):
        return "-"

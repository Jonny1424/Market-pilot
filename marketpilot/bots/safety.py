"""Monitoring & Safety Bot — stale prices, failed feeds, kill switch, auto-pause."""
from __future__ import annotations

import pandas as pd

from ..data import interval_td

KILL = "kill_switch"
PAUSE_REASON = "pause_reason"


def market_open(asset_class: str, now: pd.Timestamp) -> bool:
    wd, h = now.weekday(), now.hour
    if asset_class == "crypto":
        return True
    if asset_class == "forex":  # Sun 22:00 – Fri 21:00 UTC
        return not (wd == 5 or (wd == 4 and h >= 21) or (wd == 6 and h < 22))
    # US stocks, regular session ≈ 14:30–21:00 UTC Mon–Fri (DST ignored, conservative)
    return wd < 5


def is_stale(asset_class: str, interval: str, last_bar: pd.Timestamp, now: pd.Timestamp,
             multiplier: float) -> bool:
    step = pd.Timedelta(interval_td(interval))
    age = now - (last_bar + step)  # time since that bar closed
    if interval == "1d":
        return age > pd.Timedelta(days=4)  # covers weekends + a holiday
    if not market_open(asset_class, now):
        return age > pd.Timedelta(days=3)
    return age > step * multiplier


class SafetyBot:
    def __init__(self, store, settings):
        self.store = store
        self.cfg = settings["safety"]

    def kill_switch_on(self) -> bool:
        return self.store.get_control(KILL, "off") == "on"

    def engage(self, reason: str, ts: str | None = None) -> None:
        if not self.kill_switch_on():
            self.store.set_control(KILL, "on")
            self.store.set_control(PAUSE_REASON, reason)
            self.store.log("safety", f"KILL SWITCH ENGAGED: {reason}", "error", ts)

    def release(self) -> None:
        self.store.set_control(KILL, "off")
        self.store.set_control(PAUSE_REASON, "")
        self.store.log("safety", "kill switch released by user", "warning")

    def feed_failure(self, symbol: str, ts: str) -> None:
        key = f"feed_failures:{symbol}"
        n = int(self.store.get_control(key, "0")) + 1
        self.store.set_control(key, str(n))
        if n >= self.cfg["max_consecutive_feed_failures"]:
            self.store.log("safety", f"{symbol}: {n} consecutive feed failures — symbol skipped", "warning", ts)

    def feed_ok(self, symbol: str) -> None:
        key = f"feed_failures:{symbol}"
        if self.store.get_control(key, "0") != "0":
            self.store.set_control(key, "0")

    def end_of_cycle(self, failed: int, total: int, ts: str) -> None:
        if total and failed / total > 0.5:
            self.engage(f"{failed}/{total} data feeds failed in one cycle", ts)

"""News & Economic Events Bot — a VETO only, never a reason to enter.

1. Economic calendar blackouts from config/events.yaml (deterministic).
2. Earnings blackout for individual stocks (via yfinance, live mode only).
3. Optional LLM headline check (Claude) — flags event risk that could invalidate
   a trade. LLM judgements cannot be honestly backtested (the model may already
   know what happened next), so they are only used live and always logged.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

ETFS = {"SPY", "QQQ", "IWM", "DIA", "VUAG.L", "VOO", "VTI", "EFA", "EEM", "XLK", "XLF", "XLE", "XLV",
        "XLY", "XLP", "XLI", "XLU", "XLB", "TLT", "IEF", "GLD", "SLV", "VNQ", "MDY", "XLC", "XLRE"}


def symbol_tags(symbol: str, asset_class: str) -> set[str]:
    tags = {asset_class}
    if asset_class == "forex":
        pair = symbol.replace("=X", "")
        tags |= {pair[:3], pair[3:6]}
    else:
        tags.add("USD")
    return tags


class NewsBot:
    def __init__(self, settings, live: bool):
        self.cfg = settings["news"]
        self.live = live
        path = Path(settings.root) / self.cfg["calendar_file"]
        self.events = []
        if path.exists():
            for e in (yaml.safe_load(path.read_text()) or {}).get("events", []):
                e["time"] = datetime.fromisoformat(str(e["time"]).replace("Z", "+00:00"))
                self.events.append(e)
        self._earnings_cache: dict[str, list[datetime]] = {}

    def check(self, symbol: str, asset_class: str, now: datetime) -> list[str]:
        vetoes = []
        tags = symbol_tags(symbol, asset_class)
        for e in self.events:
            start = e["time"] - timedelta(minutes=e.get("window_minutes_before", 60))
            end = e["time"] + timedelta(minutes=e.get("window_minutes_after", 120))
            if start <= now <= end and tags & set(e.get("affects", [])):
                vetoes.append(f"event blackout: {e['name']}")
        if self.live and asset_class == "stocks" and symbol not in ETFS and self.cfg.get("use_earnings_blackout"):
            days = self.cfg.get("earnings_blackout_days", 1)
            for d in self._earnings(symbol):
                if abs((d - now).total_seconds()) <= days * 86400 + 43200:
                    vetoes.append(f"earnings blackout ({d.date()})")
        if self.live and self.cfg.get("llm_enabled") and os.environ.get("ANTHROPIC_API_KEY"):
            v = self._llm_check(symbol)
            if v:
                vetoes.append(v)
        return vetoes

    def _earnings(self, symbol: str) -> list[datetime]:
        if symbol in self._earnings_cache:
            return self._earnings_cache[symbol]
        dates = []
        try:
            import yfinance as yf
            cal = yf.Ticker(symbol).calendar or {}
            for d in cal.get("Earnings Date", []) or []:
                dates.append(datetime(d.year, d.month, d.day, 12, tzinfo=timezone.utc))
        except Exception:
            pass
        self._earnings_cache[symbol] = dates
        return dates

    def _llm_check(self, symbol: str) -> str | None:
        try:
            import anthropic
            import yfinance as yf
            items = (yf.Ticker(symbol).news or [])[:8]
            heads = [((i.get("content") or {}).get("title") or i.get("title") or "") for i in items]
            heads = [h for h in heads if h]
            if not heads:
                return None
            client = anthropic.Anthropic()
            msg = client.messages.create(
                model=self.cfg.get("llm_model", "claude-haiku-5-5"), max_tokens=200,
                messages=[{"role": "user", "content":
                    "You screen trading risk. Given these recent headlines for " + symbol +
                    ", is there a specific scheduled or breaking event (earnings, regulatory action, "
                    "halt, merger, major macro shock) in the next 24h that makes a NEW technical trade "
                    "unusually risky? Answer only JSON {\"veto\": true|false, \"reason\": \"...\"}.\n- "
                    + "\n- ".join(heads)}])
            text = msg.content[0].text
            data = json.loads(text[text.find("{"): text.rfind("}") + 1])
            return f"LLM news veto: {data.get('reason', '')}" if data.get("veto") else None
        except Exception:
            return None  # news check failing must never open a trade, only skip the veto

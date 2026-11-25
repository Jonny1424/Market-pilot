"""Daily scan (v0.14).

Every hourly cycle the bots look at all 66 markets with every strategy: the
approved ones (which may trade) and all the others (which the Learning lab
practises). This module keeps a day-by-day record of every setup found, why it
was or wasn't traded, and builds the short text for the evening phone summary.

Stored in mp_control under "scan_log" as {"YYYY-MM-DD": [item, ...]} for the
last KEEP_DAYS bar dates. An item is one setup:
  {symbol, asset_class, strategy, side, grade, confluence, p_target,
   status: "traded" | "skipped" | "practice", why, stage}
"""
from __future__ import annotations

import json

KEEP_DAYS = 10


def _key(it: dict) -> tuple:
    return (it.get("bar"), it.get("symbol"), it.get("strategy"), it.get("side"), it.get("status"))


def load(store) -> dict[str, list[dict]]:
    try:
        raw = store.get_control("scan_log", "") or ""
        data = json.loads(raw) if raw else {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def merge(store, items: list[dict]) -> int:
    """Add this cycle's setups to the log (deduplicated). Returns how many were new."""
    if not items:
        return 0
    log = load(store)
    added = 0
    for it in items:
        day = str(it.get("bar", ""))[:10]
        if not day:
            continue
        seen = {_key(x) for x in log.get(day, [])}
        if _key(it) in seen:
            continue
        log.setdefault(day, []).append(it)
        added += 1
    if added:
        for old in sorted(log)[:-KEEP_DAYS]:
            del log[old]
        store.set_control("scan_log", json.dumps(log, default=str))
    return added


def latest(store) -> tuple[str | None, list[dict]]:
    log = load(store)
    if not log:
        return None, []
    day = sorted(log)[-1]
    return day, log[day]


def summary_text(day: str | None, items: list[dict], max_lines: int = 5) -> str:
    """Short scan section for the evening phone summary."""
    if not day or not items:
        return "Scan: no setups on the latest bars."
    traded = [i for i in items if i.get("status") == "traded"]
    skipped = [i for i in items if i.get("status") == "skipped"]
    practice = [i for i in items if i.get("status") == "practice"]
    markets = len({i.get("symbol") for i in items})
    lines = [f"Scan {day}: {len(items)} setups on {markets} markets | traded {len(traded)}, "
             f"skipped {len(skipped)}, practising {len(practice)}"]
    for i in (traded + skipped)[:max_lines]:
        tag = "TRADED" if i["status"] == "traded" else "skipped"
        why = "" if i["status"] == "traded" else f" ({(i.get('why') or '')[:60]})"
        lines.append(f"- {tag} {i['symbol']} {i['side']} {i['strategy']} grade {i.get('grade') or '-'}{why}")
    return "\n".join(lines)

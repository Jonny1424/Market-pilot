"""Persistence. Two interchangeable backends with the same methods:

* SQLiteStore  — local file, used for tests, the demo and offline runs.
* SupabaseStore — your existing Supabase project (tables prefixed mp_, see
  supabase/migrations). The worker needs the SERVICE ROLE key; the dashboard
  can use the same key server-side via Streamlit secrets.

Every candidate trade, rejection reason, fill, approval and equity snapshot is saved.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any

TABLES = {
    "mp_candidates": """
        candidate_id TEXT PRIMARY KEY, created_at TEXT, bar_time TEXT, symbol TEXT,
        asset_class TEXT, strategy TEXT, direction INTEGER, regime TEXT, price REAL,
        decision TEXT, expected_r REAL, reasons TEXT, evidence TEXT""",
    "mp_trades": """
        trade_id TEXT PRIMARY KEY, candidate_id TEXT UNIQUE, symbol TEXT, asset_class TEXT,
        strategy TEXT, direction INTEGER, regime TEXT, status TEXT, opened_at TEXT,
        entry_price REAL, quantity REAL, stop REAL, target REAL, max_hold_bars INTEGER,
        bars_held INTEGER, risk_gbp REAL, risk_per_unit REAL, fx_to_gbp REAL, notional_gbp REAL,
        closed_at TEXT, exit_price REAL, exit_reason TEXT, pnl_gbp REAL, fees_gbp REAL,
        r_multiple REAL, last_bar_time TEXT, signal_bar_time TEXT, atr REAL,
        stop_atr REAL, target_atr REAL, cost_r REAL, note TEXT, trail_atr REAL,
        exit_ma INTEGER, exit_on TEXT, best_price REAL, stop_swing INTEGER, swing_level REAL,
        confluence INTEGER, tm_p_target REAL, tm_p_stop REAL, tm_exp_r REAL, grade TEXT""",
    "mp_shadow_trades": None,  # filled below: same columns as mp_trades
    "mp_strategies": """
        strategy_id TEXT PRIMARY KEY, symbol TEXT, asset_class TEXT, strategy TEXT,
        family TEXT, params TEXT, status TEXT, reasons TEXT, oos TEXT, holdout TEXT,
        avg_cost_r REAL, researched_at TEXT, data_start TEXT, data_end TEXT""",
    "mp_market": """
        symbol TEXT PRIMARY KEY, asset_class TEXT, ts TEXT, bar_time TEXT, price REAL, chg_1d REAL,
        chg_5d REAL, chg_20d REAL, above_200d INTEGER, rsi14 REAL, regime TEXT, atr_pct REAL,
        conf_long INTEGER, conf_short INTEGER""",
    "mp_equity": """
        ts TEXT PRIMARY KEY, equity_gbp REAL, realized_gbp REAL, open_positions INTEGER""",
    "mp_events": """
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, level TEXT, bot TEXT, message TEXT""",
    "mp_control": """
        key TEXT PRIMARY KEY, value TEXT, updated_at TEXT""",
}
TABLES["mp_shadow_trades"] = TABLES["mp_trades"]
JSON_FIELDS = {"reasons", "evidence", "oos", "holdout"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _enc(row: dict) -> dict:
    return {k: (json.dumps(v, default=str) if k in JSON_FIELDS and not isinstance(v, str) else v)
            for k, v in row.items()}


def _dec(row: dict) -> dict:
    out = dict(row)
    for k in JSON_FIELDS & out.keys():
        if isinstance(out[k], str):
            try:
                out[k] = json.loads(out[k])
            except ValueError:
                pass
    return out


class BaseStore:
    # --- generic primitives implemented by backends ---
    def _insert(self, table: str, row: dict, ignore_conflict: bool = False) -> bool: ...
    def _upsert(self, table: str, row: dict, key: str) -> None: ...
    def _select(self, table: str, where: dict | None = None, order: str | None = None,
                desc: bool = False, limit: int | None = None) -> list[dict]: ...
    def _update(self, table: str, key: str, value: Any, changes: dict) -> None: ...

    # --- domain API ---
    def record_candidate(self, row: dict) -> bool:
        """Returns False if this exact candidate (symbol|strategy|bar) already exists."""
        return self._insert("mp_candidates", _enc({"created_at": now_iso(), **row}), ignore_conflict=True)

    def candidates_exists(self, candidate_id: str) -> bool:
        return bool(self._select("mp_candidates", {"candidate_id": candidate_id}, limit=1))

    def candidates(self, limit: int = 500) -> list[dict]:
        return [_dec(r) for r in self._select("mp_candidates", order="created_at", desc=True, limit=limit)]

    def open_trade(self, row: dict) -> bool:
        return self._insert("mp_trades", _enc({"status": "open", "bars_held": 0, **row}), ignore_conflict=True)

    def update_trade(self, trade_id: str, changes: dict) -> None:
        self._update("mp_trades", "trade_id", trade_id, _enc(changes))

    def trades(self, status: str | None = None) -> list[dict]:
        where = {"status": status} if status else None
        return self._select("mp_trades", where, order="opened_at", desc=True)

    # --- Learning Lab shadow ledger (practice trades, never on the real paper account) ---
    def open_shadow(self, row: dict) -> bool:
        return self._insert("mp_shadow_trades", _enc({"status": "open", "bars_held": 0, **row}), ignore_conflict=True)

    def update_shadow(self, trade_id: str, changes: dict) -> None:
        self._update("mp_shadow_trades", "trade_id", trade_id, _enc(changes))

    def shadow_trades(self, status: str | None = None) -> list[dict]:
        where = {"status": status} if status else None
        return self._select("mp_shadow_trades", where, order="opened_at", desc=True)

    # --- market overview snapshot (one row per symbol, refreshed each cycle) ---
    def save_market(self, row: dict) -> None:
        self._upsert("mp_market", row, "symbol")

    def market(self) -> list[dict]:
        return self._select("mp_market")

    def save_strategy(self, row: dict) -> None:
        self._upsert("mp_strategies", _enc(row), "strategy_id")

    def strategies(self, status: str | None = None) -> list[dict]:
        where = {"status": status} if status else None
        return [_dec(r) for r in self._select("mp_strategies", where)]

    def snapshot_equity(self, ts: str, equity: float, realized: float, open_positions: int) -> None:
        self._upsert("mp_equity", {"ts": ts, "equity_gbp": equity, "realized_gbp": realized,
                                   "open_positions": open_positions}, "ts")

    def equity_curve(self) -> list[dict]:
        return self._select("mp_equity", order="ts")

    def log(self, bot: str, message: str, level: str = "info", ts: str | None = None) -> None:
        self._insert("mp_events", {"ts": ts or now_iso(), "level": level, "bot": bot, "message": message})

    def events(self, limit: int = 300) -> list[dict]:
        return self._select("mp_events", order="ts", desc=True, limit=limit)

    def get_control(self, key: str, default: str | None = None) -> str | None:
        rows = self._select("mp_control", {"key": key})
        return rows[0]["value"] if rows else default

    def set_control(self, key: str, value: str) -> None:
        self._upsert("mp_control", {"key": key, "value": value, "updated_at": now_iso()}, "key")


class SQLiteStore(BaseStore):
    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        for name, cols in TABLES.items():
            self.conn.execute(f"CREATE TABLE IF NOT EXISTS {name} ({cols})")
            have = {r[1] for r in self.conn.execute(f"PRAGMA table_info({name})")}
            for col in cols.split(","):            # older local files: add new columns
                parts = col.split()
                if len(parts) >= 2 and parts[0] not in have and "AUTOINCREMENT" not in col:
                    self.conn.execute(f"ALTER TABLE {name} ADD COLUMN {parts[0]} {parts[1]}")
        self.conn.commit()

    def _insert(self, table, row, ignore_conflict=False):
        cols = ",".join(row)
        q = f"INSERT {'OR IGNORE ' if ignore_conflict else ''}INTO {table} ({cols}) VALUES ({','.join('?' * len(row))})"
        cur = self.conn.execute(q, list(row.values()))
        self.conn.commit()
        return cur.rowcount > 0

    def _upsert(self, table, row, key):
        cols = ",".join(row)
        sets = ",".join(f"{c}=excluded.{c}" for c in row if c != key)
        q = (f"INSERT INTO {table} ({cols}) VALUES ({','.join('?' * len(row))}) "
             f"ON CONFLICT({key}) DO UPDATE SET {sets}")
        self.conn.execute(q, list(row.values()))
        self.conn.commit()

    def _select(self, table, where=None, order=None, desc=False, limit=None):
        q, args = f"SELECT * FROM {table}", []
        if where:
            q += " WHERE " + " AND ".join(f"{k}=?" for k in where)
            args = list(where.values())
        if order:
            q += f" ORDER BY {order} {'DESC' if desc else 'ASC'}"
        if limit:
            q += f" LIMIT {int(limit)}"
        return [dict(r) for r in self.conn.execute(q, args).fetchall()]

    def _update(self, table, key, value, changes):
        sets = ",".join(f"{k}=?" for k in changes)
        self.conn.execute(f"UPDATE {table} SET {sets} WHERE {key}=?", [*changes.values(), value])
        self.conn.commit()


class SupabaseStore(BaseStore):
    def __init__(self, url: str, key: str):
        from supabase import create_client
        self.db = create_client(url, key)

    def _insert(self, table, row, ignore_conflict=False):
        if ignore_conflict:
            pk = {"mp_candidates": "candidate_id", "mp_trades": "candidate_id",
                  "mp_shadow_trades": "candidate_id"}.get(table)
            if pk and row.get(pk) is not None:
                if self.db.table(table).select(pk).eq(pk, row[pk]).limit(1).execute().data:
                    return False
            try:
                self.db.table(table).insert(row).execute()
            except Exception as exc:  # unique violation from a concurrent run
                if "duplicate" in str(exc).lower() or "23505" in str(exc):
                    return False
                raise
            return True
        self.db.table(table).insert(row).execute()
        return True

    def _upsert(self, table, row, key):
        self.db.table(table).upsert(row, on_conflict=key).execute()

    def _select(self, table, where=None, order=None, desc=False, limit=None):
        q = self.db.table(table).select("*")
        for k, v in (where or {}).items():
            q = q.eq(k, v)
        if order:
            q = q.order(order, desc=desc)
        if limit:
            q = q.limit(limit)
        return q.execute().data or []

    def _update(self, table, key, value, changes):
        self.db.table(table).update(changes).eq(key, value).execute()


def make_store(database: str | None = None) -> BaseStore:
    """'supabase' (uses SUPABASE_URL + SUPABASE_SERVICE_KEY/SUPABASE_KEY env vars)
    or a sqlite path / sqlite:/// URL. Defaults to MARKETPILOT_DB env or local file."""
    database = database or os.environ.get("MARKETPILOT_DB", "sqlite:///marketpilot.db")
    if database == "supabase":
        key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY")
        return SupabaseStore(os.environ["SUPABASE_URL"], key)
    return SQLiteStore(database.removeprefix("sqlite:///"))

-- MarketPilot v0.7: Learning Lab shadow ledger (practice trades, never the real paper account).
-- Run once in Supabase: SQL Editor -> paste -> Run. Safe to run twice.
create table if not exists mp_shadow_trades (
  trade_id text primary key, candidate_id text unique, symbol text, asset_class text, strategy text,
  direction integer, regime text, status text, opened_at text, entry_price double precision,
  quantity double precision, stop double precision, target double precision, max_hold_bars integer,
  bars_held integer, risk_gbp double precision, risk_per_unit double precision, fx_to_gbp double precision,
  notional_gbp double precision, closed_at text, exit_price double precision, exit_reason text,
  pnl_gbp double precision, fees_gbp double precision, r_multiple double precision, last_bar_time text,
  signal_bar_time text, atr double precision, stop_atr double precision, target_atr double precision,
  cost_r double precision, note text, trail_atr double precision, exit_ma integer, exit_on text,
  best_price double precision
);
create index if not exists mp_shadow_status on mp_shadow_trades (status);
alter table mp_shadow_trades enable row level security;
grant select, insert, update, delete on mp_shadow_trades to service_role;

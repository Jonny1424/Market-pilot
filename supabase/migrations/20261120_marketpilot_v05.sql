-- MarketPilot v0.5: trailing stops and close-based exits need four new columns.
-- Run once in Supabase: SQL Editor -> paste -> Run. Safe to run twice.
alter table mp_trades add column if not exists trail_atr  double precision;
alter table mp_trades add column if not exists exit_ma    integer;
alter table mp_trades add column if not exists exit_on    text;
alter table mp_trades add column if not exists best_price double precision;
grant select, insert, update, delete on mp_trades to service_role;

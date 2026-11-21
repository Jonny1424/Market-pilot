-- MarketPilot v0.12: trade management mode and banked partial profit, on trades and practice trades.
-- Run once in Supabase: SQL Editor -> paste -> Run. Safe to run twice.
alter table mp_trades        add column if not exists mgmt text;
alter table mp_trades        add column if not exists partial_price double precision;
alter table mp_shadow_trades add column if not exists mgmt text;
alter table mp_shadow_trades add column if not exists partial_price double precision;
grant select, insert, update, delete on mp_trades, mp_shadow_trades to service_role;

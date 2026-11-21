-- MarketPilot v0.10: conviction grade (A+/A/B/C) on every trade and practice trade.
-- Run once in Supabase: SQL Editor -> paste -> Run. Safe to run twice.
alter table mp_trades        add column if not exists grade text;
alter table mp_shadow_trades add column if not exists grade text;
grant select, insert, update, delete on mp_trades, mp_shadow_trades to service_role;

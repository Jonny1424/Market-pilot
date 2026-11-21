-- MarketPilot v0.9: Time Machine odds saved on every trade and practice trade.
-- Run once in Supabase: SQL Editor -> paste -> Run. Safe to run twice.
alter table mp_trades        add column if not exists tm_p_target double precision;
alter table mp_trades        add column if not exists tm_p_stop   double precision;
alter table mp_trades        add column if not exists tm_exp_r    double precision;
alter table mp_shadow_trades add column if not exists tm_p_target double precision;
alter table mp_shadow_trades add column if not exists tm_p_stop   double precision;
alter table mp_shadow_trades add column if not exists tm_exp_r    double precision;
grant select, insert, update, delete on mp_trades, mp_shadow_trades to service_role;

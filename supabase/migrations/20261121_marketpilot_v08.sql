-- MarketPilot v0.8: confluence scores, structure stops, market overview.
-- Run once in Supabase: SQL Editor -> paste -> Run. Safe to run twice.
alter table mp_trades        add column if not exists stop_swing  integer;
alter table mp_trades        add column if not exists swing_level double precision;
alter table mp_trades        add column if not exists confluence  integer;
alter table mp_shadow_trades add column if not exists stop_swing  integer;
alter table mp_shadow_trades add column if not exists swing_level double precision;
alter table mp_shadow_trades add column if not exists confluence  integer;
create table if not exists mp_market (
  symbol text primary key, asset_class text, ts text, bar_time text, price double precision,
  chg_1d double precision, chg_5d double precision, chg_20d double precision, above_200d integer,
  rsi14 double precision, regime text, atr_pct double precision, conf_long integer, conf_short integer
);
alter table mp_market enable row level security;
grant select, insert, update, delete on mp_market, mp_trades, mp_shadow_trades to service_role;

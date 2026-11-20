-- MarketPilot v0.4 — multi-agent paper trading tables.
-- Run once in Supabase: Dashboard -> SQL Editor -> paste -> Run.
-- Leaves your existing v0.3 `trades` and `trade_journal` tables untouched.

create table if not exists mp_candidates (
  candidate_id text primary key,          -- symbol|strategy|bar_time (idempotency key)
  created_at   timestamptz default now(),
  bar_time     text,
  symbol       text,
  asset_class  text,
  strategy     text,
  direction    integer,
  regime       text,
  price        double precision,
  decision     text,                      -- trade | reject
  expected_r   double precision,
  reasons      jsonb,
  evidence     jsonb
);
create index if not exists mp_candidates_created on mp_candidates (created_at desc);

create table if not exists mp_trades (
  trade_id        text primary key,
  candidate_id    text unique references mp_candidates (candidate_id),
  symbol          text,
  asset_class     text,
  strategy        text,
  direction       integer,
  regime          text,
  status          text,                   -- pending | open | closed | cancelled
  opened_at       text,
  entry_price     double precision,
  quantity        double precision,
  stop            double precision,
  target          double precision,
  max_hold_bars   integer,
  bars_held       integer,
  risk_gbp        double precision,
  risk_per_unit   double precision,
  fx_to_gbp       double precision,
  notional_gbp    double precision,
  closed_at       text,
  exit_price      double precision,
  exit_reason     text,
  pnl_gbp         double precision,
  fees_gbp        double precision,
  r_multiple      double precision,
  last_bar_time   text,
  signal_bar_time text,
  atr             double precision,
  stop_atr        double precision,
  target_atr      double precision,
  cost_r          double precision,
  note            text
);
create index if not exists mp_trades_status on mp_trades (status);

create table if not exists mp_strategies (
  strategy_id   text primary key,         -- symbol|strategy family
  symbol        text,
  asset_class   text,
  strategy      text,
  family        text,
  params        text,
  status        text,                     -- approved | rejected | retired
  reasons       jsonb,
  oos           jsonb,
  holdout       jsonb,
  avg_cost_r    double precision,
  researched_at text,
  data_start    text,
  data_end      text
);

create table if not exists mp_equity (
  ts             text primary key,
  equity_gbp     double precision,
  realized_gbp   double precision,
  open_positions integer
);

create table if not exists mp_events (
  id      bigint generated always as identity primary key,
  ts      text,
  level   text,
  bot     text,
  message text
);
create index if not exists mp_events_ts on mp_events (ts desc);

create table if not exists mp_control (
  key        text primary key,
  value      text,
  updated_at text
);
insert into mp_control (key, value, updated_at) values ('kill_switch', 'off', now()::text)
on conflict (key) do nothing;

-- Lock the tables down: no anonymous access. The worker and dashboard use the
-- service-role key (server-side only, never in a browser or in the repo).
alter table mp_candidates enable row level security;
alter table mp_trades     enable row level security;
alter table mp_strategies enable row level security;
alter table mp_equity     enable row level security;
alter table mp_events     enable row level security;
alter table mp_control    enable row level security;

-- Newer Supabase projects don't auto-grant SQL-Editor tables to service_role.
grant usage on schema public to service_role;
grant select, insert, update, delete on mp_candidates, mp_trades, mp_strategies, mp_equity, mp_events, mp_control to service_role;
grant usage, select on all sequences in schema public to service_role;

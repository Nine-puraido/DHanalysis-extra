-- DHextra: create dhx schema with all tables, triggers, seed data
-- Mirrors the public schema from DHanalysis but for 22 extra leagues

begin;

-- ═══ Schema ═══
create schema if not exists dhx;

-- ═══ Helper functions (schema-local) ═══
create or replace function dhx.set_updated_at()
returns trigger
language plpgsql
as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

create or replace function dhx.prevent_update_delete()
returns trigger
language plpgsql
as $$
begin
    raise exception 'Table % is immutable: % is not allowed', tg_table_name, tg_op;
end;
$$;

-- ═══ 02: Reference & Match Lifecycle ═══

create table if not exists dhx.data_sources (
    id smallserial primary key,
    source_key text not null unique,
    name text not null,
    is_active boolean not null default true,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists dhx.leagues (
    id bigserial primary key,
    league_key text not null unique,
    name text not null,
    country text not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists dhx.seasons (
    id bigserial primary key,
    league_id bigint not null references dhx.leagues(id) on delete restrict,
    season_label text not null,
    start_date date not null,
    end_date date not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint uq_seasons_league_label unique (league_id, season_label),
    constraint ck_seasons_dates check (end_date >= start_date)
);

create table if not exists dhx.teams (
    id bigserial primary key,
    name text not null,
    short_name text,
    country text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists dhx.team_source_map (
    id bigserial primary key,
    team_id bigint not null references dhx.teams(id) on delete cascade,
    source_id smallint not null references dhx.data_sources(id) on delete cascade,
    source_team_id text not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint uq_team_source_map_source_team unique (source_id, source_team_id),
    constraint uq_team_source_map_team_source unique (team_id, source_id)
);

create table if not exists dhx.fixtures (
    id bigserial primary key,
    league_id bigint not null references dhx.leagues(id) on delete restrict,
    season_id bigint references dhx.seasons(id) on delete set null,
    home_team_id bigint not null references dhx.teams(id) on delete restrict,
    away_team_id bigint not null references dhx.teams(id) on delete restrict,
    kickoff_at timestamptz not null,
    status text not null,
    venue text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint ck_fixtures_status check (status in (
        'scheduled', 'live', 'finished', 'postponed', 'cancelled', 'abandoned', 'void'
    )),
    constraint ck_fixtures_teams_not_equal check (home_team_id <> away_team_id),
    constraint uq_fixtures_identity unique (league_id, kickoff_at, home_team_id, away_team_id)
);

create table if not exists dhx.fixture_source_map (
    id bigserial primary key,
    fixture_id bigint not null references dhx.fixtures(id) on delete cascade,
    source_id smallint not null references dhx.data_sources(id) on delete cascade,
    source_event_id text not null,
    source_custom_id text,
    raw_path text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint uq_fixture_source_map_source_event unique (source_id, source_event_id),
    constraint uq_fixture_source_map_fixture_source unique (fixture_id, source_id)
);

create table if not exists dhx.results (
    fixture_id bigint primary key references dhx.fixtures(id) on delete cascade,
    home_score integer not null check (home_score >= 0),
    away_score integer not null check (away_score >= 0),
    home_ht_score integer check (home_ht_score is null or home_ht_score >= 0),
    away_ht_score integer check (away_ht_score is null or away_ht_score >= 0),
    result_status text not null default 'final',
    settled_at timestamptz not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint ck_results_status check (result_status in ('final', 'awarded', 'void', 'abandoned'))
);

create index if not exists idx_fixtures_kickoff_at on dhx.fixtures (kickoff_at);
create index if not exists idx_fixtures_league_kickoff_at on dhx.fixtures (league_id, kickoff_at);
create index if not exists idx_fixtures_status_kickoff_at on dhx.fixtures (status, kickoff_at);
create index if not exists idx_fixture_source_map_fixture_source on dhx.fixture_source_map (fixture_id, source_id);

-- Triggers
create trigger trg_data_sources_updated_at before update on dhx.data_sources for each row execute function dhx.set_updated_at();
create trigger trg_leagues_updated_at before update on dhx.leagues for each row execute function dhx.set_updated_at();
create trigger trg_seasons_updated_at before update on dhx.seasons for each row execute function dhx.set_updated_at();
create trigger trg_teams_updated_at before update on dhx.teams for each row execute function dhx.set_updated_at();
create trigger trg_team_source_map_updated_at before update on dhx.team_source_map for each row execute function dhx.set_updated_at();
create trigger trg_fixtures_updated_at before update on dhx.fixtures for each row execute function dhx.set_updated_at();
create trigger trg_fixture_source_map_updated_at before update on dhx.fixture_source_map for each row execute function dhx.set_updated_at();
create trigger trg_results_updated_at before update on dhx.results for each row execute function dhx.set_updated_at();

-- ═══ 03: Odds Snapshots (partitioned) ═══

create table if not exists dhx.bookmakers (
    id bigserial primary key,
    bookmaker_key text not null unique,
    name text not null,
    is_active boolean not null default true,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists dhx.odds_snapshots (
    id bigint generated always as identity,
    fixture_id bigint not null references dhx.fixtures(id) on delete cascade,
    source_id smallint not null references dhx.data_sources(id) on delete restrict,
    bookmaker_id bigint not null references dhx.bookmakers(id) on delete restrict,
    market text not null,
    selection text not null,
    line numeric(6,3),
    price_decimal numeric(10,4) not null,
    implied_prob numeric(10,6),
    is_main boolean not null default false,
    is_suspended boolean not null default false,
    pulled_at timestamptz not null,
    created_at timestamptz not null default now(),
    constraint pk_odds_snapshots primary key (pulled_at, id),
    constraint ck_odds_market check (market in ('1x2', 'ah', 'totals', 'btts')),
    constraint ck_odds_price_decimal check (price_decimal > 1.0),
    constraint ck_odds_implied_prob check (implied_prob is null or (implied_prob > 0 and implied_prob <= 1))
) partition by range (pulled_at);

create table if not exists dhx.odds_snapshots_default partition of dhx.odds_snapshots default;

create index if not exists idx_odds_default_fbmp on dhx.odds_snapshots_default (fixture_id, bookmaker_id, market, pulled_at);
create index if not exists idx_odds_default_fmp_desc on dhx.odds_snapshots_default (fixture_id, market, pulled_at desc);
create index if not exists idx_odds_default_bmp_desc on dhx.odds_snapshots_default (bookmaker_id, market, pulled_at desc);
create index if not exists idx_odds_default_main on dhx.odds_snapshots_default (fixture_id, market, pulled_at desc) where is_main = true;

create or replace function dhx.create_odds_snapshots_partition(p_month_start date)
returns void
language plpgsql
as $$
declare
    month_start date := date_trunc('month', p_month_start)::date;
    part_name text := format('odds_snapshots_%s', to_char(month_start, 'YYYY_MM'));
    from_ts timestamptz := month_start::timestamptz;
    to_ts timestamptz := (month_start + interval '1 month')::timestamptz;
begin
    execute format(
        'create table if not exists dhx.%I partition of dhx.odds_snapshots for values from (%L) to (%L)',
        part_name, from_ts, to_ts
    );
    execute format('create index if not exists %I on dhx.%I (fixture_id, bookmaker_id, market, pulled_at)', part_name || '_fbmp_idx', part_name);
    execute format('create index if not exists %I on dhx.%I (fixture_id, market, pulled_at desc)', part_name || '_fmp_desc_idx', part_name);
    execute format('create index if not exists %I on dhx.%I (bookmaker_id, market, pulled_at desc)', part_name || '_bmp_desc_idx', part_name);
    execute format('create index if not exists %I on dhx.%I (fixture_id, market, pulled_at desc) where is_main = true', part_name || '_main_idx', part_name);
end;
$$;

create or replace function dhx.ensure_odds_partitions(months_back integer default 1, months_forward integer default 3)
returns void
language plpgsql
as $$
declare
    i integer;
    month_start date;
begin
    for i in -months_back..months_forward loop
        month_start := (date_trunc('month', now())::date + (i || ' month')::interval)::date;
        perform dhx.create_odds_snapshots_partition(month_start);
    end loop;
end;
$$;

select dhx.ensure_odds_partitions(1, 3);

create trigger trg_bookmakers_updated_at before update on dhx.bookmakers for each row execute function dhx.set_updated_at();

-- ═══ 04: Modeling & Predictions ═══

create table if not exists dhx.feature_set_versions (
    id bigserial primary key,
    training_window text not null,
    parquet_path text not null,
    row_count bigint not null check (row_count >= 0),
    feature_schema_hash text not null,
    notes text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists dhx.model_versions (
    id bigserial primary key,
    model_name text not null,
    model_type text not null,
    market text not null,
    feature_set_version_id bigint references dhx.feature_set_versions(id) on delete set null,
    artifact_path text not null,
    training_window text not null,
    metrics_json jsonb not null default '{}'::jsonb,
    params_json jsonb not null default '{}'::jsonb,
    is_active boolean not null default false,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    activated_at timestamptz,
    constraint ck_model_versions_market check (market in ('1x2', 'ah', 'totals', 'btts'))
);

create index if not exists idx_model_versions_market_active on dhx.model_versions (market, is_active);
create index if not exists idx_model_versions_created_at_desc on dhx.model_versions (created_at desc);
create unique index if not exists uq_model_versions_one_active_market on dhx.model_versions (market) where is_active = true;

create table if not exists dhx.predictions (
    id bigserial primary key,
    fixture_id bigint not null references dhx.fixtures(id) on delete cascade,
    model_version_id bigint not null references dhx.model_versions(id) on delete restrict,
    bookmaker_id bigint references dhx.bookmakers(id) on delete set null,
    market text not null,
    selection text not null,
    line numeric(6,3),
    probability numeric(10,6) not null,
    fair_odds numeric(10,4),
    predicted_at timestamptz not null,
    context_json jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    constraint ck_predictions_market check (market in ('1x2', 'ah', 'totals', 'btts')),
    constraint ck_predictions_probability check (probability > 0 and probability < 1),
    constraint ck_predictions_fair_odds check (fair_odds is null or fair_odds > 1.0)
);

create index if not exists idx_predictions_fixture_market on dhx.predictions (fixture_id, market, predicted_at desc);
create index if not exists idx_predictions_model on dhx.predictions (model_version_id, predicted_at desc);

create trigger trg_predictions_immutable before update or delete on dhx.predictions for each row execute function dhx.prevent_update_delete();
create trigger trg_feature_set_versions_updated_at before update on dhx.feature_set_versions for each row execute function dhx.set_updated_at();
create trigger trg_model_versions_updated_at before update on dhx.model_versions for each row execute function dhx.set_updated_at();

-- ═══ 05: Betting & Evaluation ═══

create table if not exists dhx.bet_recommendations (
    id bigserial primary key,
    prediction_id bigint not null references dhx.predictions(id) on delete cascade,
    recommended_stake numeric(10,2) not null check (recommended_stake >= 0),
    edge numeric(10,6) not null,
    reason_json jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create table if not exists dhx.bet_log (
    id bigserial primary key,
    fixture_id bigint not null references dhx.fixtures(id) on delete restrict,
    prediction_id bigint references dhx.predictions(id) on delete set null,
    bookmaker_id bigint references dhx.bookmakers(id) on delete set null,
    market text not null,
    selection text not null,
    line numeric(6,3),
    stake numeric(10,2) not null check (stake > 0),
    odds_at_prediction numeric(10,4) not null check (odds_at_prediction > 1.0),
    probability_at_bet numeric(10,6),
    expected_value numeric(12,6),
    predicted_at timestamptz,
    placed_at timestamptz not null,
    status text not null default 'placed',
    result text,
    pnl numeric(12,2),
    settled_at timestamptz,
    notes text,
    created_at timestamptz not null default now(),
    constraint ck_bet_log_market check (market in ('1x2', 'ah', 'totals', 'btts')),
    constraint ck_bet_log_probability check (probability_at_bet is null or (probability_at_bet > 0 and probability_at_bet < 1)),
    constraint ck_bet_log_status check (status in ('placed', 'void', 'settled')),
    constraint ck_bet_log_result check (result is null or result in ('win', 'loss', 'push', 'half_win', 'half_loss', 'void')),
    constraint ck_bet_log_lifecycle check (
        (status = 'placed' and settled_at is null and result is null and pnl is null)
        or (status = 'void' and settled_at is not null and result = 'void' and pnl is not null)
        or (status = 'settled' and settled_at is not null and result in ('win', 'loss', 'push', 'half_win', 'half_loss') and pnl is not null)
    )
);

create index if not exists idx_bet_log_placed_at_desc on dhx.bet_log (placed_at desc);
create index if not exists idx_bet_log_fixture_market on dhx.bet_log (fixture_id, market);
create index if not exists idx_bet_log_status_placed_at on dhx.bet_log (status, placed_at desc);

create table if not exists dhx.clv_tracking (
    bet_id bigint primary key references dhx.bet_log(id) on delete cascade,
    predicted_at timestamptz not null,
    odds_at_prediction numeric(10,4) not null check (odds_at_prediction > 1.0),
    closing_odds numeric(10,4) not null check (closing_odds > 1.0),
    clv_value numeric(10,6) not null,
    clv_percent numeric(10,4) not null,
    settled_at timestamptz not null,
    created_at timestamptz not null default now()
);

create index if not exists idx_clv_tracking_settled_at_desc on dhx.clv_tracking (settled_at desc);

create table if not exists dhx.bankroll_daily (
    date date primary key,
    opening_bankroll numeric(12,2) not null,
    closing_bankroll numeric(12,2) not null,
    daily_pnl numeric(12,2) not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists dhx.model_metrics_daily (
    id bigserial primary key,
    model_version_id bigint not null references dhx.model_versions(id) on delete cascade,
    market text not null,
    metric_date date not null,
    brier numeric(10,6),
    logloss numeric(10,6),
    calibration_error numeric(10,6),
    clv_avg numeric(10,6),
    roi numeric(10,6),
    bet_count integer not null default 0,
    stake_sum numeric(12,2) not null default 0,
    max_drawdown numeric(12,2),
    sample_size integer not null default 0,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint ck_model_metrics_market check (market in ('1x2', 'ah', 'totals', 'btts')),
    constraint uq_model_metrics_daily unique (model_version_id, market, metric_date)
);

create table if not exists dhx.edge_summary (
    id bigserial primary key,
    model_version_id bigint references dhx.model_versions(id) on delete cascade,
    market text not null,
    bucket_label text not null,
    edge_min numeric(10,6) not null,
    edge_max numeric(10,6) not null,
    sample_size integer not null default 0,
    hit_rate numeric(10,6),
    roi numeric(10,6),
    period_start date not null,
    period_end date not null,
    calculated_at timestamptz not null default now(),
    constraint ck_edge_summary_market check (market in ('1x2', 'ah', 'totals', 'btts')),
    constraint ck_edge_summary_bounds check (edge_max > edge_min),
    constraint ck_edge_summary_period check (period_end >= period_start),
    constraint uq_edge_summary_window unique (model_version_id, market, bucket_label, period_start, period_end)
);

create index if not exists idx_edge_summary_market_period on dhx.edge_summary (market, period_end desc);

create trigger trg_bankroll_daily_updated_at before update on dhx.bankroll_daily for each row execute function dhx.set_updated_at();
create trigger trg_model_metrics_daily_updated_at before update on dhx.model_metrics_daily for each row execute function dhx.set_updated_at();

-- ═══ 06: Operations & Views ═══

create table if not exists dhx.pipeline_runs (
    id bigserial primary key,
    pipeline_name text not null,
    started_at timestamptz not null,
    ended_at timestamptz,
    status text not null,
    rows_written bigint,
    details_json jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    constraint ck_pipeline_runs_status check (status in ('running', 'success', 'failed'))
);

create index if not exists idx_pipeline_runs_name_started_desc on dhx.pipeline_runs (pipeline_name, started_at desc);

create table if not exists dhx.data_quality_checks (
    id bigserial primary key,
    check_name text not null,
    check_scope text,
    run_id bigint references dhx.pipeline_runs(id) on delete set null,
    status text not null,
    details_json jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    constraint ck_data_quality_checks_status check (status in ('pass', 'warn', 'fail'))
);

create index if not exists idx_data_quality_checks_created_desc on dhx.data_quality_checks (created_at desc);

create table if not exists dhx.ingestion_errors (
    id bigserial primary key,
    source_id smallint references dhx.data_sources(id) on delete set null,
    run_id bigint references dhx.pipeline_runs(id) on delete set null,
    error_type text not null,
    error_message text not null,
    payload_ref text,
    created_at timestamptz not null default now()
);

create index if not exists idx_ingestion_errors_created_desc on dhx.ingestion_errors (created_at desc);

-- Views
create or replace view dhx.vw_odds_opening as
select distinct on (o.fixture_id, o.bookmaker_id, o.market, o.selection, coalesce(o.line, -9999.000))
    o.fixture_id, o.bookmaker_id, o.market, o.selection, o.line,
    o.price_decimal, o.implied_prob, o.is_main, o.is_suspended, o.pulled_at
from dhx.odds_snapshots o
order by o.fixture_id, o.bookmaker_id, o.market, o.selection, coalesce(o.line, -9999.000), o.pulled_at asc;

create or replace view dhx.vw_odds_latest_pre_kickoff as
select distinct on (o.fixture_id, o.bookmaker_id, o.market, o.selection, coalesce(o.line, -9999.000))
    o.fixture_id, o.bookmaker_id, o.market, o.selection, o.line,
    o.price_decimal, o.implied_prob, o.is_main, o.is_suspended, o.pulled_at
from dhx.odds_snapshots o
join dhx.fixtures f on f.id = o.fixture_id
where o.pulled_at <= f.kickoff_at
order by o.fixture_id, o.bookmaker_id, o.market, o.selection, coalesce(o.line, -9999.000), o.pulled_at desc;

create or replace view dhx.vw_latest_predictions as
select distinct on (p.fixture_id, p.model_version_id, p.market, p.selection, coalesce(p.line, -9999.000), coalesce(p.bookmaker_id, -1))
    p.id, p.fixture_id, p.model_version_id, p.bookmaker_id,
    p.market, p.selection, p.line, p.probability, p.fair_odds,
    p.predicted_at, p.context_json
from dhx.predictions p
order by p.fixture_id, p.model_version_id, p.market, p.selection, coalesce(p.line, -9999.000), coalesce(p.bookmaker_id, -1), p.predicted_at desc;

create or replace view dhx.vw_roi_by_market_30d as
select
    b.market, count(*) as settled_bets, sum(b.stake) as total_stake,
    sum(b.pnl) as total_pnl,
    case when sum(b.stake) = 0 then null else round((sum(b.pnl) / sum(b.stake)) * 100.0, 4) end as roi_percent
from dhx.bet_log b
where b.status = 'settled' and b.settled_at >= now() - interval '30 days'
group by b.market;

-- ═══ 08: Match Statistics ═══

create table if not exists dhx.match_statistics (
    fixture_id bigint primary key references dhx.fixtures(id) on delete cascade,
    home_xg numeric(6,2), away_xg numeric(6,2),
    home_shots integer, away_shots integer,
    home_shots_on_target integer, away_shots_on_target integer,
    home_shots_off_target integer, away_shots_off_target integer,
    home_blocked_shots integer, away_blocked_shots integer,
    home_possession numeric(5,2), away_possession numeric(5,2),
    home_corner_kicks integer, away_corner_kicks integer,
    home_offsides integer, away_offsides integer,
    home_fouls integer, away_fouls integer,
    home_yellow_cards integer, away_yellow_cards integer,
    home_red_cards integer, away_red_cards integer,
    home_passes integer, away_passes integer,
    home_accurate_passes integer, away_accurate_passes integer,
    home_saves integer, away_saves integer,
    home_big_chances integer, away_big_chances integer,
    home_big_chances_missed integer, away_big_chances_missed integer,
    extra_stats jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists dhx.player_match_ratings (
    id bigserial primary key,
    fixture_id bigint not null references dhx.fixtures(id) on delete cascade,
    team_id bigint not null references dhx.teams(id) on delete cascade,
    source_player_id text,
    player_name text not null,
    player_short_name text,
    position text,
    jersey_number integer,
    is_substitute boolean not null default false,
    minutes_played integer,
    rating numeric(4,2),
    goals integer default 0,
    assists integer default 0,
    shots_total integer, shots_on_target integer,
    passes_total integer, passes_accurate integer,
    key_passes integer, tackles integer, interceptions integer,
    extra_stats jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint uq_player_match unique (fixture_id, team_id, source_player_id)
);

create index if not exists idx_match_statistics_fixture on dhx.match_statistics (fixture_id);
create index if not exists idx_player_match_ratings_fixture on dhx.player_match_ratings (fixture_id);
create index if not exists idx_player_match_ratings_team on dhx.player_match_ratings (team_id);
create index if not exists idx_player_match_ratings_fixture_team on dhx.player_match_ratings (fixture_id, team_id);
create index if not exists idx_player_match_ratings_player on dhx.player_match_ratings (source_player_id);

create trigger trg_match_statistics_updated_at before update on dhx.match_statistics for each row execute function dhx.set_updated_at();
create trigger trg_player_match_ratings_updated_at before update on dhx.player_match_ratings for each row execute function dhx.set_updated_at();

-- ═══ 07: Seed Reference Data ═══

insert into dhx.data_sources (source_key, name, is_active)
values
    ('sofascore', 'SofaScore Public API', true),
    ('odds_api', 'The Odds API', true),
    ('football_data', 'Football-Data.org', true),
    ('manual', 'Manual Input', true)
on conflict (source_key) do update set name = excluded.name, is_active = excluded.is_active, updated_at = now();

-- 22 Extra Leagues
insert into dhx.leagues (league_key, name, country)
values
    -- Europe (14)
    ('POL', 'Ekstraklasa', 'Poland'),
    ('GSL', 'Super League', 'Greece'),
    ('CFL', 'Czech First League', 'Czech Republic'),
    ('DSL', 'Superliga', 'Denmark'),
    ('RSL', 'SuperLiga', 'Romania'),
    ('UPL', 'Premier League', 'Ukraine'),
    ('HNL', 'HNL', 'Croatia'),
    ('SRBL', 'Superliga', 'Serbia'),
    ('IPL', 'Premier League', 'Israel'),
    ('SB', 'Serie B', 'Italy'),
    ('LL2', 'LaLiga 2', 'Spain'),
    ('L2', 'Ligue 2', 'France'),
    ('EL1', 'League One', 'England'),
    ('GL3', '3. Liga', 'Germany'),
    -- Outside Europe (8)
    ('SPL_SA', 'Saudi Pro League', 'Saudi Arabia'),
    ('MLS', 'MLS', 'USA'),
    ('J1', 'J1 League', 'Japan'),
    ('ARG', 'Liga Profesional', 'Argentina'),
    ('ALM', 'A-League Men', 'Australia'),
    ('TL1', 'Thai League 1', 'Thailand'),
    ('QSL', 'Stars League', 'Qatar'),
    ('UAE', 'Pro League', 'UAE')
on conflict (league_key) do update set name = excluded.name, country = excluded.country, updated_at = now();

insert into dhx.bookmakers (bookmaker_key, name, is_active)
values
    ('pinnacle', 'Pinnacle', true),
    ('bet365', 'Bet365', true),
    ('1xbet', '1xBet', true),
    ('williamhill', 'William Hill', true),
    ('betfair', 'Betfair', true),
    ('sofascore_avg', 'SofaScore Average', true)
on conflict (bookmaker_key) do update set name = excluded.name, is_active = excluded.is_active, updated_at = now();

-- ═══ RLS Policies for Dashboard (anon read) ═══

alter table dhx.fixtures enable row level security;
alter table dhx.results enable row level security;
alter table dhx.odds_snapshots enable row level security;
alter table dhx.leagues enable row level security;
alter table dhx.teams enable row level security;
alter table dhx.bookmakers enable row level security;
alter table dhx.predictions enable row level security;
alter table dhx.model_versions enable row level security;
alter table dhx.feature_set_versions enable row level security;
alter table dhx.match_statistics enable row level security;
alter table dhx.bet_log enable row level security;
alter table dhx.fixture_source_map enable row level security;
alter table dhx.data_sources enable row level security;
alter table dhx.player_match_ratings enable row level security;

create policy anon_read_fixtures on dhx.fixtures for select to anon using (true);
create policy anon_read_results on dhx.results for select to anon using (true);
create policy anon_read_odds_snapshots on dhx.odds_snapshots for select to anon using (true);
create policy anon_read_leagues on dhx.leagues for select to anon using (true);
create policy anon_read_teams on dhx.teams for select to anon using (true);
create policy anon_read_bookmakers on dhx.bookmakers for select to anon using (true);
create policy anon_read_predictions on dhx.predictions for select to anon using (true);
create policy anon_read_model_versions on dhx.model_versions for select to anon using (true);
create policy anon_read_feature_set_versions on dhx.feature_set_versions for select to anon using (true);
create policy anon_read_match_statistics on dhx.match_statistics for select to anon using (true);
create policy anon_read_bet_log on dhx.bet_log for select to anon using (true);
create policy anon_read_fixture_source_map on dhx.fixture_source_map for select to anon using (true);
create policy anon_read_data_sources on dhx.data_sources for select to anon using (true);
create policy anon_read_player_match_ratings on dhx.player_match_ratings for select to anon using (true);

commit;

"""Load raw data from Supabase into Polars DataFrames for feature engineering."""

from __future__ import annotations

import logging
from typing import Any

import polars as pl

from dhx.db import get_client

logger = logging.getLogger(__name__)

PAGE_SIZE = 1000


def _paginated_select(
    table: str,
    columns: str = "*",
    filters: list[tuple[str, str, Any]] | None = None,
    order_col: str = "id",
) -> list[dict]:
    """Fetch all rows from a table with pagination (Supabase caps at 1000)."""
    client = get_client()
    all_rows: list[dict] = []
    offset = 0

    while True:
        q = client.schema("dhx").table(table).select(columns).order(order_col).range(offset, offset + PAGE_SIZE - 1)
        for col, op, val in filters or []:
            if op == "eq":
                q = q.eq(col, val)
            elif op == "is_":
                q = q.is_(col, val)
        rows = q.execute().data
        all_rows.extend(rows)
        if len(rows) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    return all_rows


def load_feature_data() -> dict[str, pl.DataFrame]:
    """Load all tables needed for feature engineering.

    Returns dict with keys: fixtures, results, stats, odds, teams, player_ratings.
    Loads ALL finished fixtures so rolling windows have full history.
    """
    logger.info("Loading data from Supabase...")

    # 1. Fixtures with league info — only finished ones
    logger.info("  Loading fixtures...")
    fixtures_raw = _paginated_select(
        "fixtures",
        "id, league_id, home_team_id, away_team_id, kickoff_at, status",
        filters=[("status", "eq", "finished")],
        order_col="kickoff_at",
    )
    # Also fetch league keys for labeling
    leagues_raw = get_client().schema("dhx").table("leagues").select("id, league_key").execute().data
    league_map = {r["id"]: r["league_key"] for r in leagues_raw}

    fixtures = pl.DataFrame(fixtures_raw).with_columns(
        pl.col("id").cast(pl.Int64).alias("fixture_id"),
        pl.col("league_id").cast(pl.Int64),
        pl.col("home_team_id").cast(pl.Int64),
        pl.col("away_team_id").cast(pl.Int64),
        pl.col("kickoff_at").str.to_datetime(time_zone="UTC"),
        pl.col("league_id").cast(pl.Int64).replace_strict(league_map, default=None).alias("league_key"),
    ).drop("id")
    logger.info(f"  Fixtures: {len(fixtures)} rows")

    # 2. Results
    logger.info("  Loading results...")
    results_raw = _paginated_select(
        "results",
        "fixture_id, home_score, away_score, home_ht_score, away_ht_score",
        order_col="fixture_id",
    )
    results = pl.DataFrame(results_raw).with_columns(
        pl.col("fixture_id").cast(pl.Int64),
        pl.col("home_score").cast(pl.Int64),
        pl.col("away_score").cast(pl.Int64),
    )
    logger.info(f"  Results: {len(results)} rows")

    # 3. Match statistics
    logger.info("  Loading match statistics...")
    stats_cols = (
        "fixture_id, home_xg, away_xg, home_shots, away_shots, "
        "home_shots_on_target, away_shots_on_target, "
        "home_possession, away_possession, "
        "home_corner_kicks, away_corner_kicks, "
        "home_fouls, away_fouls, "
        "home_yellow_cards, away_yellow_cards, "
        "home_red_cards, away_red_cards, "
        "home_passes, away_passes, "
        "home_accurate_passes, away_accurate_passes, "
        "home_big_chances, away_big_chances, "
        "home_big_chances_missed, away_big_chances_missed, "
        "home_saves, away_saves, "
        "home_offsides, away_offsides, "
        "home_blocked_shots, away_blocked_shots, "
        "home_shots_off_target, away_shots_off_target"
    )
    stats_raw = _paginated_select("match_statistics", stats_cols, order_col="fixture_id")
    stats = pl.DataFrame(stats_raw).with_columns(
        pl.col("fixture_id").cast(pl.Int64),
    )
    # Cast numeric columns to Float64
    numeric_stat_cols = [c for c in stats.columns if c != "fixture_id"]
    stats = stats.with_columns(
        [pl.col(c).cast(pl.Float64, strict=False) for c in numeric_stat_cols]
    )
    logger.info(f"  Match statistics: {len(stats)} rows")

    # 4. Odds snapshots (is_main only — much smaller)
    logger.info("  Loading odds snapshots (is_main only)...")
    odds_raw = _paginated_select(
        "odds_snapshots",
        "fixture_id, market, selection, line, price_decimal, implied_prob, pulled_at",
        filters=[("is_main", "eq", True)],
        order_col="pulled_at",
    )
    odds = pl.DataFrame(odds_raw).with_columns(
        pl.col("fixture_id").cast(pl.Int64),
        pl.col("line").cast(pl.Float64, strict=False),
        pl.col("price_decimal").cast(pl.Float64, strict=False),
        pl.col("implied_prob").cast(pl.Float64, strict=False),
        pl.col("pulled_at").str.to_datetime(time_zone="UTC"),
    )
    logger.info(f"  Odds snapshots: {len(odds)} rows")

    # 5. Teams
    logger.info("  Loading teams...")
    teams_raw = get_client().schema("dhx").table("teams").select("id, name, short_name").execute().data
    teams = pl.DataFrame(teams_raw).with_columns(
        pl.col("id").cast(pl.Int64).alias("team_id"),
    ).drop("id")
    logger.info(f"  Teams: {len(teams)} rows")

    # 6. Player match ratings — aggregated per team/fixture
    logger.info("  Loading player ratings (aggregated)...")
    ratings_raw = _paginated_select(
        "player_match_ratings",
        "fixture_id, team_id, rating, minutes_played",
        order_col="fixture_id",
    )
    if ratings_raw:
        ratings_df = pl.DataFrame(ratings_raw).with_columns(
            pl.col("fixture_id").cast(pl.Int64),
            pl.col("team_id").cast(pl.Int64),
            pl.col("rating").cast(pl.Float64, strict=False),
            pl.col("minutes_played").cast(pl.Float64, strict=False),
        )
        # Aggregate: minutes-weighted mean rating per team/fixture
        player_ratings = ratings_df.group_by(["fixture_id", "team_id"]).agg(
            (
                (pl.col("rating") * pl.col("minutes_played")).sum()
                / pl.col("minutes_played").sum()
            ).alias("avg_rating"),
            pl.col("rating").count().alias("rated_players"),
        )
    else:
        player_ratings = pl.DataFrame(
            schema={"fixture_id": pl.Int64, "team_id": pl.Int64,
                     "avg_rating": pl.Float64, "rated_players": pl.UInt32}
        )
    logger.info(f"  Player ratings (aggregated): {len(player_ratings)} rows")

    logger.info("Data loading complete.")
    return {
        "fixtures": fixtures,
        "results": results,
        "stats": stats,
        "odds": odds,
        "teams": teams,
        "player_ratings": player_ratings,
    }

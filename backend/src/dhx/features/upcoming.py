"""Compute features for scheduled (upcoming) fixtures.

Reuses the existing feature pipeline components:
- load_feature_data() for historical data
- unpivot_to_team_perspective() for team-level rows
- compute_all_rolling_features() for rolling windows
- compute_market_features() for odds-derived features

The key trick: append scheduled fixtures (with NULL match columns) to the
historical team-perspective DataFrame, so rolling window functions naturally
carry forward the latest available history via shift(1).
"""

from __future__ import annotations

import logging

import polars as pl

from dhx.db import get_client
from dhx.features.assemble import _DELTA_FEATURES, _EXCLUDE_COLS, _META_COLS
from dhx.features.loader import load_feature_data
from dhx.features.market import compute_market_features
from dhx.features.rolling import compute_all_rolling_features
from dhx.features.unpivot import unpivot_to_team_perspective

logger = logging.getLogger(__name__)


def _load_scheduled_fixtures(fixture_ids: list[int]) -> pl.DataFrame:
    """Load scheduled fixtures from Supabase by ID."""
    client = get_client()
    data = (
        client.schema("dhx").table("fixtures")
        .select("id, league_id, home_team_id, away_team_id, kickoff_at, status")
        .in_("id", fixture_ids)
        .execute()
        .data
    )
    if not data:
        return pl.DataFrame(
            schema={
                "fixture_id": pl.Int64,
                "league_id": pl.Int64,
                "home_team_id": pl.Int64,
                "away_team_id": pl.Int64,
                "kickoff_at": pl.Datetime("us", "UTC"),
                "status": pl.Utf8,
            }
        )

    leagues_raw = client.schema("dhx").table("leagues").select("id, league_key").execute().data
    league_map = {r["id"]: r["league_key"] for r in leagues_raw}

    df = pl.DataFrame(data).with_columns(
        pl.col("id").cast(pl.Int64).alias("fixture_id"),
        pl.col("league_id").cast(pl.Int64),
        pl.col("home_team_id").cast(pl.Int64),
        pl.col("away_team_id").cast(pl.Int64),
        pl.col("kickoff_at").str.to_datetime(time_zone="UTC"),
        pl.col("league_id")
        .cast(pl.Int64)
        .replace_strict(league_map, default=None)
        .alias("league_key"),
    ).drop("id")

    return df


def _build_scheduled_team_rows(scheduled: pl.DataFrame, team_df: pl.DataFrame) -> pl.DataFrame:
    """Create team-perspective rows for scheduled fixtures with NULL match columns.

    Produces 2 rows per fixture (home + away) with the same schema as unpivot output,
    but all match-outcome columns set to NULL.
    """
    # Identify all columns that are in team_df but not in the base fixture columns
    null_cols = [
        c for c in team_df.columns
        if c not in ("fixture_id", "team_id", "opponent_id", "league_id", "kickoff_at", "is_home")
    ]

    # Home perspective
    home = scheduled.select(
        pl.col("fixture_id"),
        pl.col("home_team_id").alias("team_id"),
        pl.col("away_team_id").alias("opponent_id"),
        pl.col("league_id"),
        pl.col("kickoff_at"),
        pl.lit(True).alias("is_home"),
    )

    # Away perspective
    away = scheduled.select(
        pl.col("fixture_id"),
        pl.col("away_team_id").alias("team_id"),
        pl.col("home_team_id").alias("opponent_id"),
        pl.col("league_id"),
        pl.col("kickoff_at"),
        pl.lit(False).alias("is_home"),
    )

    rows = pl.concat([home, away])

    # Add NULL columns for all match-outcome fields
    for col_name in null_cols:
        dtype = team_df[col_name].dtype
        rows = rows.with_columns(pl.lit(None).cast(dtype).alias(col_name))

    return rows


def _load_scheduled_odds(fixture_ids: list[int], all_odds: pl.DataFrame) -> pl.DataFrame:
    """Get odds for scheduled fixtures, either from the already-loaded data or DB."""
    # Check if the loaded odds already contain scheduled fixture odds
    sched_odds = all_odds.filter(pl.col("fixture_id").is_in(fixture_ids))
    if len(sched_odds) > 0:
        return sched_odds

    # If not, fetch them separately
    from dhx.features.loader import _paginated_select

    odds_raw = _paginated_select(
        "odds_snapshots",
        "fixture_id, market, selection, line, price_decimal, implied_prob, pulled_at",
        filters=[("is_main", "eq", True)],
        order_col="pulled_at",
    )
    if not odds_raw:
        return pl.DataFrame(schema=all_odds.schema)

    odds_df = pl.DataFrame(odds_raw).with_columns(
        pl.col("fixture_id").cast(pl.Int64),
        pl.col("line").cast(pl.Float64, strict=False),
        pl.col("price_decimal").cast(pl.Float64, strict=False),
        pl.col("implied_prob").cast(pl.Float64, strict=False),
        pl.col("pulled_at").str.to_datetime(time_zone="UTC"),
    )
    return odds_df.filter(pl.col("fixture_id").is_in(fixture_ids))


def compute_features_for_upcoming(fixture_ids: list[int]) -> pl.DataFrame:
    """Compute the feature matrix for scheduled (upcoming) fixtures.

    Returns a fixture-level DataFrame with the same feature columns as the
    regular pipeline (minus labels), ready for model.predict_lambdas().
    """
    logger.info(f"Computing features for {len(fixture_ids)} upcoming fixtures")

    # 1. Load all historical finished data
    logger.info("  Loading historical data...")
    data = load_feature_data()

    # 2. Load the scheduled fixtures
    logger.info("  Loading scheduled fixtures...")
    scheduled = _load_scheduled_fixtures(fixture_ids)
    found_ids = scheduled["fixture_id"].to_list()
    missing = set(fixture_ids) - set(found_ids)
    if missing:
        logger.warning(f"  Fixtures not found in DB: {missing}")
    if len(scheduled) == 0:
        raise ValueError("No scheduled fixtures found for the given IDs")
    logger.info(f"  Found {len(scheduled)} scheduled fixtures")

    # 3. Unpivot finished fixtures to team perspective
    logger.info("  Unpivoting historical data...")
    team_df = unpivot_to_team_perspective(
        data["fixtures"], data["results"], data["stats"],
        player_ratings=data["player_ratings"],
    )

    # 4. Create team-perspective rows for scheduled fixtures (NULL outcomes)
    logger.info("  Creating scheduled team rows...")
    scheduled_team_rows = _build_scheduled_team_rows(scheduled, team_df)

    # 5. Concatenate and compute rolling features on the combined set
    logger.info("  Computing rolling features on combined data...")
    combined = pl.concat([team_df, scheduled_team_rows], how="diagonal_relaxed")
    combined = combined.sort(["team_id", "kickoff_at"])
    team_features = compute_all_rolling_features(combined)

    # 6. Compute market features (odds may exist for scheduled fixtures)
    logger.info("  Computing market features...")
    sched_odds = _load_scheduled_odds(fixture_ids, data["odds"])
    # Combine historical + scheduled odds for the market feature computation
    combined_odds = pl.concat([data["odds"], sched_odds], how="diagonal_relaxed").unique()
    market_features = compute_market_features(combined_odds)

    # 7. Assemble fixture-level matrix for scheduled fixtures only
    logger.info("  Assembling feature matrix...")
    feature_cols = [
        c for c in team_features.columns
        if c not in _META_COLS and c not in _EXCLUDE_COLS
    ]

    # Home features
    home_features = (
        scheduled.select("fixture_id", "home_team_id")
        .join(
            team_features.select(["fixture_id", "team_id"] + feature_cols),
            left_on=["fixture_id", "home_team_id"],
            right_on=["fixture_id", "team_id"],
            how="left",
        )
        .drop("home_team_id")
        .rename({c: f"home_{c}" for c in feature_cols})
    )

    # Away features
    away_features = (
        scheduled.select("fixture_id", "away_team_id")
        .join(
            team_features.select(["fixture_id", "team_id"] + feature_cols),
            left_on=["fixture_id", "away_team_id"],
            right_on=["fixture_id", "team_id"],
            how="left",
        )
        .drop("away_team_id")
        .rename({c: f"away_{c}" for c in feature_cols})
    )

    # Base fixture info
    base = scheduled.select(
        "fixture_id", "home_team_id", "away_team_id", "kickoff_at",
        "league_id", "league_key",
    )

    # Join all parts
    matrix = base
    matrix = matrix.join(home_features, on="fixture_id", how="left")
    matrix = matrix.join(away_features, on="fixture_id", how="left")
    matrix = matrix.join(market_features, on="fixture_id", how="left")

    # Delta features
    delta_exprs: list[pl.Expr] = []
    for feat in _DELTA_FEATURES:
        home_col = f"home_{feat}"
        away_col = f"away_{feat}"
        if home_col in matrix.columns and away_col in matrix.columns:
            delta_exprs.append(
                (pl.col(home_col) - pl.col(away_col)).alias(f"delta_{feat}")
            )
    if delta_exprs:
        matrix = matrix.with_columns(delta_exprs)

    matrix = matrix.sort("kickoff_at")

    logger.info(f"  Feature matrix: {len(matrix)} rows x {len(matrix.columns)} cols")
    return matrix

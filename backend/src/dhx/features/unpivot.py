"""Transform fixture-level data into team-perspective rows (2 rows per fixture)."""

from __future__ import annotations

import polars as pl


def unpivot_to_team_perspective(
    fixtures: pl.DataFrame,
    results: pl.DataFrame,
    stats: pl.DataFrame,
    player_ratings: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Create 2 rows per fixture: one from home team's perspective, one from away's.

    Returns a DataFrame sorted by (team_id, kickoff_at) with columns for
    goals, xG, shots, possession, etc. from the team's perspective.
    """
    # Join fixtures with results and stats
    base = fixtures.join(results, on="fixture_id", how="inner")
    base = base.join(stats, on="fixture_id", how="left")

    # --- Home perspective ---
    home = base.select(
        pl.col("fixture_id"),
        pl.col("home_team_id").alias("team_id"),
        pl.col("away_team_id").alias("opponent_id"),
        pl.col("league_id"),
        pl.col("kickoff_at"),
        pl.lit(True).alias("is_home"),
        # Scores
        pl.col("home_score").cast(pl.Float64).alias("goals_for"),
        pl.col("away_score").cast(pl.Float64).alias("goals_against"),
        # xG
        pl.col("home_xg").alias("xg_for"),
        pl.col("away_xg").alias("xg_against"),
        # Shots
        pl.col("home_shots").alias("shots_for"),
        pl.col("away_shots").alias("shots_against"),
        pl.col("home_shots_on_target").alias("shots_on_target_for"),
        pl.col("away_shots_on_target").alias("shots_on_target_against"),
        # Possession
        pl.col("home_possession").alias("possession"),
        # Big chances
        pl.col("home_big_chances").alias("big_chances_for"),
        pl.col("home_big_chances_missed").alias("big_chances_missed"),
        # Corners
        pl.col("home_corner_kicks").alias("corner_kicks_for"),
        pl.col("away_corner_kicks").alias("corner_kicks_against"),
        # Passing
        pl.col("home_passes").alias("passes"),
        pl.col("home_accurate_passes").alias("accurate_passes"),
        # Discipline
        pl.col("home_fouls").alias("fouls_committed"),
        pl.col("away_fouls").alias("fouls_suffered"),
        pl.col("home_yellow_cards").alias("yellow_cards"),
        pl.col("home_red_cards").alias("red_cards"),
        # New: saves, offsides, blocked shots, shots off target
        pl.col("home_saves").alias("saves"),
        pl.col("home_offsides").alias("offsides"),
        pl.col("home_blocked_shots").alias("blocked_shots_for"),
        pl.col("away_blocked_shots").alias("blocked_shots_against"),
        pl.col("home_shots_off_target").alias("shots_off_target_for"),
    )

    # --- Away perspective ---
    away = base.select(
        pl.col("fixture_id"),
        pl.col("away_team_id").alias("team_id"),
        pl.col("home_team_id").alias("opponent_id"),
        pl.col("league_id"),
        pl.col("kickoff_at"),
        pl.lit(False).alias("is_home"),
        # Scores (swapped)
        pl.col("away_score").cast(pl.Float64).alias("goals_for"),
        pl.col("home_score").cast(pl.Float64).alias("goals_against"),
        # xG (swapped)
        pl.col("away_xg").alias("xg_for"),
        pl.col("home_xg").alias("xg_against"),
        # Shots (swapped)
        pl.col("away_shots").alias("shots_for"),
        pl.col("home_shots").alias("shots_against"),
        pl.col("away_shots_on_target").alias("shots_on_target_for"),
        pl.col("home_shots_on_target").alias("shots_on_target_against"),
        # Possession
        pl.col("away_possession").alias("possession"),
        # Big chances
        pl.col("away_big_chances").alias("big_chances_for"),
        pl.col("away_big_chances_missed").alias("big_chances_missed"),
        # Corners (swapped)
        pl.col("away_corner_kicks").alias("corner_kicks_for"),
        pl.col("home_corner_kicks").alias("corner_kicks_against"),
        # Passing
        pl.col("away_passes").alias("passes"),
        pl.col("away_accurate_passes").alias("accurate_passes"),
        # Discipline (swapped)
        pl.col("away_fouls").alias("fouls_committed"),
        pl.col("home_fouls").alias("fouls_suffered"),
        pl.col("away_yellow_cards").alias("yellow_cards"),
        pl.col("away_red_cards").alias("red_cards"),
        # New: saves, offsides, blocked shots, shots off target (swapped)
        pl.col("away_saves").alias("saves"),
        pl.col("away_offsides").alias("offsides"),
        pl.col("away_blocked_shots").alias("blocked_shots_for"),
        pl.col("home_blocked_shots").alias("blocked_shots_against"),
        pl.col("away_shots_off_target").alias("shots_off_target_for"),
    )

    # Stack
    team_df = pl.concat([home, away])

    # Join player ratings if provided
    if player_ratings is not None:
        team_df = team_df.join(player_ratings, on=["fixture_id", "team_id"], how="left")

    # Derive additional columns
    team_df = team_df.with_columns(
        # Pass accuracy
        (pl.col("accurate_passes") / pl.col("passes") * 100.0).alias("pass_accuracy"),
        # Goal difference
        (pl.col("goals_for") - pl.col("goals_against")).alias("goal_diff"),
        # Match outcome
        pl.when(pl.col("goals_for") > pl.col("goals_against"))
        .then(pl.lit(1, dtype=pl.Int8))
        .otherwise(pl.lit(0, dtype=pl.Int8))
        .alias("win"),
        pl.when(pl.col("goals_for") == pl.col("goals_against"))
        .then(pl.lit(1, dtype=pl.Int8))
        .otherwise(pl.lit(0, dtype=pl.Int8))
        .alias("draw"),
        pl.when(pl.col("goals_for") < pl.col("goals_against"))
        .then(pl.lit(1, dtype=pl.Int8))
        .otherwise(pl.lit(0, dtype=pl.Int8))
        .alias("loss"),
        # Points
        pl.when(pl.col("goals_for") > pl.col("goals_against"))
        .then(pl.lit(3, dtype=pl.Int8))
        .when(pl.col("goals_for") == pl.col("goals_against"))
        .then(pl.lit(1, dtype=pl.Int8))
        .otherwise(pl.lit(0, dtype=pl.Int8))
        .alias("points"),
        # New derived features
        (pl.col("shots_on_target_for") / pl.col("shots_for")).alias("shot_accuracy"),
        (pl.col("goals_for") / pl.col("shots_for")).alias("shot_conversion"),
        (pl.col("goals_against") == 0).cast(pl.Int8).alias("clean_sheet"),
        (pl.col("goals_for") - pl.col("xg_for")).alias("xg_overperformance"),
    ).sort(["team_id", "kickoff_at"])

    return team_df

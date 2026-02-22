"""Assemble the final fixture-level feature matrix from team-perspective features."""

from __future__ import annotations

from datetime import date

import polars as pl


# Feature columns to split into home_ / away_ prefixes (exclude metadata)
_META_COLS = {"fixture_id", "team_id", "opponent_id", "league_id", "kickoff_at", "is_home"}
# Raw match-outcome cols we don't need in the final feature set
_EXCLUDE_COLS = {
    "goals_for", "goals_against", "xg_for", "xg_against",
    "shots_for", "shots_against", "shots_on_target_for", "shots_on_target_against",
    "possession", "big_chances_for", "big_chances_missed",
    "corner_kicks_for", "corner_kicks_against", "passes", "accurate_passes",
    "fouls_committed", "fouls_suffered", "yellow_cards", "red_cards",
    "pass_accuracy", "goal_diff", "win", "draw", "loss", "points",
    # New raw columns (used only for rolling, not as direct features)
    "saves", "offsides", "blocked_shots_for", "blocked_shots_against",
    "shots_off_target_for", "avg_rating", "rated_players",
    "shot_accuracy", "shot_conversion", "clean_sheet", "xg_overperformance",
}

# Delta features: home minus away for these feature pairs (without prefix)
_DELTA_FEATURES = [
    "form_goals_for_r5",
    "form_xg_for_r5",
    "form_goals_against_r5",
    "form_xg_against_r5",
    "ppg",
    "gdpg",
    # New deltas
    "form_avg_rating_r5",    # player quality advantage
    "form_saves_r5",         # GK quality gap
    "form_clean_sheet_r5",   # defensive solidity gap
    "form_shot_accuracy_r5", # shooting quality gap
]


def assemble_feature_matrix(
    fixtures: pl.DataFrame,
    team_features: pl.DataFrame,
    market_features: pl.DataFrame,
    labels: pl.DataFrame,
    start_date: date,
    end_date: date,
) -> pl.DataFrame:
    """Pivot team-perspective features back to fixture-level and join everything.

    1. Split team_features into home/away by joining with fixtures
    2. Prefix feature columns as home_* / away_*
    3. Join market features + labels
    4. Compute delta features (home - away)
    5. Filter to date range and sort by kickoff_at
    """
    # Identify feature columns (everything not in meta/exclude)
    feature_cols = [
        c for c in team_features.columns
        if c not in _META_COLS and c not in _EXCLUDE_COLS
    ]

    # --- Home team features ---
    home_features = (
        fixtures.select("fixture_id", "home_team_id")
        .join(
            team_features.select(["fixture_id", "team_id"] + feature_cols),
            left_on=["fixture_id", "home_team_id"],
            right_on=["fixture_id", "team_id"],
            how="left",
        )
        .drop("home_team_id")
        .rename({c: f"home_{c}" for c in feature_cols})
    )

    # --- Away team features ---
    away_features = (
        fixtures.select("fixture_id", "away_team_id")
        .join(
            team_features.select(["fixture_id", "team_id"] + feature_cols),
            left_on=["fixture_id", "away_team_id"],
            right_on=["fixture_id", "team_id"],
            how="left",
        )
        .drop("away_team_id")
        .rename({c: f"away_{c}" for c in feature_cols})
    )

    # --- Base fixture info ---
    base = fixtures.select(
        "fixture_id", "home_team_id", "away_team_id", "kickoff_at",
        "league_id", "league_key",
    )

    # --- Join all parts ---
    matrix = base
    matrix = matrix.join(home_features, on="fixture_id", how="left")
    matrix = matrix.join(away_features, on="fixture_id", how="left")
    matrix = matrix.join(market_features, on="fixture_id", how="left")
    matrix = matrix.join(labels, on="fixture_id", how="left")

    # --- Delta features (home - away) ---
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

    # --- Filter to date range ---
    start_dt = pl.Series([start_date]).cast(pl.Date).to_list()[0]
    end_dt = pl.Series([end_date]).cast(pl.Date).to_list()[0]
    matrix = matrix.filter(
        (pl.col("kickoff_at").dt.date() >= start_dt)
        & (pl.col("kickoff_at").dt.date() <= end_dt)
    )

    # --- Sort and return ---
    matrix = matrix.sort("kickoff_at")

    return matrix

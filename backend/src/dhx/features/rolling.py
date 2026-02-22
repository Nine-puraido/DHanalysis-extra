"""Compute rolling window features with anti-leakage shift(1)."""

from __future__ import annotations

import polars as pl


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FORM_METRICS = [
    "goals_for", "goals_against", "xg_for", "xg_against",
    "shots_for", "shots_against", "shots_on_target_for", "shots_on_target_against",
    "possession", "big_chances_for",
    "corner_kicks_for", "points", "win",
    # New metrics
    "saves",              # GK quality
    "offsides",           # attacking aggression
    "blocked_shots_for",  # opponents' defensive effort against us
    "fouls_committed",    # discipline
    "yellow_cards",       # discipline severity
    "shot_accuracy",      # shooting precision (derived)
    "clean_sheet",        # defensive solidity (derived)
]

VENUE_METRICS = [
    "goals_for", "goals_against", "xg_for", "xg_against", "win", "points",
]

WINDOWS = [3, 5, 10]


# ---------------------------------------------------------------------------
# A. Team Form — overall rolling averages
# ---------------------------------------------------------------------------

def _team_form(df: pl.DataFrame) -> pl.DataFrame:
    """Compute rolling mean for FORM_METRICS across WINDOWS over all matches."""
    exprs: list[pl.Expr] = []
    for metric in FORM_METRICS:
        for w in WINDOWS:
            col_name = f"form_{metric}_r{w}"
            exprs.append(
                pl.col(metric)
                .shift(1)
                .rolling_mean(w, min_samples=1)
                .over("team_id", order_by="kickoff_at")
                .alias(col_name)
            )
    return df.with_columns(exprs)


# ---------------------------------------------------------------------------
# B. Venue-Specific Form — home-only and away-only rolling
# ---------------------------------------------------------------------------

def _venue_form(df: pl.DataFrame) -> pl.DataFrame:
    """Compute venue-specific rolling features.

    For each team, compute rolling stats using ONLY their home games or ONLY
    their away games (matching the venue of the current row). Each row gets
    its venue-specific form: home-venue form when playing at home, away-venue
    form when playing away.
    """
    # Process home and away subsets separately, using the same column names
    col_names = [f"venue_form_{m}_r{w}" for m in VENUE_METRICS for w in WINDOWS]
    parts: list[pl.DataFrame] = []

    for is_home_val in [True, False]:
        subset = df.filter(pl.col("is_home") == is_home_val)

        exprs: list[pl.Expr] = []
        for metric in VENUE_METRICS:
            for w in WINDOWS:
                col_name = f"venue_form_{metric}_r{w}"
                exprs.append(
                    pl.col(metric)
                    .shift(1)
                    .rolling_mean(w, min_samples=1)
                    .over("team_id", order_by="kickoff_at")
                    .alias(col_name)
                )

        subset = subset.with_columns(exprs).select(
            ["fixture_id", "team_id"] + col_names
        )
        parts.append(subset)

    venue_features = pl.concat(parts)
    df = df.join(venue_features, on=["fixture_id", "team_id"], how="left")

    return df


# ---------------------------------------------------------------------------
# C. Rest / Schedule
# ---------------------------------------------------------------------------

def _rest_schedule(df: pl.DataFrame) -> pl.DataFrame:
    """Compute days since last match and midweek indicator."""
    return df.with_columns(
        # Days since last match
        (
            pl.col("kickoff_at")
            - pl.col("kickoff_at").shift(1).over("team_id", order_by="kickoff_at")
        )
        .dt.total_days()
        .cast(pl.Float64)
        .alias("days_since_last"),
        # Midweek: Tue=1, Wed=2, Thu=3 → weekday() returns 1-7 (Mon=1)
        pl.col("kickoff_at")
        .dt.weekday()
        .is_in([2, 3, 4])
        .cast(pl.Int8)
        .alias("is_midweek"),
    )


# ---------------------------------------------------------------------------
# D. Head-to-Head — expanding cumulative mean (not rolling window)
# ---------------------------------------------------------------------------

def _head_to_head(df: pl.DataFrame) -> pl.DataFrame:
    """Compute H2H expanding stats over all prior meetings between same pair."""
    # Create canonical pair key so A vs B and B vs A share the same key
    df = df.with_columns(
        (
            pl.min_horizontal("team_id", "opponent_id").cast(pl.Utf8)
            + "_"
            + pl.max_horizontal("team_id", "opponent_id").cast(pl.Utf8)
        ).alias("_h2h_pair")
    )

    # Expanding cumulative mean (all prior meetings, not just last 5)
    # Polars has no cum_mean, so we compute cum_sum / cum_count
    h2h_group = ["_h2h_pair", "team_id"]
    df = df.with_columns(
        (
            pl.col("win").cast(pl.Float64).shift(1)
            .cum_sum().over(h2h_group, order_by="kickoff_at")
            / pl.col("win").cast(pl.Float64).shift(1)
            .cum_count().over(h2h_group, order_by="kickoff_at")
        ).alias("h2h_win_rate"),
        (
            pl.col("goals_for").shift(1)
            .cum_sum().over(h2h_group, order_by="kickoff_at")
            / pl.col("goals_for").shift(1)
            .cum_count().over(h2h_group, order_by="kickoff_at")
        ).alias("h2h_goals_for"),
        # Count of prior meetings
        pl.col("win")
        .cast(pl.Float64)
        .shift(1)
        .cum_count()
        .over(h2h_group, order_by="kickoff_at")
        .alias("h2h_matches"),
    ).drop("_h2h_pair")

    return df


# ---------------------------------------------------------------------------
# E. League Context — cumulative within season
# ---------------------------------------------------------------------------

def _league_context(df: pl.DataFrame) -> pl.DataFrame:
    """Compute cumulative league standings features."""
    # Create a helper column for counting games (pl.lit doesn't work with shift/over)
    df = df.with_columns(pl.lit(1.0).alias("_one"))
    df = df.with_columns(
        # Cumulative points (shift 1 to exclude current)
        pl.col("points")
        .cast(pl.Float64)
        .shift(1)
        .cum_sum()
        .over(["team_id", "league_id"], order_by="kickoff_at")
        .alias("cum_points"),
        # Cumulative games
        pl.col("_one")
        .shift(1)
        .cum_sum()
        .over(["team_id", "league_id"], order_by="kickoff_at")
        .alias("cum_games"),
        # Cumulative goal difference
        pl.col("goal_diff")
        .shift(1)
        .cum_sum()
        .over(["team_id", "league_id"], order_by="kickoff_at")
        .alias("_cum_gd"),
    ).with_columns(
        # Points per game
        (pl.col("cum_points") / pl.col("cum_games")).alias("ppg"),
        # Goal difference per game
        (pl.col("_cum_gd") / pl.col("cum_games")).alias("gdpg"),
    ).drop("_cum_gd", "_one")

    return df


# ---------------------------------------------------------------------------
# F. Player Rating Form
# ---------------------------------------------------------------------------

def _player_rating_form(df: pl.DataFrame) -> pl.DataFrame:
    """Compute rolling player rating features if avg_rating column exists."""
    if "avg_rating" not in df.columns:
        return df

    exprs: list[pl.Expr] = []
    for w in WINDOWS:
        exprs.append(
            pl.col("avg_rating")
            .shift(1)
            .rolling_mean(w, min_samples=1)
            .over("team_id", order_by="kickoff_at")
            .alias(f"form_avg_rating_r{w}")
        )
    return df.with_columns(exprs)


# ---------------------------------------------------------------------------
# G. Opponent Strength — cross-reference opponent's rolling stats
# ---------------------------------------------------------------------------

def _opponent_strength(df: pl.DataFrame) -> pl.DataFrame:
    """Add opponent's recent form as context features."""
    opp = df.select(
        "fixture_id",
        pl.col("team_id").alias("_opp_tid"),
        pl.col("ppg").alias("opponent_ppg"),
        pl.col("form_goals_for_r5").alias("opponent_form_goals_r5"),
        pl.col("form_xg_for_r5").alias("opponent_form_xg_r5"),
    )
    df = df.join(
        opp,
        left_on=["fixture_id", "opponent_id"],
        right_on=["fixture_id", "_opp_tid"],
        how="left",
    )
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_all_rolling_features(team_df: pl.DataFrame) -> pl.DataFrame:
    """Apply all rolling feature computations to the team-perspective DataFrame.

    Every rolling computation uses shift(1) to prevent data leakage.
    """
    df = team_df.sort(["team_id", "kickoff_at"])
    df = _team_form(df)
    df = _venue_form(df)
    df = _rest_schedule(df)
    df = _head_to_head(df)
    df = _league_context(df)
    df = _player_rating_form(df)
    df = _opponent_strength(df)
    return df

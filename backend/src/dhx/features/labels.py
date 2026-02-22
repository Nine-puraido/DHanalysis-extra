"""Compute target variables (labels) from results."""

from __future__ import annotations

import polars as pl


def compute_labels(fixtures: pl.DataFrame, results: pl.DataFrame) -> pl.DataFrame:
    """Build 8 label columns at the fixture level for model training.

    Returns a DataFrame with fixture_id + 8 label columns.
    """
    df = fixtures.select("fixture_id").join(results, on="fixture_id", how="inner")

    labels = df.select(
        pl.col("fixture_id"),
        pl.col("home_score").alias("label_home_score"),
        pl.col("away_score").alias("label_away_score"),
        (pl.col("home_score") + pl.col("away_score")).alias("label_total_goals"),
        # Both teams to score
        (
            (pl.col("home_score") > 0) & (pl.col("away_score") > 0)
        ).cast(pl.Int8).alias("label_btts"),
        # Match result as category
        pl.when(pl.col("home_score") > pl.col("away_score"))
        .then(pl.lit("H"))
        .when(pl.col("home_score") == pl.col("away_score"))
        .then(pl.lit("D"))
        .otherwise(pl.lit("A"))
        .alias("label_result"),
        # Binary outcome indicators
        (pl.col("home_score") > pl.col("away_score")).cast(pl.Int8).alias("label_home_win"),
        (pl.col("home_score") == pl.col("away_score")).cast(pl.Int8).alias("label_draw"),
        (pl.col("home_score") < pl.col("away_score")).cast(pl.Int8).alias("label_away_win"),
    )

    return labels

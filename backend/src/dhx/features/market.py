"""Pivot odds snapshots into fixture-level market features."""

from __future__ import annotations

import polars as pl


def compute_market_features(odds: pl.DataFrame) -> pl.DataFrame:
    """Extract market-implied probabilities from the latest pre-match odds.

    Deduplicates to the latest pulled_at per (fixture_id, market, selection, line),
    then pivots into 9 fixture-level columns.
    """
    if odds.is_empty():
        return pl.DataFrame(schema={"fixture_id": pl.Int64})

    # Dedup: keep the latest snapshot per (fixture_id, market, selection, line)
    deduped = odds.sort("pulled_at", descending=True).group_by(
        ["fixture_id", "market", "selection", "line"]
    ).first()

    # --- 1x2 probabilities ---
    odds_1x2 = deduped.filter(pl.col("market") == "1x2")
    home_1x2 = (
        odds_1x2.filter(pl.col("selection") == "home")
        .select("fixture_id", pl.col("implied_prob").alias("mkt_1x2_home_prob"))
    )
    draw_1x2 = (
        odds_1x2.filter(pl.col("selection") == "draw")
        .select("fixture_id", pl.col("implied_prob").alias("mkt_1x2_draw_prob"))
    )
    away_1x2 = (
        odds_1x2.filter(pl.col("selection") == "away")
        .select("fixture_id", pl.col("implied_prob").alias("mkt_1x2_away_prob"))
    )

    # --- Totals (over/under 2.5) ---
    odds_totals = deduped.filter(
        (pl.col("market") == "totals") & (pl.col("line") == 2.5)
    )
    over25 = (
        odds_totals.filter(pl.col("selection") == "over")
        .select("fixture_id", pl.col("implied_prob").alias("mkt_totals_over25_prob"))
    )
    under25 = (
        odds_totals.filter(pl.col("selection") == "under")
        .select("fixture_id", pl.col("implied_prob").alias("mkt_totals_under25_prob"))
    )

    # --- BTTS ---
    odds_btts = deduped.filter(pl.col("market") == "btts")
    btts_yes = (
        odds_btts.filter(pl.col("selection") == "yes")
        .select("fixture_id", pl.col("implied_prob").alias("mkt_btts_yes_prob"))
    )
    btts_no = (
        odds_btts.filter(pl.col("selection") == "no")
        .select("fixture_id", pl.col("implied_prob").alias("mkt_btts_no_prob"))
    )

    # --- Asian Handicap (main line, home team) ---
    odds_ah = deduped.filter(
        (pl.col("market") == "ah") & (pl.col("selection") == "home")
    )
    # If multiple AH lines exist per fixture, take the one closest to 0
    if not odds_ah.is_empty():
        ah_main = (
            odds_ah.with_columns(pl.col("line").abs().alias("_abs_line"))
            .sort("_abs_line")
            .group_by("fixture_id")
            .first()
            .select(
                "fixture_id",
                pl.col("line").alias("mkt_ah_line"),
                pl.col("implied_prob").alias("mkt_ah_home_prob"),
            )
        )
    else:
        ah_main = pl.DataFrame(
            schema={"fixture_id": pl.Int64, "mkt_ah_line": pl.Float64,
                     "mkt_ah_home_prob": pl.Float64}
        )

    # --- Assemble all market features ---
    # Start from unique fixture IDs in odds
    fixture_ids = deduped.select("fixture_id").unique()

    market_features = fixture_ids
    for part in [home_1x2, draw_1x2, away_1x2, over25, under25, btts_yes, btts_no, ah_main]:
        market_features = market_features.join(part, on="fixture_id", how="left")

    return market_features

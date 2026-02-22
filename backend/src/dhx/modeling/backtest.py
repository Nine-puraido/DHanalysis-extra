"""Value betting backtest engine.

Compares model probabilities against bookmaker odds to find positive-EV bets,
applies Kelly criterion staking, and simulates historical P&L.

Usage (via runner.py CLI):
    python -m dhx.modeling backtest \\
        --feature-set-version 7 --test-cutoff 2026-01-15 \\
        --model-type xgboost --kelly 0.25 --min-edge 0.03
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from dhx.db import get_client

logger = logging.getLogger(__name__)


# ======================================================================
# Data structures
# ======================================================================


@dataclass
class Bet:
    """A single simulated bet."""

    fixture_id: int
    kickoff_at: str
    market: str
    selection: str
    line: float | None
    model_prob: float
    market_odds: float
    implied_prob: float
    edge: float
    kelly_fraction: float
    stake: float
    pnl: float
    result: str  # win / loss / push / half_win / half_loss


@dataclass
class BacktestResult:
    """Aggregated backtest output."""

    model_type: str
    min_edge: float
    kelly_multiplier: float
    initial_bankroll: float
    final_bankroll: float
    total_bets: int
    total_staked: float
    total_pnl: float
    roi: float  # total_pnl / total_staked
    hit_rate: float
    avg_edge: float
    max_drawdown_pct: float
    bankroll_series: list[float]
    bets: list[Bet]
    bets_by_market: dict[str, dict]
    edge_threshold_sweep: list[dict] = field(default_factory=list)


# ======================================================================
# 1. Load and de-vig odds
# ======================================================================


def _load_odds_for_fixtures(fixture_ids: list[int]) -> pl.DataFrame:
    """Load latest pre-kickoff odds from odds_snapshots for sofascore_avg.

    Returns DataFrame: fixture_id, market, selection, line, price_decimal,
                       implied_prob (raw), devigged_prob (normalized).
    """
    client = get_client()

    # Paginated fetch — sofascore_avg bookmaker_id=6, is_main=True
    all_rows: list[dict] = []
    page_size = 1000

    for i in range(0, len(fixture_ids), page_size):
        batch = fixture_ids[i : i + page_size]
        data = (
            client.schema("dhx").table("odds_snapshots")
            .select(
                "fixture_id,market,selection,line,price_decimal,implied_prob,pulled_at"
            )
            .eq("bookmaker_id", 6)
            .eq("is_main", True)
            .in_("fixture_id", batch)
            .execute()
            .data
        )
        if data:
            all_rows.extend(data)

    if not all_rows:
        logger.warning("No odds found for the given fixtures")
        return pl.DataFrame(
            schema={
                "fixture_id": pl.Int64,
                "market": pl.String,
                "selection": pl.String,
                "line": pl.Float64,
                "price_decimal": pl.Float64,
                "implied_prob": pl.Float64,
                "devigged_prob": pl.Float64,
            }
        )

    df = pl.DataFrame(all_rows)

    # Cast types
    df = df.with_columns(
        pl.col("fixture_id").cast(pl.Int64),
        pl.col("price_decimal").cast(pl.Float64),
        pl.col("implied_prob").cast(pl.Float64),
        pl.col("line").cast(pl.Float64, strict=False),
        pl.col("pulled_at").str.to_datetime(time_zone="UTC", strict=False),
    )

    # Deduplicate to latest pulled_at per (fixture_id, market, selection, line)
    # Use sort + group_by to get last row per group
    df = df.sort("pulled_at")
    group_cols = ["fixture_id", "market", "selection", "line"]
    df = df.group_by(group_cols).last()

    # De-vig: normalize implied_prob within each (fixture_id, market) so they sum to 1
    df = df.with_columns(
        (
            pl.col("implied_prob")
            / pl.col("implied_prob").sum().over(["fixture_id", "market"])
        ).alias("devigged_prob")
    )

    return df.select(
        "fixture_id",
        "market",
        "selection",
        "line",
        "price_decimal",
        "implied_prob",
        "devigged_prob",
    )


# ======================================================================
# 2. Compute edges
# ======================================================================


def _compute_edges(
    model,
    test_df: pl.DataFrame,
    odds_df: pl.DataFrame,
) -> pl.DataFrame:
    """For each fixture x market x selection, compute model_prob and edge.

    Returns DataFrame with columns:
        fixture_id, kickoff_at, market, selection, line,
        model_prob, market_odds, implied_prob, devigged_prob, edge
    """
    from dhx.modeling.base import compute_market_probabilities
    from dhx.modeling.markets import ah_probability

    lambdas = model.predict_lambdas(test_df)

    # Build a lookup: fixture_id -> (kickoff_at, lambda_home, lambda_away)
    # Merge kickoff_at from test_df
    lambdas = lambdas.join(
        test_df.select("fixture_id", "kickoff_at"), on="fixture_id", how="left"
    )

    rows: list[dict] = []
    for lrow in lambdas.iter_rows(named=True):
        fid = int(lrow["fixture_id"])
        lh = float(lrow["lambda_home"])
        la = float(lrow["lambda_away"])
        kickoff = lrow["kickoff_at"]

        mp = compute_market_probabilities(lh, la, model.max_goals, model.rho)

        # Build model prob lookup: (market, selection, line) -> model_prob
        model_probs: list[tuple[str, str, float | None, float]] = [
            ("1x2", "home", None, mp["1x2"]["home"]),
            ("1x2", "draw", None, mp["1x2"]["draw"]),
            ("1x2", "away", None, mp["1x2"]["away"]),
            ("totals", "over", 2.5, mp["totals"]["over"]),
            ("totals", "under", 2.5, mp["totals"]["under"]),
            ("btts", "yes", None, mp["btts"]["yes"]),
            ("btts", "no", None, mp["btts"]["no"]),
        ]

        # AH: get odds lines for this fixture, compute model prob for each
        # NOTE: odds data stores lines from each side's perspective
        #   (home=-0.25, away=+0.25) but ah_probability expects
        #   the line from home perspective for both sides.
        fixture_odds = odds_df.filter(pl.col("fixture_id") == fid)
        ah_odds = fixture_odds.filter(pl.col("market") == "ah")
        for ah_row in ah_odds.iter_rows(named=True):
            odds_line = ah_row["line"]
            sel = ah_row["selection"]
            if odds_line is not None:
                hp_line = odds_line if sel == "home" else -odds_line
                r = ah_probability(lh, la, hp_line, sel)
                model_probs.append(("ah", sel, odds_line, r["effective_prob"]))

        # Match model probs against odds
        for market, selection, line, mprob in model_probs:
            # Find matching odds row
            mask = (
                (fixture_odds["market"] == market)
                & (fixture_odds["selection"] == selection)
            )
            if line is not None:
                mask = mask & (fixture_odds["line"] == line)
            else:
                mask = mask & (fixture_odds["line"].is_null())

            matched = fixture_odds.filter(mask)
            if len(matched) == 0:
                continue

            orow = matched.row(0, named=True)
            odds_decimal = float(orow["price_decimal"])
            impl_prob = float(orow["implied_prob"])
            devig_prob = float(orow["devigged_prob"])
            edge = mprob * odds_decimal - 1.0

            # For AH, store line from home perspective so settlement
            # formula works correctly for both sides.
            settle_line = line
            if market == "ah" and selection == "away" and line is not None:
                settle_line = -line

            rows.append(
                {
                    "fixture_id": fid,
                    "kickoff_at": kickoff,
                    "market": market,
                    "selection": selection,
                    "line": settle_line,
                    "model_prob": round(mprob, 6),
                    "market_odds": round(odds_decimal, 4),
                    "implied_prob": round(impl_prob, 6),
                    "devigged_prob": round(devig_prob, 6),
                    "edge": round(edge, 6),
                }
            )

    if not rows:
        return pl.DataFrame(
            schema={
                "fixture_id": pl.Int64,
                "kickoff_at": pl.String,
                "market": pl.String,
                "selection": pl.String,
                "line": pl.Float64,
                "model_prob": pl.Float64,
                "market_odds": pl.Float64,
                "implied_prob": pl.Float64,
                "devigged_prob": pl.Float64,
                "edge": pl.Float64,
            }
        )

    return pl.DataFrame(rows)


# ======================================================================
# 3. Settlement logic
# ======================================================================


def _settle_bet(
    market: str,
    selection: str,
    line: float | None,
    home_score: int,
    away_score: int,
    stake: float,
    odds: float,
) -> tuple[float, str]:
    """Settle a single bet. Returns (pnl, result_str)."""
    total_goals = home_score + away_score
    goal_diff = home_score - away_score
    btts = 1 if home_score >= 1 and away_score >= 1 else 0
    result_str = (
        "home" if home_score > away_score else ("draw" if home_score == away_score else "away")
    )

    if market == "1x2":
        if selection == result_str:
            return stake * (odds - 1), "win"
        return -stake, "loss"

    if market == "totals":
        if selection == "over" and total_goals >= 3:
            return stake * (odds - 1), "win"
        if selection == "under" and total_goals <= 2:
            return stake * (odds - 1), "win"
        return -stake, "loss"

    if market == "btts":
        if selection == "yes" and btts == 1:
            return stake * (odds - 1), "win"
        if selection == "no" and btts == 0:
            return stake * (odds - 1), "win"
        return -stake, "loss"

    if market == "ah":
        # AH settlement from home perspective
        if line is None:
            return -stake, "loss"

        if selection == "home":
            adjusted = goal_diff + line
        else:  # away
            adjusted = -goal_diff - line

        # Check for quarter lines
        frac_abs = abs(line - int(line)) if line >= 0 else abs(line - math.ceil(line - 1))
        frac_abs = abs(round(line % 1, 2)) if line >= 0 else abs(round(line % 1, 2))
        # Simpler: check if line has .25 or .75 component
        line_mod = abs(line) % 0.5
        is_quarter = abs(line_mod - 0.25) < 0.01

        if is_quarter:
            return _settle_ah_quarter(selection, goal_diff, line, stake, odds)

        # Whole or half line
        if adjusted > 0:
            return stake * (odds - 1), "win"
        if adjusted == 0:
            return 0.0, "push"
        return -stake, "loss"

    return -stake, "loss"


def _settle_ah_quarter(
    selection: str,
    goal_diff: int,
    line: float,
    stake: float,
    odds: float,
) -> tuple[float, str]:
    """Settle quarter-line AH bets (split into two half-stakes)."""
    # Split into two adjacent lines
    lower = math.floor(line * 2) / 2
    upper = math.ceil(line * 2) / 2
    half = stake / 2

    pnl = 0.0
    for sub_line in (lower, upper):
        if selection == "home":
            adjusted = goal_diff + sub_line
        else:
            adjusted = -goal_diff - sub_line

        if adjusted > 0:
            pnl += half * (odds - 1)
        elif adjusted == 0:
            pass  # push, return half stake (net 0)
        else:
            pnl -= half

    # Determine result label
    if pnl > stake * 0.01:
        result = "win" if pnl >= stake * (odds - 1) * 0.9 else "half_win"
    elif pnl < -stake * 0.01:
        result = "loss" if pnl <= -stake * 0.9 else "half_loss"
    else:
        result = "push"

    return round(pnl, 2), result


# ======================================================================
# 4. Backtest simulation
# ======================================================================


def simulate_backtest(
    edges_df: pl.DataFrame,
    labels_df: pl.DataFrame,
    initial_bankroll: float = 1000.0,
    kelly_multiplier: float = 0.25,
    min_edge: float = 0.03,
    max_stake_pct: float = 0.05,
    model_type: str = "unknown",
    staking: str = "kelly",
    flat_stake_pct: float = 0.02,
) -> BacktestResult:
    """Simulate sequential betting through time.

    Args:
        edges_df: output from _compute_edges (fixture_id, market, selection, etc.)
        labels_df: test_df with label_ columns and fixture_id, kickoff_at
        initial_bankroll: starting capital
        kelly_multiplier: fraction of Kelly (0.25 = quarter Kelly)
        min_edge: minimum edge threshold to place a bet
        max_stake_pct: max fraction of bankroll per bet
        model_type: for display purposes
        staking: "kelly" (default) or "flat"
        flat_stake_pct: flat stake as fraction of bankroll (used when staking="flat")

    Returns:
        BacktestResult with all metrics and bet details.
    """
    # Build labels lookup: fixture_id -> {home_score, away_score}
    labels_lookup: dict[int, dict] = {}
    for row in labels_df.select(
        "fixture_id",
        "label_home_score",
        "label_away_score",
    ).iter_rows(named=True):
        labels_lookup[int(row["fixture_id"])] = {
            "home_score": int(row["label_home_score"]),
            "away_score": int(row["label_away_score"]),
        }

    # Filter to positive-edge bets
    qualified = edges_df.filter(pl.col("edge") > min_edge)

    if len(qualified) == 0:
        return BacktestResult(
            model_type=model_type,
            min_edge=min_edge,
            kelly_multiplier=kelly_multiplier,
            initial_bankroll=initial_bankroll,
            final_bankroll=initial_bankroll,
            total_bets=0,
            total_staked=0.0,
            total_pnl=0.0,
            roi=0.0,
            hit_rate=0.0,
            avg_edge=0.0,
            max_drawdown_pct=0.0,
            bankroll_series=[initial_bankroll],
            bets=[],
            bets_by_market={},
        )

    # Sort by kickoff_at for chronological simulation
    qualified = qualified.sort("kickoff_at")

    bankroll = initial_bankroll
    peak_bankroll = initial_bankroll
    max_drawdown = 0.0
    bankroll_series = [initial_bankroll]
    bets: list[Bet] = []
    total_staked = 0.0

    for row in qualified.iter_rows(named=True):
        fid = int(row["fixture_id"])
        if fid not in labels_lookup:
            continue

        labels = labels_lookup[fid]
        model_prob = float(row["model_prob"])
        market_odds = float(row["market_odds"])
        edge = float(row["edge"])
        market = row["market"]
        selection = row["selection"]
        line = row["line"]
        implied_prob = float(row["implied_prob"])
        kickoff = row["kickoff_at"]

        if market_odds <= 1.0:
            continue

        if staking == "flat":
            kelly_f = 0.0
            stake = flat_stake_pct * bankroll
            stake = min(stake, max_stake_pct * bankroll)
            stake = round(stake, 2)
        else:
            # Kelly fraction: (p*b - 1) / (b - 1) where b = decimal odds, p = model_prob
            kelly_f = (model_prob * market_odds - 1.0) / (market_odds - 1.0)
            kelly_f = max(kelly_f, 0.0)
            # Apply fractional Kelly and cap
            stake = kelly_f * kelly_multiplier * bankroll
            stake = min(stake, max_stake_pct * bankroll)
            stake = round(stake, 2)

        if stake < 0.01 or bankroll < 1.0:
            continue

        # Settle
        pnl, result = _settle_bet(
            market,
            selection,
            line,
            labels["home_score"],
            labels["away_score"],
            stake,
            market_odds,
        )

        bankroll += pnl
        total_staked += stake
        bankroll_series.append(round(bankroll, 2))

        # Track drawdown
        if bankroll > peak_bankroll:
            peak_bankroll = bankroll
        dd = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
        if dd > max_drawdown:
            max_drawdown = dd

        bets.append(
            Bet(
                fixture_id=fid,
                kickoff_at=str(kickoff),
                market=market,
                selection=selection,
                line=line,
                model_prob=model_prob,
                market_odds=market_odds,
                implied_prob=implied_prob,
                edge=edge,
                kelly_fraction=round(kelly_f, 4),
                stake=stake,
                pnl=round(pnl, 2),
                result=result,
            )
        )

    # Aggregate
    total_pnl = bankroll - initial_bankroll
    roi = total_pnl / total_staked if total_staked > 0 else 0.0
    wins = sum(1 for b in bets if b.result in ("win", "half_win"))
    hit_rate = wins / len(bets) if bets else 0.0
    avg_edge = sum(b.edge for b in bets) / len(bets) if bets else 0.0

    # By-market breakdown
    bets_by_market: dict[str, dict] = {}
    for mkt in ("1x2", "totals", "btts", "ah"):
        mkt_bets = [b for b in bets if b.market == mkt]
        if not mkt_bets:
            continue
        mkt_wins = sum(1 for b in mkt_bets if b.result in ("win", "half_win"))
        mkt_staked = sum(b.stake for b in mkt_bets)
        mkt_pnl = sum(b.pnl for b in mkt_bets)
        bets_by_market[mkt] = {
            "bets": len(mkt_bets),
            "hit_rate": round(mkt_wins / len(mkt_bets), 4) if mkt_bets else 0,
            "avg_edge": round(sum(b.edge for b in mkt_bets) / len(mkt_bets), 4),
            "staked": round(mkt_staked, 2),
            "pnl": round(mkt_pnl, 2),
            "roi": round(mkt_pnl / mkt_staked, 4) if mkt_staked > 0 else 0,
        }

    return BacktestResult(
        model_type=model_type,
        min_edge=min_edge,
        kelly_multiplier=kelly_multiplier,
        initial_bankroll=initial_bankroll,
        final_bankroll=round(bankroll, 2),
        total_bets=len(bets),
        total_staked=round(total_staked, 2),
        total_pnl=round(total_pnl, 2),
        roi=round(roi, 4),
        hit_rate=round(hit_rate, 4),
        avg_edge=round(avg_edge, 4),
        max_drawdown_pct=round(max_drawdown * 100, 2),
        bankroll_series=bankroll_series,
        bets=bets,
        bets_by_market=bets_by_market,
    )


# ======================================================================
# 5. Multi-threshold sweep
# ======================================================================


def run_edge_threshold_sweep(
    edges_df: pl.DataFrame,
    labels_df: pl.DataFrame,
    thresholds: list[float] | None = None,
    initial_bankroll: float = 1000.0,
    kelly_multiplier: float = 0.25,
    max_stake_pct: float = 0.05,
    model_type: str = "unknown",
) -> list[dict]:
    """Run backtest at multiple edge thresholds and return comparison."""
    if thresholds is None:
        thresholds = [0.01, 0.03, 0.05, 0.10]

    results = []
    for t in thresholds:
        r = simulate_backtest(
            edges_df,
            labels_df,
            initial_bankroll=initial_bankroll,
            kelly_multiplier=kelly_multiplier,
            min_edge=t,
            max_stake_pct=max_stake_pct,
            model_type=model_type,
        )
        results.append(
            {
                "min_edge": t,
                "bets": r.total_bets,
                "staked": r.total_staked,
                "pnl": r.total_pnl,
                "roi": r.roi,
                "hit_rate": r.hit_rate,
                "avg_edge": r.avg_edge,
                "max_drawdown": r.max_drawdown_pct,
                "final_bankroll": r.final_bankroll,
            }
        )

    return results


# ======================================================================
# 6. Full pipeline
# ======================================================================


def run_backtest(
    model,
    test_df: pl.DataFrame,
    model_type: str = "unknown",
    kelly_multiplier: float = 0.25,
    min_edge: float = 0.03,
    max_stake_pct: float = 0.05,
    initial_bankroll: float = 1000.0,
) -> tuple[BacktestResult, list[dict], pl.DataFrame]:
    """End-to-end backtest: load odds → compute edges → simulate → sweep.

    Returns:
        (primary_result, threshold_sweep, edges_df)
    """
    # 1. Get fixture IDs from test set
    fixture_ids = test_df["fixture_id"].to_list()
    fixture_ids = [int(fid) for fid in fixture_ids]
    logger.info(f"Loading odds for {len(fixture_ids)} test fixtures...")

    # 2. Load and de-vig odds
    odds_df = _load_odds_for_fixtures(fixture_ids)
    fixtures_with_odds = odds_df["fixture_id"].n_unique()
    logger.info(
        f"Loaded {len(odds_df)} odds rows for {fixtures_with_odds} fixtures"
    )

    if len(odds_df) == 0:
        raise ValueError("No odds data found for test fixtures")

    # 3. Compute edges
    logger.info("Computing edges...")
    edges_df = _compute_edges(model, test_df, odds_df)
    logger.info(
        f"Computed {len(edges_df)} edge rows "
        f"({len(edges_df.filter(pl.col('edge') > min_edge))} above {min_edge:.0%} threshold)"
    )

    # 4. Primary simulation
    result = simulate_backtest(
        edges_df,
        test_df,
        initial_bankroll=initial_bankroll,
        kelly_multiplier=kelly_multiplier,
        min_edge=min_edge,
        max_stake_pct=max_stake_pct,
        model_type=model_type,
    )

    # 5. Threshold sweep
    sweep = run_edge_threshold_sweep(
        edges_df,
        test_df,
        initial_bankroll=initial_bankroll,
        kelly_multiplier=kelly_multiplier,
        max_stake_pct=max_stake_pct,
        model_type=model_type,
    )

    return result, sweep, edges_df


# ======================================================================
# 7. Disagreement strategy (AH line gap)
# ======================================================================


def _compute_disagreement_bets(
    model,
    test_df: pl.DataFrame,
    odds_df: pl.DataFrame,
) -> pl.DataFrame:
    """Select AH bets based on model vs market line disagreement.

    For each fixture with AH odds:
    1. Compute model's fair AH line via find_fair_ah_line(lh, la)
    2. Get market's home AH line from odds data
    3. gap = fair_line - market_home_line
    4. If gap == 0 (agree): bet favorite (side with negative handicap)
    5. If gap != 0 (disagree): bet underdog (side with positive handicap)

    Returns DataFrame with columns:
        fixture_id, kickoff_at, market, selection, line,
        model_prob, market_odds, implied_prob, devigged_prob, edge,
        fair_line, market_line, gap
    """
    from dhx.modeling.markets import ah_probability, find_fair_ah_line

    lambdas = model.predict_lambdas(test_df)
    lambdas = lambdas.join(
        test_df.select("fixture_id", "kickoff_at"), on="fixture_id", how="left"
    )

    rows: list[dict] = []
    for lrow in lambdas.iter_rows(named=True):
        fid = int(lrow["fixture_id"])
        lh = float(lrow["lambda_home"])
        la = float(lrow["lambda_away"])
        kickoff = lrow["kickoff_at"]

        # Model's fair AH line (home perspective)
        fair_line = find_fair_ah_line(lh, la)

        # Get market home AH line from odds
        fixture_odds = odds_df.filter(pl.col("fixture_id") == fid)
        ah_home = fixture_odds.filter(
            (pl.col("market") == "ah") & (pl.col("selection") == "home")
        )
        if len(ah_home) == 0:
            continue

        market_home_line = float(ah_home.row(0, named=True)["line"])

        # Gap: positive means market more aggressive on fav, negative means model more aggressive
        gap = fair_line - market_home_line

        # Decision: agree (gap == 0) → bet favorite; disagree → bet underdog
        # Favorite has negative handicap line, underdog has positive
        if gap == 0.0:
            # Agree: bet the favorite (side with negative handicap)
            if market_home_line <= 0:
                bet_selection = "home"
            else:
                bet_selection = "away"
        else:
            # Disagree: bet the underdog (side with positive handicap)
            if market_home_line <= 0:
                bet_selection = "away"
            else:
                bet_selection = "home"

        # Find odds for the chosen side
        bet_odds_row = fixture_odds.filter(
            (pl.col("market") == "ah") & (pl.col("selection") == bet_selection)
        )
        if len(bet_odds_row) == 0:
            continue

        orow = bet_odds_row.row(0, named=True)
        odds_decimal = float(orow["price_decimal"])
        impl_prob = float(orow["implied_prob"])
        devig_prob = float(orow["devigged_prob"])
        odds_line = orow["line"]

        # Compute model probability for the chosen side at the market line
        hp_line = float(odds_line) if bet_selection == "home" else -float(odds_line)
        r = ah_probability(lh, la, hp_line, bet_selection)
        model_prob = r["effective_prob"]
        edge = model_prob * odds_decimal - 1.0

        # Store line from home perspective for settlement
        settle_line = float(odds_line) if bet_selection == "home" else -float(odds_line)

        rows.append(
            {
                "fixture_id": fid,
                "kickoff_at": kickoff,
                "market": "ah",
                "selection": bet_selection,
                "line": settle_line,
                "model_prob": round(model_prob, 6),
                "market_odds": round(odds_decimal, 4),
                "implied_prob": round(impl_prob, 6),
                "devigged_prob": round(devig_prob, 6),
                "edge": round(edge, 6),
                "fair_line": fair_line,
                "market_line": market_home_line,
                "gap": round(gap, 4),
            }
        )

    if not rows:
        return pl.DataFrame(
            schema={
                "fixture_id": pl.Int64,
                "kickoff_at": pl.String,
                "market": pl.String,
                "selection": pl.String,
                "line": pl.Float64,
                "model_prob": pl.Float64,
                "market_odds": pl.Float64,
                "implied_prob": pl.Float64,
                "devigged_prob": pl.Float64,
                "edge": pl.Float64,
                "fair_line": pl.Float64,
                "market_line": pl.Float64,
                "gap": pl.Float64,
            }
        )

    return pl.DataFrame(rows)


def run_disagreement_backtest(
    model,
    test_df: pl.DataFrame,
    model_type: str = "unknown",
    flat_stake_pct: float = 0.02,
    max_stake_pct: float = 0.05,
    initial_bankroll: float = 1000.0,
) -> tuple[BacktestResult, pl.DataFrame]:
    """End-to-end disagreement backtest.

    1. Load odds
    2. Compute disagreement bets (AH only)
    3. Simulate with flat staking
    4. Return (result, edges_df with gap info)
    """
    # 1. Get fixture IDs from test set
    fixture_ids = test_df["fixture_id"].to_list()
    fixture_ids = [int(fid) for fid in fixture_ids]
    logger.info(f"Loading odds for {len(fixture_ids)} test fixtures...")

    # 2. Load and de-vig odds
    odds_df = _load_odds_for_fixtures(fixture_ids)
    fixtures_with_odds = odds_df["fixture_id"].n_unique()
    logger.info(
        f"Loaded {len(odds_df)} odds rows for {fixtures_with_odds} fixtures"
    )

    if len(odds_df) == 0:
        raise ValueError("No odds data found for test fixtures")

    # 3. Compute disagreement bets
    logger.info("Computing disagreement bets (AH only)...")
    edges_df = _compute_disagreement_bets(model, test_df, odds_df)
    logger.info(f"Generated {len(edges_df)} AH disagreement bets")

    # 4. Simulate with flat staking (include all bets, no min_edge filter)
    result = simulate_backtest(
        edges_df,
        test_df,
        initial_bankroll=initial_bankroll,
        kelly_multiplier=0.0,
        min_edge=-1.0,  # include all bets
        max_stake_pct=max_stake_pct,
        model_type=model_type,
        staking="flat",
        flat_stake_pct=flat_stake_pct,
    )

    return result, edges_df

"""Checklist-based strategy simulation for AH betting.

Combines four signals to decide whether to follow model or market:
  1. S1 League-conditional – model accuracy varies by league
  2. S2 Price chain – HFA-adjusted implied AH via common opponents (>=3 proxies)
  3. S3 Team trends – sliding window (last 15) model-vs-market accuracy
  4. S4 Totals agreement – model & market agree on O/U 2.5?

Majority vote (tie = no bet) -> flat stake at market AH line -> settle.

Usage (via runner.py CLI):
    python -m dhx.modeling strategy-sim \\
        --feature-set-version 7 --test-cutoff 2026-01-15 \\
        --model-type xgboost --stake 100
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import polars as pl

from dhx.db import get_client

logger = logging.getLogger(__name__)

# ======================================================================
# Constants
# ======================================================================

HFA = 0.5  # Home field advantage in AH terms (industry standard)
LOOKBACK_DAYS = 60  # 2-month window for price chain proxy search
TREND_MIN_MATCHES = 3  # Minimum decided matches before team trend signal is valid
TREND_WINDOW = 15  # Sliding window size for S3 team trends
MIN_PROXIES = 3  # Minimum proxy teams for S2 price chain signal
DEFAULT_MIN_EDGE = 0.0  # Minimum EV edge (model_prob × odds - 1) to place bet

# S1: League-conditional model direction
# Model zone: model accuracy > 53% historically -> S1 votes "model"
S1_MODEL_LEAGUES = {"EPL", "LL", "TSL", "SPL", "SA", "L1"}
# Market zone: model accuracy < 49% historically -> S1 votes "market"
S1_MARKET_LEAGUES = {"BL", "BL2", "PPL", "SSL"}
# Dead zone: leagues in neither set -> S1 is None


# ======================================================================
# Data structures
# ======================================================================


@dataclass
class StrategyBet:
    """A single simulated bet from the checklist strategy."""

    fixture_id: int
    kickoff_at: str
    league_key: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    model_line: float
    market_line: float
    line_diff: float
    # Signals
    signal_s1: str | None  # "model", "market", or None (dead zone)
    signal_chain: str | None  # "model", "market", or None
    chain_implied_ah: float | None
    proxy_count: int  # number of proxy teams used for S2
    signal_trend: str | None  # "model", "market", or None
    home_team_model_pct: float | None
    away_team_model_pct: float | None
    signal_totals: str | None  # "model" or "market" (S4)
    # Vote counts
    model_votes: int
    market_votes: int
    signals_available: int
    # Decision
    decision: str  # "model" or "market"
    bet_side: str  # "home" or "away"
    # Settlement
    adjusted: float
    result: str  # "win", "loss", "push"
    pnl: float  # +stake, -stake, or 0
    edge: float | None = None  # EV edge: model_prob × odds - 1


@dataclass
class StrategyResult:
    """Aggregated strategy simulation output."""

    model_type: str
    total_fixtures: int
    fixtures_with_odds: int
    fixtures_with_disagreement: int
    total_bets: int
    skipped: int
    wins: int
    losses: int
    pushes: int
    total_pnl: float
    roi: float  # pnl / total_staked
    hit_rate: float
    stake_per_bet: float
    bets: list[StrategyBet]
    # Breakdown
    by_decision: dict[str, dict]  # "model" and "market" sub-stats
    by_signal_count: dict[str, dict]  # "4_signals", "3_signals", "2_signals", "1_signal"
    by_league: dict[str, dict]
    by_signal: dict[str, dict]  # per-signal accuracy: {s1: {fired, correct, pct}, ...}
    by_combination: dict[str, dict]  # per-combo pattern stats


# ======================================================================
# 1. Data loading
# ======================================================================


def _load_predictions() -> pl.DataFrame:
    """Load finished predictions with AH lines from vw_predictions_summary."""
    client = get_client()
    all_rows: list[dict] = []
    for offset in range(0, 5000, 1000):
        data = (
            client.schema("dhx").table("vw_predictions_summary")
            .select(
                "fixture_id,kickoff_at,status,league_key,league_name,"
                "home_team,away_team,home_score,away_score,"
                "ah_fair_line,ah_closing_line,"
                "ah_home_prob,ah_away_prob,"
                "ah_closing_home,ah_closing_away,"
                "prob_over25,closing_over25"
            )
            .eq("status", "finished")
            .order("kickoff_at")
            .range(offset, offset + 999)
            .execute()
            .data
        )
        if data:
            all_rows.extend(data)
        if not data or len(data) < 1000:
            break
    logger.info(f"Loaded {len(all_rows)} finished predictions")
    return pl.DataFrame(all_rows) if all_rows else pl.DataFrame()


def _load_fixtures() -> pl.DataFrame:
    """Load fixture details (team IDs, league IDs)."""
    client = get_client()
    all_rows: list[dict] = []
    for offset in range(0, 5000, 1000):
        data = (
            client.schema("dhx").table("fixtures")
            .select("id,league_id,home_team_id,away_team_id,kickoff_at,status")
            .eq("status", "finished")
            .order("kickoff_at")
            .range(offset, offset + 999)
            .execute()
            .data
        )
        if data:
            all_rows.extend(data)
        if not data or len(data) < 1000:
            break
    logger.info(f"Loaded {len(all_rows)} fixtures")
    return pl.DataFrame(all_rows) if all_rows else pl.DataFrame()


def _load_ah_lines() -> dict[int, float]:
    """Load AH home lines from vw_odds_latest_pre_kickoff (sofascore_avg).

    Returns {fixture_id: home_ah_line}.
    """
    client = get_client()
    all_rows: list[dict] = []
    for offset in range(0, 10000, 1000):
        data = (
            client.schema("dhx").table("vw_odds_latest_pre_kickoff")
            .select("fixture_id,line")
            .eq("bookmaker_id", 6)
            .eq("market", "ah")
            .eq("selection", "home")
            .order("fixture_id")
            .range(offset, offset + 999)
            .execute()
            .data
        )
        if data:
            all_rows.extend(data)
        if not data or len(data) < 1000:
            break

    result: dict[int, float] = {}
    for row in all_rows:
        if row.get("line") is not None:
            result[int(row["fixture_id"])] = float(row["line"])
    logger.info(f"Loaded AH lines for {len(result)} fixtures")
    return result


# ======================================================================
# 2. Signal computation
# ======================================================================


def _compute_s1_signal(league_key: str) -> str | None:
    """Signal 1: League-conditional model direction.

    Returns "model", "market", or None (dead zone).
    """
    if league_key in S1_MODEL_LEAGUES:
        return "model"
    if league_key in S1_MARKET_LEAGUES:
        return "market"
    return None


def _compute_chain_signal(
    target_fixture_id: int,
    home_team_id: int,
    away_team_id: int,
    league_id: int,
    kickoff_dt: datetime,
    model_line: float,
    market_line: float,
    fixtures_df: pl.DataFrame,
    ah_lines: dict[int, float],
) -> tuple[str | None, float | None, int]:
    """Signal 2: Price chain via common opponents.

    Returns (signal, implied_ah, proxy_count) where signal is "model",
    "market", or None.  Requires >= MIN_PROXIES proxy teams with AH lines.
    """
    cutoff = kickoff_dt - timedelta(days=LOOKBACK_DAYS)

    # Filter to same league, finished, before this match, within lookback
    league_fix = fixtures_df.filter(
        (pl.col("league_id") == league_id)
        & (pl.col("kickoff_at") < kickoff_dt)
        & (pl.col("kickoff_at") >= cutoff)
        & (pl.col("id") != target_fixture_id)
    )

    if len(league_fix) == 0:
        return None, None, 0

    # Build team match lists
    team_a_matches: list[dict] = []  # home team's matches
    team_b_matches: list[dict] = []  # away team's matches

    for row in league_fix.iter_rows(named=True):
        fid = int(row["id"])
        htid = int(row["home_team_id"])
        atid = int(row["away_team_id"])

        if htid == home_team_id or atid == home_team_id:
            team_a_matches.append(row)
        if htid == away_team_id or atid == away_team_id:
            team_b_matches.append(row)

    # Find proxy teams (opponents of both A and B)
    a_opponents: dict[int, dict] = {}
    for m in team_a_matches:
        opp_id = m["away_team_id"] if m["home_team_id"] == home_team_id else m["home_team_id"]
        if opp_id not in (home_team_id, away_team_id):
            a_opponents[int(opp_id)] = m

    b_opponents: dict[int, dict] = {}
    for m in team_b_matches:
        opp_id = m["away_team_id"] if m["home_team_id"] == away_team_id else m["home_team_id"]
        if opp_id not in (home_team_id, away_team_id):
            b_opponents[int(opp_id)] = m

    proxy_ids = set(a_opponents.keys()) & set(b_opponents.keys())
    if len(proxy_ids) < MIN_PROXIES:
        return None, None, len(proxy_ids)

    # Compute HFA-adjusted implied AH for each proxy
    implied_ahs: list[float] = []
    for pid in proxy_ids:
        vs_a = a_opponents[pid]
        vs_b = b_opponents[pid]

        fid_a = int(vs_a["id"])
        fid_b = int(vs_b["id"])

        ah_a = ah_lines.get(fid_a)
        ah_b = ah_lines.get(fid_b)

        if ah_a is None or ah_b is None:
            continue

        # TeamA's neutral strength advantage over proxy
        if vs_a["home_team_id"] == home_team_id:
            team_a_vs_proxy = -ah_a - HFA  # teamA was home
        else:
            team_a_vs_proxy = ah_a + HFA  # teamA was away

        # TeamB's neutral strength advantage over proxy
        if vs_b["home_team_id"] == away_team_id:
            team_b_vs_proxy = -ah_b - HFA  # teamB was home
        else:
            team_b_vs_proxy = ah_b + HFA  # teamB was away

        # Neutral strength diff (positive = teamA stronger)
        neutral_diff = team_a_vs_proxy - team_b_vs_proxy

        # Implied AH for actual match (teamA is home)
        implied_ah = -(neutral_diff + HFA)
        implied_ahs.append(implied_ah)

    if len(implied_ahs) < MIN_PROXIES:
        return None, None, len(implied_ahs)

    avg_implied = sum(implied_ahs) / len(implied_ahs)

    # Is implied closer to model or market?
    dist_to_model = abs(avg_implied - model_line)
    dist_to_market = abs(avg_implied - market_line)

    if dist_to_model < dist_to_market:
        return "model", avg_implied, len(implied_ahs)
    elif dist_to_market < dist_to_model:
        return "market", avg_implied, len(implied_ahs)
    else:
        return None, avg_implied, len(implied_ahs)  # equidistant -> neutral


def _compute_trend_signal(
    home_team_name: str,
    away_team_name: str,
    team_accuracy: dict[str, list[bool]],
) -> tuple[str | None, float | None, float | None]:
    """Signal 3: Per-team rolling model accuracy (sliding window).

    Returns (signal, home_model_pct, away_model_pct).
    """
    home_results = team_accuracy.get(home_team_name, [])
    away_results = team_accuracy.get(away_team_name, [])

    home_decided = len(home_results)
    away_decided = len(away_results)

    # Need minimum matches for signal to be valid
    if home_decided < TREND_MIN_MATCHES or away_decided < TREND_MIN_MATCHES:
        return None, None, None

    home_model_wins = sum(home_results)
    away_model_wins = sum(away_results)

    home_model_pct = home_model_wins / home_decided * 100
    away_model_pct = away_model_wins / away_decided * 100

    # Weighted average to decide signal
    total_model = home_model_wins + away_model_wins
    total_decided = home_decided + away_decided
    combined_pct = total_model / total_decided * 100

    if combined_pct > 50:
        return "model", home_model_pct, away_model_pct
    elif combined_pct < 50:
        return "market", home_model_pct, away_model_pct
    else:
        return None, home_model_pct, away_model_pct  # exactly 50% -> neutral


def _update_team_accuracy(
    team_accuracy: dict[str, list[bool]],
    home_team: str,
    away_team: str,
    model_bets_home: bool,
    adjusted: float,
) -> None:
    """Update sliding-window team accuracy after a match settles."""
    if adjusted == 0:
        return  # push -- don't update

    # Determine if model's bet was profitable
    if model_bets_home:
        model_profitable = adjusted > 0
    else:
        model_profitable = adjusted < 0

    # Update both teams with sliding window
    for team in (home_team, away_team):
        if team not in team_accuracy:
            team_accuracy[team] = []
        team_accuracy[team].append(model_profitable)
        # Trim to sliding window
        if len(team_accuracy[team]) > TREND_WINDOW:
            team_accuracy[team].pop(0)


def _compute_totals_signal(
    prob_over25: float | None,
    closing_over25: float | None,
) -> str | None:
    """Signal 4: Totals agreement.

    Model and market agree on O/U 2.5 direction -> "model" (agreement = confidence).
    Disagree -> "market". Missing data -> None.
    """
    if prob_over25 is None or closing_over25 is None:
        return None
    model_over = prob_over25 > 0.5
    market_over = 1 / closing_over25 > 0.5
    return "model" if model_over == market_over else "market"


# ======================================================================
# 3. Main simulation
# ======================================================================


def run_strategy_sim(
    model_type: str = "xgboost",
    test_cutoff: str | None = None,
    stake: float = 100.0,
    feature_set_version: int | None = None,
    min_signals: int = 1,
    min_edge: float = DEFAULT_MIN_EDGE,
    only_leagues: list[str] | None = None,
    exclude_leagues: list[str] | None = None,
) -> StrategyResult:
    """Run the full checklist strategy simulation.

    Processes ALL fixtures chronologically to build team trend counters,
    but only places bets on fixtures after test_cutoff.
    """
    # 1. Load data
    logger.info("Loading predictions...")
    preds_df = _load_predictions()
    if len(preds_df) == 0:
        raise ValueError("No predictions found")

    logger.info("Loading fixtures...")
    fixtures_df = _load_fixtures()

    logger.info("Loading AH odds lines...")
    ah_lines = _load_ah_lines()

    # 1b. Filter by league if specified
    if only_leagues:
        preds_df = preds_df.filter(pl.col("league_key").is_in(only_leagues))
        logger.info(f"Filtered to leagues: {only_leagues} -> {len(preds_df)} fixtures")
    if exclude_leagues:
        preds_df = preds_df.filter(~pl.col("league_key").is_in(exclude_leagues))
        logger.info(f"Excluded leagues: {exclude_leagues} -> {len(preds_df)} fixtures")

    # 2. Filter to fixtures with both model and market AH lines + results
    preds_df = preds_df.filter(
        pl.col("ah_fair_line").is_not_null()
        & pl.col("ah_closing_line").is_not_null()
        & pl.col("home_score").is_not_null()
        & pl.col("away_score").is_not_null()
    )

    total_with_odds = len(preds_df)

    # Filter to lines that disagree
    preds_df = preds_df.filter(
        pl.col("ah_fair_line") != pl.col("ah_closing_line")
    )

    total_disagree = len(preds_df)
    logger.info(
        f"Found {total_with_odds} fixtures with odds, "
        f"{total_disagree} with disagreement"
    )

    # Parse cutoff
    cutoff_dt = None
    if test_cutoff:
        cutoff_dt = datetime.strptime(test_cutoff, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )

    # 3. Parse dates
    preds_df = preds_df.with_columns(
        pl.col("kickoff_at").str.to_datetime(time_zone="UTC", strict=False)
        .alias("kickoff_dt")
    )

    fixtures_df = fixtures_df.with_columns(
        pl.col("kickoff_at").str.to_datetime(time_zone="UTC", strict=False)
        .alias("kickoff_at")
    )

    # 4. Join fixture team IDs to predictions
    fixture_info = fixtures_df.select(
        pl.col("id").alias("fixture_id"),
        pl.col("league_id"),
        pl.col("home_team_id"),
        pl.col("away_team_id"),
    )
    preds_df = preds_df.with_columns(
        pl.col("fixture_id").cast(pl.Int64)
    )
    fixture_info = fixture_info.with_columns(
        pl.col("fixture_id").cast(pl.Int64)
    )

    preds_df = preds_df.join(fixture_info, on="fixture_id", how="left")

    # Sort ALL fixtures chronologically (warmup + test)
    preds_df = preds_df.sort("kickoff_dt")

    # Count test-period fixtures for reporting
    if cutoff_dt:
        test_count = len(preds_df.filter(pl.col("kickoff_dt") >= cutoff_dt))
        warmup_count = len(preds_df) - test_count
        logger.info(
            f"Warmup (pre-cutoff): {warmup_count} fixtures, "
            f"Test (post-cutoff): {test_count} fixtures"
        )

    # 5. Run simulation -- process ALL rows, only bet after cutoff
    team_accuracy: dict[str, list[bool]] = {}
    bets: list[StrategyBet] = []
    skipped = 0

    # Per-signal accuracy tracking: {signal_name: {correct: int, incorrect: int}}
    signal_accuracy: dict[str, dict[str, int]] = {
        "s1": {"correct": 0, "incorrect": 0},
        "s2": {"correct": 0, "incorrect": 0},
        "s3": {"correct": 0, "incorrect": 0},
        "s4": {"correct": 0, "incorrect": 0},
    }
    # Per-combination tracking: {pattern: {bets: int, wins: int, losses: int, pnl: float}}
    combo_stats: dict[str, dict] = {}

    for row in preds_df.iter_rows(named=True):
        fid = int(row["fixture_id"])
        model_line = float(row["ah_fair_line"])
        market_line = float(row["ah_closing_line"])
        home_score = int(row["home_score"])
        away_score = int(row["away_score"])
        kickoff = row["kickoff_at"] if isinstance(row["kickoff_at"], str) else str(row["kickoff_at"])
        home_team = row["home_team"]
        away_team = row["away_team"]
        league_key = row["league_key"]

        league_id = row.get("league_id")
        home_team_id = row.get("home_team_id")
        away_team_id = row.get("away_team_id")

        # Totals data (may be None)
        prob_over25 = row.get("prob_over25")
        closing_over25 = row.get("closing_over25")
        if prob_over25 is not None:
            prob_over25 = float(prob_over25)
        if closing_over25 is not None:
            closing_over25 = float(closing_over25)

        kickoff_dt = row["kickoff_dt"]
        if isinstance(kickoff_dt, str):
            kickoff_dt = datetime.fromisoformat(kickoff_dt)

        line_diff = abs(model_line - market_line)
        goal_diff = home_score - away_score
        adjusted = goal_diff + market_line

        # Model direction (used for settlement tracking, not as a signal itself)
        model_bets_home = model_line < market_line

        # Check if this match is in test period (should we bet?)
        is_test = cutoff_dt is None or kickoff_dt >= cutoff_dt

        if is_test:
            # --- Signal 1: League-conditional ---
            signal_s1 = _compute_s1_signal(league_key)

            # --- Signal 2: Price chain ---
            signal_chain = None
            chain_implied = None
            proxy_count = 0
            if league_id is not None and home_team_id is not None and away_team_id is not None:
                signal_chain, chain_implied, proxy_count = _compute_chain_signal(
                    fid,
                    int(home_team_id),
                    int(away_team_id),
                    int(league_id),
                    kickoff_dt,
                    model_line,
                    market_line,
                    fixtures_df,
                    ah_lines,
                )

            # --- Signal 3: Team trends (sliding window) ---
            signal_trend, home_pct, away_pct = _compute_trend_signal(
                home_team, away_team, team_accuracy
            )

            # --- Signal 4: Totals agreement ---
            signal_totals = _compute_totals_signal(prob_over25, closing_over25)

            # Collect signals, filter nulls
            signals = [signal_s1, signal_chain, signal_trend, signal_totals]
            active_signals = [s for s in signals if s is not None]
            signals_available = len(active_signals)

            # Check minimum signals requirement
            if signals_available < min_signals:
                skipped += 1
                # Still update team accuracy below
                _update_team_accuracy(
                    team_accuracy, home_team, away_team,
                    model_bets_home, adjusted,
                )
                continue

            # --- Majority vote (tie = no bet) ---
            model_votes = sum(1 for s in active_signals if s == "model")
            market_votes = sum(1 for s in active_signals if s == "market")

            if model_votes == market_votes:
                # Tie -> no bet (matches dashboard behavior)
                skipped += 1
                _update_team_accuracy(
                    team_accuracy, home_team, away_team,
                    model_bets_home, adjusted,
                )
                continue

            if model_votes > market_votes:
                decision = "model"
            else:
                decision = "market"

            # Determine which side to bet
            if decision == "model":
                bet_home = model_bets_home
            else:
                bet_home = not model_bets_home

            bet_side = "home" if bet_home else "away"

            # --- EV edge filter ---
            edge = None
            ah_home_prob = row.get("ah_home_prob")
            ah_away_prob = row.get("ah_away_prob")
            ah_closing_home_odds = row.get("ah_closing_home")
            ah_closing_away_odds = row.get("ah_closing_away")

            if bet_side == "home" and ah_home_prob is not None and ah_closing_home_odds is not None:
                edge = float(ah_home_prob) * float(ah_closing_home_odds) - 1
            elif bet_side == "away" and ah_away_prob is not None and ah_closing_away_odds is not None:
                edge = float(ah_away_prob) * float(ah_closing_away_odds) - 1

            if min_edge > 0 and (edge is None or edge < min_edge):
                skipped += 1
                _update_team_accuracy(
                    team_accuracy, home_team, away_team,
                    model_bets_home, adjusted,
                )
                continue

            # --- Settlement ---
            if adjusted == 0:
                result = "push"
                pnl = 0.0
            elif bet_home:
                if adjusted > 0:
                    result = "win"
                    pnl = stake
                else:
                    result = "loss"
                    pnl = -stake
            else:
                if adjusted < 0:
                    result = "win"
                    pnl = stake
                else:
                    result = "loss"
                    pnl = -stake

            bet = StrategyBet(
                fixture_id=fid,
                kickoff_at=kickoff,
                league_key=league_key,
                home_team=home_team,
                away_team=away_team,
                home_score=home_score,
                away_score=away_score,
                model_line=model_line,
                market_line=market_line,
                line_diff=round(line_diff, 4),
                signal_s1=signal_s1,
                signal_chain=signal_chain,
                chain_implied_ah=round(chain_implied, 4) if chain_implied is not None else None,
                proxy_count=proxy_count,
                signal_trend=signal_trend,
                home_team_model_pct=round(home_pct, 1) if home_pct is not None else None,
                away_team_model_pct=round(away_pct, 1) if away_pct is not None else None,
                signal_totals=signal_totals,
                model_votes=model_votes,
                market_votes=market_votes,
                signals_available=signals_available,
                decision=decision,
                bet_side=bet_side,
                adjusted=round(adjusted, 2),
                result=result,
                pnl=round(pnl, 2),
                edge=round(edge, 4) if edge is not None else None,
            )
            bets.append(bet)

            # --- Per-signal accuracy tracking ---
            # For each active signal, did its vote match the actual outcome?
            # "correct" means: if the signal voted X, and following X would have won
            if result != "push":
                bet_won = result == "win"
                for sig_name, sig_val in [
                    ("s1", signal_s1),
                    ("s2", signal_chain),
                    ("s3", signal_trend),
                    ("s4", signal_totals),
                ]:
                    if sig_val is None:
                        continue
                    # Would following this signal's vote have won?
                    if sig_val == decision:
                        # Signal agreed with the actual decision
                        signal_correct = bet_won
                    else:
                        # Signal disagreed with the actual decision
                        signal_correct = not bet_won
                    if signal_correct:
                        signal_accuracy[sig_name]["correct"] += 1
                    else:
                        signal_accuracy[sig_name]["incorrect"] += 1

            # --- Per-combination tracking ---
            pattern = ",".join(
                s if s is not None else "-"
                for s in [signal_s1, signal_chain, signal_trend, signal_totals]
            )
            if pattern not in combo_stats:
                combo_stats[pattern] = {
                    "bets": 0, "wins": 0, "losses": 0, "pushes": 0, "pnl": 0.0
                }
            combo_stats[pattern]["bets"] += 1
            if result == "win":
                combo_stats[pattern]["wins"] += 1
            elif result == "loss":
                combo_stats[pattern]["losses"] += 1
            else:
                combo_stats[pattern]["pushes"] += 1
            combo_stats[pattern]["pnl"] += pnl

        # ALWAYS update team accuracy counters (warmup + test), for future matches
        _update_team_accuracy(
            team_accuracy, home_team, away_team,
            model_bets_home, adjusted,
        )

    # 6. Aggregate
    total_bets = len(bets)
    wins = sum(1 for b in bets if b.result == "win")
    losses = sum(1 for b in bets if b.result == "loss")
    pushes = sum(1 for b in bets if b.result == "push")
    total_pnl = sum(b.pnl for b in bets)
    total_staked = (wins + losses) * stake  # pushes don't count
    roi = total_pnl / total_staked if total_staked > 0 else 0.0
    hit_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.0

    # By decision
    by_decision: dict[str, dict] = {}
    for dec in ("model", "market"):
        dec_bets = [b for b in bets if b.decision == dec]
        dec_wins = sum(1 for b in dec_bets if b.result == "win")
        dec_losses = sum(1 for b in dec_bets if b.result == "loss")
        dec_pushes = sum(1 for b in dec_bets if b.result == "push")
        dec_decided = dec_wins + dec_losses
        by_decision[dec] = {
            "bets": len(dec_bets),
            "wins": dec_wins,
            "losses": dec_losses,
            "pushes": dec_pushes,
            "pnl": round(sum(b.pnl for b in dec_bets), 2),
            "hit_rate": round(dec_wins / dec_decided, 4) if dec_decided > 0 else 0.0,
        }

    # By signal count (how many of the 4 signals were active)
    by_signal_count: dict[str, dict] = {}
    for label, target_count in [
        ("4_signals", 4),
        ("3_signals", 3),
        ("2_signals", 2),
        ("1_signal", 1),
    ]:
        sg_bets = [b for b in bets if b.signals_available == target_count]
        sg_wins = sum(1 for b in sg_bets if b.result == "win")
        sg_losses = sum(1 for b in sg_bets if b.result == "loss")
        sg_decided = sg_wins + sg_losses
        by_signal_count[label] = {
            "bets": len(sg_bets),
            "wins": sg_wins,
            "losses": sg_losses,
            "pnl": round(sum(b.pnl for b in sg_bets), 2),
            "hit_rate": round(sg_wins / sg_decided, 4) if sg_decided > 0 else 0.0,
        }

    # By league
    by_league: dict[str, dict] = {}
    league_keys = sorted(set(b.league_key for b in bets))
    for lk in league_keys:
        lg_bets = [b for b in bets if b.league_key == lk]
        lg_wins = sum(1 for b in lg_bets if b.result == "win")
        lg_losses = sum(1 for b in lg_bets if b.result == "loss")
        lg_decided = lg_wins + lg_losses
        by_league[lk] = {
            "bets": len(lg_bets),
            "wins": lg_wins,
            "losses": lg_losses,
            "pnl": round(sum(b.pnl for b in lg_bets), 2),
            "hit_rate": round(lg_wins / lg_decided, 4) if lg_decided > 0 else 0.0,
        }

    # Per-signal accuracy
    by_signal: dict[str, dict] = {}
    for sig_name in ("s1", "s2", "s3", "s4"):
        c = signal_accuracy[sig_name]["correct"]
        i = signal_accuracy[sig_name]["incorrect"]
        fired = c + i
        by_signal[sig_name] = {
            "fired": fired,
            "correct": c,
            "incorrect": i,
            "pct": round(c / fired * 100, 1) if fired > 0 else 0.0,
        }

    # Per-combination (round pnl)
    by_combination: dict[str, dict] = {}
    for pattern, stats in combo_stats.items():
        decided = stats["wins"] + stats["losses"]
        by_combination[pattern] = {
            **stats,
            "pnl": round(stats["pnl"], 2),
            "hit_rate": round(stats["wins"] / decided, 4) if decided > 0 else 0.0,
        }

    return StrategyResult(
        model_type=model_type,
        total_fixtures=len(preds_df) + (total_with_odds - total_disagree),
        fixtures_with_odds=total_with_odds,
        fixtures_with_disagreement=total_disagree,
        total_bets=total_bets,
        skipped=skipped,
        wins=wins,
        losses=losses,
        pushes=pushes,
        total_pnl=round(total_pnl, 2),
        roi=round(roi, 4),
        hit_rate=round(hit_rate, 4),
        stake_per_bet=stake,
        bets=bets,
        by_decision=by_decision,
        by_signal_count=by_signal_count,
        by_league=by_league,
        by_signal=by_signal,
        by_combination=by_combination,
    )

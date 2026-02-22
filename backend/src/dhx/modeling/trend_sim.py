"""Team-trend + implied-odds strategy simulation for AH betting.

Decision logic:
  1. Team trends first (primary signal):
     - Both teams favor model → bet model side
     - Both teams favor market → bet market side
  2. If mixed/50-50 → check price chain implied AH:
     - Closer to model → bet model side
     - Closer to market → bet market side
     - No implied available → trust market
  3. If no trend data (< min matches per team) → skip

Usage (via runner.py CLI):
    python -m dhx.modeling trend-sim \\
        --test-cutoff 2026-01-15 --stake 100
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import polars as pl

from dhx.db import get_client

logger = logging.getLogger(__name__)

HFA = 0.5  # Home field advantage in AH terms
LOOKBACK_DAYS = 60
TREND_MIN_MATCHES = 3


# ======================================================================
# Data structures
# ======================================================================


@dataclass
class TrendBet:
    """A single simulated bet from the trend strategy."""

    fixture_id: int
    kickoff_at: str
    league_key: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    model_line: float
    market_line: float
    # Signal info
    reason: str  # "chain→market", "trend→model", "trend→market"
    chain_implied: float | None
    trend_signal: str | None
    home_model_pct: float | None
    away_model_pct: float | None
    # Decision
    decision: str  # "model" or "market"
    bet_side: str  # "home" or "away"
    bet_team: str
    # Settlement
    adjusted: float
    result: str  # "win", "loss", "push"
    pnl: float


@dataclass
class TrendResult:
    """Aggregated trend simulation output."""

    total_fixtures: int
    fixtures_with_odds: int
    fixtures_with_disagreement: int
    total_bets: int
    skipped_no_trend: int
    skipped_chain_market: int  # skipped because implied was closer to market
    wins: int
    losses: int
    pushes: int
    total_pnl: float
    roi: float
    hit_rate: float
    stake_per_bet: float
    bets: list[TrendBet]
    by_reason: dict[str, dict]
    by_league: dict[str, dict]
    by_league_reason: dict[str, dict[str, dict]]  # league -> reason -> stats


# ======================================================================
# Data loading
# ======================================================================


def _load_predictions() -> pl.DataFrame:
    """Load finished predictions with AH lines from vw_predictions_summary."""
    client = get_client()
    all_rows: list[dict] = []
    for offset in range(0, 10000, 1000):
        data = (
            client.schema("dhx").table("vw_predictions_summary")
            .select(
                "fixture_id,kickoff_at,status,league_key,league_name,"
                "home_team,away_team,home_score,away_score,"
                "ah_fair_line,ah_closing_line"
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
    for offset in range(0, 10000, 1000):
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
# Price chain (implied odds from common opponents)
# ======================================================================


def _compute_chain_implied(
    target_fixture_id: int,
    home_team_id: int,
    away_team_id: int,
    league_id: int,
    kickoff_dt: datetime,
    fixtures_df: pl.DataFrame,
    ah_lines: dict[int, float],
) -> float | None:
    """Compute HFA-adjusted implied AH from common opponents.

    Returns implied AH line (home perspective), or None if no proxy data.
    """
    cutoff = kickoff_dt - timedelta(days=LOOKBACK_DAYS)

    league_fix = fixtures_df.filter(
        (pl.col("league_id") == league_id)
        & (pl.col("kickoff_at") < kickoff_dt)
        & (pl.col("kickoff_at") >= cutoff)
        & (pl.col("id") != target_fixture_id)
    )

    if len(league_fix) == 0:
        return None

    team_a_matches: list[dict] = []
    team_b_matches: list[dict] = []

    for row in league_fix.iter_rows(named=True):
        htid = int(row["home_team_id"])
        atid = int(row["away_team_id"])
        if htid == home_team_id or atid == home_team_id:
            team_a_matches.append(row)
        if htid == away_team_id or atid == away_team_id:
            team_b_matches.append(row)

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
    if not proxy_ids:
        return None

    implied_ahs: list[float] = []
    for pid in proxy_ids:
        vs_a = a_opponents[pid]
        vs_b = b_opponents[pid]

        ah_a = ah_lines.get(int(vs_a["id"]))
        ah_b = ah_lines.get(int(vs_b["id"]))

        if ah_a is None or ah_b is None:
            continue

        if vs_a["home_team_id"] == home_team_id:
            team_a_vs_proxy = -ah_a - HFA
        else:
            team_a_vs_proxy = ah_a + HFA

        if vs_b["home_team_id"] == away_team_id:
            team_b_vs_proxy = -ah_b - HFA
        else:
            team_b_vs_proxy = ah_b + HFA

        neutral_diff = team_a_vs_proxy - team_b_vs_proxy
        implied_ah = -(neutral_diff + HFA)
        implied_ahs.append(implied_ah)

    if not implied_ahs:
        return None

    return sum(implied_ahs) / len(implied_ahs)


# ======================================================================
# Team trend logic
# ======================================================================


def _compute_trend_signal(
    home_team: str,
    away_team: str,
    team_accuracy: dict[str, dict[str, int]],
) -> tuple[str | None, float | None, float | None, int, int]:
    """Compute team trend signal.

    Returns (signal, home_model_pct, away_model_pct, home_decided, away_decided).
    """
    home_stats = team_accuracy.get(home_team, {"model": 0, "market": 0})
    away_stats = team_accuracy.get(away_team, {"model": 0, "market": 0})

    home_decided = home_stats["model"] + home_stats["market"]
    away_decided = away_stats["model"] + away_stats["market"]

    if home_decided < TREND_MIN_MATCHES or away_decided < TREND_MIN_MATCHES:
        return None, None, None, home_decided, away_decided

    home_model_pct = home_stats["model"] / home_decided * 100
    away_model_pct = away_stats["model"] / away_decided * 100

    home_favors_model = home_model_pct > 50
    away_favors_model = away_model_pct > 50

    if home_favors_model and away_favors_model:
        return "model", home_model_pct, away_model_pct, home_decided, away_decided
    elif not home_favors_model and not away_favors_model:
        return "market", home_model_pct, away_model_pct, home_decided, away_decided
    else:
        total_model = home_stats["model"] + away_stats["model"]
        total_decided = home_decided + away_decided
        combined_pct = total_model / total_decided * 100
        if combined_pct >= 50:
            return "model", home_model_pct, away_model_pct, home_decided, away_decided
        else:
            return "market", home_model_pct, away_model_pct, home_decided, away_decided


def _update_team_accuracy(
    team_accuracy: dict[str, dict[str, int]],
    home_team: str,
    away_team: str,
    model_bets_home: bool,
    adjusted: float,
) -> None:
    """Update running team accuracy counters after a match settles."""
    if adjusted == 0:
        return

    if model_bets_home:
        model_profitable = adjusted > 0
    else:
        model_profitable = adjusted < 0

    result = "model" if model_profitable else "market"

    for team in (home_team, away_team):
        if team not in team_accuracy:
            team_accuracy[team] = {"model": 0, "market": 0}
        team_accuracy[team][result] += 1


# ======================================================================
# Main simulation
# ======================================================================


def run_trend_sim(
    test_cutoff: str | None = None,
    stake: float = 100.0,
    min_trend_matches: int = 3,
    only_leagues: list[str] | None = None,
    exclude_leagues: list[str] | None = None,
) -> TrendResult:
    """Run the trend + implied odds strategy simulation.

    Decision priority:
      1. Team trends first:
         - Both teams favor model → bet model side
         - Both teams favor market → bet market side
      2. If mixed/50-50 → check price chain implied AH:
         - Closer to model → bet model side
         - Closer to market → bet market side
         - No implied available → trust market
      3. If no trend data (< min matches) → skip
    """
    global TREND_MIN_MATCHES
    TREND_MIN_MATCHES = min_trend_matches

    # 1. Load data
    logger.info("Loading predictions...")
    preds_df = _load_predictions()
    if len(preds_df) == 0:
        raise ValueError("No predictions found")

    total_fixtures = len(preds_df)

    # Filter by league if specified
    if only_leagues:
        preds_df = preds_df.filter(pl.col("league_key").is_in(only_leagues))
        logger.info(f"Filtered to leagues: {only_leagues} → {len(preds_df)} fixtures")
    if exclude_leagues:
        preds_df = preds_df.filter(~pl.col("league_key").is_in(exclude_leagues))
        logger.info(f"Excluded leagues: {exclude_leagues} → {len(preds_df)} fixtures")

    logger.info("Loading fixtures...")
    fixtures_df = _load_fixtures()

    logger.info("Loading AH odds lines...")
    ah_lines = _load_ah_lines()

    # 2. Filter
    preds_df = preds_df.filter(
        pl.col("ah_fair_line").is_not_null()
        & pl.col("ah_closing_line").is_not_null()
        & pl.col("home_score").is_not_null()
        & pl.col("away_score").is_not_null()
    )
    total_with_odds = len(preds_df)

    preds_df = preds_df.filter(
        pl.col("ah_fair_line") != pl.col("ah_closing_line")
    )
    total_disagree = len(preds_df)

    logger.info(f"Found {total_with_odds} with odds, {total_disagree} with disagreement")

    cutoff_dt = None
    if test_cutoff:
        cutoff_dt = datetime.strptime(test_cutoff, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    # Parse dates
    preds_df = preds_df.with_columns(
        pl.col("kickoff_at").str.to_datetime(time_zone="UTC", strict=False).alias("kickoff_dt")
    )

    # Join fixture info
    fixture_info = fixtures_df.select(
        pl.col("id").alias("fixture_id"),
        pl.col("league_id"),
        pl.col("home_team_id"),
        pl.col("away_team_id"),
    )
    preds_df = preds_df.with_columns(pl.col("fixture_id").cast(pl.Int64))
    fixture_info = fixture_info.with_columns(pl.col("fixture_id").cast(pl.Int64))
    preds_df = preds_df.join(fixture_info, on="fixture_id", how="left")

    # Parse fixture dates for price chain lookback
    fixtures_df = fixtures_df.with_columns(
        pl.col("kickoff_at").str.to_datetime(time_zone="UTC", strict=False).alias("kickoff_at")
    )

    preds_df = preds_df.sort("kickoff_dt")

    if cutoff_dt:
        test_count = len(preds_df.filter(pl.col("kickoff_dt") >= cutoff_dt))
        warmup_count = len(preds_df) - test_count
        logger.info(f"Warmup: {warmup_count}, Test: {test_count}")

    # 3. Run simulation
    team_accuracy: dict[str, dict[str, int]] = {}
    bets: list[TrendBet] = []
    skipped_no_trend = 0
    skipped_chain_market = 0

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
        kickoff_dt_val = row["kickoff_dt"]

        if isinstance(kickoff_dt_val, str):
            kickoff_dt_val = datetime.fromisoformat(kickoff_dt_val)

        goal_diff = home_score - away_score
        adjusted = goal_diff + market_line
        model_bets_home = model_line < market_line

        is_test = cutoff_dt is None or kickoff_dt_val >= cutoff_dt

        if is_test:
            league_id = row.get("league_id")
            home_team_id = row.get("home_team_id")
            away_team_id = row.get("away_team_id")

            # --- Step 1: Team trends (primary signal) ---
            trend_signal, home_pct, away_pct, home_decided, away_decided = _compute_trend_signal(
                home_team, away_team, team_accuracy
            )

            if trend_signal is None:
                # Not enough trend data → skip
                skipped_no_trend += 1
                _update_team_accuracy(
                    team_accuracy, home_team, away_team,
                    model_bets_home, adjusted,
                )
                continue

            # Check if trends are unanimous or mixed
            home_stats = team_accuracy.get(home_team, {"model": 0, "market": 0})
            away_stats = team_accuracy.get(away_team, {"model": 0, "market": 0})
            home_model_pct = home_stats["model"] / home_decided if home_decided > 0 else 0
            away_model_pct = away_stats["model"] / away_decided if away_decided > 0 else 0
            home_favors_model = home_model_pct > 0.5
            away_favors_model = away_model_pct > 0.5
            trends_unanimous = (home_favors_model and away_favors_model) or (not home_favors_model and not away_favors_model)

            # Compute chain implied (needed for mixed case)
            chain_implied = None
            if league_id is not None and home_team_id is not None and away_team_id is not None:
                chain_implied = _compute_chain_implied(
                    fid, int(home_team_id), int(away_team_id),
                    int(league_id), kickoff_dt_val,
                    fixtures_df, ah_lines,
                )

            if trends_unanimous:
                # Both teams agree → follow trends directly
                decision = trend_signal
                reason = f"trend→{decision}"
                if decision == "model":
                    bet_home = model_bets_home
                else:
                    bet_home = not model_bets_home
            else:
                # Mixed/50-50 → use implied to break tie
                if chain_implied is not None:
                    dist_to_model = abs(chain_implied - model_line)
                    dist_to_market = abs(chain_implied - market_line)
                    if dist_to_model < dist_to_market:
                        decision = "model"
                        reason = "mixed→chain→model"
                        bet_home = model_bets_home
                    else:
                        decision = "market"
                        reason = "mixed→chain→market"
                        bet_home = not model_bets_home
                else:
                    # No implied → trust model
                    decision = "model"
                    reason = "mixed→no_chain→model"
                    bet_home = model_bets_home

            bet_side = "home" if bet_home else "away"
            bet_team = home_team if bet_home else away_team

            # Settlement
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

            bets.append(
                TrendBet(
                    fixture_id=fid,
                    kickoff_at=kickoff,
                    league_key=league_key,
                    home_team=home_team,
                    away_team=away_team,
                    home_score=home_score,
                    away_score=away_score,
                    model_line=model_line,
                    market_line=market_line,
                    reason=reason,
                    chain_implied=round(chain_implied, 4) if chain_implied is not None else None,
                    trend_signal=trend_signal,
                    home_model_pct=round(home_model_pct * 100, 1),
                    away_model_pct=round(away_model_pct * 100, 1),
                    decision=decision,
                    bet_side=bet_side,
                    bet_team=bet_team,
                    adjusted=round(adjusted, 2),
                    result=result,
                    pnl=round(pnl, 2),
                )
            )

        # ALWAYS update team accuracy (warmup + test)
        _update_team_accuracy(
            team_accuracy, home_team, away_team,
            model_bets_home, adjusted,
        )

    # 4. Aggregate
    total_bets = len(bets)
    wins = sum(1 for b in bets if b.result == "win")
    losses = sum(1 for b in bets if b.result == "loss")
    pushes = sum(1 for b in bets if b.result == "push")
    total_pnl = sum(b.pnl for b in bets)
    total_staked = (wins + losses) * stake
    roi = total_pnl / total_staked if total_staked > 0 else 0.0
    hit_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.0

    # By reason
    by_reason: dict[str, dict] = {}
    for reason_key in sorted(set(b.reason for b in bets)):
        r_bets = [b for b in bets if b.reason == reason_key]
        r_wins = sum(1 for b in r_bets if b.result == "win")
        r_losses = sum(1 for b in r_bets if b.result == "loss")
        r_decided = r_wins + r_losses
        by_reason[reason_key] = {
            "bets": len(r_bets),
            "wins": r_wins,
            "losses": r_losses,
            "pushes": sum(1 for b in r_bets if b.result == "push"),
            "pnl": round(sum(b.pnl for b in r_bets), 2),
            "hit_rate": round(r_wins / r_decided, 4) if r_decided > 0 else 0.0,
        }

    # By league
    by_league: dict[str, dict] = {}
    for lk in sorted(set(b.league_key for b in bets)):
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

    # By league + reason
    by_league_reason: dict[str, dict[str, dict]] = {}
    for lk in sorted(set(b.league_key for b in bets)):
        by_league_reason[lk] = {}
        lg_bets = [b for b in bets if b.league_key == lk]
        for reason_key in sorted(set(b.reason for b in lg_bets)):
            r_bets = [b for b in lg_bets if b.reason == reason_key]
            r_wins = sum(1 for b in r_bets if b.result == "win")
            r_losses = sum(1 for b in r_bets if b.result == "loss")
            r_decided = r_wins + r_losses
            by_league_reason[lk][reason_key] = {
                "bets": len(r_bets),
                "wins": r_wins,
                "losses": r_losses,
                "pnl": round(sum(b.pnl for b in r_bets), 2),
                "hit_rate": round(r_wins / r_decided, 4) if r_decided > 0 else 0.0,
            }

    return TrendResult(
        total_fixtures=total_fixtures,
        fixtures_with_odds=total_with_odds,
        fixtures_with_disagreement=total_disagree,
        total_bets=total_bets,
        skipped_no_trend=skipped_no_trend,
        skipped_chain_market=skipped_chain_market,
        wins=wins,
        losses=losses,
        pushes=pushes,
        total_pnl=round(total_pnl, 2),
        roi=round(roi, 4),
        hit_rate=round(hit_rate, 4),
        stake_per_bet=stake,
        bets=bets,
        by_reason=by_reason,
        by_league=by_league,
        by_league_reason=by_league_reason,
    )

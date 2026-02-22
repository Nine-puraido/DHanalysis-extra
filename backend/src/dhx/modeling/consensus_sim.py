"""Consensus simulation: Model-Market agreement analysis.

When model and market agree on AH direction (same side favored):
1. AH coverage: Does the consensus favorite cover the spread?
2. O/U disagreement: Among agreed-AH matches where model & market
   disagree on O/U, does following market or model win more often?

Usage (via runner.py CLI):
    python -m dhx.modeling consensus-sim
    python -m dhx.modeling consensus-sim --test-cutoff 2026-01-15
    python -m dhx.modeling consensus-sim --only-leagues EPL,LL
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from dhx.db import get_client

logger = logging.getLogger(__name__)


# ======================================================================
# Data structures
# ======================================================================


@dataclass
class LeagueAHStats:
    """AH coverage stats for one league."""

    fav_covers: int = 0
    dog_covers: int = 0
    pushes: int = 0

    @property
    def total(self) -> int:
        return self.fav_covers + self.dog_covers + self.pushes

    @property
    def fav_cover_rate(self) -> float:
        decided = self.fav_covers + self.dog_covers
        return self.fav_covers / decided if decided > 0 else 0.0


@dataclass
class LeagueOUStats:
    """O/U disagreement stats for one league."""

    model_correct: int = 0
    market_correct: int = 0

    @property
    def total(self) -> int:
        return self.model_correct + self.market_correct

    @property
    def model_win_rate(self) -> float:
        return self.model_correct / self.total if self.total > 0 else 0.0


@dataclass
class ConsensusResult:
    """Aggregated consensus simulation output."""

    # Counts
    total_finished: int
    total_with_ah: int
    total_agreed: int

    # AH coverage (overall)
    fav_covers: int
    dog_covers: int
    pushes: int
    fav_cover_rate: float

    # AH coverage by league
    ah_by_league: dict[str, LeagueAHStats]

    # O/U disagreement (overall)
    total_with_ou: int
    disagree_count: int
    model_correct: int
    market_correct: int
    model_win_rate: float

    # O/U by league
    ou_by_league: dict[str, LeagueOUStats]


# ======================================================================
# Data loading
# ======================================================================


def _load_predictions_for_consensus() -> list[dict]:
    """Load finished predictions with AH + O/U data from vw_predictions_summary."""
    client = get_client()
    all_rows: list[dict] = []
    page_size = 1000

    for offset in range(0, 10000, page_size):
        data = (
            client.schema("dhx").table("vw_predictions_summary")
            .select(
                "fixture_id,kickoff_at,status,league_key,league_name,"
                "home_team,away_team,home_score,away_score,"
                "ah_fair_line,ah_closing_line,"
                "prob_over25,closing_over25"
            )
            .eq("status", "finished")
            .order("kickoff_at")
            .range(offset, offset + page_size - 1)
            .execute()
            .data
        )
        if data:
            all_rows.extend(data)
        if not data or len(data) < page_size:
            break

    logger.info(f"Loaded {len(all_rows)} finished predictions")
    return all_rows


# ======================================================================
# Simulation
# ======================================================================


def run_consensus_sim(
    test_cutoff: str | None = None,
    only_leagues: list[str] | None = None,
    exclude_leagues: list[str] | None = None,
) -> ConsensusResult:
    """Run the consensus simulation.

    Args:
        test_cutoff: Only analyze fixtures after this date (YYYY-MM-DD).
        only_leagues: Comma-separated league keys to include.
        exclude_leagues: Comma-separated league keys to exclude.

    Returns:
        ConsensusResult with AH coverage and O/U disagreement analysis.
    """
    rows = _load_predictions_for_consensus()
    total_finished = len(rows)

    # Apply cutoff filter
    if test_cutoff:
        rows = [r for r in rows if (r.get("kickoff_at") or "") >= test_cutoff]
        logger.info(f"After cutoff {test_cutoff}: {len(rows)} fixtures")

    # Apply league filters
    if only_leagues:
        rows = [r for r in rows if r.get("league_key") in only_leagues]
        logger.info(f"After only_leagues filter: {len(rows)} fixtures")
    if exclude_leagues:
        rows = [r for r in rows if r.get("league_key") not in exclude_leagues]
        logger.info(f"After exclude_leagues filter: {len(rows)} fixtures")

    # Filter to fixtures with both AH lines and results
    ah_rows = [
        r for r in rows
        if r.get("ah_fair_line") is not None
        and r.get("ah_closing_line") is not None
        and r.get("home_score") is not None
        and r.get("away_score") is not None
    ]
    total_with_ah = len(ah_rows)

    # ---- AH Agreement Filter ----
    # Agree = sign(ah_fair_line) == sign(ah_closing_line)
    # Skip if either is exactly 0
    agreed_rows = []
    for r in ah_rows:
        fair = float(r["ah_fair_line"])
        closing = float(r["ah_closing_line"])
        if fair == 0.0 or closing == 0.0:
            continue
        if math.copysign(1, fair) == math.copysign(1, closing):
            agreed_rows.append(r)

    total_agreed = len(agreed_rows)
    logger.info(
        f"AH agreement: {total_agreed}/{total_with_ah} fixtures "
        f"({total_agreed / total_with_ah * 100:.1f}%)" if total_with_ah > 0 else
        f"AH agreement: 0 fixtures"
    )

    # ---- Analysis 1: Favorite vs Underdog Coverage ----
    overall_ah = LeagueAHStats()
    ah_by_league: dict[str, LeagueAHStats] = {}

    for r in agreed_rows:
        closing = float(r["ah_closing_line"])
        home_score = int(r["home_score"])
        away_score = int(r["away_score"])
        league_key = r.get("league_key", "?")

        if league_key not in ah_by_league:
            ah_by_league[league_key] = LeagueAHStats()

        # Favorite = team both agree is favored
        # Negative AH line = home is favorite
        fav_is_home = closing < 0

        # Settle at closing line
        adjusted = (home_score - away_score) + closing

        if fav_is_home:
            # Home is favorite: adjusted > 0 means home covers (fav covers)
            if adjusted > 0:
                overall_ah.fav_covers += 1
                ah_by_league[league_key].fav_covers += 1
            elif adjusted < 0:
                overall_ah.dog_covers += 1
                ah_by_league[league_key].dog_covers += 1
            else:
                overall_ah.pushes += 1
                ah_by_league[league_key].pushes += 1
        else:
            # Away is favorite: adjusted < 0 means away covers (fav covers)
            if adjusted < 0:
                overall_ah.fav_covers += 1
                ah_by_league[league_key].fav_covers += 1
            elif adjusted > 0:
                overall_ah.dog_covers += 1
                ah_by_league[league_key].dog_covers += 1
            else:
                overall_ah.pushes += 1
                ah_by_league[league_key].pushes += 1

    # ---- Analysis 2: O/U Disagreement ----
    # Filter agreed rows that also have O/U data
    ou_rows = [
        r for r in agreed_rows
        if r.get("prob_over25") is not None
        and r.get("closing_over25") is not None
    ]
    total_with_ou = len(ou_rows)

    overall_ou = LeagueOUStats()
    ou_by_league: dict[str, LeagueOUStats] = {}
    disagree_count = 0

    for r in ou_rows:
        prob_over = float(r["prob_over25"])
        closing_over_odds = float(r["closing_over25"])
        home_score = int(r["home_score"])
        away_score = int(r["away_score"])
        league_key = r.get("league_key", "?")
        total_goals = home_score + away_score

        # Model direction
        model_over = prob_over > 0.5

        # Market direction: closing_over25 is decimal odds for over 2.5
        # implied prob = 1 / odds; if implied > 0.5 then market says over
        if closing_over_odds <= 0:
            continue
        implied_over_prob = 1.0 / closing_over_odds
        market_over = implied_over_prob > 0.5

        # Only interested in disagreements
        if model_over == market_over:
            continue

        disagree_count += 1

        if league_key not in ou_by_league:
            ou_by_league[league_key] = LeagueOUStats()

        # Actual result
        actual_over = total_goals >= 3

        # Who was correct?
        model_right = (model_over == actual_over)
        market_right = (market_over == actual_over)

        if model_right:
            overall_ou.model_correct += 1
            ou_by_league[league_key].model_correct += 1
        if market_right:
            overall_ou.market_correct += 1
            ou_by_league[league_key].market_correct += 1

    return ConsensusResult(
        total_finished=total_finished,
        total_with_ah=total_with_ah,
        total_agreed=total_agreed,
        fav_covers=overall_ah.fav_covers,
        dog_covers=overall_ah.dog_covers,
        pushes=overall_ah.pushes,
        fav_cover_rate=overall_ah.fav_cover_rate,
        ah_by_league=ah_by_league,
        total_with_ou=total_with_ou,
        disagree_count=disagree_count,
        model_correct=overall_ou.model_correct,
        market_correct=overall_ou.market_correct,
        model_win_rate=overall_ou.model_win_rate,
        ou_by_league=ou_by_league,
    )

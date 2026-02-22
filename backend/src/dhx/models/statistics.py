"""Models for match statistics and player match ratings."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from pydantic import BaseModel


class MatchStatistics(BaseModel):
    fixture_id: int

    # Expected goals
    home_xg: Decimal | None = None
    away_xg: Decimal | None = None

    # Shots
    home_shots: int | None = None
    away_shots: int | None = None
    home_shots_on_target: int | None = None
    away_shots_on_target: int | None = None
    home_shots_off_target: int | None = None
    away_shots_off_target: int | None = None
    home_blocked_shots: int | None = None
    away_blocked_shots: int | None = None

    # Possession
    home_possession: Decimal | None = None
    away_possession: Decimal | None = None

    # Set pieces
    home_corner_kicks: int | None = None
    away_corner_kicks: int | None = None
    home_offsides: int | None = None
    away_offsides: int | None = None

    # Discipline
    home_fouls: int | None = None
    away_fouls: int | None = None
    home_yellow_cards: int | None = None
    away_yellow_cards: int | None = None
    home_red_cards: int | None = None
    away_red_cards: int | None = None

    # Passing
    home_passes: int | None = None
    away_passes: int | None = None
    home_accurate_passes: int | None = None
    away_accurate_passes: int | None = None

    # Goalkeeper
    home_saves: int | None = None
    away_saves: int | None = None

    # Big chances
    home_big_chances: int | None = None
    away_big_chances: int | None = None
    home_big_chances_missed: int | None = None
    away_big_chances_missed: int | None = None

    # Catch-all
    extra_stats: dict[str, Any] | None = None


class PlayerMatchRating(BaseModel):
    fixture_id: int
    team_id: int
    source_player_id: str | None = None
    player_name: str
    player_short_name: str | None = None
    position: str | None = None
    jersey_number: int | None = None
    is_substitute: bool = False
    minutes_played: int | None = None
    rating: Decimal | None = None

    # Key individual stats
    goals: int = 0
    assists: int = 0
    shots_total: int | None = None
    shots_on_target: int | None = None
    passes_total: int | None = None
    passes_accurate: int | None = None
    key_passes: int | None = None
    tackles: int | None = None
    interceptions: int | None = None

    # Catch-all
    extra_stats: dict[str, Any] | None = None

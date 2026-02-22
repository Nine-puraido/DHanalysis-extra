"""Models for teams, fixtures, results, and source mappings."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class Team(BaseModel):
    id: int | None = None
    name: str
    short_name: str | None = None
    country: str | None = None


class TeamSourceMap(BaseModel):
    id: int | None = None
    team_id: int
    source_id: int
    source_team_id: str


class Fixture(BaseModel):
    id: int | None = None
    league_id: int
    season_id: int | None = None
    home_team_id: int
    away_team_id: int
    kickoff_at: datetime
    status: str = "scheduled"
    venue: str | None = None


class FixtureSourceMap(BaseModel):
    id: int | None = None
    fixture_id: int
    source_id: int
    source_event_id: str
    source_custom_id: str | None = None
    raw_path: str | None = None


class Result(BaseModel):
    fixture_id: int
    home_score: int
    away_score: int
    home_ht_score: int | None = None
    away_ht_score: int | None = None
    result_status: str = "final"
    settled_at: datetime | None = None

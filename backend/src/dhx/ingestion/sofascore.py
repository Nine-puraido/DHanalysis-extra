"""SofaScore ingestion worker.

Flow:
  1. Fetch scheduled events for a date
  2. Archive raw JSON to Storage
  3. Filter to tracked leagues, normalize teams/fixtures
  4. For each fixture, fetch odds, archive, normalize into odds_snapshots
  5. Track pipeline run and errors
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

from dhx import db, storage
from dhx.ingestion.sofascore_client import (
    TOURNAMENT_TO_LEAGUE,
    TRACKED_TOURNAMENT_IDS,
    fetch_event_odds,
    fetch_scheduled_events,
)

logger = logging.getLogger(__name__)

SOURCE_ID = 1  # sofascore in data_sources

# ── Market name mapping (SofaScore market name → our market key) ────────────
MARKET_MAP: dict[str, str] = {
    "full time": "1x2",
    "full time result": "1x2",
    "asian handicap": "ah",
    "asian handicap, including overtime": "ah",
    "total": "totals",
    "total goals": "totals",
    "match goals": "totals",
    "over/under": "totals",
    "both teams to score": "btts",
    "both teams score": "btts",
}

# ── Selection normalization ──────────────────────────────────────────────────
SELECTION_1X2 = {"1": "home", "x": "draw", "2": "away"}
SELECTION_BTTS = {"yes": "yes", "no": "no"}


def _parse_fractional(frac: str) -> Decimal | None:
    """Convert fractional odds '7/4' to decimal 2.75 → (7/4)+1."""
    try:
        parts = frac.split("/")
        if len(parts) != 2:
            return None
        num, den = Decimal(parts[0]), Decimal(parts[1])
        if den == 0:
            return None
        return (num / den) + 1
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def _resolve_league_id(tournament_id: int) -> int | None:
    """Map SofaScore uniqueTournament ID to our leagues.id via league_key."""
    league_key = TOURNAMENT_TO_LEAGUE.get(tournament_id)
    if league_key is None:
        return None
    rows = db.select_rows("leagues", "id", {"league_key": league_key})
    return rows[0]["id"] if rows else None


# ── Caches (populated per-run) ───────────────────────────────────────────────
_league_cache: dict[int, int] = {}  # sofascore tournament_id → leagues.id
_team_cache: dict[int, int] = {}    # sofascore team_id → teams.id
_fixture_cache: dict[int, int] = {} # sofascore event_id → fixtures.id
_bookmaker_id: int | None = None


def _get_bookmaker_id() -> int:
    """Get our bookmaker ID for sofascore_avg."""
    global _bookmaker_id
    if _bookmaker_id is None:
        rows = db.select_rows("bookmakers", "id", {"bookmaker_key": "sofascore_avg"})
        _bookmaker_id = rows[0]["id"]
    return _bookmaker_id


def _get_league_id(tournament_id: int) -> int | None:
    if tournament_id not in _league_cache:
        lid = _resolve_league_id(tournament_id)
        if lid is not None:
            _league_cache[tournament_id] = lid
    return _league_cache.get(tournament_id)


def _upsert_team(sf_team: dict) -> int:
    """Upsert a team from SofaScore data, return teams.id."""
    sf_id = sf_team["id"]
    if sf_id in _team_cache:
        return _team_cache[sf_id]

    # Check if we already have a mapping for this source team
    existing = db.select_rows(
        "team_source_map", "team_id",
        {"source_id": SOURCE_ID, "source_team_id": str(sf_id)},
    )
    if existing:
        _team_cache[sf_id] = existing[0]["team_id"]
        return _team_cache[sf_id]

    # Upsert the team itself
    team_row = {
        "name": sf_team.get("name", f"Unknown-{sf_id}"),
        "short_name": sf_team.get("shortName"),
        "country": None,
    }
    # Try to get country from team.country or category
    if "country" in sf_team:
        team_row["country"] = sf_team["country"].get("name") if isinstance(sf_team["country"], dict) else sf_team["country"]

    inserted = db.insert_rows("teams", [team_row])
    team_id = inserted[0]["id"]

    # Create source mapping
    db.insert_rows("team_source_map", [{
        "team_id": team_id,
        "source_id": SOURCE_ID,
        "source_team_id": str(sf_id),
    }])
    _team_cache[sf_id] = team_id
    return team_id


def _upsert_fixture(event: dict, league_id: int) -> int | None:
    """Upsert a fixture from a SofaScore event, return fixtures.id."""
    sf_event_id = event["id"]
    if sf_event_id in _fixture_cache:
        return _fixture_cache[sf_event_id]

    # Check existing mapping
    existing = db.select_rows(
        "fixture_source_map", "fixture_id",
        {"source_id": SOURCE_ID, "source_event_id": str(sf_event_id)},
    )
    if existing:
        _fixture_cache[sf_event_id] = existing[0]["fixture_id"]
        return _fixture_cache[sf_event_id]

    home_team_id = _upsert_team(event["homeTeam"])
    away_team_id = _upsert_team(event["awayTeam"])
    kickoff = datetime.fromtimestamp(event["startTimestamp"], tz=timezone.utc)

    # Determine fixture status from SofaScore status
    sf_status = event.get("status", {})
    sf_type = sf_status.get("type", "notstarted")
    status_map = {
        "notstarted": "scheduled",
        "inprogress": "live",
        "finished": "finished",
        "postponed": "postponed",
        "canceled": "cancelled",
        "cancelled": "cancelled",
        "abandoned": "abandoned",
    }
    status = status_map.get(sf_type, "scheduled")

    venue_name = None
    if "venue" in event and event["venue"]:
        venue_name = event["venue"].get("stadium", {}).get("name") or event["venue"].get("name")

    fixture_row = {
        "league_id": league_id,
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "kickoff_at": kickoff.isoformat(),
        "status": status,
        "venue": venue_name,
    }
    inserted = db.insert_rows("fixtures", [fixture_row])
    fixture_id = inserted[0]["id"]

    # Create source mapping
    custom_id = event.get("customId")
    db.insert_rows("fixture_source_map", [{
        "fixture_id": fixture_id,
        "source_id": SOURCE_ID,
        "source_event_id": str(sf_event_id),
        "source_custom_id": custom_id,
    }])

    _fixture_cache[sf_event_id] = fixture_id
    return fixture_id


def _normalize_market(market_name: str) -> str | None:
    """Map a SofaScore market name to our market key."""
    key = market_name.strip().lower()
    return MARKET_MAP.get(key)


def _normalize_ah_choice(
    choice_name: str, choices: list[dict], choice_idx: int,
) -> tuple[str, Decimal | None]:
    """Normalize an Asian Handicap choice.

    SofaScore format: '(-0.25) Cagliari' or '(0.25) Lecce'
    First choice in the list = home, second = away.
    """
    line = None
    line_match = re.search(r"\(([+-]?\d+(?:\.\d+)?)\)", choice_name)
    if line_match:
        try:
            line = Decimal(line_match.group(1))
        except InvalidOperation:
            pass

    # SofaScore lists home first, away second
    selection = "home" if choice_idx == 0 else "away"
    return (selection, line)


def _normalize_odds(
    raw: dict[str, Any],
    fixture_id: int,
    pulled_at: datetime,
) -> list[dict]:
    """Convert SofaScore odds payload to list of odds_snapshot rows."""
    bookmaker_id = _get_bookmaker_id()
    snapshots: list[dict] = []

    markets = raw.get("markets", [])
    for mkt in markets:
        market_name = mkt.get("marketName", "")
        our_market = _normalize_market(market_name)
        if our_market is None:
            continue  # skip markets we don't track

        is_suspended = mkt.get("suspended", False)
        choices = mkt.get("choices", [])

        # For totals, the goal line is in choiceGroup (e.g., "2.5")
        choice_group = mkt.get("choiceGroup", "")
        market_line: Decimal | None = None
        if our_market == "totals" and choice_group:
            try:
                market_line = Decimal(choice_group)
            except InvalidOperation:
                pass

        for idx, choice in enumerate(choices):
            frac = choice.get("fractionalValue")
            if not frac:
                continue

            price = _parse_fractional(frac)
            if price is None or price <= 1:
                continue

            # Determine selection and line based on market type
            name_lower = choice.get("name", "").strip().lower()
            selection: str
            line: Decimal | None = None

            if our_market == "1x2":
                sel = SELECTION_1X2.get(name_lower)
                selection = sel or name_lower

            elif our_market == "ah":
                selection, line = _normalize_ah_choice(
                    choice.get("name", ""), choices, idx,
                )

            elif our_market == "totals":
                if "over" in name_lower:
                    selection = "over"
                elif "under" in name_lower:
                    selection = "under"
                else:
                    selection = name_lower
                line = market_line  # from choiceGroup

            elif our_market == "btts":
                sel = SELECTION_BTTS.get(name_lower)
                selection = sel or name_lower

            else:
                selection = name_lower

            implied_prob = Decimal(1) / price

            # Mark the standard 2.5 total and the main AH line as is_main
            is_main = False
            if our_market == "1x2" or our_market == "btts":
                is_main = True
            elif our_market == "totals" and market_line == Decimal("2.5"):
                is_main = True
            elif our_market == "ah":
                is_main = True  # SofaScore typically returns only the main AH line

            snapshots.append({
                "fixture_id": fixture_id,
                "source_id": SOURCE_ID,
                "bookmaker_id": bookmaker_id,
                "market": our_market,
                "selection": selection,
                "line": str(line) if line is not None else None,
                "price_decimal": str(price.quantize(Decimal("0.001"))),
                "implied_prob": str(implied_prob.quantize(Decimal("0.0001"))),
                "is_main": is_main,
                "is_suspended": is_suspended,
                "pulled_at": pulled_at.isoformat(),
            })

    return snapshots


def _log_error(run_id: int, error_type: str, message: str, ref: str | None = None) -> None:
    db.insert_rows("ingestion_errors", [{
        "source_id": SOURCE_ID,
        "run_id": run_id,
        "error_type": error_type,
        "error_message": message[:2000],
        "payload_ref": ref,
    }])


# ── Main pipeline entry point ────────────────────────────────────────────────

def run(date_str: str) -> dict[str, Any]:
    """Run the SofaScore ingestion pipeline for a single date.

    Returns a summary dict with counts.
    """
    now = datetime.now(timezone.utc)

    # Reset per-run caches
    _league_cache.clear()
    _team_cache.clear()
    _fixture_cache.clear()

    # Start pipeline run
    run_rows = db.insert_rows("pipeline_runs", [{
        "pipeline_name": f"sofascore_ingest_{date_str}",
        "started_at": now.isoformat(),
        "status": "running",
        "details_json": {"date": date_str},
    }])
    run_id = run_rows[0]["id"]

    stats = {
        "events_fetched": 0,
        "events_tracked": 0,
        "fixtures_upserted": 0,
        "odds_snapshots_inserted": 0,
        "errors": 0,
    }

    try:
        # 1. Fetch scheduled events
        raw_events = fetch_scheduled_events(date_str)
        events = raw_events.get("events", [])
        stats["events_fetched"] = len(events)

        # 2. Archive raw response
        storage.archive_raw_response(
            source_key="sofascore",
            endpoint_label="scheduled-events",
            identifier=date_str,
            payload=raw_events,
            ts=now,
        )

        # 3. Filter to tracked leagues (accept all events from tracked tournaments;
        #    SofaScore may include neighboring-date events — idempotent upsert handles dupes)
        tracked_events = []
        for ev in events:
            tournament = ev.get("tournament", {})
            unique_tournament = tournament.get("uniqueTournament", {})
            ut_id = unique_tournament.get("id")
            if ut_id in TRACKED_TOURNAMENT_IDS:
                tracked_events.append((ev, ut_id))

        stats["events_tracked"] = len(tracked_events)
        logger.info(
            "Date %s: %d total events, %d in tracked leagues for this date",
            date_str, len(events), len(tracked_events),
        )

        # 4. Upsert fixtures and fetch odds
        for ev, ut_id in tracked_events:
            sf_event_id = ev["id"]
            try:
                league_id = _get_league_id(ut_id)
                if league_id is None:
                    logger.warning("Could not resolve league for tournament %d", ut_id)
                    continue

                fixture_id = _upsert_fixture(ev, league_id)
                if fixture_id is None:
                    continue
                stats["fixtures_upserted"] += 1

                # Fetch odds for this event
                try:
                    raw_odds = fetch_event_odds(sf_event_id)
                except Exception as e:
                    logger.warning("No odds for event %d: %s", sf_event_id, e)
                    _log_error(run_id, "odds_fetch", str(e), f"event:{sf_event_id}")
                    stats["errors"] += 1
                    continue

                # Archive raw odds
                storage.archive_raw_response(
                    source_key="sofascore",
                    endpoint_label="event-odds",
                    identifier=str(sf_event_id),
                    payload=raw_odds,
                    ts=now,
                )

                # Normalize and insert odds
                odds_rows = _normalize_odds(raw_odds, fixture_id, now)
                if odds_rows:
                    db.insert_rows("odds_snapshots", odds_rows)
                    stats["odds_snapshots_inserted"] += len(odds_rows)

            except Exception as e:
                logger.error("Error processing event %d: %s", sf_event_id, e, exc_info=True)
                _log_error(run_id, "event_processing", str(e), f"event:{sf_event_id}")
                stats["errors"] += 1
                continue

        # Mark pipeline success
        ended = datetime.now(timezone.utc)
        db.get_client().schema("dhx").table("pipeline_runs").update({
            "status": "success",
            "ended_at": ended.isoformat(),
            "rows_written": stats["odds_snapshots_inserted"] + stats["fixtures_upserted"],
            "details_json": {**stats, "date": date_str},
        }).eq("id", run_id).execute()

    except Exception as e:
        logger.error("Pipeline failed for %s: %s", date_str, e, exc_info=True)
        _log_error(run_id, "pipeline_fatal", str(e))
        ended = datetime.now(timezone.utc)
        db.get_client().schema("dhx").table("pipeline_runs").update({
            "status": "failed",
            "ended_at": ended.isoformat(),
            "details_json": {**stats, "date": date_str, "fatal_error": str(e)},
        }).eq("id", run_id).execute()
        stats["errors"] += 1

    return stats

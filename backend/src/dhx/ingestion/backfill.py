"""Historical backfill pipeline — populate a full season of data from SofaScore.

Flow per league:
  1. Fetch current season ID via /unique-tournament/{id}/seasons
  2. Paginate through past events via /unique-tournament/{id}/season/{sid}/events/last/{page}
  3. For each finished event within the requested date range:
     a. Upsert team + fixture (reuse helpers from sofascore.py)
     b. Extract result from event object (no extra API call needed)
     c. Fetch + store closing odds   (skip if already present)
     d. Fetch + store match statistics (skip if already present)
     e. Fetch + store player lineups   (skip if already present)
  4. Track progress, errors, pipeline run

The pipeline is idempotent — safe to re-run. It checks for existing data
before making API calls, so interrupted runs can be resumed cheaply.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from dhx import db, storage
from dhx.ingestion.sofascore_client import (
    TOURNAMENT_TO_LEAGUE,
    fetch_event_lineups,
    fetch_event_odds,
    fetch_event_statistics,
    fetch_tournament_events_last,
    fetch_tournament_seasons,
)
from dhx.ingestion.sofascore import (
    SOURCE_ID,
    _get_bookmaker_id,
    _get_league_id,
    _league_cache,
    _team_cache,
    _fixture_cache,
    _upsert_team,
    _upsert_fixture,
    _normalize_odds,
    _log_error as _sofascore_log_error,
)

logger = logging.getLogger(__name__)

# ── Stat name mapping: SofaScore stat name (lowercase) → our column name ────
# Only "ALL" period stats are stored in dedicated columns.
# The key is matched with `in` so partial matches work.

_STAT_COLUMN_MAP: dict[str, tuple[str, str]] = {
    # stat_name_fragment → (home_column, away_column)
    "expected goals": ("home_xg", "away_xg"),
    "ball possession": ("home_possession", "away_possession"),
    "total shots": ("home_shots", "away_shots"),
    "shots on target": ("home_shots_on_target", "away_shots_on_target"),
    "shots off target": ("home_shots_off_target", "away_shots_off_target"),
    "blocked shots": ("home_blocked_shots", "away_blocked_shots"),
    "corner kicks": ("home_corner_kicks", "away_corner_kicks"),
    "offsides": ("home_offsides", "away_offsides"),
    "fouls": ("home_fouls", "away_fouls"),
    "yellow cards": ("home_yellow_cards", "away_yellow_cards"),
    "red cards": ("home_red_cards", "away_red_cards"),
    "total passes": ("home_passes", "away_passes"),
    "accurate passes": ("home_accurate_passes", "away_accurate_passes"),
    "goalkeeper saves": ("home_saves", "away_saves"),
    "big chances": ("home_big_chances", "away_big_chances"),
    "big chances missed": ("home_big_chances_missed", "away_big_chances_missed"),
}

# Player stat keys we store in dedicated columns (SofaScore statistics keys)
_PLAYER_STAT_DEDICATED = {
    "minutesPlayed", "rating", "goals", "assists", "totalShots",
    "shotsOnTarget", "totalPass", "accuratePass", "keyPass",
    "tackles", "interceptions",
}


# ── Helpers ──────────────────────────────────────────────────────────────────


def _has_rows(table: str, fixture_id: int) -> bool:
    """Check if data already exists for a fixture in a given table."""
    rows = db.select_rows(table, "fixture_id", {"fixture_id": fixture_id})
    return len(rows) > 0


def _get_current_season_id(tournament_id: int) -> int | None:
    """Fetch the most recent (current) season ID for a tournament."""
    data = fetch_tournament_seasons(tournament_id)
    seasons = data.get("seasons", [])
    if not seasons:
        return None
    # SofaScore returns seasons in reverse chronological order
    return seasons[0]["id"]


def _log_error(run_id: int, error_type: str, message: str, ref: str | None = None) -> None:
    db.insert_rows("ingestion_errors", [{
        "source_id": SOURCE_ID,
        "run_id": run_id,
        "error_type": error_type,
        "error_message": message[:2000],
        "payload_ref": ref,
    }])


def _extract_scores_from_event(event: dict) -> dict | None:
    """Extract scores directly from a tournament events list item.

    The event object from /events/last already contains homeScore/awayScore,
    so no additional API call is needed for results.
    """
    home_score_obj = event.get("homeScore", {})
    away_score_obj = event.get("awayScore", {})

    home_score = home_score_obj.get("current")
    away_score = away_score_obj.get("current")

    if home_score is None or away_score is None:
        return None

    return {
        "home_score": int(home_score),
        "away_score": int(away_score),
        "home_ht_score": (
            int(home_score_obj["period1"])
            if home_score_obj.get("period1") is not None
            else None
        ),
        "away_ht_score": (
            int(away_score_obj["period1"])
            if away_score_obj.get("period1") is not None
            else None
        ),
    }


# ── Statistics normalization ─────────────────────────────────────────────────


def _safe_numeric(value: Any) -> Any:
    """Convert a value to a number suitable for DB insertion."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        # Handle percentage strings like "65%"
        cleaned = value.replace("%", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _normalize_statistics(raw: dict, fixture_id: int) -> dict | None:
    """Parse SofaScore /event/{id}/statistics response into a match_statistics row.

    Only uses the 'ALL' period. Known stats go into dedicated columns;
    everything else goes into extra_stats JSONB.
    """
    row: dict[str, Any] = {"fixture_id": fixture_id}
    extra: dict[str, Any] = {}
    found_stats = False

    for period in raw.get("statistics", []):
        if period.get("period") != "ALL":
            continue

        for group in period.get("groups", []):
            for item in group.get("statisticsItems", []):
                stat_name = item.get("name", "").strip().lower()
                home_val = _safe_numeric(item.get("homeValue", item.get("home")))
                away_val = _safe_numeric(item.get("awayValue", item.get("away")))
                found_stats = True

                # Try to match to a dedicated column
                matched = False
                for fragment, (home_col, away_col) in _STAT_COLUMN_MAP.items():
                    if fragment in stat_name:
                        # Avoid overwriting a more specific match
                        # e.g. "shots on target" should not be overwritten by "total shots"
                        if home_col not in row or len(fragment) > len(
                            next(
                                (f for f, (hc, _) in _STAT_COLUMN_MAP.items() if hc == home_col and f != fragment),
                                "",
                            )
                        ):
                            row[home_col] = home_val
                            row[away_col] = away_val
                        matched = True
                        break

                if not matched:
                    extra[stat_name] = {"home": home_val, "away": away_val}

        break  # Only process ALL period

    if not found_stats:
        return None

    if extra:
        row["extra_stats"] = json.dumps(extra)

    return row


# ── Lineups normalization ────────────────────────────────────────────────────


def _normalize_lineups(
    raw: dict,
    fixture_id: int,
    home_team_id: int,
    away_team_id: int,
) -> list[dict]:
    """Parse SofaScore /event/{id}/lineups into player_match_ratings rows."""
    rows: list[dict] = []

    for side, team_id in [("home", home_team_id), ("away", away_team_id)]:
        lineup = raw.get(side, {})
        players = lineup.get("players", [])

        for p in players:
            player_info = p.get("player", {})
            stats = p.get("statistics", {})

            # Build extra_stats from anything not in dedicated columns
            extra = {k: v for k, v in stats.items() if k not in _PLAYER_STAT_DEDICATED}

            rows.append({
                "fixture_id": fixture_id,
                "team_id": team_id,
                "source_player_id": str(player_info.get("id", "")),
                "player_name": player_info.get("name", "Unknown"),
                "player_short_name": player_info.get("shortName"),
                "position": p.get("position") or player_info.get("position"),
                "jersey_number": p.get("shirtNumber"),
                "is_substitute": p.get("substitute", False),
                "minutes_played": stats.get("minutesPlayed"),
                "rating": stats.get("rating"),
                "goals": stats.get("goals", 0),
                "assists": stats.get("assists", 0),
                "shots_total": stats.get("totalShots"),
                "shots_on_target": stats.get("shotsOnTarget"),
                "passes_total": stats.get("totalPass"),
                "passes_accurate": stats.get("accuratePass"),
                "key_passes": stats.get("keyPass"),
                "tackles": stats.get("tackles"),
                "interceptions": stats.get("interceptions"),
                "extra_stats": json.dumps(extra) if extra else None,
            })

    return rows


# ── Event pagination ─────────────────────────────────────────────────────────


def _collect_events(
    tournament_id: int,
    season_id: int,
    start_ts: float,
    end_ts: float,
) -> list[dict]:
    """Paginate through all past events for a tournament season.

    Returns events whose startTimestamp falls within [start_ts, end_ts],
    sorted oldest-first.
    """
    all_events: list[dict] = []
    page = 0
    max_pages = 100  # safety limit

    while page < max_pages:
        data = fetch_tournament_events_last(tournament_id, season_id, page)
        events = data.get("events", [])
        if not events:
            break

        for ev in events:
            ts = ev.get("startTimestamp", 0)
            if ts < start_ts:
                # Events are newest-first; once we're before start, stop
                return sorted(all_events, key=lambda e: e["startTimestamp"])
            if ts <= end_ts:
                all_events.append(ev)

        has_next = data.get("hasNextPage", False)
        if not has_next:
            break
        page += 1

    return sorted(all_events, key=lambda e: e["startTimestamp"])


# ── Main pipeline entry point ────────────────────────────────────────────────


def run(
    start_date: str,
    end_date: str,
    *,
    skip_odds: bool = False,
    skip_stats: bool = False,
    skip_lineups: bool = False,
    progress_callback: Any = None,
    only_league: str | None = None,
) -> dict[str, Any]:
    """Run the historical backfill pipeline.

    Args:
        start_date: Start of backfill range (YYYY-MM-DD, inclusive).
        end_date: End of backfill range (YYYY-MM-DD, inclusive).
        skip_odds: Skip fetching odds (saves API calls).
        skip_stats: Skip fetching match statistics.
        skip_lineups: Skip fetching player lineups/ratings.
        progress_callback: Optional callable(league_key, event_idx, total, event)
            invoked after each event is processed.

    Returns a summary dict with counts.
    """
    now = datetime.now(timezone.utc)

    # Parse date range to Unix timestamps
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
        hour=23, minute=59, second=59, tzinfo=timezone.utc,
    )
    start_ts = start_dt.timestamp()
    end_ts = end_dt.timestamp()

    # Reset per-run caches
    _league_cache.clear()
    _team_cache.clear()
    _fixture_cache.clear()

    # Start pipeline run
    run_rows = db.insert_rows("pipeline_runs", [{
        "pipeline_name": f"backfill_{start_date}_to_{end_date}",
        "started_at": now.isoformat(),
        "status": "running",
        "details_json": {
            "start_date": start_date,
            "end_date": end_date,
            "skip_odds": skip_odds,
            "skip_stats": skip_stats,
            "skip_lineups": skip_lineups,
        },
    }])
    run_id = run_rows[0]["id"]

    stats: dict[str, Any] = {
        "leagues_processed": 0,
        "events_found": 0,
        "fixtures_upserted": 0,
        "results_upserted": 0,
        "odds_snapshots_inserted": 0,
        "statistics_upserted": 0,
        "player_ratings_upserted": 0,
        "skipped_odds": 0,
        "skipped_stats": 0,
        "skipped_lineups": 0,
        "errors": 0,
        "per_league": {},
    }

    try:
        for tournament_id, league_key in TOURNAMENT_TO_LEAGUE.items():
            if only_league and league_key.upper() != only_league.upper():
                continue
            league_stats = {
                "events_found": 0,
                "fixtures": 0,
                "results": 0,
                "odds": 0,
                "statistics": 0,
                "lineups": 0,
                "errors": 0,
            }

            logger.info("── Backfill: %s (tournament %d) ──", league_key, tournament_id)

            # 1. Get current season ID
            try:
                season_id = _get_current_season_id(tournament_id)
            except Exception as e:
                logger.error("Failed to get season for %s: %s", league_key, e)
                _log_error(run_id, "season_fetch", str(e), f"tournament:{tournament_id}")
                stats["errors"] += 1
                league_stats["errors"] += 1
                stats["per_league"][league_key] = league_stats
                continue

            if season_id is None:
                logger.warning("No seasons found for %s", league_key)
                stats["per_league"][league_key] = league_stats
                continue

            logger.info("  Season ID: %d", season_id)

            # 2. Collect all events in date range
            try:
                events = _collect_events(tournament_id, season_id, start_ts, end_ts)
            except Exception as e:
                logger.error("Failed to paginate events for %s: %s", league_key, e)
                _log_error(run_id, "events_fetch", str(e), f"tournament:{tournament_id}")
                stats["errors"] += 1
                league_stats["errors"] += 1
                stats["per_league"][league_key] = league_stats
                continue

            league_stats["events_found"] = len(events)
            stats["events_found"] += len(events)
            logger.info("  Found %d events in date range", len(events))

            # 3. Process each event
            for idx, event in enumerate(events):
                sf_event_id = event["id"]
                sf_status_type = event.get("status", {}).get("type", "notstarted")

                try:
                    # 3a. Upsert fixture
                    league_id = _get_league_id(tournament_id)
                    if league_id is None:
                        continue

                    fixture_id = _upsert_fixture(event, league_id)
                    if fixture_id is None:
                        continue
                    league_stats["fixtures"] += 1
                    stats["fixtures_upserted"] += 1

                    # Look up team IDs for lineups (from the event, already cached)
                    home_team_id = _upsert_team(event["homeTeam"])
                    away_team_id = _upsert_team(event["awayTeam"])

                    # 3b. Extract and store result (from event object, no API call)
                    if sf_status_type == "finished":
                        scores = _extract_scores_from_event(event)
                        if scores is not None:
                            db.upsert_rows("results", [{
                                "fixture_id": fixture_id,
                                "home_score": scores["home_score"],
                                "away_score": scores["away_score"],
                                "home_ht_score": scores["home_ht_score"],
                                "away_ht_score": scores["away_ht_score"],
                                "result_status": "final",
                                "settled_at": now.isoformat(),
                            }], on_conflict="fixture_id")
                            league_stats["results"] += 1
                            stats["results_upserted"] += 1

                    # 3c. Fetch + store odds (if not already present)
                    if not skip_odds:
                        if _has_rows("odds_snapshots", fixture_id):
                            stats["skipped_odds"] += 1
                        else:
                            try:
                                raw_odds = fetch_event_odds(sf_event_id)
                                storage.archive_raw_response(
                                    "sofascore", "event-odds",
                                    str(sf_event_id), raw_odds, ts=now,
                                )
                                odds_rows = _normalize_odds(raw_odds, fixture_id, now)
                                if odds_rows:
                                    db.insert_rows("odds_snapshots", odds_rows)
                                    league_stats["odds"] += len(odds_rows)
                                    stats["odds_snapshots_inserted"] += len(odds_rows)
                            except Exception as e:
                                logger.warning(
                                    "  No odds for event %d: %s", sf_event_id, e,
                                )
                                _log_error(
                                    run_id, "odds_fetch", str(e),
                                    f"event:{sf_event_id}",
                                )
                                stats["errors"] += 1
                                league_stats["errors"] += 1

                    # 3d. Fetch + store match statistics
                    if not skip_stats and sf_status_type == "finished":
                        if _has_rows("match_statistics", fixture_id):
                            stats["skipped_stats"] += 1
                        else:
                            try:
                                raw_stats = fetch_event_statistics(sf_event_id)
                                storage.archive_raw_response(
                                    "sofascore", "event-statistics",
                                    str(sf_event_id), raw_stats, ts=now,
                                )
                                stat_row = _normalize_statistics(raw_stats, fixture_id)
                                if stat_row:
                                    db.upsert_rows(
                                        "match_statistics", [stat_row],
                                        on_conflict="fixture_id",
                                    )
                                    league_stats["statistics"] += 1
                                    stats["statistics_upserted"] += 1
                            except Exception as e:
                                logger.warning(
                                    "  No stats for event %d: %s", sf_event_id, e,
                                )
                                _log_error(
                                    run_id, "stats_fetch", str(e),
                                    f"event:{sf_event_id}",
                                )
                                stats["errors"] += 1
                                league_stats["errors"] += 1

                    # 3e. Fetch + store player lineups/ratings
                    if not skip_lineups and sf_status_type == "finished":
                        if _has_rows("player_match_ratings", fixture_id):
                            stats["skipped_lineups"] += 1
                        else:
                            try:
                                raw_lineups = fetch_event_lineups(sf_event_id)
                                storage.archive_raw_response(
                                    "sofascore", "event-lineups",
                                    str(sf_event_id), raw_lineups, ts=now,
                                )
                                lineup_rows = _normalize_lineups(
                                    raw_lineups, fixture_id,
                                    home_team_id, away_team_id,
                                )
                                if lineup_rows:
                                    db.upsert_rows(
                                        "player_match_ratings", lineup_rows,
                                        on_conflict="fixture_id,team_id,source_player_id",
                                    )
                                    league_stats["lineups"] += len(lineup_rows)
                                    stats["player_ratings_upserted"] += len(lineup_rows)
                            except Exception as e:
                                logger.warning(
                                    "  No lineups for event %d: %s", sf_event_id, e,
                                )
                                _log_error(
                                    run_id, "lineups_fetch", str(e),
                                    f"event:{sf_event_id}",
                                )
                                stats["errors"] += 1
                                league_stats["errors"] += 1

                    # Progress callback
                    if progress_callback:
                        progress_callback(league_key, idx + 1, len(events), event)

                except Exception as e:
                    logger.error(
                        "  Error processing event %d: %s", sf_event_id, e,
                        exc_info=True,
                    )
                    _log_error(
                        run_id, "event_processing", str(e),
                        f"event:{sf_event_id}",
                    )
                    stats["errors"] += 1
                    league_stats["errors"] += 1
                    continue

            stats["leagues_processed"] += 1
            stats["per_league"][league_key] = league_stats
            logger.info(
                "  Done: %d fixtures, %d results, %d odds, %d stats, %d lineups, %d errors",
                league_stats["fixtures"],
                league_stats["results"],
                league_stats["odds"],
                league_stats["statistics"],
                league_stats["lineups"],
                league_stats["errors"],
            )

        # Mark pipeline success
        ended = datetime.now(timezone.utc)
        db.get_client().schema("dhx").table("pipeline_runs").update({
            "status": "success",
            "ended_at": ended.isoformat(),
            "rows_written": (
                stats["fixtures_upserted"]
                + stats["results_upserted"]
                + stats["odds_snapshots_inserted"]
                + stats["statistics_upserted"]
                + stats["player_ratings_upserted"]
            ),
            "details_json": stats,
        }).eq("id", run_id).execute()

    except Exception as e:
        logger.error("Backfill pipeline failed: %s", e, exc_info=True)
        _log_error(run_id, "pipeline_fatal", str(e))
        ended = datetime.now(timezone.utc)
        db.get_client().schema("dhx").table("pipeline_runs").update({
            "status": "failed",
            "ended_at": ended.isoformat(),
            "details_json": {**stats, "fatal_error": str(e)},
        }).eq("id", run_id).execute()
        stats["errors"] += 1

    return stats

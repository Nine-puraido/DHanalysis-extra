"""Results ingestion — fetch finished match scores and settle fixtures.

Flow:
  1. Query fixtures with status='scheduled' (or 'live') and kickoff in the past
  2. Look up SofaScore event IDs via fixture_source_map
  3. Fetch event details from SofaScore API
  4. If finished: upsert result, update fixture status
  5. Archive raw responses, track pipeline run + errors
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from dhx import db, storage
from dhx.ingestion.sofascore_client import fetch_event_details, fetch_scheduled_events

logger = logging.getLogger(__name__)

SOURCE_ID = 1  # sofascore

# SofaScore status.type → our fixture status
_STATUS_MAP: dict[str, str] = {
    "finished": "finished",
    "postponed": "postponed",
    "canceled": "cancelled",
    "cancelled": "cancelled",
    "abandoned": "abandoned",
}

# SofaScore status.type → our result_status
_RESULT_STATUS_MAP: dict[str, str] = {
    "finished": "final",
    "abandoned": "abandoned",
}


def _get_pending_fixtures(date_str: str | None = None) -> list[dict]:
    """Get fixtures that may need result settlement.

    Returns fixtures with status in ('scheduled', 'live') whose kickoff is in the past,
    along with their SofaScore event IDs from fixture_source_map.

    If date_str is provided, only return fixtures whose kickoff_at date matches.
    """
    client = db.get_client()

    # Query fixtures that could have finished
    q = (
        client.schema("dhx").table("fixtures")
        .select("id, league_id, home_team_id, away_team_id, kickoff_at, status")
        .in_("status", ["scheduled", "live"])
        .lt("kickoff_at", datetime.now(timezone.utc).isoformat())
        .order("kickoff_at")
    )

    if date_str:
        # Filter by kickoff date (start/end of day in UTC)
        q = q.gte("kickoff_at", f"{date_str}T00:00:00Z")
        q = q.lt("kickoff_at", f"{date_str}T23:59:59Z")

    fixtures = q.execute().data
    if not fixtures:
        return []

    # Enrich with SofaScore event IDs
    fixture_ids = [f["id"] for f in fixtures]
    mappings = (
        client.schema("dhx").table("fixture_source_map")
        .select("fixture_id, source_event_id")
        .eq("source_id", SOURCE_ID)
        .in_("fixture_id", fixture_ids)
        .execute()
        .data
    )
    mapping_lookup = {m["fixture_id"]: m["source_event_id"] for m in mappings}

    for f in fixtures:
        f["source_event_id"] = mapping_lookup.get(f["id"])

    return fixtures


def _extract_scores(event: dict) -> dict | None:
    """Extract scores from a SofaScore event response.

    Returns a dict with home_score, away_score, home_ht_score, away_ht_score
    or None if scores aren't available.
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


def _log_error(run_id: int, error_type: str, message: str, ref: str | None = None) -> None:
    db.insert_rows("ingestion_errors", [{
        "source_id": SOURCE_ID,
        "run_id": run_id,
        "error_type": error_type,
        "error_message": message[:2000],
        "payload_ref": ref,
    }])


def run_bulk(date_str: str) -> dict[str, Any]:
    """Settle results using a single scheduled-events API call (fast).

    Instead of fetching each event individually (~1s per fixture),
    fetches all events for the date in one bulk call and matches
    them to pending fixtures in the DB.

    Requires date_str (cannot settle "all pending" in bulk mode).
    """
    now = datetime.now(timezone.utc)

    run_rows = db.insert_rows("pipeline_runs", [{
        "pipeline_name": f"results_settle_bulk_{date_str}",
        "started_at": now.isoformat(),
        "status": "running",
        "details_json": {"date": date_str, "mode": "results_bulk"},
    }])
    run_id = run_rows[0]["id"]

    stats: dict[str, Any] = {
        "fixtures_checked": 0,
        "results_inserted": 0,
        "fixtures_updated": 0,
        "still_pending": 0,
        "errors": 0,
    }

    try:
        # 1. Get pending fixtures from DB
        fixtures = _get_pending_fixtures(date_str)
        stats["fixtures_checked"] = len(fixtures)
        logger.info("Found %d pending fixtures to check for results", len(fixtures))

        if not fixtures:
            db.get_client().schema("dhx").table("pipeline_runs").update({
                "status": "success",
                "ended_at": now.isoformat(),
                "rows_written": 0,
                "details_json": {**stats, "date": date_str},
            }).eq("id", run_id).execute()
            return stats

        # 2. Fetch all events for the date in one API call
        raw_events = fetch_scheduled_events(date_str)
        events = raw_events.get("events", [])

        # Archive raw response
        storage.archive_raw_response(
            source_key="sofascore",
            endpoint_label="scheduled-events-settle",
            identifier=date_str,
            payload=raw_events,
            ts=now,
        )

        # 3. Build lookup: SofaScore event ID → event data
        event_lookup: dict[int, dict] = {}
        for ev in events:
            eid = ev.get("id")
            if eid is not None:
                event_lookup[int(eid)] = ev

        # 4. Match fixtures to events and settle
        for fix in fixtures:
            fixture_id = fix["id"]
            sf_event_id = fix.get("source_event_id")

            if sf_event_id is None:
                logger.warning("No SofaScore mapping for fixture %d, skipping", fixture_id)
                stats["still_pending"] += 1
                continue

            event = event_lookup.get(int(sf_event_id))
            if event is None:
                # Event not found in scheduled-events response — still pending
                stats["still_pending"] += 1
                continue

            try:
                sf_status = event.get("status", {})
                sf_type = sf_status.get("type", "notstarted")
                our_status = _STATUS_MAP.get(sf_type)

                if our_status is None:
                    stats["still_pending"] += 1
                    continue

                db.update_rows("fixtures", {"status": our_status}, {"id": fixture_id})
                stats["fixtures_updated"] += 1

                if our_status in ("finished", "abandoned"):
                    scores = _extract_scores(event)
                    if scores is not None:
                        result_status = _RESULT_STATUS_MAP.get(our_status, "final")
                        db.upsert_rows("results", [{
                            "fixture_id": fixture_id,
                            "home_score": scores["home_score"],
                            "away_score": scores["away_score"],
                            "home_ht_score": scores["home_ht_score"],
                            "away_ht_score": scores["away_ht_score"],
                            "result_status": result_status,
                            "settled_at": now.isoformat(),
                        }], on_conflict="fixture_id")
                        stats["results_inserted"] += 1
                        logger.info(
                            "Settled fixture %d: %d-%d (%s)",
                            fixture_id, scores["home_score"], scores["away_score"], result_status,
                        )
                    else:
                        logger.warning("Fixture %d status=%s but no scores", fixture_id, our_status)
                        _log_error(run_id, "missing_scores",
                                   f"Status {our_status} but no scores for fixture {fixture_id}",
                                   f"event:{sf_event_id}")

            except Exception as e:
                logger.error("Error settling fixture %d: %s", fixture_id, e, exc_info=True)
                _log_error(run_id, "settle_error", str(e), f"event:{sf_event_id}")
                stats["errors"] += 1

        ended = datetime.now(timezone.utc)
        db.get_client().schema("dhx").table("pipeline_runs").update({
            "status": "success",
            "ended_at": ended.isoformat(),
            "rows_written": stats["results_inserted"],
            "details_json": {**stats, "date": date_str},
        }).eq("id", run_id).execute()

    except Exception as e:
        logger.error("Bulk settle pipeline failed: %s", e, exc_info=True)
        _log_error(run_id, "pipeline_fatal", str(e))
        ended = datetime.now(timezone.utc)
        db.get_client().schema("dhx").table("pipeline_runs").update({
            "status": "failed",
            "ended_at": ended.isoformat(),
            "details_json": {**stats, "date": date_str, "fatal_error": str(e)},
        }).eq("id", run_id).execute()
        stats["errors"] += 1

    return stats


def run(date_str: str | None = None) -> dict[str, Any]:
    """Run the results settlement pipeline.

    If date_str is provided, only settle fixtures from that date.
    Otherwise, settle all past fixtures that are still 'scheduled' or 'live'.

    Returns a summary dict with counts.
    """
    now = datetime.now(timezone.utc)
    label = date_str or "all-pending"

    # Start pipeline run
    run_rows = db.insert_rows("pipeline_runs", [{
        "pipeline_name": f"results_settle_{label}",
        "started_at": now.isoformat(),
        "status": "running",
        "details_json": {"date": label, "mode": "results"},
    }])
    run_id = run_rows[0]["id"]

    stats: dict[str, Any] = {
        "fixtures_checked": 0,
        "results_inserted": 0,
        "fixtures_updated": 0,
        "still_pending": 0,
        "errors": 0,
    }

    try:
        fixtures = _get_pending_fixtures(date_str)
        stats["fixtures_checked"] = len(fixtures)
        logger.info("Found %d pending fixtures to check for results", len(fixtures))

        for fix in fixtures:
            fixture_id = fix["id"]
            sf_event_id = fix.get("source_event_id")

            if sf_event_id is None:
                logger.warning("No SofaScore mapping for fixture %d, skipping", fixture_id)
                stats["still_pending"] += 1
                continue

            try:
                raw = fetch_event_details(int(sf_event_id))
                event = raw.get("event", raw)

                # Archive raw response
                storage.archive_raw_response(
                    source_key="sofascore",
                    endpoint_label="event-details",
                    identifier=sf_event_id,
                    payload=raw,
                    ts=now,
                )

                # Check SofaScore status
                sf_status = event.get("status", {})
                sf_type = sf_status.get("type", "notstarted")
                our_status = _STATUS_MAP.get(sf_type)

                if our_status is None:
                    # Still in progress or not started — skip
                    stats["still_pending"] += 1
                    continue

                # Update fixture status
                db.update_rows(
                    "fixtures",
                    {"status": our_status},
                    {"id": fixture_id},
                )
                stats["fixtures_updated"] += 1

                # If finished or abandoned with scores, insert result
                if our_status in ("finished", "abandoned"):
                    scores = _extract_scores(event)
                    if scores is not None:
                        result_status = _RESULT_STATUS_MAP.get(our_status, "final")

                        # Use upsert so re-running is idempotent
                        db.upsert_rows("results", [{
                            "fixture_id": fixture_id,
                            "home_score": scores["home_score"],
                            "away_score": scores["away_score"],
                            "home_ht_score": scores["home_ht_score"],
                            "away_ht_score": scores["away_ht_score"],
                            "result_status": result_status,
                            "settled_at": now.isoformat(),
                        }], on_conflict="fixture_id")
                        stats["results_inserted"] += 1
                        logger.info(
                            "Settled fixture %d: %d-%d (%s)",
                            fixture_id,
                            scores["home_score"],
                            scores["away_score"],
                            result_status,
                        )
                    else:
                        logger.warning(
                            "Fixture %d status=%s but no scores available",
                            fixture_id, our_status,
                        )
                        _log_error(
                            run_id, "missing_scores",
                            f"Status {our_status} but no scores for fixture {fixture_id}",
                            f"event:{sf_event_id}",
                        )

            except Exception as e:
                logger.error(
                    "Error settling fixture %d (event %s): %s",
                    fixture_id, sf_event_id, e, exc_info=True,
                )
                _log_error(run_id, "settle_error", str(e), f"event:{sf_event_id}")
                stats["errors"] += 1
                continue

        # Mark pipeline success
        ended = datetime.now(timezone.utc)
        db.get_client().schema("dhx").table("pipeline_runs").update({
            "status": "success",
            "ended_at": ended.isoformat(),
            "rows_written": stats["results_inserted"],
            "details_json": {**stats, "date": label},
        }).eq("id", run_id).execute()

    except Exception as e:
        logger.error("Results pipeline failed: %s", e, exc_info=True)
        _log_error(run_id, "pipeline_fatal", str(e))
        ended = datetime.now(timezone.utc)
        db.get_client().schema("dhx").table("pipeline_runs").update({
            "status": "failed",
            "ended_at": ended.isoformat(),
            "details_json": {**stats, "date": label, "fatal_error": str(e)},
        }).eq("id", run_id).execute()
        stats["errors"] += 1

    return stats

"""HTTP client for the SofaScore public API."""

from __future__ import annotations

import logging
import time
from typing import Any

import cloudscraper
from requests.exceptions import ConnectionError, HTTPError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from dhx.config import get_settings

logger = logging.getLogger(__name__)

# SofaScore uniqueTournament IDs → our league_key mapping (22 extra leagues)
TOURNAMENT_TO_LEAGUE: dict[int, str] = {
    # Europe (14)
    202: "POL",     # Ekstraklasa (Poland)
    185: "GSL",     # Super League (Greece)
    172: "CFL",     # Czech First League
    39: "DSL",      # Superliga (Denmark)
    152: "RSL",     # SuperLiga (Romania)
    218: "UPL",     # Premier League (Ukraine)
    170: "HNL",     # HNL (Croatia)
    210: "SRBL",    # Superliga (Serbia)
    266: "IPL",     # Premier League (Israel)
    53: "SB",       # Serie B (Italy)
    54: "LL2",      # LaLiga 2 (Spain)
    182: "L2",      # Ligue 2 (France)
    24: "EL1",      # League One (England)
    491: "GL3",     # 3. Liga (Germany)
    # Outside Europe (8)
    955: "SPL_SA",  # Saudi Pro League
    242: "MLS",     # MLS (USA)
    196: "J1",      # J1 League (Japan)
    155: "ARG",     # Liga Profesional (Argentina)
    136: "ALM",     # A-League Men (Australia)
    1032: "TL1",    # Thai League 1
    825: "QSL",     # Stars League (Qatar)
    971: "UAE",     # Pro League (UAE)
}

TRACKED_TOURNAMENT_IDS: set[int] = set(TOURNAMENT_TO_LEAGUE.keys())

# Rate limiting: pause between API calls to be polite
_REQUEST_DELAY_S = 1.0
_last_request_time: float = 0.0

_scraper: cloudscraper.CloudScraper | None = None


def _get_scraper() -> cloudscraper.CloudScraper:
    global _scraper
    if _scraper is None:
        _scraper = cloudscraper.create_scraper()
    return _scraper


def _rate_limited_get(url: str) -> Any:
    """GET with rate limiting and raise on HTTP errors."""
    global _last_request_time
    elapsed = time.monotonic() - _last_request_time
    if elapsed < _REQUEST_DELAY_S:
        time.sleep(_REQUEST_DELAY_S - elapsed)

    resp = _get_scraper().get(url, timeout=30)
    _last_request_time = time.monotonic()
    resp.raise_for_status()
    return resp.json()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((HTTPError, ConnectionError)),
    reraise=True,
)
def fetch_scheduled_events(date_str: str) -> dict[str, Any]:
    """GET /sport/football/scheduled-events/{date}

    Returns raw JSON response.
    """
    base = get_settings().sofascore_base_url
    url = f"{base}/sport/football/scheduled-events/{date_str}"
    logger.info("Fetching scheduled events for %s", date_str)
    return _rate_limited_get(url)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((HTTPError, ConnectionError)),
    reraise=True,
)
def fetch_event_odds(event_id: int) -> dict[str, Any]:
    """GET /event/{eventId}/odds/1/all

    Returns raw JSON response with all odds markets.
    """
    base = get_settings().sofascore_base_url
    url = f"{base}/event/{event_id}/odds/1/all"
    logger.debug("Fetching odds for event %d", event_id)
    return _rate_limited_get(url)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((HTTPError, ConnectionError)),
    reraise=True,
)
def fetch_event_details(event_id: int) -> dict[str, Any]:
    """GET /event/{eventId}

    Returns full event details including scores and status.
    """
    base = get_settings().sofascore_base_url
    url = f"{base}/event/{event_id}"
    logger.debug("Fetching event details for %d", event_id)
    return _rate_limited_get(url)


# ── Additional endpoints for backfill ────────────────────────────────────────


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((HTTPError, ConnectionError)),
    reraise=True,
)
def fetch_event_statistics(event_id: int) -> dict[str, Any]:
    """GET /event/{eventId}/statistics

    Returns match statistics (xG, shots, possession, etc.) grouped by period.
    """
    base = get_settings().sofascore_base_url
    url = f"{base}/event/{event_id}/statistics"
    logger.debug("Fetching statistics for event %d", event_id)
    return _rate_limited_get(url)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((HTTPError, ConnectionError)),
    reraise=True,
)
def fetch_event_lineups(event_id: int) -> dict[str, Any]:
    """GET /event/{eventId}/lineups

    Returns player lineups with ratings and individual stats.
    """
    base = get_settings().sofascore_base_url
    url = f"{base}/event/{event_id}/lineups"
    logger.debug("Fetching lineups for event %d", event_id)
    return _rate_limited_get(url)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((HTTPError, ConnectionError)),
    reraise=True,
)
def fetch_tournament_seasons(tournament_id: int) -> dict[str, Any]:
    """GET /unique-tournament/{tournamentId}/seasons

    Returns all available seasons for a tournament.
    """
    base = get_settings().sofascore_base_url
    url = f"{base}/unique-tournament/{tournament_id}/seasons"
    logger.info("Fetching seasons for tournament %d", tournament_id)
    return _rate_limited_get(url)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((HTTPError, ConnectionError)),
    reraise=True,
)
def fetch_tournament_events_last(
    tournament_id: int,
    season_id: int,
    page: int = 0,
) -> dict[str, Any]:
    """GET /unique-tournament/{id}/season/{seasonId}/events/last/{page}

    Returns past events (most recent first) for a tournament season.
    Each page has ~20 events. Response includes 'hasNextPage' boolean.
    """
    base = get_settings().sofascore_base_url
    url = f"{base}/unique-tournament/{tournament_id}/season/{season_id}/events/last/{page}"
    logger.debug(
        "Fetching past events for tournament %d season %d page %d",
        tournament_id, season_id, page,
    )
    return _rate_limited_get(url)

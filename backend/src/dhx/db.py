"""Supabase client helpers – DHextra uses the 'dhx' schema."""

from __future__ import annotations

from supabase import Client, create_client

from dhx.config import get_settings

SCHEMA = "dhx"

_client: Client | None = None


def get_client() -> Client:
    """Return a singleton Supabase client (service-role for backend workers)."""
    global _client
    if _client is None:
        s = get_settings()
        _client = create_client(s.supabase_url, s.supabase_service_role_key)
    return _client


def _table(name: str):
    """Return a table query builder scoped to the dhx schema."""
    return get_client().schema(SCHEMA).table(name)


def upsert_rows(table: str, rows: list[dict], on_conflict: str) -> list[dict]:
    """Upsert rows into a table, returning the upserted records."""
    if not rows:
        return []
    return (
        _table(table)
        .upsert(rows, on_conflict=on_conflict)
        .execute()
        .data
    )


def insert_rows(table: str, rows: list[dict]) -> list[dict]:
    """Insert rows into a table, returning inserted records."""
    if not rows:
        return []
    return _table(table).insert(rows).execute().data


def select_rows(
    table: str,
    columns: str = "*",
    filters: dict | None = None,
) -> list[dict]:
    """Simple select with optional equality filters."""
    q = _table(table).select(columns)
    for k, v in (filters or {}).items():
        q = q.eq(k, v)
    return q.execute().data


def update_rows(
    table: str,
    values: dict,
    filters: dict,
) -> list[dict]:
    """Update rows matching equality filters. Returns updated records."""
    q = _table(table).update(values)
    for k, v in filters.items():
        q = q.eq(k, v)
    return q.execute().data

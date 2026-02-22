"""Supabase Storage helpers for raw JSON archival and binary uploads."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from dhx.config import get_settings
from dhx.db import get_client


def upload_bytes(
    bucket: str,
    path: str,
    data: bytes,
    content_type: str = "application/octet-stream",
) -> str:
    """Upload raw bytes to Supabase Storage. Returns the storage path."""
    get_client().storage.from_(bucket).upload(
        path,
        data,
        file_options={"content-type": content_type, "upsert": "true"},
    )
    return f"{bucket}/{path}"


def upload_json(
    bucket: str,
    path: str,
    data: dict | list,
) -> str:
    """Upload a JSON payload to Supabase Storage. Returns the storage path."""
    body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
    get_client().storage.from_(bucket).upload(
        path,
        body,
        file_options={"content-type": "application/json", "upsert": "true"},
    )
    return f"{bucket}/{path}"


def archive_raw_response(
    source_key: str,
    endpoint_label: str,
    identifier: str,
    payload: dict | list,
    ts: datetime | None = None,
) -> str:
    """Archive a raw API response to Storage and return the path.

    Path format: raw-api/{source}/{endpoint}/{identifier}/{timestamp}.json
    """
    ts = ts or datetime.now(timezone.utc)
    ts_str = ts.strftime("%Y%m%dT%H%M%SZ")
    path = f"{source_key}/{endpoint_label}/{identifier}/{ts_str}.json"
    bucket = get_settings().supabase_bucket_raw_api
    return upload_json(bucket, path, payload)

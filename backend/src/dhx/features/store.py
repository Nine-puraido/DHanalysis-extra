"""Write feature matrix to Supabase Storage and register in feature_set_versions."""

from __future__ import annotations

import hashlib
import io
import logging
from datetime import datetime, timezone

import polars as pl

from dhx.config import get_settings
from dhx.db import insert_rows
from dhx.storage import upload_bytes

logger = logging.getLogger(__name__)


def _schema_hash(df: pl.DataFrame) -> str:
    """Compute a deterministic hash of column names + dtypes."""
    parts = sorted(f"{c}:{df[c].dtype}" for c in df.columns)
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def save_feature_set(
    matrix: pl.DataFrame,
    start_date: str,
    end_date: str,
) -> int:
    """Write Parquet to Supabase Storage and insert a feature_set_versions row.

    Returns the version ID.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = f"feature_matrix/{start_date}_{end_date}/{ts}.parquet"
    bucket = get_settings().supabase_bucket_features

    # Write Parquet to bytes buffer
    buf = io.BytesIO()
    matrix.write_parquet(buf, compression="zstd")
    parquet_bytes = buf.getvalue()
    logger.info(f"Parquet size: {len(parquet_bytes):,} bytes")

    # Upload to Storage
    full_path = upload_bytes(bucket, path, parquet_bytes, "application/octet-stream")
    logger.info(f"Uploaded to {full_path}")

    # Register in feature_set_versions
    schema_hash = _schema_hash(matrix)
    row = {
        "training_window": f"{start_date}/{end_date}",
        "parquet_path": full_path,
        "row_count": len(matrix),
        "feature_schema_hash": schema_hash,
        "notes": f"Columns: {len(matrix.columns)}, "
                 f"features: {len([c for c in matrix.columns if not c.startswith('label_')])}",
    }
    inserted = insert_rows("feature_set_versions", [row])
    version_id = inserted[0]["id"]
    logger.info(f"Registered feature_set_versions id={version_id}")

    return version_id

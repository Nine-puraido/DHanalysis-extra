"""Models for pipeline_runs and ingestion_errors."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class PipelineRun(BaseModel):
    id: int | None = None
    pipeline_name: str
    started_at: datetime
    ended_at: datetime | None = None
    status: str = "running"  # running, success, failed
    rows_written: int | None = None
    details_json: dict | None = None


class IngestionError(BaseModel):
    id: int | None = None
    source_id: int | None = None
    run_id: int | None = None
    error_type: str
    error_message: str
    payload_ref: str | None = None

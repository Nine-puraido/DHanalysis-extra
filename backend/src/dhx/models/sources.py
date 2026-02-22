"""Models for data_sources and bookmakers."""

from __future__ import annotations

from pydantic import BaseModel


class DataSource(BaseModel):
    id: int | None = None
    source_key: str
    name: str
    is_active: bool = True


class Bookmaker(BaseModel):
    id: int | None = None
    bookmaker_key: str
    name: str
    is_active: bool = True

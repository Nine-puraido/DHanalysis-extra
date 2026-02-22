"""Models for odds_snapshots."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel


class OddsSnapshot(BaseModel):
    fixture_id: int
    source_id: int
    bookmaker_id: int
    market: str  # 1x2, ah, totals, btts
    selection: str
    line: Decimal | None = None
    price_decimal: Decimal
    implied_prob: Decimal | None = None
    is_main: bool = False
    is_suspended: bool = False
    pulled_at: datetime

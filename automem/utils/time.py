from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Tuple


def utc_now() -> str:
    """Return an ISO formatted UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _parse_iso_datetime(value: Optional[Any]) -> Optional[datetime]:
    """Parse ISO strings or unix timestamps into timezone-aware datetimes (UTC fallback for naive)."""
    if value is None:
        return None

    # Handle numeric timestamps (unix epoch)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc)
        except (ValueError, OSError):
            return None

    # Handle string timestamps
    if not isinstance(value, str):
        return None

    candidate = value.strip()
    if not candidate:
        return None

    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(candidate)
        # Ensure timezone-aware (assume UTC if naive)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def _normalize_timestamp(raw: Any) -> str:
    """Validate and normalise an incoming timestamp string to UTC ISO format."""
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("Timestamp must be a non-empty ISO formatted string")

    candidate = raw.strip()
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:  # pragma: no cover - validation path
        raise ValueError("Invalid ISO timestamp") from exc

    return parsed.astimezone(timezone.utc).isoformat()


def _parse_time_expression(expression: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not expression:
        return None, None

    expr = expression.strip().lower()
    if not expr:
        return None, None

    now = datetime.now(timezone.utc)

    def start_of_day(dt: datetime) -> datetime:
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)

    def end_of_day(dt: datetime) -> datetime:
        return start_of_day(dt) + timedelta(days=1)

    if expr in {"today", "this day"}:
        start = start_of_day(now)
        end = end_of_day(now)
    elif expr in {"yesterday"}:
        start = start_of_day(now - timedelta(days=1))
        end = start + timedelta(days=1)
    elif expr in {"last 24 hours", "past 24 hours"}:
        end = now
        start = now - timedelta(hours=24)
    elif expr in {"last 48 hours", "past 48 hours"}:
        end = now
        start = now - timedelta(hours=48)
    elif expr in {"this week"}:
        start = start_of_day(now - timedelta(days=now.weekday()))
        end = start + timedelta(days=7)
    elif expr in {"last week", "past week"}:
        end = start_of_day(now - timedelta(days=now.weekday()))
        start = end - timedelta(days=7)
    elif expr in {"this month"}:
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if start.month == 12:
            end = start.replace(year=start.year + 1, month=1)
        else:
            end = start.replace(month=start.month + 1)
    elif expr in {"last month", "past month"}:
        current_month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if current_month_start.month == 1:
            previous_month_start = current_month_start.replace(
                year=current_month_start.year - 1, month=12
            )
        else:
            previous_month_start = current_month_start.replace(month=current_month_start.month - 1)
        start = previous_month_start
        end = current_month_start
    elif expr.startswith("last ") and expr.endswith(" days"):
        try:
            days = int(expr.split()[1])
            end = now
            start = now - timedelta(days=days)
        except ValueError:
            return None, None
    elif expr in {"last year", "past year", "this year"}:
        start = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        if expr.startswith("last") or expr.startswith("past"):
            end = start
            start = start.replace(year=start.year - 1)
        else:
            if start.year == 9999:
                end = now
            else:
                end = start.replace(year=start.year + 1)
    else:
        return None, None

    return start.isoformat(), end.isoformat()

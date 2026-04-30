"""Small SQL extraction and validation helpers for Text-to-SQL nodes."""

from __future__ import annotations

import re


_SQL_BLOCK_RE = re.compile(r"```(?:sql|sqlite)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_SQL_START_RE = re.compile(r"\b(?:WITH|SELECT)\b", re.IGNORECASE)


def extract_sql(text: str, fallback: str = "") -> str:
    """Extract the first SELECT/WITH query from an LLM response."""
    if not text:
        return fallback

    match = _SQL_BLOCK_RE.search(text)
    candidate = match.group(1).strip() if match else text.strip()
    candidate = candidate.replace("\r", " ").strip()

    start = _SQL_START_RE.search(candidate)
    if start:
        candidate = candidate[start.start() :]

    # Keep a single statement. SQLite accepts trailing semicolons, but extra prose
    # or multiple statements make sandbox feedback noisy and can be unsafe.
    parts = [part.strip() for part in candidate.split(";") if part.strip()]
    if parts:
        candidate = parts[0] + ";"

    if not is_readonly_select(candidate):
        return fallback
    return re.sub(r"\s+", " ", candidate).strip()


def is_readonly_select(sql: str) -> bool:
    """Return True for a single read-only SELECT/CTE query."""
    if not sql:
        return False
    cleaned = sql.strip().strip(";").strip()
    if not _SQL_START_RE.match(cleaned):
        return False
    forbidden = re.search(
        r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH|PRAGMA|VACUUM)\b",
        cleaned,
        re.IGNORECASE,
    )
    return forbidden is None

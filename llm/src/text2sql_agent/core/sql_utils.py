"""Small SQL extraction and validation helpers for Text-to-SQL nodes.

Extraction priority (highest → lowest):
  1. ``<sql>...</sql>`` XML tags  — enforced by corrector prompt
  2. Triple-backtick sql/sqlite fenced blocks
  3. Raw string starting with SELECT/WITH after stripping markdown noise
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# Priority 1: explicit <sql>...</sql> XML tags (case-insensitive, multiline)
_XML_TAG_RE = re.compile(
    r"<sql>\s*(.*?)\s*</sql>",
    re.DOTALL | re.IGNORECASE,
)

# Priority 2: fenced markdown code blocks (```sql / ```sqlite / ``` alone)
_SQL_BLOCK_RE = re.compile(
    r"```(?:sql|sqlite)?\s*(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

# Detect WHERE a SELECT/WITH query starts within a larger string
_SQL_START_RE = re.compile(r"\b(?:WITH|SELECT)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_sql(text: str, fallback: str = "") -> str:
    """Extract the first valid SELECT/WITH query from an LLM response.

    Tries three strategies in priority order:

    1. ``<sql>...</sql>`` XML tags.
    2. Fenced markdown code blocks (`` ```sql `` / `` ```sqlite `` / `` ``` ``).
    3. Bare text: strips leading/trailing prose and looks for a SELECT/WITH.

    In all cases:
    - Trailing semicolons are normalised to a single one.
    - Multiple statements are trimmed to the first one.
    - Consecutive whitespace is collapsed.
    - Non-SELECT / non-CTE outputs return ``fallback``.

    Args:
        text: Raw LLM response string.
        fallback: Value returned when no valid SQL can be extracted.
            Defaults to ``""``.

    Returns:
        Cleaned SQL string, or ``fallback`` if extraction fails.
    """
    if not text:
        return fallback

    candidate: str | None = None

    # --- Strategy 1: <sql> XML tags ---
    xml_match = _XML_TAG_RE.search(text)
    if xml_match:
        candidate = xml_match.group(1).strip()

    # --- Strategy 2: fenced code block ---
    if not candidate:
        block_match = _SQL_BLOCK_RE.search(text)
        if block_match:
            candidate = block_match.group(1).strip()

    # --- Strategy 3: raw text, find first SELECT/WITH ---
    if not candidate:
        candidate = text.strip()

    candidate = candidate.replace("\r", " ")

    # Advance to the first SELECT or WITH keyword
    start = _SQL_START_RE.search(candidate)
    if start:
        candidate = candidate[start.start():]

    # Keep only the first SQL statement (split on ";")
    parts = [part.strip() for part in candidate.split(";") if part.strip()]
    if parts:
        candidate = parts[0] + ";"

    if not is_readonly_select(candidate):
        return fallback

    return re.sub(r"\s+", " ", candidate).strip()


def is_readonly_select(sql: str) -> bool:
    """Return True for a single read-only SELECT/CTE query.

    Args:
        sql: SQL string to validate.

    Returns:
        ``True`` if the query is a safe read-only SELECT or CTE.
    """
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

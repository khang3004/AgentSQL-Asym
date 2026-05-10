"""Module 2: SemanticErrorChecker — MCI-SQL Logic-Aware SQL Execution Wrapper.

This module implements a Semantic State Evaluation layer that wraps raw
SQLite execution with three tiers of error classification:

1. **Syntax / Runtime Error** — ``sqlite3.OperationalError``: the query is
   structurally invalid or references non-existent objects.
2. **Empty Result Error** — ``EmptyResultError``: the query executed
   successfully but returned zero rows.  This is a *semantic* failure that
   vanilla try/except cannot detect.
3. **Null Result Error** — ``NullResultError``: every cell across every
   returned row is ``None`` / ``NULL``, indicating a broken JOIN or
   incorrect NULL-handling in the predicate.

Each custom exception carries an actionable ``suggestion`` attribute that is
formatted for direct inclusion in the LLM correction prompt, guiding the
critic model toward a concrete fix without an additional API round-trip.

Typical usage::

    from text2sql_agent.tools.semantic_error_checker import (
        SemanticErrorChecker,
        EmptyResultError,
        NullResultError,
    )

    checker = SemanticErrorChecker(db_path="path/to/database.sqlite")

    try:
        rows = checker.execute(
            "SELECT name FROM customers WHERE segment = 'vip'"
        )
        print("Success:", rows)
    except EmptyResultError as e:
        print("Empty:", e.suggestion)
    except NullResultError as e:
        print("Null:", e.suggestion)
    except sqlite3.OperationalError as e:
        print("Syntax:", e)
"""

import logging
import sqlite3
from contextlib import contextmanager
from typing import Any, Generator, List, Optional, Tuple
from langsmith import traceable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom Semantic Exception Hierarchy
# ---------------------------------------------------------------------------


class SemanticSQLError(Exception):
    """Base class for semantic SQL execution errors.

    Extends ``Exception`` with a structured ``suggestion`` attribute so that
    upstream pipeline components can embed the guidance directly into an LLM
    correction prompt without any additional processing.

    Attributes:
        suggestion (str): A concrete, human-readable remediation hint targeted
            at the LLM critic, describing what to change in the SQL query.
    """

    def __init__(self, message: str, suggestion: str) -> None:
        """Initialises the error with a message and a correction suggestion.

        Args:
            message (str): Short description of the semantic failure.
            suggestion (str): Actionable correction hint for the LLM critic.
        """
        super().__init__(message)
        self.suggestion: str = suggestion

    def __str__(self) -> str:
        """Returns a combined error description including the suggestion.

        Returns:
            str: Multi-line string joining the base message and the suggestion.
        """
        return f"{super().__str__()}\nSuggestion: {self.suggestion}"


class EmptyResultError(SemanticSQLError):
    """Raised when a syntactically valid query returns exactly zero rows.

    An empty result set is a *semantic* failure: the query executed without
    error but did not satisfy the information need expressed in the natural
    language question.  Common causes include:

    - Exact-match string predicates (``=``) against values with different
      casing or surrounding whitespace.
    - Overly restrictive compound predicates that exclude all rows.
    - Referencing a date/category value that does not exist verbatim in the
      database.

    The embedded ``suggestion`` guides the critic LLM toward LIKE-based fuzzy
    matching, LOWER()/UPPER() normalisation, or predicate relaxation.
    """

    _DEFAULT_SUGGESTION: str = (
        "Query returned 0 rows. Consider using 'LIKE' with wildcards instead "
        "of exact '=' matching, or check for case sensitivity. For example, "
        "replace `col = 'value'` with `LOWER(col) LIKE LOWER('%value%')`. "
        "Also verify that the literal value actually exists in the database "
        "by cross-checking sample values from the metadata context."
    )

    def __init__(self, sql: str, suggestion: Optional[str] = None) -> None:
        """Initialises EmptyResultError with the offending SQL query.

        Args:
            sql (str): The SQL query that produced an empty result set.
            suggestion (Optional[str]): Override for the default suggestion.
                If ``None``, the class-level ``_DEFAULT_SUGGESTION`` is used.
        """
        super().__init__(
            message=f"Query returned 0 rows. OffendingSQL: {sql[:300]}",
            suggestion=suggestion or self._DEFAULT_SUGGESTION,
        )
        self.sql: str = sql


class NullResultError(SemanticSQLError):
    """Raised when every cell in every returned row evaluates to NULL.

    A fully-null result indicates that the query structure is syntactically
    valid but semantically broken.  Common causes include:

    - A LEFT JOIN that does not match any rows in the right-hand table,
      causing all projected columns from that table to be NULL.
    - Incorrect or missing ``IS NOT NULL`` / ``COALESCE`` guards in
      aggregation expressions.
    - Type mismatches between a foreign key and a primary key that prevent
      rows from joining.

    The embedded ``suggestion`` instructs the critic to re-examine JOIN
    predicates and to add NULL-guards where appropriate.
    """

    _DEFAULT_SUGGESTION: str = (
        "Query returned NULL for all values. Re-evaluate the JOIN conditions: "
        "ensure the ON clause correctly links the primary key to the foreign "
        "key (e.g., `table_a.id = table_b.a_id`). Consider switching from "
        "LEFT JOIN to INNER JOIN if unmatched rows should be excluded, or add "
        "'WHERE <joined_col> IS NOT NULL' to filter nullified join results. "
        "Also verify COALESCE/NULLIF usage in aggregation expressions."
    )

    def __init__(self, sql: str, suggestion: Optional[str] = None) -> None:
        """Initialises NullResultError with the offending SQL query.

        Args:
            sql (str): The SQL query that returned an all-NULL result set.
            suggestion (Optional[str]): Override for the default suggestion.
                If ``None``, the class-level ``_DEFAULT_SUGGESTION`` is used.
        """
        super().__init__(
            message=f"Query returned NULL for all values. OffendingSQL: {sql[:300]}",
            suggestion=suggestion or self._DEFAULT_SUGGESTION,
        )
        self.sql: str = sql


# ---------------------------------------------------------------------------
# SemanticErrorChecker
# ---------------------------------------------------------------------------


class SemanticErrorChecker:
    """Wraps local SQLite query execution with three-tier semantic evaluation.

    The checker intercepts query results *after* successful execution and
    applies rule-based Semantic State Evaluation:

    +------------------+----------------------------------------------+
    | Execution State  | Exception Raised                             |
    +==================+==============================================+
    | Syntax error     | ``sqlite3.OperationalError`` (re-raised)     |
    +------------------+----------------------------------------------+
    | 0 rows returned  | ``EmptyResultError``                         |
    +------------------+----------------------------------------------+
    | All NULLs        | ``NullResultError``                          |
    +------------------+----------------------------------------------+
    | Valid data       | Returns ``List[Tuple[Any, ...]]`` normally   |
    +------------------+----------------------------------------------+

    This three-tier model enables the orchestrating pipeline to route each
    failure type to an appropriate correction strategy without incurring
    additional LLM API calls for trivial semantic diagnosis.

    Attributes:
        db_path (str): Path to the target SQLite database file.
    """

    def __init__(self, db_path: str) -> None:
        """Initialises the checker and validates the database path.

        Args:
            db_path (str): Path to the target SQLite database file.

        Raises:
            FileNotFoundError: If ``db_path`` does not point to an existing
                file on disk.
        """
        import os
        if not os.path.isfile(db_path):
            raise FileNotFoundError(
                f"SemanticErrorChecker: Database not found at '{db_path}'."
            )
        self.db_path: str = db_path
        logger.debug(
            "[SemanticErrorChecker] Initialised for database: %s", db_path
        )

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager that yields a read-only SQLite connection.

        Opens the database in URI read-only mode to prevent any accidental
        mutations by LLM-generated DML or DDL statements.

        Yields:
            sqlite3.Connection: An open, read-only database connection.
        """
        conn = sqlite3.connect(
            f"file:{self.db_path}?mode=ro",
            uri=True,
        )
        try:
            yield conn
        finally:
            conn.close()

    @staticmethod
    def _is_all_null(rows: List[Tuple[Any, ...]]) -> bool:
        """Determines whether every cell in every row is ``None``.

        Args:
            rows (List[Tuple[Any, ...]]): The raw result set returned by the
                SQLite cursor, where each element is a row tuple.

        Returns:
            bool: ``True`` if ``rows`` is non-empty and every single value
                across all rows and all columns is ``None``.  Returns
                ``False`` if ``rows`` is empty (deferred to ``EmptyResultError``)
                or if at least one non-null value exists.
        """
        if not rows:
            return False
        return all(
            cell is None
            for row in rows
            for cell in row
        )

    @traceable(run_type="tool")
    def execute(self, sql: str) -> List[Tuple[Any, ...]]:
        """Executes a SQL query and applies Semantic State Evaluation.

        This is the primary public method.  It delegates raw execution to
        ``sqlite3``, then evaluates the result against three semantic states:
        syntax error, empty result, and all-null result.

        Args:
            sql (str): A read-only SQL query to execute against the database.
                The query must be a SELECT statement.

        Returns:
            List[Tuple[Any, ...]]: The raw row tuples returned by the query
                when the result is semantically valid (non-empty, non-null).

        Raises:
            sqlite3.OperationalError: Propagated directly if SQLite raises a
                syntax or runtime error during execution.
            EmptyResultError: If the query succeeds but returns zero rows.
            NullResultError: If the query succeeds, returns rows, but every
                cell value is ``None`` / ``NULL``.
        """
        logger.info(
            "[SemanticErrorChecker] Executing SQL: %s",
            sql[:120].replace("\n", " "),
        )

        with self._connect() as conn:
            # Phase 1: Raw execution — may raise sqlite3.OperationalError.
            cursor = conn.execute(sql)
            rows: List[Tuple[Any, ...]] = cursor.fetchall()

        # Phase 2: Semantic State Evaluation.
        if len(rows) == 0:
            logger.warning(
                "[SemanticErrorChecker] EmptyResultError for query: %s",
                sql[:120],
            )
            raise EmptyResultError(sql=sql)

        if self._is_all_null(rows):
            logger.warning(
                "[SemanticErrorChecker] NullResultError for query: %s",
                sql[:120],
            )
            raise NullResultError(sql=sql)

        logger.info(
            "[SemanticErrorChecker] Query returned %d valid rows.", len(rows)
        )
        return rows

    @traceable(run_type="tool")
    def execute_safe(
        self, sql: str
    ) -> Tuple[Optional[List[Tuple[Any, ...]]], Optional[str]]:
        """Non-raising variant of ``execute`` that returns a structured result tuple.

        Provides a ``(rows, error_message)`` interface for callers that prefer
        to handle all failure modes uniformly (e.g., the pipeline integration
        layer) without a try/except block at the call site.

        Args:
            sql (str): A read-only SQL query to execute.

        Returns:
            Tuple[Optional[List[Tuple[Any, ...]]], Optional[str]]:
                - ``(rows, None)`` on success — ``rows`` is the valid result set.
                - ``(None, error_message)`` on any failure — ``error_message``
                  embeds the full ``SemanticSQLError.__str__`` (including the
                  suggestion) or the sqlite3 error string, ready for direct
                  injection into an LLM correction prompt.
        """
        try:
            rows = self.execute(sql)
            return rows, None
        except (EmptyResultError, NullResultError) as exc:
            return None, str(exc)
        except sqlite3.OperationalError as exc:
            error_msg = (
                f"SQLite OperationalError: {exc}. "
                f"Check the syntax, table names, and column references. "
                f"OffendingSQL: {sql[:300]}"
            )
            logger.error("[SemanticErrorChecker] %s", error_msg)
            return None, error_msg
        except Exception as exc:
            error_msg = f"Unexpected execution error: {type(exc).__name__}: {exc}"
            logger.error("[SemanticErrorChecker] %s", error_msg)
            return None, error_msg

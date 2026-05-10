"""Module 1: MetadataExtractor — MCI-SQL Metadata-Complete Context Provider.

This module implements a local, offline metadata extraction layer that
enriches Text-to-SQL prompts with statistically rich database metadata,
inspired by the Metadata-Complete Context (MCI) principle from the MCI-SQL
paper. By running all analysis directly on the local SQLite file **before**
any API call is made, it eliminates the need for extra LLM round-trips,
thereby minimising quota consumption.

The extracted context includes:
    - Numerical column statistics (MIN, MAX values).
    - Cardinality inference (Primary-Key 1:1 vs. Foreign-Key 1:N ratio).
    - Random non-null sample values for text/blob columns.

Typical usage::

    from text2sql_agent.tools.metadata_extractor import MetadataExtractor

    extractor = MetadataExtractor(db_path="path/to/database.sqlite")
    context_json = extractor.build_context(
        tables=["customers", "transactions_1k"],
        columns={"customers": ["Segment", "Currency"],
                 "transactions_1k": ["Amount", "Date"]},
    )
    # Append context_json to the LLM prompt string.
"""

import json
import logging
import random
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional
from langsmith import traceable

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Connects to a local SQLite database and extracts Metadata-Complete Context.

    The extractor operates entirely offline using Python's built-in
    ``sqlite3`` module.  No external dependencies are required, which makes it
    perfectly safe to invoke synchronously in the prompt-building phase,
    before any LLM API call is dispatched.

    Attributes:
        db_path (str): Absolute or relative path to the target SQLite file.
        _NUMERIC_AFFINITY (frozenset): SQLite type-affinity tokens that map to
            numeric storage classes (INTEGER and REAL).
        _TEXT_AFFINITY (frozenset): SQLite type-affinity tokens that map to the
            TEXT storage class (TEXT, BLOB, NONE).
        _SAMPLE_SIZE (int): Number of random non-null sample values fetched for
            text columns.
    """

    _NUMERIC_AFFINITY: frozenset = frozenset({
        "INT", "INTEGER", "TINYINT", "SMALLINT", "MEDIUMINT", "BIGINT",
        "UNSIGNED BIG INT", "INT2", "INT8",
        "REAL", "DOUBLE", "DOUBLE PRECISION", "FLOAT",
        "NUMERIC", "DECIMAL", "BOOLEAN", "DATE", "DATETIME",
    })

    _TEXT_AFFINITY: frozenset = frozenset({
        "TEXT", "CHARACTER", "VARCHAR", "VARYING CHARACTER",
        "NCHAR", "NATIVE CHARACTER", "NVARCHAR", "CLOB", "BLOB", "NONE",
    })

    _SAMPLE_SIZE: int = 3

    def __init__(self, db_path: str) -> None:
        """Initialises the extractor and validates that the database file exists.

        Args:
            db_path (str): Path to the target SQLite database file.

        Raises:
            FileNotFoundError: If the specified ``db_path`` does not point to an
                existing file on disk.
        """
        import os
        if not os.path.isfile(db_path):
            raise FileNotFoundError(
                f"MetadataExtractor: SQLite database not found at '{db_path}'."
            )
        self.db_path: str = db_path
        logger.debug("[MetadataExtractor] Initialised for database: %s", db_path)

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager that yields a read-only SQLite connection.

        Uses ``sqlite3.PARSE_DECLTYPES`` to let Python infer Python types from
        the declared column types.  The connection is always closed, even if an
        exception is raised inside the ``with`` block.

        Yields:
            sqlite3.Connection: An open, read-only database connection.
        """
        conn = sqlite3.connect(
            f"file:{self.db_path}?mode=ro",
            uri=True,
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_affinity(self, declared_type: str) -> str:
        """Resolves a SQLite declared type string to a broad affinity class.

        Implements the SQLite type-affinity rules (§3.1 of the SQLite spec)
        in a simplified, case-insensitive fashion sufficient for prompt
        construction purposes.

        Args:
            declared_type (str): The raw declared type string from
                ``PRAGMA table_info``, e.g. ``"VARCHAR(255)"``, ``"REAL"``.

        Returns:
            str: Either ``"NUMERIC"``, ``"TEXT"``, or ``"UNKNOWN"``.
        """
        normalised: str = declared_type.upper().strip()

        for token in self._NUMERIC_AFFINITY:
            if token in normalised:
                return "NUMERIC"

        for token in self._TEXT_AFFINITY:
            if token in normalised:
                return "TEXT"

        # SQLite default affinity for unrecognised types is NUMERIC
        return "NUMERIC" if normalised else "UNKNOWN"

    def _get_columns_info(
        self, conn: sqlite3.Connection, table: str
    ) -> List[Dict[str, Any]]:
        """Fetches column metadata for a given table via PRAGMA.

        Args:
            conn (sqlite3.Connection): An open database connection.
            table (str): The table name to inspect.

        Returns:
            List[Dict[str, Any]]: A list of dicts, each describing one column
                with keys: ``cid``, ``name``, ``type``, ``notnull``, ``pk``.

        Raises:
            sqlite3.OperationalError: If the specified table does not exist.
        """
        cursor = conn.execute(f"PRAGMA table_info(\"{table}\");")
        return [dict(row) for row in cursor.fetchall()]

    def _get_row_count(self, conn: sqlite3.Connection, table: str) -> int:
        """Counts total rows in a table.

        Args:
            conn (sqlite3.Connection): An open database connection.
            table (str): The target table name.

        Returns:
            int: Total row count, or 0 if the table is empty or inaccessible.
        """
        try:
            row = conn.execute(
                f"SELECT COUNT(*) FROM \"{table}\";"
            ).fetchone()
            return int(row[0]) if row else 0
        except sqlite3.OperationalError as exc:
            logger.warning(
                "[MetadataExtractor] Cannot count rows in '%s': %s", table, exc
            )
            return 0

    def _extract_numeric_stats(
        self,
        conn: sqlite3.Connection,
        table: str,
        column: str,
    ) -> Dict[str, Any]:
        """Queries MIN and MAX values for a numeric column.

        Args:
            conn (sqlite3.Connection): An open database connection.
            table (str): The table containing the column.
            column (str): The numeric column name to analyse.

        Returns:
            Dict[str, Any]: A dict with keys ``"min"`` and ``"max"``.
                Values are ``None`` if the column contains only NULLs.
        """
        try:
            row = conn.execute(
                f"SELECT MIN(\"{column}\"), MAX(\"{column}\") FROM \"{table}\";"
            ).fetchone()
            return {
                "min": row[0] if row else None,
                "max": row[1] if row else None,
            }
        except sqlite3.OperationalError as exc:
            logger.warning(
                "[MetadataExtractor] Numeric stats failed for '%s.%s': %s",
                table,
                column,
                exc,
            )
            return {"min": None, "max": None}

    def _infer_cardinality(
        self,
        conn: sqlite3.Connection,
        table: str,
        column: str,
        total_rows: int,
        is_pk: bool,
    ) -> Dict[str, Any]:
        """Infers cardinality relationship from DISTINCT count vs. total row count.

        The heuristic used is:
            - ``DISTINCT == total_rows`` → the column functions as a **1:1 PK**.
            - ``DISTINCT < total_rows`` → the column functions as a **1:N FK**
              (or a low-cardinality categorical).
            - Zero total rows → ``"UNKNOWN"``.

        Args:
            conn (sqlite3.Connection): An open database connection.
            table (str): The table containing the column.
            column (str): The column to evaluate.
            total_rows (int): Pre-computed total row count for the table.
            is_pk (bool): Whether the column is already marked as a primary key
                by SQLite's PRAGMA.

        Returns:
            Dict[str, Any]: A dict with keys:
                - ``"distinct_count"`` (int): Number of unique non-null values.
                - ``"total_rows"`` (int): Total rows in the table.
                - ``"cardinality"`` (str): One of ``"1:1 (PK-like)"``,
                  ``"1:N (FK/Categorical)"``, or ``"UNKNOWN"``.
        """
        distinct_count: int = 0
        try:
            row = conn.execute(
                f"SELECT COUNT(DISTINCT \"{column}\") FROM \"{table}\";"
            ).fetchone()
            distinct_count = int(row[0]) if row else 0
        except sqlite3.OperationalError as exc:
            logger.warning(
                "[MetadataExtractor] Cardinality query failed for '%s.%s': %s",
                table,
                column,
                exc,
            )

        if total_rows == 0:
            cardinality = "UNKNOWN"
        elif is_pk or distinct_count == total_rows:
            cardinality = "1:1 (PK-like)"
        else:
            cardinality = "1:N (FK/Categorical)"

        return {
            "distinct_count": distinct_count,
            "total_rows": total_rows,
            "cardinality": cardinality,
        }

    def _fetch_text_samples(
        self,
        conn: sqlite3.Connection,
        table: str,
        column: str,
    ) -> List[Any]:
        """Fetches up to ``_SAMPLE_SIZE`` random non-null values from a text column.

        Uses ``ORDER BY RANDOM()`` to obtain an unbiased sample while keeping
        the query simple and dependency-free.  This is intentionally limited to
        a small number of values to remain token-efficient.

        Args:
            conn (sqlite3.Connection): An open database connection.
            table (str): The table containing the column.
            column (str): The text column to sample.

        Returns:
            List[Any]: A list of up to ``_SAMPLE_SIZE`` distinct, non-null
                sample values.  Returns an empty list on failure.
        """
        try:
            rows = conn.execute(
                f"""
                SELECT DISTINCT "{column}"
                FROM   "{table}"
                WHERE  "{column}" IS NOT NULL
                ORDER  BY RANDOM()
                LIMIT  {self._SAMPLE_SIZE};
                """
            ).fetchall()
            return [row[0] for row in rows]
        except sqlite3.OperationalError as exc:
            logger.warning(
                "[MetadataExtractor] Text sample failed for '%s.%s': %s",
                table,
                column,
                exc,
            )
            return []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_column_metadata(
        self,
        table: str,
        column: str,
        conn: sqlite3.Connection,
        total_rows: int,
    ) -> Dict[str, Any]:
        """Extracts full MCI-style metadata for a single column.

        Automatically dispatches to numeric or text extraction strategies
        based on the column's SQLite type affinity.

        Args:
            table (str): The table containing the column.
            column (str): The target column name.
            conn (sqlite3.Connection): An open, reusable database connection.
            total_rows (int): Pre-computed total row count for the table,
                passed in to avoid redundant COUNT queries.

        Returns:
            Dict[str, Any]: A structured metadata dict for the column.  Its
                exact shape depends on the column affinity:

                For NUMERIC columns::

                    {
                        "affinity": "NUMERIC",
                        "min": <value>,
                        "max": <value>,
                        "distinct_count": <int>,
                        "total_rows": <int>,
                        "cardinality": "1:1 (PK-like)" | "1:N (FK/Categorical)",
                    }

                For TEXT columns::

                    {
                        "affinity": "TEXT",
                        "samples": [<val1>, <val2>, <val3>],
                        "distinct_count": <int>,
                        "total_rows": <int>,
                        "cardinality": "1:1 (PK-like)" | "1:N (FK/Categorical)",
                    }
        """
        cols_info = self._get_columns_info(conn, table)
        col_meta: Optional[Dict[str, Any]] = next(
            (c for c in cols_info if c["name"] == column), None
        )

        declared_type: str = col_meta["type"] if col_meta else ""
        is_pk: bool = bool(col_meta["pk"]) if col_meta else False
        affinity: str = self._resolve_affinity(declared_type)

        cardinality_info = self._infer_cardinality(
            conn, table, column, total_rows, is_pk
        )

        result: Dict[str, Any] = {
            "affinity": affinity,
            **cardinality_info,
        }

        if affinity == "NUMERIC":
            result.update(self._extract_numeric_stats(conn, table, column))
        else:
            result["samples"] = self._fetch_text_samples(conn, table, column)

        return result

    @traceable(run_type="tool")
    def build_context(
        self,
        tables: List[str],
        columns: Optional[Dict[str, List[str]]] = None,
    ) -> str:
        """Builds a compressed, token-efficient JSON metadata context string.

        This is the primary entry point for the pipeline.  It accepts a list
        of table names and an optional mapping of ``{table: [columns]}``.  If
        ``columns`` is omitted for a given table, **all** columns in that table
        are profiled automatically.

        The returned JSON string is intentionally compact (no extra whitespace)
        to minimise token consumption when appended to an LLM prompt.

        Args:
            tables (List[str]): Names of the tables to profile.  Must exist
                in the target database.
            columns (Optional[Dict[str, List[str]]]): A mapping from table name
                to a list of column names to profile.  If a table is absent from
                this mapping, all columns in the table are profiled.

        Returns:
            str: A compact JSON string encoding the metadata context, suitable
                for direct concatenation into an LLM prompt.  Structure::

                    {
                        "<table_name>": {
                            "<column_name>": { ...metadata... },
                            ...
                        },
                        ...
                    }

        Raises:
            sqlite3.OperationalError: If a specified table does not exist in
                the database (propagated from ``_get_columns_info``).

        Example::

            extractor = MetadataExtractor("path/to/db.sqlite")
            ctx = extractor.build_context(
                tables=["customers"],
                columns={"customers": ["Segment", "Amount"]},
            )
            # ctx == '{"customers":{"Segment":{...},"Amount":{...}}}'
        """
        context: Dict[str, Dict[str, Any]] = {}
        columns = columns or {}

        with self._connect() as conn:
            for table in tables:
                logger.info(
                    "[MetadataExtractor] Profiling table: '%s'", table
                )
                cols_info = self._get_columns_info(conn, table)
                total_rows = self._get_row_count(conn, table)

                target_columns: List[str] = columns.get(
                    table,
                    [c["name"] for c in cols_info],
                )

                table_context: Dict[str, Any] = {}
                for col in target_columns:
                    logger.debug(
                        "[MetadataExtractor] Profiling column: '%s.%s'",
                        table,
                        col,
                    )
                    table_context[col] = self.extract_column_metadata(
                        table=table,
                        column=col,
                        conn=conn,
                        total_rows=total_rows,
                    )

                context[table] = table_context

        # Compact serialisation: no separators beyond the minimum required.
        return json.dumps(context, separators=(",", ":"), default=str)

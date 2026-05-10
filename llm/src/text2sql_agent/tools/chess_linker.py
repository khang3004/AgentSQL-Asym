"""CHESS-Inspired Semantic Linker for schema pruning.

Implements the CHESS (Context-Aware Schema Linking) pruning strategy as a
local, offline pre-filter that scores every table and column definition against
the user's natural language question using cosine similarity computed from a
local ``sentence-transformers`` model.

Only the Top-K most semantically relevant tables survive pruning.  This
dramatically reduces the token footprint of the downstream MCI metadata
extraction and LLM prompts, preventing context overflow on databases with
large schemas (>20 tables).

Key design decisions:

- **No API call**: The embedding model runs entirely on-device.  Latency is
  sub-second for typical BIRD/Spider schema sizes.
- **Lazy model loading**: The ``SentenceTransformer`` is loaded once per
  ``ChessLinker`` instance and cached as ``self._model``, avoiding repeated
  warm-up overhead inside the pipeline loop.
- **Composite table representation**: Each table's textual representation
  combines its name, all column names, declared types, and BIRD-style
  ``database_description`` CSV annotations (when present) into a single
  string.  This gives the similarity scorer maximum signal.
- **Guaranteed minimum**: If ``top_k >= len(all_tables)``, all tables are
  returned, preventing accidental empty schemas on tiny databases.

Typical usage::

    linker = ChessLinker(model_name="all-MiniLM-L6-v2")
    result = linker.prune(
        question="How many customers are in the VIP segment?",
        db_path="path/to/database.sqlite",
        top_k=3,
    )
    print(result.selected_tables)          # ["customers"]
    print(result.similarity_scores)        # {"customers": 0.812, ...}
    print(result.pruned_schema_ddl)        # CREATE TABLE customers ...
"""

import csv
import logging
import os
import sqlite3
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
from langsmith import traceable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class PruningResult:
    """Result produced by ``ChessLinker.prune``.

    Attributes:
        question (str): The original natural language question.
        all_tables (List[str]): Every table name found in the database.
        selected_tables (List[str]): Top-K tables after semantic pruning,
            ordered by descending similarity score.
        similarity_scores (Dict[str, float]): Cosine similarity score for
            every table keyed by table name.
        pruned_schema_ddl (str): CREATE TABLE DDL concatenated only for the
            selected tables, ready to inject into the LLM prompt.
        reduction_ratio (float): Fraction of tables removed, e.g. 0.70 means
            70 % of tables were pruned away.
    """

    question: str
    all_tables: List[str]
    selected_tables: List[str]
    similarity_scores: Dict[str, float]
    pruned_schema_ddl: str
    reduction_ratio: float = field(init=False)

    def __post_init__(self) -> None:
        """Computes the reduction ratio after all fields are set."""
        total = len(self.all_tables)
        self.reduction_ratio = (
            round(1.0 - len(self.selected_tables) / total, 4)
            if total > 0
            else 0.0
        )


# ---------------------------------------------------------------------------
# ChessLinker
# ---------------------------------------------------------------------------


class ChessLinker:
    """Local semantic schema linker powered by sentence-transformers.

    Scores every table in a SQLite database against the natural language
    question using cosine similarity over dense sentence embeddings.  Returns
    only the Top-K most relevant tables together with their pruned DDL.

    Attributes:
        model_name (str): HuggingFace model ID for the sentence-transformers
            encoder.  Defaults to ``"all-MiniLM-L6-v2"`` — a 22 M-parameter
            model that offers an excellent speed / accuracy trade-off for
            schema similarity tasks.
        _model: The loaded ``SentenceTransformer`` instance (lazy-loaded on
            first call to ``prune``).
    """

    _DEFAULT_MODEL: str = "all-MiniLM-L6-v2"

    def __init__(self, model_name: Optional[str] = None) -> None:
        """Initialises the linker and defers model loading.

        Args:
            model_name (Optional[str]): HuggingFace model identifier.  If
                ``None``, ``"all-MiniLM-L6-v2"`` is used.
        """
        self.model_name: str = model_name or self._DEFAULT_MODEL
        self._model = None  # lazy-loaded on first prune() call
        logger.debug(
            "[ChessLinker] Configured with model: %s", self.model_name
        )

    # ------------------------------------------------------------------
    # Private: model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Loads the sentence-transformers model if not already cached.

        This method is idempotent; calling it multiple times is safe.

        Raises:
            ImportError: If ``sentence-transformers`` is not installed in the
                active Python environment.
        """
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for ChessLinker.  "
                "Install it with: pip install sentence-transformers"
            ) from exc

        logger.info(
            "[ChessLinker] Loading embedding model '%s' ...", self.model_name
        )
        self._model = SentenceTransformer(self.model_name)
        logger.info("[ChessLinker] Model loaded successfully.")

    # ------------------------------------------------------------------
    # Private: database introspection helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _connect(
        self, db_path: str
    ) -> Generator[sqlite3.Connection, None, None]:
        """Context manager yielding a read-only SQLite connection.

        Args:
            db_path (str): Path to the SQLite file.

        Yields:
            sqlite3.Connection: An open, read-only connection.
        """
        conn = sqlite3.connect(
            f"file:{db_path}?mode=ro", uri=True
        )
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    @staticmethod
    def _load_column_descriptions(
        db_path: str,
    ) -> Dict[str, Dict[str, str]]:
        """Loads BIRD-style ``database_description`` CSV annotations.

        Args:
            db_path (str): Path to the SQLite database file.  The
                ``database_description/`` folder is expected to be a sibling
                of this file.

        Returns:
            Dict[str, Dict[str, str]]: Nested dict keyed
                ``{table_lower: {column_lower: description_text}}``.
        """
        desc_dir = os.path.join(
            os.path.dirname(db_path), "database_description"
        )
        descriptions: Dict[str, Dict[str, str]] = defaultdict(dict)
        if not os.path.isdir(desc_dir):
            return descriptions

        for filename in os.listdir(desc_dir):
            if not filename.endswith(".csv"):
                continue
            table_name = filename[:-4].lower()
            path = os.path.join(desc_dir, filename)
            try:
                with open(path, newline="", encoding="latin-1") as fh:
                    reader = csv.reader(fh)
                    header = next(reader, [])
                    norm = [
                        h.strip().lstrip("\ufeff").lower() for h in header
                    ]

                    def _get(
                        names: Tuple[str, ...], default_idx: int
                    ) -> str:
                        for name in names:
                            if name in norm:
                                idx = norm.index(name)
                                if idx < len(row) and row[idx].strip():
                                    return row[idx].strip()
                        return (
                            row[default_idx].strip()
                            if default_idx < len(row)
                            else ""
                        )

                    for row in reader:
                        if not row:
                            continue
                        col = _get(
                            ("original_column_name", "column_name"), 0
                        )
                        desc = _get(
                            ("column_description", "description"), 2
                        )
                        val_desc = _get(("value_description",), 4)
                        if col:
                            combined = "; ".join(
                                p for p in [desc, val_desc] if p
                            )
                            descriptions[table_name][col.lower()] = (
                                combined[:400]
                            )
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "[ChessLinker] Skipped description file %s: %s",
                    path,
                    exc,
                )
        return descriptions

    def _build_table_representation(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        descriptions: Dict[str, Dict[str, str]],
    ) -> str:
        """Constructs a rich textual representation of a table for embedding.

        The representation combines:
        - The table name itself.
        - Every column name and its declared type.
        - BIRD-style column descriptions (if available).

        This composite string gives the encoder maximum signal about the
        table's semantic domain.

        Args:
            conn (sqlite3.Connection): Open read-only database connection.
            table_name (str): Name of the table to represent.
            descriptions (Dict[str, Dict[str, str]]): Pre-loaded BIRD
                column descriptions.

        Returns:
            str: A whitespace-normalised textual representation of the table.
        """
        cursor = conn.execute(f'PRAGMA table_info("{table_name}");')
        columns = cursor.fetchall()

        table_desc_dict = descriptions.get(table_name.lower(), {})
        parts: List[str] = [f"table {table_name}"]

        for col in columns:
            col_name: str = col["name"]
            col_type: str = col["type"] or "TEXT"
            annotation: str = table_desc_dict.get(col_name.lower(), "")
            col_repr = (
                f"column {col_name} type {col_type}"
                + (f" meaning {annotation}" if annotation else "")
            )
            parts.append(col_repr)

        return " , ".join(parts)

    def _get_table_ddl(
        self, conn: sqlite3.Connection, table_name: str
    ) -> str:
        """Fetches the CREATE TABLE DDL statement for a single table.

        Args:
            conn (sqlite3.Connection): Open read-only connection.
            table_name (str): Target table name.

        Returns:
            str: The raw DDL string, or an empty string if not found.
        """
        row = conn.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type='table' AND name=?;",
            (table_name,),
        ).fetchone()
        return (row["sql"] + ";") if row and row["sql"] else ""

    # ------------------------------------------------------------------
    # Private: similarity computation
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(
        query_vec: "np.ndarray", corpus_vecs: "np.ndarray"
    ) -> "np.ndarray":
        """Computes cosine similarity between one query vector and a matrix.

        Uses numerically stable L2-normalised dot product without any
        external scipy dependency.

        Args:
            query_vec (np.ndarray): Shape ``(D,)`` — the question embedding.
            corpus_vecs (np.ndarray): Shape ``(N, D)`` — table embeddings.

        Returns:
            np.ndarray: Shape ``(N,)`` — similarity score per table,
                clamped to ``[-1.0, 1.0]``.
        """
        q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        c_norms = corpus_vecs / (
            np.linalg.norm(corpus_vecs, axis=1, keepdims=True) + 1e-10
        )
        return np.clip(c_norms @ q_norm, -1.0, 1.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @traceable(run_type="tool")
    def prune(
        self,
        question: str,
        db_path: str,
        top_k: int = 3,
    ) -> PruningResult:
        """Runs CHESS semantic pruning and returns the Top-K relevant tables.

        Pipeline inside this method (all offline):

        1. Load the sentence-transformer model (cached after first call).
        2. Introspect all tables in the SQLite database.
        3. Build composite textual representations for every table.
        4. Embed the question and all table representations.
        5. Rank tables by cosine similarity; keep Top-K.
        6. Assemble the pruned DDL string.

        Args:
            question (str): The user's natural language question.
            db_path (str): Path to the SQLite database file.
            top_k (int): Maximum number of tables to retain.  If the
                database has fewer tables than ``top_k``, all tables are
                kept.  Defaults to ``3``.

        Returns:
            PruningResult: A fully populated result object including
                selected table names, per-table similarity scores, and the
                pruned DDL string.

        Raises:
            FileNotFoundError: If ``db_path`` does not point to an existing
                file.
            ImportError: If ``sentence-transformers`` is not installed.
        """
        if not os.path.isfile(db_path):
            raise FileNotFoundError(
                f"[ChessLinker] Database not found: '{db_path}'"
            )

        self._load_model()

        descriptions = self._load_column_descriptions(db_path)

        with self._connect(db_path) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name;"
            ).fetchall()
            all_tables: List[str] = [r["name"] for r in rows]

            if not all_tables:
                logger.warning(
                    "[ChessLinker] No tables found in '%s'.", db_path
                )
                return PruningResult(
                    question=question,
                    all_tables=[],
                    selected_tables=[],
                    similarity_scores={},
                    pruned_schema_ddl="",
                )

            effective_k = min(top_k, len(all_tables))
            logger.info(
                "[ChessLinker] Scoring %d tables → selecting Top-%d.",
                len(all_tables),
                effective_k,
            )

            # Build composite textual representations
            table_reprs: List[str] = [
                self._build_table_representation(conn, t, descriptions)
                for t in all_tables
            ]

            # Embed question and table representations
            all_texts: List[str] = [question] + table_reprs
            embeddings: "np.ndarray" = self._model.encode(
                all_texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=False,
            )
            question_vec: "np.ndarray" = embeddings[0]
            table_vecs: "np.ndarray" = embeddings[1:]

            # Score and rank
            scores: "np.ndarray" = self._cosine_similarity(
                question_vec, table_vecs
            )
            ranked_indices = np.argsort(scores)[::-1][:effective_k]

            selected_tables: List[str] = [
                all_tables[i] for i in ranked_indices
            ]
            similarity_scores: Dict[str, float] = {
                all_tables[i]: float(round(scores[i], 6))
                for i in range(len(all_tables))
            }

            for t in selected_tables:
                logger.info(
                    "[ChessLinker]   ✓ %-30s  similarity=%.4f",
                    t,
                    similarity_scores[t],
                )
            pruned_tables = set(all_tables) - set(selected_tables)
            for t in sorted(pruned_tables):
                logger.info(
                    "[ChessLinker]   ✗ %-30s  similarity=%.4f  (pruned)",
                    t,
                    similarity_scores[t],
                )

            # Assemble pruned DDL
            ddl_parts: List[str] = []
            for t in selected_tables:
                ddl = self._get_table_ddl(conn, t)
                if ddl:
                    ddl_parts.append(ddl)
            pruned_schema_ddl: str = "\n\n".join(ddl_parts)

        return PruningResult(
            question=question,
            all_tables=all_tables,
            selected_tables=selected_tables,
            similarity_scores=similarity_scores,
            pruned_schema_ddl=pruned_schema_ddl,
        )

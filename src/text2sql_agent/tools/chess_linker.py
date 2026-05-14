"""CHESS-Inspired Semantic Linker — Online Inference Phase.

Implements the **Online Retriever** half of the two-phase CHESS embedding
pipeline.  This module assumes that the **Offline Indexing** phase has already
been executed via ``build_offline_index.py``, which produced:

- ``schema_index.faiss``: A pre-built ``faiss.IndexFlatIP`` of L2-normalised
  table-representation embeddings (one vector per table across all databases).
- ``metadata.pkl``: A ``List[IndexRecord]`` parallel to the FAISS vectors,
  mapping each vector position to its ``(db_id, table_name, table_repr)``.

At runtime :class:`ChessLinker`:

1. Loads the ``BAAI/bge-small-en-v1.5`` model **once** (for the user question
   only — no schema re-embedding ever happens online).
2. Memory-maps the pre-built FAISS index from disk (zero re-computation).
3. Calls :py:meth:`ChessLinker.prune` to embed the question, query the FAISS
   index for the Top-K most similar tables in the target database, and
   assemble the pruned DDL string — all in sub-millisecond latency.

Design decisions
----------------
- **Index type**: ``faiss.IndexFlatIP`` with L2-normalised vectors gives
  exact cosine similarity search with 100 % recall.  No approximate methods
  (IVF/HNSW/LSH) are used, as the vector space is small (≤ 500 tables) and
  accuracy is paramount.
- **Database isolation**: The FAISS index spans all databases.  Filtering to
  the current ``db_id`` is performed post-search on the metadata list, so the
  single query still benefits from FAISS's SIMD-accelerated inner-product
  kernel.  An ``oversampling_factor`` controls how many extra candidates are
  fetched to ensure enough per-DB results survive the filter.
- **BGE instruction prefix**: BGE asymmetric retrieval models expect a query
  prefix ``"Represent this sentence: "`` for queries (not for documents /
  table representations embedded offline).  This is applied automatically.
- **LangSmith observability**: :py:meth:`ChessLinker.prune` is decorated with
  ``@traceable(run_type="tool")`` so every retrieval call appears as a
  discrete, inspectable tool span in LangSmith.

Typical usage::

    linker = ChessLinker(
        index_path="src/text2sql_agent/index/schema_index.faiss",
        metadata_path="src/text2sql_agent/index/metadata.pkl",
    )
    result = linker.prune(
        question="How many customers are in the VIP segment?",
        db_path="data_minidev/MINIDEV/dev_databases/"
                "debit_card_specializing/debit_card_specializing.sqlite",
        top_k=3,
    )
    print(result.selected_tables)    # ["customers", ...]
    print(result.similarity_scores)  # {"customers": 0.874, ...}
    print(result.pruned_schema_ddl)  # CREATE TABLE customers ...

Run the indexer first::

    python src/build_offline_index.py
"""

from __future__ import annotations

import logging
import os
import pickle
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional

import numpy as np
from langsmith import traceable

# Import IndexRecord from the indexer; fall back to a local stub so this
# module remains importable even if build_offline_index is not on the path.
try:
    from build_offline_index import (  # type: ignore[import]
        INDEX_FILENAME,
        METADATA_FILENAME,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_MODEL,
        IndexRecord,
    )
except ImportError:
    # Minimal stub — only used when chess_linker is imported standalone in
    # environments where build_offline_index.py is not on sys.path.
    @dataclass
    class IndexRecord:  # type: ignore[no-redef]
        """Metadata record parallel to each FAISS vector."""

        db_id: str
        table_name: str
        table_repr: str

    INDEX_FILENAME: str = "schema_index.faiss"
    METADATA_FILENAME: str = "metadata.pkl"
    DEFAULT_OUTPUT_DIR: str = "src/text2sql_agent/index"
    DEFAULT_MODEL: str = "BAAI/bge-small-en-v1.5"

logger = logging.getLogger(__name__)

# BGE asymmetric retrieval prefix — applied to query strings only (not to the
# offline-indexed table representations, which are treated as documents).
_BGE_QUERY_PREFIX: str = "Represent this sentence: "

# Default oversampling multiplier.  The FAISS search fetches
# top_k * OVERSAMPLING_FACTOR candidates before filtering by db_id.
# Keeps recall high even when the target DB has only a fraction of the index.
_OVERSAMPLING_FACTOR: int = 10


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class PruningResult:
    """Result produced by :py:meth:`ChessLinker.prune`.

    Attributes:
        question (str): The original natural language question.
        all_tables (List[str]): Every table name found in the database.
        selected_tables (List[str]): Top-K tables after FAISS retrieval,
            ordered by descending similarity score.
        similarity_scores (Dict[str, float]): Cosine similarity score for
            every table in the database keyed by table name.
        pruned_schema_ddl (str): Concatenated CREATE TABLE DDL for the
            selected tables, ready to inject into the LLM prompt.
        reduction_ratio (float): Fraction of tables removed, e.g. ``0.70``
            means 70 % of tables were pruned away.
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
            round(1.0 - len(self.selected_tables) / total, 4) if total > 0 else 0.0
        )


# ---------------------------------------------------------------------------
# ChessLinker — Online FAISS Retriever
# ---------------------------------------------------------------------------


class ChessLinker:
    """Online FAISS-backed semantic schema linker.

    Loads the pre-built offline FAISS index and performs sub-millisecond
    schema retrieval for each user question.  The embedding model is loaded
    once per :class:`ChessLinker` instance and cached, so successive calls
    to :py:meth:`prune` inside a batch evaluation loop pay the model warm-up
    cost only once.

    Attributes:
        index_path (str): Path to ``schema_index.faiss``.
        metadata_path (str): Path to ``metadata.pkl``.
        model_name (str): HuggingFace model ID for query embedding.
        _model: Cached ``SentenceTransformer`` instance (lazy-loaded on first
            call to :py:meth:`prune`).
        _index: Cached ``faiss.Index`` (lazy-loaded on first call).
        _metadata (List[IndexRecord]): Cached metadata parallel to the index
            (lazy-loaded on first call).
    """

    _DEFAULT_INDEX_DIR: str = DEFAULT_OUTPUT_DIR
    _DEFAULT_MODEL: str = DEFAULT_MODEL

    def __init__(
        self,
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """Initialises the linker.  All heavy assets are loaded lazily.

        Args:
            index_path (Optional[str]): Path to ``schema_index.faiss``.
                Defaults to ``<DEFAULT_OUTPUT_DIR>/schema_index.faiss``.
            metadata_path (Optional[str]): Path to ``metadata.pkl``.
                Defaults to ``<DEFAULT_OUTPUT_DIR>/metadata.pkl``.
            model_name (Optional[str]): Sentence-transformers model ID.
                Defaults to ``BAAI/bge-small-en-v1.5``.
        """
        self.index_path: str = index_path or os.path.join(
            self._DEFAULT_INDEX_DIR, INDEX_FILENAME
        )
        self.metadata_path: str = metadata_path or os.path.join(
            self._DEFAULT_INDEX_DIR, METADATA_FILENAME
        )
        self.model_name: str = model_name or self._DEFAULT_MODEL

        # Lazily loaded on first prune() call
        self._model = None
        self._index = None
        self._metadata: Optional[List[IndexRecord]] = None

        logger.debug(
            "[ChessLinker] Configured — model=%s, index=%s",
            self.model_name,
            self.index_path,
        )

    # ------------------------------------------------------------------
    # Private: lazy asset loaders
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the sentence-transformer model if not already cached.

        Only the **query** embedding model is loaded here.  Schema embeddings
        were computed offline and are stored in the FAISS index.

        Raises:
            ImportError: If ``sentence-transformers`` is not installed.
        """
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for ChessLinker.\n"
                "Install it with:  pip install sentence-transformers"
            ) from exc

        logger.info(
            "[ChessLinker] Loading query embedding model '%s' ...",
            self.model_name,
        )
        self._model = SentenceTransformer(self.model_name)
        logger.info("[ChessLinker] Query model loaded.")

    def _load_index(self) -> None:
        """Load the pre-built FAISS index and metadata from disk.

        Raises:
            ImportError: If ``faiss-cpu`` is not installed.
            FileNotFoundError: If the index or metadata file is missing.
                Run ``build_offline_index.py`` first.
        """
        if self._index is not None and self._metadata is not None:
            return

        try:
            import faiss  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is required for ChessLinker.\n"
                "Install it with:  pip install faiss-cpu"
            ) from exc

        if not os.path.isfile(self.index_path):
            raise FileNotFoundError(
                f"[ChessLinker] FAISS index not found: '{self.index_path}'.\n"
                "Run 'python src/build_offline_index.py' to build it first."
            )
        if not os.path.isfile(self.metadata_path):
            raise FileNotFoundError(
                f"[ChessLinker] Metadata not found: '{self.metadata_path}'.\n"
                "Run 'python src/build_offline_index.py' to build it first."
            )

        logger.info(
            "[ChessLinker] Loading FAISS index from '%s' ...", self.index_path
        )
        self._index = faiss.read_index(self.index_path)
        logger.info(
            "[ChessLinker] FAISS index loaded — ntotal=%d, dim=%d.",
            self._index.ntotal,
            self._index.d,
        )

        class _ChessUnpickler(pickle.Unpickler):
            def find_class(self, module: str, name: str):
                if name == "IndexRecord":
                    return IndexRecord
                return super().find_class(module, name)

        with open(self.metadata_path, "rb") as fh:
            self._metadata = _ChessUnpickler(fh).load()
        logger.info(
            "[ChessLinker] Metadata loaded — %d records.", len(self._metadata)
        )

    # ------------------------------------------------------------------
    # Private: SQLite helpers
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
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _get_all_tables(self, db_path: str) -> List[str]:
        """Return all user-defined table names in the database.

        Args:
            db_path (str): Path to the SQLite file.

        Returns:
            List[str]: Table names ordered alphabetically.
        """
        with self._connect(db_path) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name;"
            ).fetchall()
        return [r["name"] for r in rows]

    def _get_table_ddl(
        self, conn: sqlite3.Connection, table_name: str
    ) -> str:
        """Fetch the CREATE TABLE DDL for a single table.

        Args:
            conn (sqlite3.Connection): Open read-only connection.
            table_name (str): Target table name.

        Returns:
            str: The raw DDL string terminated with ``;``, or ``""`` if not
                found.
        """
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?;",
            (table_name,),
        ).fetchone()
        return (row["sql"] + ";") if row and row["sql"] else ""

    # ------------------------------------------------------------------
    # Private: query embedding
    # ------------------------------------------------------------------

    def _embed_query(self, question: str) -> np.ndarray:
        """Embed the user question with the BGE query prefix.

        BGE asymmetric models distinguish between *query* and *document*
        encoding.  The query prefix ``"Represent this sentence: "`` is applied
        here.  Table representations were embedded **without** this prefix
        during the offline phase, matching the BGE document encoding path.

        Args:
            question (str): The natural language question.

        Returns:
            np.ndarray: L2-normalised query embedding, shape ``(1, D)`` in
                ``float32``.
        """
        prefixed_query = _BGE_QUERY_PREFIX + question
        vec: np.ndarray = self._model.encode(
            [prefixed_query],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        norm = np.where(norm == 0.0, 1.0, norm)
        return (vec / norm).astype(np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @traceable(run_type="tool")
    def prune(
        self,
        question: str,
        db_path: str,
        top_k: int = 3,
        oversampling_factor: int = _OVERSAMPLING_FACTOR,
    ) -> PruningResult:
        """Run online FAISS-based schema pruning for a single question.

        This is the **lightning-fast** online path:

        1. Load the query embedding model (cached after first call).
        2. Load the pre-built FAISS index + metadata (cached after first call).
        3. Embed only the user question (~5 ms).
        4. FAISS inner-product search for top candidates (~0.1 ms).
        5. Filter candidates to the current database and keep Top-K.
        6. Fetch DDL for selected tables from SQLite and assemble pruned DDL.

        Args:
            question (str): The user's natural language question.
            db_path (str): Path to the target SQLite database.
            top_k (int): Maximum number of tables to retain.  Defaults to
                ``3``.  If the database has fewer tables, all are kept.
            oversampling_factor (int): Multiplier applied to ``top_k`` when
                fetching FAISS candidates before DB-ID filtering.  Increase if
                the target database's tables are under-represented in the index
                (e.g., a very large multi-DB index).  Defaults to ``10``.

        Returns:
            PruningResult: Fully populated result including selected tables,
                per-table similarity scores, and pruned DDL.

        Raises:
            FileNotFoundError: If ``db_path`` or the FAISS index files do not
                exist.
            ImportError: If ``sentence-transformers`` or ``faiss-cpu`` are not
                installed.
        """
        if not os.path.isfile(db_path):
            raise FileNotFoundError(
                f"[ChessLinker] Database not found: '{db_path}'"
            )

        # ── Load assets (cached after first call) ─────────────────────
        self._load_model()
        self._load_index()

        # ── Introspect target database ─────────────────────────────────
        all_tables: List[str] = self._get_all_tables(db_path)
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
        # Derive target db_id from the directory name (standard BIRD layout)
        db_id: str = os.path.basename(os.path.dirname(db_path))

        logger.info(
            "[ChessLinker] db_id='%s' — %d tables, selecting Top-%d via FAISS.",
            db_id,
            len(all_tables),
            effective_k,
        )

        # ── Embed the question ─────────────────────────────────────────
        query_vec: np.ndarray = self._embed_query(question)  # shape (1, D)

        # ── FAISS search — oversample to survive db_id filter ──────────
        n_candidates: int = min(
            effective_k * oversampling_factor,
            self._index.ntotal,  # type: ignore[union-attr]
        )
        scores_raw, indices = self._index.search(  # type: ignore[union-attr]
            query_vec, n_candidates
        )
        # scores_raw: (1, n_candidates), indices: (1, n_candidates)
        flat_scores: np.ndarray = scores_raw[0]
        flat_indices: np.ndarray = indices[0]

        # ── Filter to current db_id and build score dict ───────────────
        # table_name → best cosine score found in FAISS results
        faiss_scores: Dict[str, float] = {}
        for raw_idx, score in zip(flat_indices, flat_scores):
            if raw_idx == -1:
                continue  # FAISS pads with -1 when ntotal < n_candidates
            record: IndexRecord = self._metadata[raw_idx]  # type: ignore[index]
            if record.db_id != db_id:
                continue
            tname_lower = record.table_name.lower()
            # Keep best score if table appears multiple times (shouldn't happen
            # with a clean index, but defensive here)
            if tname_lower not in faiss_scores or score > faiss_scores[tname_lower]:
                faiss_scores[tname_lower] = float(score)

        # ── Map FAISS scores back to canonical (case-preserving) table names
        all_tables_lower = {t.lower(): t for t in all_tables}

        similarity_scores: Dict[str, float] = {}
        for t in all_tables:
            key = t.lower()
            similarity_scores[t] = round(faiss_scores.get(key, -1.0), 6)

        # ── Rank and select Top-K ──────────────────────────────────────
        ranked_tables: List[str] = sorted(
            all_tables,
            key=lambda t: similarity_scores[t],
            reverse=True,
        )
        selected_tables: List[str] = ranked_tables[:effective_k]

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

        # ── Fetch DDL for selected tables ──────────────────────────────
        ddl_parts: List[str] = []
        with self._connect(db_path) as conn:
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


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="ChessLinker online retrieval smoke-test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--db_path", type=str, required=True)
    parser.add_argument(
        "--question",
        type=str,
        default="How many customers are in the SME segment?",
    )
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument(
        "--index_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory containing schema_index.faiss and metadata.pkl.",
    )
    args = parser.parse_args()

    linker = ChessLinker(
        index_path=os.path.join(args.index_dir, INDEX_FILENAME),
        metadata_path=os.path.join(args.index_dir, METADATA_FILENAME),
    )
    result = linker.prune(
        question=args.question,
        db_path=args.db_path,
        top_k=args.top_k,
    )

    sep = "=" * 70
    print(f"\n{sep}\nCHESSLinker FAISS Result\n{sep}")
    print(f"Question        : {result.question}")
    print(f"All tables      : {result.all_tables}")
    print(f"Selected tables : {result.selected_tables}")
    print(f"Reduction ratio : {result.reduction_ratio:.0%}")
    print(f"\nSimilarity scores:")
    for t, s in sorted(
        result.similarity_scores.items(), key=lambda x: x[1], reverse=True
    ):
        marker = "✓" if t in result.selected_tables else "✗"
        print(f"  {marker} {t:<35} {s:.6f}")
    print(f"\nPruned DDL ({len(result.pruned_schema_ddl)} chars):")
    print(result.pruned_schema_ddl[:600] + ("..." if len(result.pruned_schema_ddl) > 600 else ""))
    print(sep)

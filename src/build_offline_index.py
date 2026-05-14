"""Offline Schema Embedding Indexer — Phase 0: Build FAISS Index.

This script implements the **Offline Indexing** phase of the two-phase
Text-to-SQL embedding pipeline.  It should be run **once** (or whenever the
underlying schema changes) to pre-compute and persist:

- ``schema_index.faiss``: An exact inner-product FAISS index containing
  L2-normalised table-representation embeddings.
- ``metadata.pkl``: A pickle mapping every FAISS vector position to its
  ``(db_id, table_name, table_representation)`` metadata tuple.

The companion module ``chess_linker.py`` then loads these artifacts at startup
and performs sub-millisecond online retrieval without ever touching the
embedding model during inference.

Architecture rationale
----------------------
- **Model**: ``BAAI/bge-small-en-v1.5`` — a 33 M-parameter, BGE-family model
  fine-tuned for symmetric semantic similarity and passage retrieval.  It
  natively outputs L2-normalised vectors, making inner product identical to
  cosine similarity, which is exactly what ``IndexFlatIP`` computes.
- **Index type**: ``faiss.IndexFlatIP`` — exact brute-force inner-product
  search.  No approximation (IVF / HNSW / LSH).  On the BIRD Mini-Dev
  dataset (~500 tables across ~11 databases) this is effectively free at
  query time (<1 ms) while guaranteeing 100% recall.
- **Normalisation**: All vectors are explicitly L2-normalised before being
  added to the index so that ``IndexFlatIP`` scores are mathematically
  equivalent to cosine similarity in ``[-1, 1]``.

Usage::

    # Index every database found in the default BIRD Mini-Dev data directory
    python src/build_offline_index.py

    # Custom data root and output paths
    python src/build_offline_index.py \\
        --tables_json data_minidev/MINIDEV/dev_tables.json \\
        --db_root     data_minidev/MINIDEV/dev_databases \\
        --output_dir  src/text2sql_agent/index \\
        --model       BAAI/bge-small-en-v1.5 \\
        --batch_size  64

Output files (written to ``--output_dir``)::

    schema_index.faiss   — FAISS IndexFlatIP (float32, D=384 for bge-small)
    metadata.pkl         — List[IndexRecord] parallel to the FAISS vectors
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import pickle
import sqlite3
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constants — kept in sync with chess_linker.py
# ---------------------------------------------------------------------------

DEFAULT_MODEL: str = "BAAI/bge-small-en-v1.5"
DEFAULT_TABLES_JSON: str = "data_minidev/MINIDEV/dev_tables.json"
DEFAULT_DB_ROOT: str = "data_minidev/MINIDEV/dev_databases"
DEFAULT_OUTPUT_DIR: str = "src/text2sql_agent/index"
INDEX_FILENAME: str = "schema_index.faiss"
METADATA_FILENAME: str = "metadata.pkl"


# ---------------------------------------------------------------------------
# Metadata record (one per FAISS vector)
# ---------------------------------------------------------------------------


@dataclass
class IndexRecord:
    """Metadata stored parallel to each FAISS vector.

    Attributes:
        db_id (str): Database identifier, e.g. ``"financial"``.
        table_name (str): Canonical table name as it appears in SQLite master.
        table_repr (str): The composite textual representation that was
            embedded.  Kept for debugging and offline introspection.
    """

    db_id: str
    table_name: str
    table_repr: str


# ---------------------------------------------------------------------------
# Schema helpers (reused from chess_linker logic, kept independent here)
# ---------------------------------------------------------------------------


@contextmanager
def _connect_ro(db_path: str) -> Generator[sqlite3.Connection, None, None]:
    """Yield a read-only SQLite connection, closing it on exit.

    Args:
        db_path (str): Absolute path to the ``.sqlite`` file.

    Yields:
        sqlite3.Connection: Open, read-only connection with ``Row`` factory.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _load_column_descriptions(db_path: str) -> Dict[str, Dict[str, str]]:
    """Load BIRD-style ``database_description`` CSV annotations.

    Args:
        db_path (str): Path to the SQLite database.  The
            ``database_description/`` folder is expected to be a sibling.

    Returns:
        Dict[str, Dict[str, str]]: Nested dict
            ``{table_lower: {column_lower: description_text}}``.
    """
    desc_dir = os.path.join(os.path.dirname(db_path), "database_description")
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
                norm = [h.strip().lstrip("\ufeff").lower() for h in header]

                def _get(row: List[str], names: Tuple[str, ...], default_idx: int) -> str:
                    for name in names:
                        if name in norm:
                            idx = norm.index(name)
                            if idx < len(row) and row[idx].strip():
                                return row[idx].strip()
                    return row[default_idx].strip() if default_idx < len(row) else ""

                for row in reader:
                    if not row:
                        continue
                    col = _get(row, ("original_column_name", "column_name"), 0)
                    desc = _get(row, ("column_description", "description"), 2)
                    val_desc = _get(row, ("value_description",), 4)
                    if col:
                        combined = "; ".join(p for p in [desc, val_desc] if p)
                        descriptions[table_name][col.lower()] = combined[:400]
        except Exception:  # noqa: BLE001
            logger.debug("Skipped description file '%s'.", path)

    return descriptions


def _build_table_repr(
    conn: sqlite3.Connection,
    table_name: str,
    descriptions: Dict[str, Dict[str, str]],
) -> str:
    """Build the composite textual representation of a table for embedding.

    The representation combines:
    - The table name.
    - Every column name and its declared SQL type.
    - BIRD-style column descriptions (when available).

    Args:
        conn (sqlite3.Connection): Open read-only connection.
        table_name (str): Target table name.
        descriptions (Dict[str, Dict[str, str]]): Pre-loaded column annotations.

    Returns:
        str: Whitespace-normalised composite text ready for embedding.
    """
    cursor = conn.execute(f'PRAGMA table_info("{table_name}");')
    columns = cursor.fetchall()
    table_desc = descriptions.get(table_name.lower(), {})
    parts: List[str] = [f"table {table_name}"]
    for col in columns:
        col_name: str = col["name"]
        col_type: str = col["type"] or "TEXT"
        annotation: str = table_desc.get(col_name.lower(), "")
        col_repr = f"column {col_name} type {col_type}"
        if annotation:
            col_repr += f" meaning {annotation}"
        parts.append(col_repr)
    return " , ".join(parts)


# ---------------------------------------------------------------------------
# Core indexing logic
# ---------------------------------------------------------------------------


def _enumerate_tables(
    tables_json_path: str,
    db_root: str,
) -> List[Tuple[str, str, str]]:
    """Enumerate all (db_id, table_name, db_path) triples from the schema JSON.

    If a database's ``.sqlite`` file is missing from ``db_root``, the database
    is skipped with a warning instead of raising an error, so partial datasets
    still produce a usable index.

    Args:
        tables_json_path (str): Path to ``dev_tables.json``.
        db_root (str): Root directory containing per-database sub-folders.

    Returns:
        List[Tuple[str, str, str]]: Triples of ``(db_id, table_name, db_path)``.
    """
    with open(tables_json_path, encoding="utf-8") as fh:
        schemas = json.load(fh)

    triples: List[Tuple[str, str, str]] = []
    for schema in schemas:
        db_id: str = schema["db_id"]
        db_path = os.path.join(db_root, db_id, f"{db_id}.sqlite")
        if not os.path.isfile(db_path):
            logger.warning(
                "SQLite file not found for db_id='%s' at '%s'. Skipping.",
                db_id,
                db_path,
            )
            continue
        for table_name in schema.get("table_names_original", []):
            triples.append((db_id, table_name, db_path))

    logger.info(
        "Enumerated %d (db_id, table) pairs across %d databases.",
        len(triples),
        len({t[0] for t in triples}),
    )
    return triples


def build_index(
    tables_json_path: str = DEFAULT_TABLES_JSON,
    db_root: str = DEFAULT_DB_ROOT,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 64,
) -> None:
    """Build and persist the offline FAISS index.

    Pipeline:
    1. Load all (db_id, table_name, db_path) triples from ``tables_json_path``.
    2. For each table, open the SQLite file and build its composite text
       representation (table name + columns + BIRD descriptions).
    3. Embed all representations in batches using ``BAAI/bge-small-en-v1.5``.
    4. L2-normalise all embeddings so inner-product == cosine similarity.
    5. Build a ``faiss.IndexFlatIP`` and add all normalised embeddings.
    6. Save the FAISS index and the parallel ``List[IndexRecord]`` metadata
       to ``output_dir``.

    Args:
        tables_json_path (str): Path to the BIRD ``dev_tables.json`` file.
        db_root (str): Root directory that contains per-database sub-folders.
        output_dir (str): Directory where ``schema_index.faiss`` and
            ``metadata.pkl`` are written.  Created if it does not exist.
        model_name (str): HuggingFace sentence-transformers model ID.
            Defaults to ``BAAI/bge-small-en-v1.5``.
        batch_size (int): Embedding batch size.  Tune down if memory is tight.
            Defaults to ``64``.

    Raises:
        ImportError: If ``sentence-transformers`` or ``faiss-cpu`` are not
            installed.
        FileNotFoundError: If ``tables_json_path`` does not exist.
    """
    # ── Validate dependencies early ────────────────────────────────────────
    try:
        import faiss  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "faiss-cpu is required to build the index.\n"
            "Install it with:  pip install faiss-cpu"
        ) from exc

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required to build the index.\n"
            "Install it with:  pip install sentence-transformers"
        ) from exc

    if not os.path.isfile(tables_json_path):
        raise FileNotFoundError(
            f"Schema JSON not found: '{tables_json_path}'. "
            "Run from the project root or pass --tables_json explicitly."
        )

    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Enumerate tables ───────────────────────────────────────────
    logger.info("=== Step 1/5: Enumerating tables ===")
    triples = _enumerate_tables(tables_json_path, db_root)
    if not triples:
        raise RuntimeError(
            "No tables found.  Check --tables_json and --db_root paths."
        )

    # ── Step 2: Build composite text representations ───────────────────────
    logger.info("=== Step 2/5: Building table representations ===")
    records: List[IndexRecord] = []
    table_reprs: List[str] = []

    # Cache open connections per db_path to avoid repeated open/close
    db_descriptions: Dict[str, Dict[str, Dict[str, str]]] = {}
    db_connections: Dict[str, sqlite3.Connection] = {}

    try:
        for db_id, table_name, db_path in triples:
            if db_path not in db_connections:
                conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
                conn.row_factory = sqlite3.Row
                db_connections[db_path] = conn
                db_descriptions[db_path] = _load_column_descriptions(db_path)

            conn = db_connections[db_path]
            descriptions = db_descriptions[db_path]
            try:
                repr_text = _build_table_repr(conn, table_name, descriptions)
            except Exception:  # noqa: BLE001
                logger.debug(
                    "Failed to build repr for %s.%s, using fallback.",
                    db_id,
                    table_name,
                )
                repr_text = f"table {table_name}"

            records.append(
                IndexRecord(
                    db_id=db_id,
                    table_name=table_name,
                    table_repr=repr_text,
                )
            )
            table_reprs.append(repr_text)
    finally:
        for conn in db_connections.values():
            conn.close()

    logger.info("Built %d table representations.", len(records))

    # ── Step 3: Embed with BGE ─────────────────────────────────────────────
    logger.info(
        "=== Step 3/5: Embedding with '%s' (batch_size=%d) ===",
        model_name,
        batch_size,
    )
    model = SentenceTransformer(model_name)
    # BGE models benefit from the instruction prefix for asymmetric tasks,
    # but for symmetric schema-to-schema similarity we omit it.
    embeddings: np.ndarray = model.encode(
        table_reprs,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=False,  # We normalise manually below
    )
    logger.info(
        "Embeddings shape: %s, dtype: %s", embeddings.shape, embeddings.dtype
    )

    # ── Step 4: L2-normalise (cosine → inner product equivalence) ─────────
    logger.info("=== Step 4/5: L2-normalising embeddings ===")
    norms: np.ndarray = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Guard against degenerate zero vectors (should never happen with BGE)
    norms = np.where(norms == 0.0, 1.0, norms)
    embeddings_norm: np.ndarray = (embeddings / norms).astype(np.float32)
    logger.info(
        "Normalised embeddings — mean norm: %.6f (should be ~1.0)",
        float(np.linalg.norm(embeddings_norm, axis=1).mean()),
    )

    # ── Step 5: Build FAISS index and persist ─────────────────────────────
    logger.info("=== Step 5/5: Building FAISS IndexFlatIP and saving ===")
    dim: int = embeddings_norm.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_norm)

    index_path = os.path.join(output_dir, INDEX_FILENAME)
    metadata_path = os.path.join(output_dir, METADATA_FILENAME)

    faiss.write_index(index, index_path)
    logger.info("FAISS index saved → '%s'  (ntotal=%d)", index_path, index.ntotal)

    with open(metadata_path, "wb") as fh:
        pickle.dump(records, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(
        "Metadata pickle saved → '%s'  (%d records)", metadata_path, len(records)
    )

    # ── Summary ───────────────────────────────────────────────────────────
    print(
        f"\n{'='*60}\n"
        f"  Offline Index Build — COMPLETE\n"
        f"{'='*60}\n"
        f"  Model       : {model_name}\n"
        f"  Dimension   : {dim}\n"
        f"  Vectors     : {index.ntotal}\n"
        f"  Databases   : {len({r.db_id for r in records})}\n"
        f"  FAISS index : {index_path}\n"
        f"  Metadata    : {metadata_path}\n"
        f"{'='*60}\n"
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="build_offline_index",
        description=(
            "Build a FAISS offline schema index for the CHESS linker.\n"
            "Run from the project root directory."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tables_json",
        type=str,
        default=DEFAULT_TABLES_JSON,
        help="Path to the BIRD dev_tables.json schema file.",
    )
    parser.add_argument(
        "--db_root",
        type=str,
        default=DEFAULT_DB_ROOT,
        help="Root directory containing per-database SQLite sub-folders.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where schema_index.faiss and metadata.pkl are saved.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="HuggingFace sentence-transformers model ID for embedding.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Embedding batch size (reduce if OOM).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()
    build_index(
        tables_json_path=args.tables_json,
        db_root=args.db_root,
        output_dir=args.output_dir,
        model_name=args.model,
        batch_size=args.batch_size,
    )

"""Schema Explorer Node — CHESS Pruning + MCI Metadata + Context Assembly.

This is the unified schema enrichment node for the AgentSQL LangGraph workflow.
It absorbs three formerly separate pipeline phases into one offline (zero LLM API cost) step:

Phase 1 — CHESS Semantic Pruning
    Uses a pre-built FAISS index to select the Top-K tables most relevant to the question.
    Falls back to all tables if the index is unavailable.

Phase 2 — MCI Metadata Enrichment (from MCI-SQL paper)
    Calls ``MetadataExtractor.build_context()`` on CHESS-selected tables only.
    Extracts: MIN/MAX for numeric columns, cardinality (1:1 vs 1:N), 3 random
    non-null text samples, distinct_count, total_rows.

Phase 3 — Rich Context Assembly
    Assembles a single, token-optimised ``schema_context`` string combining:
      - Raw DDL with PK/FK annotation and cardinality/null_ratio inline hints
      - MCI metadata JSON (from Phase 2)
      - MAGIC correction guidelines (for the generator to self-check)

This node only runs for COMPLEX queries (SIMPLE queries bypass it via the router).
"""

import os
import sqlite3
import logging
from typing import Any, Dict, List, Set

from ..core.state import AgentState
from ..tools.chess_linker import ChessLinker
from ..tools.metadata_extractor import MetadataExtractor

logger = logging.getLogger(__name__)

# MAGIC correction checklist — embedded into schema context so the generator
# can self-check before outputting SQL (reduces correction iterations).
_MAGIC_GUIDELINES: str = """\
MAGIC SQL Self-Check Checklist:
1. LIMIT/ORDER BY: for "top/most/least/highest/lowest" queries.
2. Aggregation: per-entity averages need a subquery before averaging.
3. Ratios: use conditional SUM/COUNT with CAST(... AS FLOAT).
4. Filters: verify literal values against MCI metadata samples below.
5. Dates: match storage format; use strftime()/SUBSTR() when needed.
6. Joins: use FK/PK columns explicitly; avoid spurious DISTINCT.
7. Projection: SELECT only the columns the question explicitly requests.
8. Extremes: prefer LIMIT 1 + ORDER BY over MIN/MAX subqueries.
"""


def _analyze_column_cardinality(
    cursor: sqlite3.Cursor, table_name: str, col_name: str
) -> str:
    """Analyses column cardinality and null density directly from SQLite.

    Args:
        cursor (sqlite3.Cursor): Open SQLite cursor.
        table_name (str): The table containing the column.
        col_name (str): Column to analyse.

    Returns:
        str: A metadata annotation like ``'[cardinality: 1:N, null_ratio: 0.15]'``.
    """
    try:
        cursor.execute(
            f'SELECT COUNT(*), COUNT(DISTINCT "{col_name}"), '
            f'SUM(CASE WHEN "{col_name}" IS NULL THEN 1 ELSE 0 END) '
            f'FROM "{table_name}"'
        )
        row = cursor.fetchone()
        if not row:
            return "[cardinality: UNKNOWN, null_ratio: 0.00]"

        total_count, distinct_count, null_count = row
        total_count = total_count or 0
        distinct_count = distinct_count or 0
        null_count = null_count or 0

        if total_count == 0:
            cardinality = "EMPTY"
        elif distinct_count == total_count:
            cardinality = "1:1 (PK-like)"
        else:
            cardinality = "1:N (FK/Categorical)"

        null_ratio = null_count / total_count if total_count > 0 else 0.0
        null_flag = ", predominantly NULL" if null_ratio > 0.50 else ""
        return f"[cardinality: {cardinality}, null_ratio: {null_ratio:.2f}{null_flag}]"
    except Exception as exc:
        logger.debug(
            "Failed to analyse column %s.%s metadata: %s", table_name, col_name, exc
        )
        return "[cardinality: UNKNOWN, null_ratio: 0.00]"


def explorer_node(state: AgentState) -> Dict[str, Any]:
    """Explores, prunes, and richly enriches the database schema context.

    Executes three offline phases (zero LLM API cost):
      1. CHESS semantic pruning via FAISS index.
      2. MCI metadata extraction via ``MetadataExtractor``.
      3. Context assembly into a single ``schema_context`` string.

    Args:
        state (AgentState): The current LangGraph workflow state.

    Returns:
        Dict[str, Any]: State update dict with ``schema_context`` (rich, assembled)
            and ``metadata_context`` (raw MCI JSON, stored separately for the
            corrector node to reference).
    """
    question = state["question"]
    db_path = state["db_path"]
    evidence = state.get("evidence", "") or ""
    top_k = int(os.environ.get("CHESS_TOP_K", "3"))

    logger.info("[schema_explorer] Starting schema enrichment for db: %s", db_path)

    # ── Phase 1: CHESS Semantic Pruning ──────────────────────────────────────
    selected_tables: List[str] = []
    try:
        index_dir = "src/text2sql_agent/index"
        linker = ChessLinker(
            index_path=os.path.join(index_dir, "schema_index.faiss"),
            metadata_path=os.path.join(index_dir, "metadata.pkl"),
        )
        pruned_res = linker.prune(question=question, db_path=db_path, top_k=top_k)
        selected_tables = pruned_res.selected_tables
        logger.info(
            "[schema_explorer] CHESS pruned: %d/%d tables retained: %s",
            len(selected_tables),
            len(pruned_res.all_tables),
            selected_tables,
        )
    except Exception as exc:
        logger.warning(
            "[schema_explorer] ChessLinker failed (%s). Falling back to all tables.",
            exc,
        )

    # ── Phase 1b: Fallback — use all tables if CHESS did not run ─────────────
    ddl_dump: List[str] = []
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if not selected_tables:
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%'"
            )
            selected_tables = [r["name"] for r in cursor.fetchall()]
            logger.info(
                "[schema_explorer] Using all %d tables (no CHESS index).",
                len(selected_tables),
            )

        # ── Phase 1c: DDL + PK/FK Retention + Cardinality Annotation ─────────
        for table_name in selected_tables:
            ddl_dump.append(f"----- Table: {table_name} -----")

            cursor.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            )
            table_sql = cursor.fetchone()
            if table_sql and table_sql["sql"]:
                ddl_dump.append(f"{table_sql['sql']};")

            # PK and FK extraction for Mandatory Key Retention
            cursor.execute(f'PRAGMA table_info("{table_name}")')
            columns = cursor.fetchall()
            pk_cols: Set[str] = {r["name"].lower() for r in columns if r["pk"] > 0}

            cursor.execute(f'PRAGMA foreign_key_list("{table_name}")')
            fks = cursor.fetchall()
            fk_cols: Set[str] = {fk["from"].lower() for fk in fks}

            column_lines: List[str] = []
            pruned_count = 0

            for col in columns:
                col_name = col["name"]
                col_name_lower = col_name.lower()
                is_pk = col_name_lower in pk_cols
                is_fk = col_name_lower in fk_cols
                is_relevant = col_name_lower in question.lower() or (
                    evidence and col_name_lower in evidence.lower()
                )

                # Keep: PK, FK, or semantically relevant
                if is_pk or is_fk or is_relevant:
                    pk_suffix = " PRIMARY KEY" if col["pk"] else ""
                    cardinality_hint = _analyze_column_cardinality(
                        cursor, table_name, col_name
                    )
                    column_lines.append(
                        f"  - {col_name} ({col['type'] or 'UNKNOWN'}{pk_suffix}) "
                        f"{cardinality_hint}"
                    )
                else:
                    pruned_count += 1

            if column_lines:
                ddl_dump.append("Columns (retained):\n" + "\n".join(column_lines))
            if pruned_count > 0:
                ddl_dump.append(
                    f"  [{pruned_count} non-key irrelevant columns pruned to save tokens]"
                )

            if fks:
                fk_lines = [
                    f"  - {table_name}.{fk['from']} → {fk['table']}.{fk['to']}"
                    for fk in fks
                ]
                ddl_dump.append("Foreign Keys:\n" + "\n".join(fk_lines))

            # Sample rows for literal value grounding
            cursor.execute(f'SELECT * FROM "{table_name}" LIMIT 3')
            samples = [dict(row) for row in cursor.fetchall()]
            ddl_dump.append(f"Sample rows: {samples}\n")

        conn.close()

    except Exception as exc:
        logger.error("[schema_explorer] DDL extraction failed: %s", exc)
        return {"schema_context": f"Error extracting schema: {exc}"}

    pruned_ddl = "\n".join(ddl_dump)
    logger.info(
        "[schema_explorer] DDL context built: %d chars.", len(pruned_ddl)
    )

    # ── Phase 2: MCI Metadata Enrichment ────────────────────────────────────
    metadata_context = ""
    try:
        extractor = MetadataExtractor(db_path=db_path)
        metadata_context = extractor.build_context(tables=selected_tables)
        logger.info(
            "[schema_explorer] MCI metadata extracted: %d chars.", len(metadata_context)
        )
    except Exception as exc:
        logger.warning(
            "[schema_explorer] MetadataExtractor failed (%s). Proceeding without MCI.", exc
        )

    # ── Phase 3: Context Assembly ─────────────────────────────────────────────
    evidence_block = f"-- Evidence/Hint: {evidence}\n" if evidence else ""

    schema_context = (
        f"-- === PRUNED SCHEMA (CHESS Top-{top_k}, DDL + PK/FK + Cardinality) ===\n"
        f"{pruned_ddl}\n\n"
        + (
            f"-- === MCI METADATA (MIN/MAX, samples, cardinality — offline extracted) ===\n"
            f"-- Keys: affinity, min, max, distinct_count, total_rows, cardinality, samples.\n"
            f"-- Use these to validate literal values and predicates.\n"
            f"{metadata_context}\n\n"
            if metadata_context
            else ""
        )
        + f"-- === MAGIC SELF-CHECK GUIDELINES ===\n"
        f"{_MAGIC_GUIDELINES}\n"
        + evidence_block
    )

    logger.info(
        "[schema_explorer] Assembled schema_context: %d chars.", len(schema_context)
    )

    return {
        "schema_context": schema_context,
        "metadata_context": metadata_context,
    }

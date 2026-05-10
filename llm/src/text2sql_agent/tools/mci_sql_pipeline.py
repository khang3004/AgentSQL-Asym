"""Module 3: MCI-SQL Integration Pipeline.

This module demonstrates how the ``MetadataExtractor`` (Module 1) and the
``SemanticErrorChecker`` (Module 2) integrate into the existing MAGIC-based
AgentSQL framework to form a quota-efficient, semantically aware Text-to-SQL
pipeline.

Pipeline flow (all offline steps precede any API call):

::

    ┌─────────────────────────────────────────────────────────────────┐
    │               OFFLINE / LOCAL PHASE (zero API cost)            │
    │                                                                 │
    │  1. MetadataExtractor.build_context()                          │
    │     → Queries MIN/MAX, cardinality, text samples locally        │
    │     → Returns compact JSON metadata string                      │
    │                                                                 │
    └────────────────────────────┬────────────────────────────────────┘
                                 │ metadata_context injected into prompt
    ┌────────────────────────────▼────────────────────────────────────┐
    │                   LLM API CALL 1 (Groq)                        │
    │         Generator: meta-llama/llama-4-scout-17b-16e-instruct    │
    │         Receives enriched prompt → returns candidate SQL        │
    └────────────────────────────┬────────────────────────────────────┘
                                 │ candidate_sql
    ┌────────────────────────────▼────────────────────────────────────┐
    │          OFFLINE / LOCAL PHASE (zero API cost)                  │
    │                                                                 │
    │  2. SemanticErrorChecker.execute_safe()                         │
    │     → Executes SQL locally on SQLite                            │
    │     → Classifies: SUCCESS | EmptyResultError | NullResultError  │
    │       | sqlite3.OperationalError                                │
    │                                                                 │
    └───────────────┬──────────────────────┬───────────────────────── ┘
                    │ SUCCESS              │ ERROR (with suggestion)
                    ▼                      ▼
             Return rows          LLM API CALL 2 (Gemini 2.5 Flash)
                                  Critic: corrects SQL using the
                                  embedded suggestion in error_msg
                                          │
                                          ▼
                                   Corrected SQL returned

Design Principles:
    - **Offline-First**: Both metadata extraction and semantic checking run
      entirely on the local SQLite file.  No extra LLM calls are made solely
      to gather database facts or diagnose empty/null results.
    - **Decoupled Modules**: ``MetadataExtractor`` and ``SemanticErrorChecker``
      are independent of each other and of the LLM clients.  Each can be
      tested in isolation.
    - **Actionable Feedback**: Custom exceptions carry ``suggestion`` strings
      formatted specifically for the critic LLM's correction prompt, enabling
      the critic to fix semantic issues without additional back-and-forth.
    - **PEP 8 / PEP 526 Compliant**: Full type annotations throughout.
"""

import logging
import os
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from langsmith import traceable
from text2sql_agent.tools.metadata_extractor import MetadataExtractor
from text2sql_agent.tools.semantic_error_checker import (
    EmptyResultError,
    NullResultError,
    SemanticErrorChecker,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Containers
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Immutable result container produced by ``run_mci_sql_pipeline``.

    Captures the full execution trace of one pipeline invocation, including
    whether correction was triggered and what the critic LLM returned.

    Attributes:
        question (str): The original natural language question.
        db_path (str): Path to the SQLite database used.
        final_sql (str): The definitive SQL query (post-correction if any).
        rows (Optional[List[Tuple[Any, ...]]]): The result rows returned by the
            final SQL execution.  ``None`` if the pipeline ended in an
            unrecoverable error.
        metadata_context (str): The compact JSON metadata context that was
            injected into the generator prompt.
        generator_raw_sql (str): The raw SQL returned by the generator LLM
            (Llama-4-Scout) before any correction.
        semantic_error_message (Optional[str]): The full error string (including
            the suggestion) produced by ``SemanticErrorChecker`` when the
            generator SQL failed.  ``None`` on first-attempt success.
        critic_corrected_sql (Optional[str]): The SQL produced by the critic
            LLM (Gemini 2.5 Flash) after receiving the semantic error.
            ``None`` if no correction was necessary.
        api_calls_made (int): Total number of LLM API calls made during this
            pipeline run (minimum 1 for the generator, maximum 2 when
            correction is triggered).
    """

    question: str
    db_path: str
    final_sql: str
    rows: Optional[List[Tuple[Any, ...]]]
    metadata_context: str
    generator_raw_sql: str
    semantic_error_message: Optional[str] = field(default=None)
    critic_corrected_sql: Optional[str] = field(default=None)
    api_calls_made: int = field(default=1)


# ---------------------------------------------------------------------------
# Dummy LLM Stubs
# (Replace with real ``groq_request.connect_groq`` /
#  ``gemini_request.connect_gemini`` in production.)
# ---------------------------------------------------------------------------


@traceable(run_type="llm")
def _stub_generator_llm(
    enriched_prompt: str,
    model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
) -> str:
    """Stub simulating the Groq Llama-4-Scout SQL generator LLM call.

    In production this function is replaced by a call to
    ``llm.src.groq_request.connect_groq`` (or the ``get_llm`` factory in
    ``text2sql_agent.core.llm_factory``).  The stub is intentionally
    deterministic so that the integration logic can be unit-tested offline.

    Args:
        enriched_prompt (str): The fully assembled prompt, including the
            MCI metadata context and the natural language question.
        model (str): Groq model identifier.  Defaults to the fast
            ``llama-4-scout-17b`` model used in the project's groq_request.py.

    Returns:
        str: A dummy SQL SELECT string.
    """
    logger.info(
        "[STUB] Generator LLM (%s) invoked. Prompt length: %d chars.",
        model,
        len(enriched_prompt),
    )
    # Return a deliberately imprecise query to exercise the semantic checker.
    return "SELECT name FROM customers WHERE segment = 'VIP'"


@traceable(run_type="llm")
def _stub_critic_llm(
    correction_prompt: str,
    model: str = "gemini-2.5-flash",
) -> str:
    """Stub simulating the Gemini 2.5 Flash critic LLM correction call.

    In production this function delegates to
    ``llm.src.gemini_request.connect_gemini``.

    Args:
        correction_prompt (str): The correction prompt including the semantic
            error description and its embedded suggestion string.
        model (str): Gemini model identifier.  Defaults to ``gemini-2.5-flash``
            as used in the project's gemini_request.py.

    Returns:
        str: A corrected SQL SELECT string.
    """
    logger.info(
        "[STUB] Critic LLM (%s) invoked for correction. Prompt length: %d chars.",
        model,
        len(correction_prompt),
    )
    # Return a query that applies the LIKE suggestion from EmptyResultError.
    return "SELECT name FROM customers WHERE LOWER(segment) LIKE LOWER('%vip%')"


# ---------------------------------------------------------------------------
# Prompt Assembly Helpers
# ---------------------------------------------------------------------------


@traceable(run_type="tool")
def build_enriched_generator_prompt(
    question: str,
    schema_ddl: str,
    metadata_context: str,
    evidence: str = "",
    sql_dialect: str = "SQLite",
) -> str:
    """Assembles the enriched generator prompt with injected MCI metadata.

    Combines the standard DDL schema (used by the existing ``generate_combined_
    prompts_one`` function) with the compact JSON metadata context produced by
    ``MetadataExtractor.build_context``.  The metadata section is placed
    between the DDL and the question to preserve the existing prompt structure
    while adding richer context.

    Args:
        question (str): The natural language question to be answered.
        schema_ddl (str): The CREATE TABLE DDL string for all relevant tables,
            as produced by ``table_schema.generate_schema_prompt_sqlite``.
        metadata_context (str): The compact JSON metadata string from
            ``MetadataExtractor.build_context``.
        evidence (str): Optional domain knowledge / hint string.
        sql_dialect (str): Target SQL dialect label.  Defaults to ``"SQLite"``.

    Returns:
        str: The fully assembled prompt string ready for LLM submission.
    """
    evidence_section: str = (
        f"-- External Knowledge: {evidence}\n" if evidence else ""
    )

    prompt = (
        f"-- Using valid {sql_dialect}, answer the question below.\n\n"
        f"-- === DATABASE SCHEMA (DDL) ===\n"
        f"{schema_ddl}\n\n"
        f"-- === METADATA-COMPLETE CONTEXT (MCI) ===\n"
        f"-- This JSON encodes statistical metadata for each column.\n"
        f"-- Keys: affinity, min, max, distinct_count, total_rows,\n"
        f"--       cardinality ('1:1 (PK-like)' | '1:N (FK/Categorical)'),\n"
        f"--       samples (text columns only).\n"
        f"-- Use this to validate literal values and choose correct predicates.\n"
        f"{metadata_context}\n\n"
        f"-- === QUESTION ===\n"
        f"-- {question}\n"
        f"{evidence_section}"
        f"\nGenerate the {sql_dialect} query after thinking step by step:\n"
        f"In your response, do not include comments. "
        f"Return only the SQL starting from SELECT.\n"
    )
    return prompt


@traceable(run_type="tool")
def build_correction_prompt(
    question: str,
    schema_ddl: str,
    metadata_context: str,
    bad_sql: str,
    error_message: str,
    evidence: str = "",
) -> str:
    """Assembles the critic correction prompt embedding the semantic error.

    The error message produced by ``SemanticErrorChecker`` already contains
    the actionable suggestion (e.g., "use LIKE instead of =").  This function
    embeds it verbatim so the critic LLM receives structured, specific guidance
    rather than a generic "your query failed" message.

    Args:
        question (str): The original natural language question.
        schema_ddl (str): DDL context for all relevant tables.
        metadata_context (str): MCI metadata JSON context.
        bad_sql (str): The SQL query that failed semantic evaluation.
        error_message (str): The full error string from ``SemanticErrorChecker``
            including the embedded suggestion.
        evidence (str): Optional domain knowledge hint.

    Returns:
        str: The correction prompt for the critic LLM (Gemini 2.5 Flash).
    """
    evidence_section: str = (
        f"Evidence/Hint: {evidence}\n" if evidence else ""
    )

    prompt = (
        f"You are an expert SQLite Text-to-SQL correction agent.\n"
        f"A previously generated query failed semantic validation.\n"
        f"Your task: produce one corrected, read-only SQLite SELECT query.\n\n"
        f"=== QUESTION ===\n{question}\n\n"
        f"{evidence_section}"
        f"=== DATABASE SCHEMA ===\n{schema_ddl}\n\n"
        f"=== METADATA-COMPLETE CONTEXT ===\n{metadata_context}\n\n"
        f"=== FAILED SQL ===\n```sql\n{bad_sql}\n```\n\n"
        f"=== SEMANTIC ERROR & CORRECTION HINT ===\n{error_message}\n\n"
        f"Apply the correction hint above. Output only the corrected SQL "
        f"starting from SELECT, with no comments or markdown fences.\n"
    )
    return prompt


# ---------------------------------------------------------------------------
# Main Pipeline Function
# ---------------------------------------------------------------------------


@traceable(run_type="chain")
def run_mci_sql_pipeline(
    question: str,
    db_path: str,
    tables: List[str],
    schema_ddl: str,
    evidence: str = "",
    columns: Optional[Dict[str, List[str]]] = None,
    generator_model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
    critic_model: str = "gemini-2.5-flash",
    sql_dialect: str = "SQLite",
    generator_fn: Any = _stub_generator_llm,
    critic_fn: Any = _stub_critic_llm,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> PipelineResult:
    """Runs the full MCI-SQL augmented Text-to-SQL pipeline for one question.

    This function orchestrates the three-phase pipeline:

    Phase 1 — *Offline Context Enrichment* (zero LLM API cost):
        ``MetadataExtractor`` profiles the target tables locally and returns a
        compact JSON metadata string.

    Phase 2 — *Generator LLM Call* (1 API call to Groq Llama-4-Scout):
        The enriched prompt (DDL + MCI metadata + question) is sent to the
        generator model.  The raw SQL candidate is extracted.

    Phase 3 — *Offline Semantic Validation* (zero LLM API cost):
        ``SemanticErrorChecker.execute_safe`` runs the candidate SQL against
        the local SQLite file and classifies the result as SUCCESS, EMPTY, or
        NULL.

    Phase 4 (conditional) — *Critic LLM Correction* (0 or 1 API calls):
        If Phase 3 detected a semantic error, the correction prompt (including
        the embedded suggestion) is sent to the Gemini 2.5 Flash critic.  The
        corrected SQL is re-executed for final validation.

    The function is deliberately **decoupled** from the actual LLM clients:
    ``generator_fn`` and ``critic_fn`` accept callable stubs for testing and
    real API functions (``connect_groq``, ``connect_gemini``) in production.

    Args:
        question (str): The natural language question to answer with SQL.
        db_path (str): Path to the local SQLite database file.
        tables (List[str]): Table names to profile with ``MetadataExtractor``.
        schema_ddl (str): Pre-generated DDL string for the schema context.
        evidence (str): Optional domain knowledge hint.  Defaults to ``""``.
        columns (Optional[Dict[str, List[str]]]): Optional column subset to
            profile per table.  ``None`` means all columns are profiled.
        generator_model (str): Identifier of the generator LLM.
        critic_model (str): Identifier of the critic LLM.
        sql_dialect (str): Target SQL dialect label.  Defaults to ``"SQLite"``.
        generator_fn (Any): Callable with signature
            ``(enriched_prompt: str, model: str) -> str``.
            Defaults to the dummy stub ``_stub_generator_llm``.
        critic_fn (Any): Callable with signature
            ``(correction_prompt: str, model: str) -> str``.
            Defaults to the dummy stub ``_stub_critic_llm``.

    Returns:
        PipelineResult: A fully populated result container capturing the
            entire execution trace, including whether correction was triggered
            and the total number of API calls made.

    Raises:
        FileNotFoundError: If ``db_path`` does not point to a valid file
            (propagated from ``MetadataExtractor`` or ``SemanticErrorChecker``).

    Example::

        result = run_mci_sql_pipeline(
            question="What are the names of customers in the VIP segment?",
            db_path="data_minidev/MINIDEV/dev_databases/debit_card/debit_card.sqlite",
            tables=["customers"],
            schema_ddl=generate_schema_prompt_sqlite(db_path),
        )
        print(result.final_sql)
        print(f"API calls consumed: {result.api_calls_made}")
    """
    logger.info("[Pipeline] ===== MCI-SQL Pipeline Start =====")
    logger.info("[Pipeline] Question: %s", question)

    # ------------------------------------------------------------------
    # Phase 1: Offline Metadata Extraction (0 API calls)
    # ------------------------------------------------------------------
    logger.info("[Pipeline] Phase 1 — Extracting MCI metadata locally...")
    extractor = MetadataExtractor(db_path=db_path)
    metadata_context: str = extractor.build_context(
        tables=tables,
        columns=columns,
    )
    logger.info(
        "[Pipeline] Metadata context built (%d chars).",
        len(metadata_context),
    )

    # ------------------------------------------------------------------
    # Phase 2: Generator LLM Call (1 API call)
    # ------------------------------------------------------------------
    logger.info(
        "[Pipeline] Phase 2 — Invoking Generator LLM: %s", generator_model
    )
    enriched_prompt: str = build_enriched_generator_prompt(
        question=question,
        schema_ddl=schema_ddl,
        metadata_context=metadata_context,
        evidence=evidence,
        sql_dialect=sql_dialect,
    )
    generator_raw_sql: str = generator_fn(enriched_prompt, generator_model)
    api_calls_made: int = 1
    logger.info(
        "[Pipeline] Generator returned SQL: %s",
        generator_raw_sql[:120].replace("\n", " "),
    )

    # ------------------------------------------------------------------
    # Phase 3: Offline Semantic Validation (0 API calls)
    # ------------------------------------------------------------------
    logger.info(
        "[Pipeline] Phase 3 — Running SemanticErrorChecker locally..."
    )
    checker = SemanticErrorChecker(db_path=db_path)
    rows, error_message = checker.execute_safe(generator_raw_sql)

    # ------------------------------------------------------------------
    # Phase 4 (Conditional): Critic LLM Correction (0 or 1 API calls)
    # ------------------------------------------------------------------
    final_sql: str = generator_raw_sql
    critic_corrected_sql: Optional[str] = None

    if error_message is not None:
        logger.warning(
            "[Pipeline] Phase 4 — Semantic error detected. "
            "Invoking Critic LLM: %s",
            critic_model,
        )
        correction_prompt: str = build_correction_prompt(
            question=question,
            schema_ddl=schema_ddl,
            metadata_context=metadata_context,
            bad_sql=generator_raw_sql,
            error_message=error_message,
            evidence=evidence,
        )
        critic_corrected_sql = critic_fn(correction_prompt, critic_model)
        api_calls_made += 1
        final_sql = critic_corrected_sql

        # Re-execute the corrected SQL to confirm it passes semantic checks.
        rows, final_error = checker.execute_safe(critic_corrected_sql)
        if final_error:
            logger.error(
                "[Pipeline] Critic correction still failed: %s",
                final_error[:200],
            )
        else:
            logger.info(
                "[Pipeline] Critic correction succeeded. "
                "Rows returned: %d",
                len(rows) if rows else 0,
            )
    else:
        logger.info(
            "[Pipeline] Phase 3 passed. No correction needed. "
            "Rows returned: %d",
            len(rows) if rows else 0,
        )

    logger.info(
        "[Pipeline] ===== Pipeline Complete. "
        "Total API calls: %d =====",
        api_calls_made,
    )

    return PipelineResult(
        question=question,
        db_path=db_path,
        final_sql=final_sql,
        rows=rows,
        metadata_context=metadata_context,
        generator_raw_sql=generator_raw_sql,
        semantic_error_message=error_message,
        critic_corrected_sql=critic_corrected_sql,
        api_calls_made=api_calls_made,
    )


# ---------------------------------------------------------------------------
# CLI demo entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description=(
            "MCI-SQL Integration Pipeline demo.  "
            "Runs the full offline + LLM pipeline on a single question."
        )
    )
    parser.add_argument(
        "--db_path",
        type=str,
        required=True,
        help="Path to the target SQLite database file.",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="How many customers are in the SME segment?",
        help="Natural language question to answer.",
    )
    parser.add_argument(
        "--tables",
        type=str,
        nargs="+",
        default=["customers"],
        help="Table names to profile with MetadataExtractor.",
    )
    args = parser.parse_args()

    # Build a minimal DDL for the demo (normally from table_schema.py)
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from table_schema import generate_schema_prompt_sqlite
        ddl = generate_schema_prompt_sqlite(args.db_path, num_rows=3)
    except ImportError:
        ddl = "-- DDL not available in standalone mode."

    result: PipelineResult = run_mci_sql_pipeline(
        question=args.question,
        db_path=args.db_path,
        tables=args.tables,
        schema_ddl=ddl,
    )

    print("\n" + "=" * 70)
    print("PIPELINE RESULT SUMMARY")
    print("=" * 70)
    print(f"Question       : {result.question}")
    print(f"Final SQL      : {result.final_sql}")
    print(f"Rows returned  : {len(result.rows) if result.rows else 0}")
    print(f"API calls made : {result.api_calls_made}  (max=2 per question)")
    if result.semantic_error_message:
        print(f"Semantic error : {result.semantic_error_message[:200]}")
    print("=" * 70)

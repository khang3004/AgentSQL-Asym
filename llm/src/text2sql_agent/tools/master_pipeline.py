"""MasterPipeline — Unified CHESS + MCI-SQL + MAGIC orchestrator.

This module integrates the three upstream modules into a single, traceable
end-to-end Text-to-SQL pipeline:

::

    ┌──────────────────────────────────────────────────────────────────┐
    │  PHASE 1 — CHESS SEMANTIC PRUNING               [OFFLINE / FREE] │
    │  ChessLinker.prune(question, db_path, top_k)                     │
    │  • sentence-transformers cosine similarity on all tables          │
    │  • Returns Top-K tables + pruned DDL + similarity scores         │
    │  • Logs reduction ratio (e.g., "10 tables → 3, pruned 70 %")    │
    └───────────────────────────┬──────────────────────────────────────┘
                                │ selected_tables, pruned_ddl
    ┌───────────────────────────▼──────────────────────────────────────┐
    │  PHASE 2 — MCI-SQL METADATA ENRICHMENT          [OFFLINE / FREE] │
    │  MetadataExtractor.build_context(selected_tables)                │
    │  • MIN/MAX for numeric columns                                    │
    │  • DISTINCT vs. COUNT(*) → cardinality label                     │
    │  • 3 random non-null samples for text columns                    │
    │  • Compact JSON string (no whitespace)                           │
    └───────────────────────────┬──────────────────────────────────────┘
                                │ metadata_context JSON
    ┌───────────────────────────▼──────────────────────────────────────┐
    │  PHASE 3 — CONTEXT ASSEMBLY                     [OFFLINE / FREE] │
    │  • Pruned DDL + MCI metadata JSON → single optimised prompt      │
    │  • MAGIC self-check guidelines appended to generation prompt      │
    └───────────────────────────┬──────────────────────────────────────┘
                                │ enriched_prompt
    ┌───────────────────────────▼──────────────────────────────────────┐
    │  PHASE 4a — SQL GENERATION          [LLM API CALL #1 Groq Gen]   │
    │  Generator: meta-llama/llama-4-scout-17b-16e-instruct            │
    │  → candidate SQL                                                 │
    └───────────────────────────┬──────────────────────────────────────┘
                                │ candidate_sql
    ┌───────────────────────────▼──────────────────────────────────────┐
    │  PHASE 4ab — REFLECTION          [LLM API CALL #2 Groq Scout]    │
    │  Reflector: meta-llama/llama-4-scout-17b-16e-instruct            │
    │  • Back-translate SQL → English and compare with question         │
    │  • If match → <ok>   If mismatch → <error>logical mismatch</error>│
    └──────────────┬────────────────────────────┬───────────────────────┘
               <ok>                        <error> logical mismatch
                   │                             │
    ┌──────────────▼──────────────────────────────────────┐
    │  PHASE 4b — SEMANTIC VALIDATION                 [OFFLINE / FREE] │
    │  SemanticErrorChecker.execute_safe(candidate_sql)                │
    └──────────────┬───────────────────────────────────────┘
              SUCCESS                  FAILURE + suggestion string
                  │                             │
                  └──── merge with reflection ────┘
                                    │ any error (semantic OR logical)
                         ┌──────────▼─────────────────────────────┐
                         │  PHASE 4c — MAGIC CORRECTION             │
                         │         [LLM API CALL #3 Groq Critic]    │
                         │  Critic: openai/gpt-oss-20b              │
                         │  • XML-tagged output: <sql>…</sql>         │
                         │  • Re-validated locally (free)           │
                         └─────────────────────────────────────────┘
                                   (api_calls ≤ 3)

Design Invariants:
    - Maximum 3 LLM API calls per question (Gen + Reflect + Correct).
    - Reflection catches silent logical errors SQLite cannot see.
    - All schema pruning, metadata extraction, and validation are local.
    - All public methods carry full PEP 526 type annotations.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from langsmith import traceable
from text2sql_agent.core.llm_factory import get_llm
from text2sql_agent.core.sql_utils import extract_sql
from text2sql_agent.tools.chess_linker import (
    ChessLinker,
    PruningResult,
    INDEX_FILENAME,
    METADATA_FILENAME,
)
from text2sql_agent.tools.metadata_extractor import MetadataExtractor
from text2sql_agent.tools.semantic_error_checker import (
    SemanticErrorChecker,
)

logger = logging.getLogger(__name__)

# MAGIC correction checklist (sourced from corrector.py, centralised here).
_MAGIC_GUIDELINES: str = """MAGIC SQL Correction Checklist:
1. LIMIT/ORDER BY: for "top/most/least/highest/lowest" queries.
2. Aggregation: per-entity averages need a subquery before averaging.
3. Ratios: use conditional SUM/COUNT with CAST(... AS FLOAT).
4. Filters: verify literal values against metadata samples.
5. Dates: match storage format; use strftime()/SUBSTR() when needed.
6. Joins: use FK/PK columns; avoid spurious DISTINCT.
7. Projection: select only the columns the question requests.
8. Extremes: prefer LIMIT 1 + ORDER BY over MIN/MAX subqueries.
"""


# ---------------------------------------------------------------------------
# Master result container
# ---------------------------------------------------------------------------


@dataclass
class MasterPipelineResult:
    """Complete execution trace produced by ``MasterPipeline.run``.

    Attributes:
        question (str): Original natural language question.
        db_path (str): Path to the SQLite database.
        pruning (PruningResult): Full CHESS pruning trace including
            similarity scores and reduction ratio.
        metadata_context (str): Compact MCI JSON string covering only
            the pruned tables.
        final_sql (str): Definitive SQL after all pipeline phases.
        rows (Optional[List[Tuple[Any, ...]]]): Result rows from the
            final SQL execution; ``None`` if unrecoverable error.
        generator_raw_sql (str): First-pass SQL from the generator LLM.
        semantic_error_message (Optional[str]): Error + suggestion from
            ``SemanticErrorChecker``; ``None`` on first-attempt success.
        critic_corrected_sql (Optional[str]): Critic output; ``None``
            when no correction was required.
        api_calls_made (int): Total LLM API calls (1 or 2).
        prompt_char_count (int): Character length of the enriched prompt
            sent to the generator, for token-budget auditing.
    """

    question: str
    db_path: str
    pruning: PruningResult
    metadata_context: str
    final_sql: str
    rows: Optional[List[Tuple[Any, ...]]]
    generator_raw_sql: str
    semantic_error_message: Optional[str] = field(default=None)
    reflection_error_message: Optional[str] = field(default=None)
    critic_corrected_sql: Optional[str] = field(default=None)
    api_calls_made: int = field(default=1)
    prompt_char_count: int = field(default=0)


# ---------------------------------------------------------------------------
# MasterPipeline
# ---------------------------------------------------------------------------


class MasterPipeline:
    """Orchestrates the four-phase CHESS + MCI-SQL + MAGIC pipeline.

    Instantiate once and call ``run()`` for each question.  The FAISS index
    and the ``BAAI/bge-small-en-v1.5`` query encoder are loaded lazily on the
    first call and cached for the lifetime of the instance, avoiding repeated
    warm-up overhead in batch evaluation loops.

    .. note::
        The offline FAISS index must be built before running any pipeline
        queries.  Run ``make build-index`` (Docker) or
        ``python llm/src/build_offline_index.py`` (local) once.

    Attributes:
        top_k (int): Number of tables to retain after CHESS pruning.
        embedding_model (str): HuggingFace model ID used for **query**
            embedding at inference time.  Must match the model used when
            building the offline index.
        index_dir (str): Directory containing the pre-built FAISS artifacts.
        generator_provider (str): LLM provider for the generator role
            (``"groq"`` by default).
        generator_model (str): Model name for the generator LLM.
        critic_provider (str): LLM provider for the critic role
            (``"google"`` by default).
        critic_model (str): Model name for the critic LLM.
        sql_dialect (str): SQL dialect label injected into prompts.
        _linker (ChessLinker): Cached FAISS-backed CHESS linker instance.
    """

    def __init__(
        self,
        top_k: int = 3,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        index_dir: str = "llm/src/text2sql_agent/index",
        generator_provider: str = "groq",
        generator_model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
        critic_provider: str = "google",
        critic_model: str = "gemini-2.5-flash",
        sql_dialect: str = "SQLite",
    ) -> None:
        """Initialises all components with configurable hyperparameters.

        Args:
            top_k (int): Maximum tables to retain after CHESS pruning.
                Defaults to ``3``.
            embedding_model (str): Sentence-transformers model used for
                **query** embedding at inference time.  Defaults to
                ``"BAAI/bge-small-en-v1.5"``.  Must match the model used
                when running ``build_offline_index.py``.
            index_dir (str): Directory containing the pre-built FAISS index
                (``schema_index.faiss``) and metadata (``metadata.pkl``).
                Build these with ``make build-index`` or
                ``python llm/src/build_offline_index.py``.
                Defaults to ``"llm/src/text2sql_agent/index"``.
            generator_provider (str): Provider for the generator LLM.
                Defaults to ``"groq"``.
            generator_model (str): Generator model identifier.
                Defaults to ``"meta-llama/llama-4-scout-17b-16e-instruct"``.
            critic_provider (str): Provider for the critic LLM.
                Defaults to ``"google"``.
            critic_model (str): Critic model identifier.
                Defaults to ``"gemini-2.5-flash"``.
            sql_dialect (str): SQL dialect label used in prompt templates.
                Defaults to ``"SQLite"``.
        """
        self.top_k: int = top_k
        self.embedding_model: str = embedding_model
        self.index_dir: str = index_dir
        self.generator_provider: str = generator_provider
        self.generator_model: str = generator_model
        self.critic_provider: str = critic_provider
        self.critic_model: str = critic_model
        self.sql_dialect: str = sql_dialect

        # ChessLinker loads FAISS index + query model lazily on first prune().
        self._linker: ChessLinker = ChessLinker(
            index_path=os.path.join(index_dir, INDEX_FILENAME),
            metadata_path=os.path.join(index_dir, METADATA_FILENAME),
            model_name=embedding_model,
        )

        logger.info(
            "[MasterPipeline] Initialised — top_k=%d, embed=%s, "
            "index_dir=%s, gen=%s/%s, critic=%s/%s",
            top_k,
            embedding_model,
            index_dir,
            generator_provider,
            generator_model,
            critic_provider,
            critic_model,
        )

    # ------------------------------------------------------------------
    # Phase 1: CHESS Semantic Pruning
    # ------------------------------------------------------------------

    @traceable(run_type="tool")
    def _phase1_chess_prune(
        self, question: str, db_path: str
    ) -> PruningResult:
        """Executes CHESS semantic schema pruning.

        Scores every table against the question with cosine similarity and
        retains only the Top-K most relevant tables.

        Args:
            question (str): The natural language question.
            db_path (str): Path to the SQLite database.

        Returns:
            PruningResult: Pruning trace with selected tables, scores,
                and pruned DDL.
        """
        logger.info(
            "[MasterPipeline | Phase 1] CHESS pruning — top_k=%d", self.top_k
        )
        result: PruningResult = self._linker.prune(
            question=question, db_path=db_path, top_k=self.top_k
        )
        logger.info(
            "[MasterPipeline | Phase 1] %d → %d tables retained "
            "(reduction=%.0f%%, schema_ddl=%d chars).",
            len(result.all_tables),
            len(result.selected_tables),
            result.reduction_ratio * 100,
            len(result.pruned_schema_ddl),
        )
        return result

    # ------------------------------------------------------------------
    # Phase 2: MCI-SQL Metadata Enrichment
    # ------------------------------------------------------------------

    @traceable(run_type="tool")
    def _phase2_mci_enrich(
        self,
        db_path: str,
        selected_tables: List[str],
    ) -> str:
        """Extracts MCI metadata ONLY for the CHESS-pruned tables.

        Args:
            db_path (str): Path to the SQLite database.
            selected_tables (List[str]): Tables returned by Phase 1.

        Returns:
            str: Compact JSON metadata string covering all selected
                tables and their columns.
        """
        logger.info(
            "[MasterPipeline | Phase 2] MCI extraction for tables: %s",
            selected_tables,
        )
        extractor = MetadataExtractor(db_path=db_path)
        context: str = extractor.build_context(tables=selected_tables)
        logger.info(
            "[MasterPipeline | Phase 2] Metadata context: %d chars.",
            len(context),
        )
        return context

    # ------------------------------------------------------------------
    # Phase 3: Context Assembly
    # ------------------------------------------------------------------

    @traceable(run_type="tool")
    def _phase3_assemble_prompt(
        self,
        question: str,
        pruned_ddl: str,
        metadata_context: str,
        evidence: str,
    ) -> str:
        """Assembles the final, token-optimised generator prompt.

        Combines:
        - Pruned DDL (CHESS-filtered schema)
        - MCI metadata JSON (only pruned tables)
        - MAGIC self-check guidelines
        - The natural language question and optional evidence hint

        Args:
            question (str): The natural language question.
            pruned_ddl (str): DDL for selected tables only.
            metadata_context (str): Compact MCI JSON string.
            evidence (str): Optional domain knowledge hint.

        Returns:
            str: The fully assembled enriched prompt string.
        """
        evidence_block: str = (
            f"-- Evidence/Hint: {evidence}\n" if evidence else ""
        )
        prompt: str = (
            f"-- Using valid {self.sql_dialect}, answer the question below.\n\n"
            f"-- === PRUNED SCHEMA (CHESS Top-{self.top_k} tables) ===\n"
            f"{pruned_ddl}\n\n"
            f"-- === MCI METADATA (offline-extracted stats) ===\n"
            f"-- JSON keys: affinity, min, max, distinct_count, "
            f"total_rows, cardinality, samples.\n"
            f"-- Use these to validate literal values and predicates.\n"
            f"{metadata_context}\n\n"
            f"-- === MAGIC CORRECTION GUIDELINES ===\n"
            f"{_MAGIC_GUIDELINES}\n"
            f"-- === QUESTION ===\n"
            f"-- {question}\n"
            f"{evidence_block}"
            f"\nThink step by step, then output ONLY the final "
            f"{self.sql_dialect} SELECT query — no comments, no markdown.\n"
        )
        logger.info(
            "[MasterPipeline | Phase 3] Assembled prompt: %d chars.",
            len(prompt),
        )
        return prompt

    # ------------------------------------------------------------------
    # Phase 4a: SQL Generation
    # ------------------------------------------------------------------

    @traceable(run_type="tool")
    def _phase4a_generate(self, enriched_prompt: str) -> str:
        """Invokes the generator LLM (Groq Llama-4-Scout) — API call #1.

        Args:
            enriched_prompt (str): The fully assembled enriched prompt.

        Returns:
            str: Extracted SQL from the LLM response, or ``"SELECT 1;"``
                as a safe fallback if extraction fails.
        """
        logger.info(
            "[MasterPipeline | Phase 4a] Generator LLM: %s / %s",
            self.generator_provider,
            self.generator_model,
        )
        llm = get_llm(
            role="generator",
            model_name=self.generator_model,
        )
        try:
            raw: str = llm.generate(enriched_prompt)
            sql: str = extract_sql(raw)
            logger.info(
                "[MasterPipeline | Phase 4a] Generator SQL: %s",
                sql[:120].replace("\n", " "),
            )
            return sql or "SELECT 1;"
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[MasterPipeline | Phase 4a] Generator error: %s", exc
            )
            return "SELECT 1;"

    # ------------------------------------------------------------------
    # Phase 4ab: Reflection (logical self-consistency check)
    # ------------------------------------------------------------------

    @traceable(run_type="tool")
    def _phase4ab_reflect(self, question: str, candidate_sql: str) -> Optional[str]:
        """Back-translate candidate SQL and check logical consistency — API call #2.

        Uses ``meta-llama/llama-4-scout-17b-16e-instruct`` (blazing fast, 512
        tokens max) to translate the SQL back to English and compare it with
        the original question.  If they do not match, returns a logical error
        description so the corrector can be triggered without an SQLite error.

        Args:
            question (str): The original natural language question.
            candidate_sql (str): SQL produced by the generator.

        Returns:
            Optional[str]: ``None`` when the SQL is logically consistent
                (reflector replied ``<ok>``); otherwise a non-empty logical
                mismatch string to feed into the corrector.
        """
        reflection_prompt: str = (
            "You are a SQL semantic auditor.\n"
            "Given a SQL query, translate it to plain English, then decide if it "
            "fully and correctly satisfies the user question below.\n\n"
            f"User question: {question}\n\n"
            f"SQL:\n```sql\n{candidate_sql}\n```\n\n"
            "Instructions:\n"
            "  • If the SQL correctly answers the question, reply ONLY with: <ok>\n"
            "  • If there is ANY logical mismatch (wrong aggregation, wrong filter, "
            "wrong column, MIN vs MAX, etc.), reply with:\n"
            "    <error>One sentence describing the specific logical mismatch.</error>\n"
            "Do NOT output anything else."
        )
        try:
            llm = get_llm(role="reflector")
            raw: str = llm.generate(reflection_prompt)
            raw_stripped = raw.strip()
            logger.info(
                "[MasterPipeline | Phase 4ab] Reflection response: %s",
                raw_stripped[:200],
            )
            if "<ok>" in raw_stripped.lower():
                logger.info("[MasterPipeline | Phase 4ab] ✓ SQL passed reflection.")
                return None
            # Extract <error>...</error> content
            import re as _re
            err_match = _re.search(
                r"<error>\s*(.*?)\s*</error>", raw_stripped, _re.DOTALL | _re.IGNORECASE
            )
            if err_match:
                mismatch = err_match.group(1).strip()
            else:
                # Fallback: treat the whole response as the error
                mismatch = raw_stripped
            logger.warning(
                "[MasterPipeline | Phase 4ab] ✗ Logical mismatch detected: %s", mismatch
            )
            return mismatch
        except Exception as exc:  # noqa: BLE001
            # Reflection is best-effort — never block the pipeline
            logger.warning(
                "[MasterPipeline | Phase 4ab] Reflection failed (skipping): %s", exc
            )
            return None

    # ------------------------------------------------------------------
    # Phase 4b: Semantic Validation
    # ------------------------------------------------------------------

    @traceable(run_type="tool")
    def _phase4b_validate(
        self,
        db_path: str,
        sql: str,
    ) -> Tuple[Optional[List[Tuple[Any, ...]]], Optional[str]]:
        """Runs the SemanticErrorChecker locally — zero API cost.

        Args:
            db_path (str): Path to the SQLite database.
            sql (str): Candidate SQL to validate.

        Returns:
            Tuple[Optional[List[Tuple]], Optional[str]]:
                ``(rows, None)`` on success; ``(None, error_with_suggestion)``
                on any semantic failure.
        """
        logger.info(
            "[MasterPipeline | Phase 4b] Semantic validation (local)..."
        )
        checker = SemanticErrorChecker(db_path=db_path)
        rows, error_msg = checker.execute_safe(sql)
        if error_msg:
            logger.warning(
                "[MasterPipeline | Phase 4b] Semantic error: %s",
                error_msg[:200],
            )
        else:
            logger.info(
                "[MasterPipeline | Phase 4b] ✓ Valid — %d rows returned.",
                len(rows) if rows else 0,
            )
        return rows, error_msg

    # ------------------------------------------------------------------
    # Phase 4c: MAGIC Critic Correction
    # ------------------------------------------------------------------

    @traceable(run_type="tool")
    def _phase4c_correct(
        self,
        question: str,
        pruned_ddl: str,
        metadata_context: str,
        bad_sql: str,
        error_message: str,
        evidence: str,
        guideline_memory: str,
    ) -> str:
        """Invokes the critic LLM (Gemini 2.5 Flash) — API call #2.

        Constructs a correction prompt embedding the semantic error's
        suggestion text, the MAGIC checklist, and accumulated guideline
        memory from previous correction iterations.

        Args:
            question (str): The original question.
            pruned_ddl (str): Pruned DDL schema context.
            metadata_context (str): MCI metadata JSON string.
            bad_sql (str): The failing SQL query.
            error_message (str): Full error string including suggestion.
            evidence (str): Optional domain knowledge hint.
            guideline_memory (str): Accumulated MAGIC correction history
                from prior iterations (for iterative correction loops).

        Returns:
            str: Corrected SQL extracted from the critic's response, or
                ``bad_sql`` as a safe fallback.
        """
        logger.info(
            "[MasterPipeline | Phase 4c] Critic LLM: %s / %s",
            self.critic_provider,
            self.critic_model,
        )
        evidence_block: str = (
            f"Evidence/Hint: {evidence}\n" if evidence else ""
        )
        correction_prompt: str = (
            f"You are an expert {self.sql_dialect} Text-to-SQL correction agent.\n"
            f"Repair the failed query using the schema, MCI metadata, the error "
            f"description, and the MAGIC checklist below.\n\n"
            f"Question: {question}\n"
            f"{evidence_block}"
            f"\nPruned Schema:\n{pruned_ddl}\n\n"
            f"MCI Metadata:\n{metadata_context}\n\n"
            f"Failed SQL:\n```sql\n{bad_sql}\n```\n\n"
            f"Error / Logical Mismatch:\n{error_message}\n\n"
            f"MAGIC Checklist:\n{_MAGIC_GUIDELINES}\n"
            f"Past Guidelines:\n{guideline_memory or 'None'}\n\n"
            f"CRITICAL OUTPUT FORMAT: You MUST wrap the corrected SQL inside "
            f"<sql> and </sql> XML tags.\n"
            f"Example: <sql>SELECT COUNT(*) FROM customers WHERE segment = 'SME';</sql>\n"
            f"Output ONLY the XML-tagged SQL — no prose, no markdown, no explanation.\n"
        )
        llm = get_llm(
            role="critic",
            model_name=self.critic_model,
        )
        try:
            raw: str = llm.generate(correction_prompt)
            corrected: str = extract_sql(raw, fallback=bad_sql)
            logger.info(
                "[MasterPipeline | Phase 4c] Corrected SQL: %s",
                corrected[:120].replace("\n", " "),
            )
            return corrected
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[MasterPipeline | Phase 4c] Critic error: %s", exc
            )
            return bad_sql

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    @traceable(run_type="chain")
    def run(
        self,
        question: str,
        db_path: str,
        evidence: str = "",
        guideline_memory: str = "",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MasterPipelineResult:
        """Executes the full pipeline for a single question.

        Phases summary:
            1.  CHESS pruning → Top-K tables (offline).
            2.  MCI enrichment → compact JSON metadata (offline).
            3.  Prompt assembly → token-optimised enriched prompt (offline).
            4a. Generator LLM → candidate SQL (API call #1).
            4ab.Reflection LLM (scout) → logical self-check (API call #2).
            4b. SemanticErrorChecker → syntax/empty/null validation (offline).
            4c. [Conditional] Critic LLM → corrected SQL (API call #3 max).

        Args:
            question (str): Natural language question to answer with SQL.
            db_path (str): Absolute or relative path to the SQLite file.
            evidence (str): Optional domain knowledge / hint string.
                Defaults to ``""``.
            guideline_memory (str): Accumulated MAGIC correction history
                from prior pipeline iterations for the same question.
                Defaults to ``""``.

        Returns:
            MasterPipelineResult: Complete execution trace including
                pruning details, metadata, SQL candidates, row results,
                and the exact number of API calls consumed.

        Raises:
            FileNotFoundError: If ``db_path`` does not exist (propagated
                from ``ChessLinker`` or ``MetadataExtractor``).
            ImportError: If ``sentence-transformers`` is not installed
                (propagated from ``ChessLinker``).

        Example::

            pipeline = MasterPipeline(top_k=3)
            result = pipeline.run(
                question="How many customers are in the VIP segment?",
                db_path="data_minidev/MINIDEV/dev_databases/"
                        "debit_card_specializing/"
                        "debit_card_specializing.sqlite",
            )
            print(result.final_sql)
            print(f"API calls: {result.api_calls_made}")
            print(f"Tables pruned: {result.pruning.reduction_ratio:.0%}")
        """
        logger.info(
            "[MasterPipeline] ══════ START ══════ question=%s",
            question[:80],
        )

        # ── Phase 1: CHESS Semantic Pruning ──────────────────────────
        pruning: PruningResult = self._phase1_chess_prune(question, db_path)

        # ── Phase 2: MCI Metadata Enrichment ─────────────────────────
        metadata_context: str = self._phase2_mci_enrich(
            db_path=db_path,
            selected_tables=pruning.selected_tables,
        )

        # ── Phase 3: Context Assembly ─────────────────────────────────
        enriched_prompt: str = self._phase3_assemble_prompt(
            question=question,
            pruned_ddl=pruning.pruned_schema_ddl,
            metadata_context=metadata_context,
            evidence=evidence,
        )

        # ── Phase 4a: SQL Generation (API call #1) ────────────────────
        generator_raw_sql: str = self._phase4a_generate(enriched_prompt)
        api_calls_made: int = 1

        # ── Phase 4ab: Reflection — logical self-consistency (API call #2)
        reflection_error: Optional[str] = self._phase4ab_reflect(
            question=question, candidate_sql=generator_raw_sql
        )
        api_calls_made += 1  # always counts (scout is cheap)

        # ── Phase 4b: Semantic Validation (offline) ───────────────────
        rows, semantic_error = self._phase4b_validate(
            db_path=db_path, sql=generator_raw_sql
        )

        # Merge errors: prefer semantic error (concrete), then logical
        error_message: Optional[str] = semantic_error or reflection_error

        final_sql: str = generator_raw_sql
        critic_corrected_sql: Optional[str] = None

        # ── Phase 4c: MAGIC Critic Correction (API call #3, conditional)
        if error_message is not None:
            logger.info(
                "[MasterPipeline] Triggering corrector — reason: %s",
                "semantic" if semantic_error else "logical reflection",
            )
            critic_corrected_sql = self._phase4c_correct(
                question=question,
                pruned_ddl=pruning.pruned_schema_ddl,
                metadata_context=metadata_context,
                bad_sql=generator_raw_sql,
                error_message=error_message,
                evidence=evidence,
                guideline_memory=guideline_memory,
            )
            api_calls_made += 1
            final_sql = critic_corrected_sql

            # Re-validate the critic's output locally — still no API cost.
            rows, final_err = self._phase4b_validate(
                db_path=db_path, sql=critic_corrected_sql
            )
            if final_err:
                logger.error(
                    "[MasterPipeline] Critic correction still failing: %s",
                    final_err[:200],
                )
            else:
                logger.info(
                    "[MasterPipeline] Critic correction validated — rows: %d",
                    len(rows) if rows else 0,
                )

        logger.info(
            "[MasterPipeline] ══════ END ══════ api_calls=%d | "
            "tables=%d/%d | meta=%d chars | prompt=%d chars",
            api_calls_made,
            len(pruning.selected_tables),
            len(pruning.all_tables),
            len(metadata_context),
            len(enriched_prompt),
        )

        return MasterPipelineResult(
            question=question,
            db_path=db_path,
            pruning=pruning,
            metadata_context=metadata_context,
            final_sql=final_sql,
            rows=rows,
            generator_raw_sql=generator_raw_sql,
            semantic_error_message=semantic_error,
            reflection_error_message=reflection_error,
            critic_corrected_sql=critic_corrected_sql,
            api_calls_made=api_calls_made,
            prompt_char_count=len(enriched_prompt),
        )


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="MasterPipeline demo (CHESS + MCI-SQL + MAGIC)."
    )
    parser.add_argument("--db_path", type=str, required=True)
    parser.add_argument(
        "--question",
        type=str,
        default="How many customers are in the SME segment?",
    )
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--evidence", type=str, default="")
    args = parser.parse_args()

    pipeline = MasterPipeline(top_k=args.top_k)
    result: MasterPipelineResult = pipeline.run(
        question=args.question,
        db_path=args.db_path,
        evidence=args.evidence,
    )

    sep = "=" * 70
    print(f"\n{sep}\nMASTER PIPELINE RESULT\n{sep}")
    print(f"Question        : {result.question}")
    print(f"Final SQL       : {result.final_sql}")
    print(f"Rows returned   : {len(result.rows) if result.rows else 0}")
    print(f"API calls       : {result.api_calls_made}  (max 2)")
    print(
        f"Tables selected : {result.pruning.selected_tables} "
        f"({len(result.pruning.selected_tables)}/{len(result.pruning.all_tables)})"
    )
    print(f"Reduction ratio : {result.pruning.reduction_ratio:.0%}")
    print(f"Prompt length   : {result.prompt_char_count} chars")
    if result.semantic_error_message:
        print(f"Semantic error  : {result.semantic_error_message[:180]}")
    print(sep)

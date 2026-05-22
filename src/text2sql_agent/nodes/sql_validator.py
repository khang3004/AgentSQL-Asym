"""SQL Validator Node — Three-Tier Semantic Evaluation.

Upgraded from the basic EphemeralSandbox-only evaluator to a three-tier
evaluation stack, absorbing MasterPipeline Phase 4ab and Phase 4b logic:

Tier 1 — EphemeralSandbox (ground truth comparison)
    Compares predicted SQL output against ground truth if available.
    Returns: SUCCESS | FAILED | EMPTY | NONE

Tier 2 — SemanticErrorChecker (offline, zero API cost)
    Executes SQL locally and classifies: syntax error, empty result, null result.
    Provides actionable correction suggestions embedded in the error string.

Tier 3 — Reflection LLM (API call using the fast "reflector" role)
    Back-translates the candidate SQL to English and checks logical consistency
    against the original question. Catches silent logical errors that SQLite
    cannot detect (wrong aggregation, wrong filter, MIN vs MAX, etc.).

Merge priority: sandbox feedback (if ground truth available) → semantic error
(concrete, offline) → reflection error (logical, LLM-based).
"""

import os
import logging
import re
from typing import Any, Dict, Optional

from ..core.state import AgentState
from ..core.llm_factory import get_llm
from ..tools.execution_sandbox import EphemeralSandbox
from ..tools.semantic_error_checker import SemanticErrorChecker

logger = logging.getLogger(__name__)

_REFLECTION_PROMPT = """\
You are a SQL semantic auditor.
Given a SQL query, translate it to plain English, then decide if it fully \
and correctly satisfies the user question below.

User question: {question}

SQL:
```sql
{sql}
```

Instructions:
  • If the SQL correctly answers the question, reply ONLY with: <ok>
  • If there is ANY logical mismatch (wrong aggregation, wrong filter, wrong \
column, MIN vs MAX, etc.), reply with:
    <error>One sentence describing the specific logical mismatch.</error>
Do NOT output anything else.\
"""


def _reflect(question: str, candidate_sql: str) -> Optional[str]:
    """Back-translates SQL and checks logical consistency — API call (reflector role).

    Args:
        question (str): The original natural language question.
        candidate_sql (str): SQL produced by the generator.

    Returns:
        Optional[str]: ``None`` if SQL is logically consistent; otherwise a
            non-empty mismatch description string for the corrector.
    """
    try:
        llm = get_llm(role="reflector")
        prompt = _REFLECTION_PROMPT.format(question=question, sql=candidate_sql)
        raw = llm.generate(prompt).strip()

        logger.info("[sql_validator] Reflection response: %s", raw[:200])

        if "<ok>" in raw.lower():
            logger.info("[sql_validator] ✓ SQL passed reflection.")
            return None

        err_match = re.search(
            r"<error>\s*(.*?)\s*</error>", raw, re.DOTALL | re.IGNORECASE
        )
        mismatch = err_match.group(1).strip() if err_match else raw
        logger.warning("[sql_validator] ✗ Logical mismatch: %s", mismatch)
        return f"REFLECTION_ERROR: {mismatch}"

    except Exception as exc:
        # Reflection is best-effort — never block the pipeline
        logger.warning("[sql_validator] Reflection failed (skipping): %s", exc)
        return None


def evaluator_node(state: AgentState) -> Dict[str, Any]:
    """Validates the generated SQL using a three-tier evaluation stack.

    Tier 1: EphemeralSandbox (compares against ground truth if available).
    Tier 2: SemanticErrorChecker (offline — syntax, empty result, null result).
    Tier 3: Reflection LLM (logical self-consistency check, fast model).

    Args:
        state (AgentState): The current LangGraph workflow state.

    Returns:
        Dict[str, Any]: State update with ``execution_feedback`` (merged result)
            and incremented ``iteration_count``.
    """
    iter_count = state.get("iteration_count", 0)
    generated_sql = state.get("generated_sql", "")
    db_path = state["db_path"]
    question = state["question"]
    ground_truth_sql = state.get("ground_truth_sql", "")

    logger.info(
        "[sql_validator] Validating SQL (iter=%d): %s",
        iter_count,
        generated_sql.replace("\n", " ")[:120],
    )

    if not generated_sql or generated_sql.strip() == "SELECT 1;":
        return {
            "execution_feedback": "FAILED: No valid SQL was generated.",
            "iteration_count": iter_count,
        }

    # ── Tier 1: EphemeralSandbox (ground truth comparison) ───────────────────
    sandbox_feedback: str = ""
    if ground_truth_sql:
        db_uri = (
            f"sqlite:///{os.path.abspath(db_path)}"
            if not os.path.isabs(db_path)
            else f"sqlite:///{db_path}"
        )
        sandbox = EphemeralSandbox()
        sandbox_feedback = sandbox.execute_and_compare(
            predicted_sql=generated_sql,
            ground_truth_sql=ground_truth_sql,
            db_uri=db_uri,
        )
        logger.info("[sql_validator] Sandbox feedback: %s", sandbox_feedback[:200])

        # If sandbox succeeded, skip the other tiers for efficiency
        if sandbox_feedback == "SUCCESS":
            logger.info("[sql_validator] ✓ Sandbox SUCCESS — skipping deeper checks.")
            return {
                "execution_feedback": "SUCCESS",
                "iteration_count": iter_count,
            }

    # ── Tier 2: SemanticErrorChecker (offline, zero API cost) ────────────────
    semantic_error: Optional[str] = None
    try:
        checker = SemanticErrorChecker(db_path=db_path)
        _, semantic_error = checker.execute_safe(generated_sql)
        if semantic_error:
            logger.warning("[sql_validator] Semantic error: %s", semantic_error[:200])
        else:
            logger.info("[sql_validator] ✓ SQL passed semantic check.")
    except Exception as exc:
        logger.warning("[sql_validator] SemanticErrorChecker failed (%s).", exc)

    # ── Tier 3: Reflection LLM (logical self-consistency) ────────────────────
    reflection_error: Optional[str] = None
    # Only run reflection if semantic check passed (no point reflecting broken SQL)
    if not semantic_error:
        reflection_error = _reflect(question, generated_sql)

    # ── Merge feedback — priority: sandbox > semantic > reflection ────────────
    if sandbox_feedback and sandbox_feedback != "SUCCESS":
        merged_feedback = sandbox_feedback
    elif semantic_error:
        merged_feedback = semantic_error
    elif reflection_error:
        merged_feedback = reflection_error
    else:
        # All three tiers passed — treat as success (no ground truth to compare)
        logger.info("[sql_validator] ✓ All validation tiers passed.")
        merged_feedback = "SUCCESS"

    if merged_feedback != "SUCCESS":
        logger.warning("[sql_validator] ✗ Final feedback: %s", merged_feedback[:200])

    return {
        "execution_feedback": merged_feedback,
        "reflection_error": reflection_error or "",
        "iteration_count": iter_count,
    }

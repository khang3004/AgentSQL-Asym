"""Corrector Node — The Syntax Fixer.

Primary model : openai/gpt-oss-20b   (fast, agentic, cost-efficient)
Fallback model: meta-llama/llama-4-scout-17b-16e-instruct  (auto, on key exhaustion)

Design philosophy — minimal patch, no hallucinated intent:
  The corrector receives the original question, the schema subset, the
  failing SQL, and the DB error message.  Its ONLY mandate is to fix the
  immediate defect (syntax error, wrong alias, wrong column name, bad join
  condition) while preserving the original query's semantic intent verbatim.
  It must NOT invent new filters, new aggregations, or new table references.
"""

import logging
import os
from typing import Any, Dict

from ..core.state import AgentState
from ..core.llm_factory import get_llm
from ..core.sql_utils import extract_sql

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_CORRECTOR_SYSTEM_PROMPT = """\
You are a precise SQL Syntax Fixer for SQLite databases.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR ONLY JOB: MINIMAL PATCH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You receive:
  (A) The original natural-language question — the true intent, do NOT change it.
  (B) The schema — ground truth for table/column names.
  (C) A candidate SQL query that failed or produced wrong output.
  (D) The exact database error or execution feedback.

You MUST apply the smallest possible patch to fix the error.
You MUST NOT:
  • Add new tables that do not appear in the original query (unless the error
    explicitly requires a missing JOIN to resolve an unknown column).
  • Add or remove WHERE / HAVING filters that change the question's scope.
  • Change aggregation logic unless the error message specifically indicates
    a wrong aggregate.
  • Hallucinate column values, category names, or status codes.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE-BASED CORRECTION CHECKLIST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Apply the FIRST matching rule and stop:

  RULE 1  — "no such column":      Correct the column name to match the schema exactly.
  RULE 2  — "no such table":       Correct the table name or alias to match the schema.
  RULE 3  — "ambiguous column":    Qualify with the correct table alias or name.
  RULE 4  — "syntax error":        Fix the specific token; do not restructure the query.
  RULE 5  — Division by zero risk: Wrap denominator with NULLIF(…, 0).
  RULE 6  — Wrong CAST:            Ensure CAST(… AS FLOAT) is applied to the right operand.
  RULE 7  — Date format mismatch:  Use date() / strftime() / SUBSTR() to match storage format.
  RULE 8  — DISTINCT misuse:       Add or remove DISTINCT only if duplicate rows cause the error.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Question  : {question}
Evidence  : {evidence}

Schema and sample values:
{schema}

Candidate SQL (iteration #{iteration}):
```sql
{bad_sql}
```

DB error / execution feedback:
{error_feedback}

Past correction notes:
{past_guideline}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Output ONE corrected read-only SQLite query inside a ```sql block.
No explanation. No prose after the closing code fence.\
"""


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def corrector_node(state: AgentState) -> Dict[str, Any]:
    """Apply a minimal syntactic patch to a failing SQL query.

    Args:
        state: The current LangGraph agent state.

    Returns:
        State update dict with corrected ``generated_sql``, updated ``guideline``,
        and incremented ``iteration_count``.
    """
    question        = state["question"]
    schema          = state["schema_context"]
    evidence        = state.get("evidence", "") or "None"
    error_feedback  = state["execution_feedback"]
    bad_sql         = state["generated_sql"]
    current_guideline = state.get("guideline", "") or "None"
    iteration_count   = state.get("iteration_count", 0)

    logger.info("[corrector_node] Correction attempt #%d.", iteration_count + 1)

    # Model can be overridden via env var
    corr_model = os.environ.get("CORRECTOR_MODEL")  # None → factory default

    llm = get_llm(role="corrector", model_name=corr_model)

    prompt = _CORRECTOR_SYSTEM_PROMPT.format(
        question=question,
        evidence=evidence,
        schema=schema,
        iteration=iteration_count + 1,
        bad_sql=bad_sql,
        error_feedback=error_feedback,
        past_guideline=current_guideline,
    )

    corrected_sql = bad_sql  # safe default: keep last query if LLM fails
    correction_log = f"[iter={iteration_count + 1}] {error_feedback[:600]}"

    try:
        raw_content   = llm.generate(prompt)
        corrected_sql = extract_sql(raw_content, fallback=bad_sql)
        # Append a compact excerpt of the response for downstream diagnostics
        correction_log += f"\n[response_excerpt] {raw_content[:800]}"
    except Exception as exc:
        logger.error("[corrector_node] Corrector LLM failed: %s", exc)

    updated_guideline = (
        (current_guideline + "\n" + correction_log)[-5000:]
    )

    logger.info(
        "[corrector_node] Corrected SQL: %s",
        corrected_sql.replace("\n", " ")[:120],
    )

    return {
        "generated_sql":  corrected_sql,
        "guideline":      updated_guideline,
        "iteration_count": iteration_count + 1,
    }

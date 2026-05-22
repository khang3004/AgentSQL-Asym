"""SQL Corrector Node — Targeted Syntax & Logic Fixer with MAGIC Guidelines.

Primary model : openai/gpt-oss-20b   (fast, agentic, cost-efficient)
Fallback model: meta-llama/llama-4-scout-17b-16e-instruct  (auto, on key exhaustion)

Features:
  - 4-state targeted repair strategy: FAILED / EMPTY / NONE / REFLECTION_ERROR.
  - MAGIC correction checklist embedded in every prompt (sourced from MCI-SQL paper).
  - Minimal patch principle: applies smallest possible fix without restructuring query.
"""

import logging
import os
from typing import Any, Dict

from ..core.state import AgentState
from ..core.llm_factory import get_llm
from ..core.sql_utils import extract_sql

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MAGIC correction checklist (MCI-SQL paper)
# ---------------------------------------------------------------------------

_MAGIC_GUIDELINES: str = """\
MAGIC SQL Correction Checklist:
1. LIMIT/ORDER BY: for "top/most/least/highest/lowest" queries.
2. Aggregation: per-entity averages need a subquery before averaging.
3. Ratios: use conditional SUM/COUNT with CAST(... AS FLOAT).
4. Filters: verify literal values against MCI metadata samples.
5. Dates: match storage format; use strftime()/SUBSTR() when needed.
6. Joins: use FK/PK columns; avoid spurious DISTINCT.
7. Projection: select only the columns the question requests.
8. Extremes: prefer LIMIT 1 + ORDER BY over MIN/MAX subqueries.
"""

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_CORRECTOR_SYSTEM_PROMPT = """\
You are a precise SQL Syntax and Logic Fixer for SQLite databases.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR ONLY JOB: MINIMAL TARGETED PATCH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You receive:
  (A) The original natural-language question — the true intent, do NOT change it.
  (B) The schema — ground truth for table/column names.
  (C) A candidate SQL query that failed or produced wrong output.
  (D) The exact database error or execution feedback.
  (E) Targeted repair strategy based on execution state.

You MUST apply the smallest possible patch to fix the error.
You MUST NOT:
  • Add new tables that do not appear in the original query (unless the error
    explicitly requires a missing JOIN to resolve an unknown column).
  • Add or remove WHERE / HAVING filters that change the question's scope.
  • Change aggregation logic unless the error message specifically indicates
    a wrong aggregate.
  • Hallucinate column values, category names, or status codes.

{targeted_strategy}

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

DB error / execution feedback (State: {state_label}):
{error_feedback}

Past correction notes:
{past_guideline}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MAGIC CORRECTION CHECKLIST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{magic_guidelines}

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
    """Apply a minimal targeted patch to a failing SQL query based on state.

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

    logger.info("[sql_corrector] Correction attempt #%d.", iteration_count + 1)

    # 1. Parse SOTA Error State Label and construct targeted prompt strategy
    state_label = "FAILED"
    targeted_strategy = ""
    
    if error_feedback.startswith("FAILED"):
        state_label = "FAILED (Syntax/Runtime Error)"
        targeted_strategy = """\
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TARGETED REPAIR STRATEGY: SYNTAX & RESOLUTION (FAILED)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Focus on:
• Look for compile/syntax errors (e.g., mismatched parentheses, wrong keyword placement).
• Check for 'no such column' or 'no such table'. Correct spelling to match the schema exactly.
• If columns are ambiguous, ensure you qualify them with table aliases (e.g., 't1.col_name').
"""
    elif error_feedback.startswith("EMPTY"):
        state_label = "EMPTY (Zero Row Output)"
        targeted_strategy = """\
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TARGETED REPAIR STRATEGY: DATA ALIGNMENT (EMPTY)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Focus on:
• The SQL executed successfully but returned 0 rows, whereas it was expected to return data.
• This is typically a format/filter mismatch (e.g., filtering on 'usd' when data contains 'USD').
• Use 'LIKE' with wildcards (e.g., 'col LIKE "%val%"') instead of exact equality '=' for string values.
• Verify literals and filters against the sample values in the schema context.
"""
    elif error_feedback.startswith("NONE"):
        state_label = "NONE (Logical Mismatch / NULL)"
        targeted_strategy = """\
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TARGETED REPAIR STRATEGY: LOGICAL CORRECTNESS (NONE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Focus on:
• The query executed but returned incorrect columns, mismatched counts, or nulls.
• Check your JOIN path: are you connecting tables on the correct matching PK/FK column fields?
• Check your SELECT projection: did you select the correct target column requested by the question?
• Do NOT blindly add 'IS NOT NULL' filters; look for join mistakes or aggregate logical flaws.
"""

    # Model can be overridden via env var
    corr_model = os.environ.get("CORRECTOR_MODEL")
    llm = get_llm(role="corrector", model_name=corr_model)

    prompt = _CORRECTOR_SYSTEM_PROMPT.format(
        question=question,
        evidence=evidence,
        schema=schema,
        iteration=iteration_count + 1,
        bad_sql=bad_sql,
        state_label=state_label,
        error_feedback=error_feedback,
        targeted_strategy=targeted_strategy,
        past_guideline=current_guideline,
        magic_guidelines=_MAGIC_GUIDELINES,
    )

    corrected_sql = bad_sql  # Safe default
    correction_log = f"[iter={iteration_count + 1}][State={state_label}] {error_feedback[:500]}"

    try:
        raw_content = llm.generate(prompt)
        corrected_sql = extract_sql(raw_content, fallback=bad_sql)
        correction_log += f"\n[response_excerpt] {raw_content[:400]}"
    except Exception as exc:
        logger.error("[sql_corrector] Corrector LLM failed: %s", exc)

    updated_guideline = (
        (current_guideline + "\n" + correction_log)[-5000:]
    )

    logger.info(
        "[sql_corrector] Corrected SQL: %s",
        corrected_sql.replace("\n", " ")[:120],
    )

    return {
        "generated_sql": corrected_sql,
        "guideline": updated_guideline,
        "iteration_count": iteration_count + 1,
    }

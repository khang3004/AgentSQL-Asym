"""Generator Node — The Senior Data Engineer.

Primary model : openai/gpt-oss-120b   (complex schema reasoning & SQL drafting)
Fallback model: meta-llama/llama-4-scout-17b-16e-instruct  (auto, on key exhaustion)

The prompt forces the model to:
  1. Declare JOIN paths, selected tables, and result grain inside a <thought> block.
  2. Emit the final query in a ```sql block only after the chain-of-thought.
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

_GENERATOR_SYSTEM_PROMPT = """\
You are a Senior Data Engineer with deep expertise in SQL query authoring for \
analytical workloads on relational databases.

Your task is to translate a natural-language question into a single, read-only \
SQLite query that is both syntactically correct and semantically faithful to the question.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MANDATORY REASONING PROTOCOL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Before writing the final SQL you MUST produce a <thought> block.
Inside the <thought> block explicitly answer these three questions:

  1. SELECTED TABLES  — Which tables are needed and why?
  2. JOIN PATHS       — What foreign-key or id-column links connect them?
  3. RESULT GRAIN     — What is one row in the final result set?
                        (e.g., "one row per customer", "one aggregate row")

Only after closing </thought> may you emit the final ```sql block.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT CODING RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Use ONLY tables and columns present in the schema below.
• Prefer explicit JOINs via foreign keys or matching id columns.
• For ratios / averages: CAST(... AS FLOAT) for the numerator or denominator.
• For dates: use date(), strftime(), SUBSTR(), or half-open range predicates.
• Backtick SQLite reserved identifiers when required.
• Encode "top / most / least / ranking" questions with ORDER BY + LIMIT.
• Return exactly ONE read-only SQLite query inside a ```sql block — no prose after it.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCHEMA AND SAMPLE VALUES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{schema}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TASK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Question : {question}
Evidence : {evidence}

Now reason inside <thought>…</thought>, then output one ```sql block.\
"""


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def generator_node(state: AgentState) -> Dict[str, Any]:
    """Generate initial SQL using the Senior Data Engineer LLM.

    Args:
        state: The current LangGraph agent state.

    Returns:
        State update dict with ``generated_sql`` (str) and reset ``execution_feedback``.
    """
    question = state["question"]
    schema   = state["schema_context"]
    evidence = state.get("evidence", "") or "None"

    logger.info("[generator_node] Generating SQL for question: %s", question)

    # Model / provider can be overridden via env vars for experimentation
    gen_model = os.environ.get("GENERATOR_MODEL")  # None → factory default

    llm = get_llm(role="generator", model_name=gen_model)

    prompt = _GENERATOR_SYSTEM_PROMPT.format(
        schema=schema,
        question=question,
        evidence=evidence,
    )

    generated_sql = ""
    try:
        raw_content   = llm.generate(prompt)
        generated_sql = extract_sql(raw_content)
    except Exception as exc:
        logger.error("[generator_node] LLM generation failed: %s", exc)

    if not generated_sql:
        logger.warning("[generator_node] Extraction yielded empty SQL; using placeholder.")
        generated_sql = "SELECT 1;"

    logger.info(
        "[generator_node] Generated SQL: %s",
        generated_sql.replace("\n", " ")[:120],
    )

    return {
        "generated_sql": generated_sql,
        "execution_feedback": "",  # Reset on each new generation pass
    }

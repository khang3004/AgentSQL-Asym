"""Module containing the Corrector Node using the LLM factory."""

import logging
from typing import Dict, Any

from ..core.state import AgentState
from ..core.llm_factory import get_llm
from ..core.sql_utils import extract_sql

logger = logging.getLogger(__name__)

MAGIC_SELF_CHECK_GUIDELINES = """Common Text-to-SQL correction checks:
1. LIMIT/ORDER BY: for "top", "most", "least", "highest", "lowest", order by the metric and limit when one entity is requested.
2. Aggregation: for averages per entity, aggregate per entity in a subquery before averaging.
3. Ratios/division: use conditional SUM/COUNT and CAST the numerator or denominator to FLOAT.
4. Filters: verify every literal against sample values and column descriptions; do not invent status/category values.
5. Dates: match the storage format; use date()/SUBSTR()/strftime() or half-open ranges instead of fragile LIKE when appropriate.
6. Joins: use foreign keys or id columns, and avoid DISTINCT unless duplicate join rows would change the requested semantics.
7. Projection: select only the columns requested by the question; count the requested unit, not arbitrary rows.
8. Extremes: if ties are semantically possible, prefer equality to MIN/MAX subqueries unless the benchmark expects one row via LIMIT.
"""

def corrector_node(state: AgentState) -> Dict[str, Any]:
    """
    Corrects the generated SQL query based on feedback using the Critic LLM.
    
    Args:
        state (AgentState): The current state.
        
    Returns:
        Dict[str, Any]: A state update dict with new generated_sql, guideline, and incremented iteration_count.
    """
    question = state["question"]
    schema = state["schema_context"]
    evidence = state.get("evidence", "")
    error_feedback = state["execution_feedback"]
    bad_sql = state["generated_sql"]
    current_guideline = state.get("guideline", "")
    iteration_count = state.get("iteration_count", 0)
    
    logger.info("[corrector_node] Correction attempt #%d.", iteration_count + 1)
    
    import os
    
    critic_provider = os.environ.get("CRITIC_PROVIDER", "google")
    critic_model = os.environ.get("CRITIC_MODEL", "gemini-2.5-flash")
    
    # Asymmetric Architecture: Powerful reasoning model for Correction
    llm = get_llm(role="critic", provider=critic_provider, model_name=critic_model)
    
    correction_prompt = f"""You are an expert SQLite Text-to-SQL self-correction agent.
Your goal is execution accuracy. Repair the failed query using the schema, hint, execution feedback, and MAGIC-style checklist.

Question:
{question}

Evidence/hint:
{evidence or "None"}

Schema and sample values:
{schema}

Previous SQL:
```sql
{bad_sql}
```

Execution feedback:
{error_feedback}

MAGIC correction checklist:
{MAGIC_SELF_CHECK_GUIDELINES}

Past local feedback:
{current_guideline or "None"}

Think through the mismatch privately, then output only one corrected read-only SQLite query in a ```sql block."""

    corrected_sql = bad_sql
    analytical_feedback = self_feedback = f"iter={iteration_count + 1}: {error_feedback[:700]}"
    try:
        content_corr = llm.generate(correction_prompt)
        corrected_sql = extract_sql(content_corr, fallback=bad_sql)
        self_feedback = content_corr[:1200]
    except Exception as exc:
        logger.error("[corrector_node] Critic correction phase error: %s", exc)

    return {
        "generated_sql": corrected_sql,
        "guideline": (current_guideline + "\n" + analytical_feedback + "\n" + self_feedback)[-5000:],
        "iteration_count": iteration_count + 1
    }

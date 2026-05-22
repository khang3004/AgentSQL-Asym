"""Module containing the Output Alignment Node for AgentSQL.

Post-processes the generated SQL query to ensure it only projects columns requested 
by the natural language question, avoiding redundant columns.
"""

import os
import logging
from typing import Dict, Any

from ..core.state import AgentState
from ..core.llm_factory import get_llm
from ..core.sql_utils import extract_sql

logger = logging.getLogger(__name__)

_ALIGNMENT_SYSTEM_PROMPT = """\
You are an SQL Output Alignment assistant.
Your task is to ensure the SELECT clause of the final SQL query selects ONLY the columns explicitly or implicitly requested by the natural language question.

Rules:
1. If the SQL projects redundant fields (such as primary keys, extra ID columns, or descriptions) that the question did not ask for, rewrite the SQL query to project ONLY the requested columns.
2. Do NOT alter the WHERE, JOIN, GROUP BY, HAVING, or ORDER BY clauses.
3. If no redundant columns are found and the projection perfectly matches the user question, return the original SQL query exactly.
4. Output EXACTLY one read-only SQLite query inside a ```sql ... ``` block. No explanations.

Question : {question}
Evidence : {evidence}
Input SQL:
```sql
{sql}
```

Now, output the aligned SQL query inside the ```sql block.
"""

def alignment_node(state: AgentState) -> Dict[str, Any]:
    """Post-processes and cleans up the SQL projection clause to strictly align with user request.

    Args:
        state (AgentState): The current state of the workflow.

    Returns:
        Dict[str, Any]: A state update dict with the final aligned ``generated_sql``.
    """
    question = state["question"]
    evidence = state.get("evidence", "") or "None"
    generated_sql = state.get("generated_sql", "")
    
    if not generated_sql:
        return {}
        
    logger.info("[output_aligner] Aligning final SQL output...")

    # Use the dedicated 'aligner' role — ALIGNMENT_MODEL and ALIGNMENT_PROVIDER
    # are resolved from env by llm_factory (defaults: openai/gpt-oss-20b via groq)
    llm = get_llm(role="aligner")

    prompt = _ALIGNMENT_SYSTEM_PROMPT.format(
        question=question,
        evidence=evidence,
        sql=generated_sql
    )

    aligned_sql = generated_sql  # Safe default
    try:
        raw_content = llm.generate(prompt)
        aligned_sql = extract_sql(raw_content, fallback=generated_sql)
    except Exception as exc:
        logger.error("[output_aligner] Alignment LLM failed: %s. Using original SQL.", exc)

    logger.info(
        "[output_aligner] Aligned SQL: %s",
        aligned_sql.replace("\n", " ")[:120]
    )

    return {"generated_sql": aligned_sql}

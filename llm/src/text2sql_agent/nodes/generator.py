"""Module containing the Generator Node."""

import logging
from typing import Dict, Any

from ..core.state import AgentState
from ..core.llm_factory import get_llm
from ..core.sql_utils import extract_sql

logger = logging.getLogger(__name__)

def generator_node(state: AgentState) -> Dict[str, Any]:
    """
    Generates initial SQL context using the dependency-injected Generator LLM.
    
    Args:
        state (AgentState): The current state.
        
    Returns:
        Dict[str, Any]: A state update dict containing generated_sql and reset execution_feedback.
    """
    question = state["question"]
    schema = state["schema_context"]
    evidence = state.get("evidence", "")
    
    logger.info("[generator_node] Generating initial SQL for question: %s", question)
    
    import os
    
    gen_provider = os.environ.get("GENERATOR_PROVIDER", "groq")
    gen_model = os.environ.get("GENERATOR_MODEL", "llama3-70b-8192")
    
    # Asymmetric Architecture: Fast/Cheap model for Generation
    llm = get_llm(role="generator", provider=gen_provider, model_name=gen_model)
    
    prompt = f"""You are an expert SQLite Text-to-SQL system for BIRD/Spider-style execution accuracy.
Write one read-only SQLite query that answers the question exactly.

Rules:
- Use only tables and columns present in the schema.
- Prefer explicit joins through foreign keys or matching id columns.
- If the question asks for a ratio, difference, average per group, max/min entity, ranking, or "top/most/least", encode that operation directly.
- Use CAST(... AS FLOAT) for division ratios.
- Use date(), strftime(), SUBSTR(), or range predicates for date/time questions when the sample values show textual timestamps.
- Backtick SQLite reserved identifiers when needed.
- Return only the final SQL in a ```sql block.

Schema and sample values:
{schema}

Question: {question}
Evidence/hint: {evidence or "None"}

SQL:"""

    generated_sql = ""
    try:
        content = llm.generate(prompt)
        generated_sql = extract_sql(content)
    except Exception as exc:
        logger.error("[generator_node] LLM generation error: %s", exc)

    if not generated_sql:
        generated_sql = "SELECT 1;"

    logger.info("[generator_node] Generated SQL: %s", generated_sql.replace('\\n', ' ')[:100])
    
    return {
        "generated_sql": generated_sql,
        "execution_feedback": "" # Reset feedback
    }

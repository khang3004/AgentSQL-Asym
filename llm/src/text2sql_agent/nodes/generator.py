"""Module containing the Generator Node."""

import re
import logging
from typing import Dict, Any

from ..core.state import AgentState
from ..core.llm_factory import get_llm

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
    
    logger.info("[generator_node] Generating initial SQL for question: %s", question)
    
    import os
    
    gen_provider = os.environ.get("GENERATOR_PROVIDER", "groq")
    gen_model = os.environ.get("GENERATOR_MODEL", "llama3-70b-8192")
    
    # Asymmetric Architecture: Fast/Cheap model for Generation
    llm = get_llm(role="generator", provider=gen_provider, model_name=gen_model)
    
    prompt = f"""You are an Expert AI Database Engineer.
Your task is to write a syntactically correct SQL query.

Schema:
{schema}

Question: {question}

Provide ONLY the SQL inside a ```sql ... ``` block."""

    generated_sql = "SELECT 1;"
    try:
        content = llm.generate(prompt)
        
        match = re.search(r"```sql\n(.*?)```", content, re.DOTALL | re.IGNORECASE)
        if match:
            generated_sql = match.group(1).strip()
        else:
            generated_sql = content.strip().replace("\n", " ")
            
    except Exception as exc:
        logger.error("[generator_node] LLM generation error: %s", exc)
        generated_sql = f"SELECT 1; -- Error generating SQL: {exc}"

    logger.info("[generator_node] Generated SQL: %s", generated_sql.replace('\\n', ' ')[:100])
    
    return {
        "generated_sql": generated_sql,
        "execution_feedback": "" # Reset feedback
    }

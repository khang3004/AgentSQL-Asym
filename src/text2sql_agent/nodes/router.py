"""Module containing the Router Node for AgentSQL.

Classifies the natural language question to route the workflow dynamically.
"""

import os
import logging
from typing import Dict, Any, Literal

from ..core.state import AgentState
from ..core.llm_factory import get_llm

logger = logging.getLogger(__name__)

_ROUTER_SYSTEM_PROMPT = """\
You are an expert query router designed to analyze the complexity of a natural language question before executing Text-to-SQL.
Your ONLY job is to classify the question as either "SIMPLE" or "COMPLEX".

Definition:
- "SIMPLE": The question requires querying exactly 1 table, does not require complex JOINs, does not require complex calculations, aggregate grouping, or subqueries.
- "COMPLEX": The question requires JOINing multiple tables, complex filters, aggregates (e.g., conditional sum, window functions), subqueries, or math.

Examples:
1. Question: "What is the name of the teacher who works in department 5?" -> SIMPLE
2. Question: "Find the average salary of teachers in each department where the average is greater than 50000." -> COMPLEX
3. Question: "How many users registered in 2023?" -> SIMPLE

Input:
Question: {question}
Evidence: {evidence}

Return EXACTLY one of the following words inside the XML tags: <complexity>SIMPLE</complexity> or <complexity>COMPLEX</complexity>. Do NOT output any other text or explanation.
"""

def router_node(state: AgentState) -> Dict[str, Any]:
    """Classifies the complexity of the query to optimize workflow routing.

    Args:
        state (AgentState): The current state of the workflow.

    Returns:
        Dict[str, Any]: A state update dict with the query_complexity.
    """
    question = state["question"]
    evidence = state.get("evidence", "") or "None"
    
    logger.info("[router] Routing question: %s", question)

    prompt = _ROUTER_SYSTEM_PROMPT.format(question=question, evidence=evidence)

    # Use the dedicated 'router' role — ROUTER_MODEL and ROUTER_PROVIDER are
    # resolved from env by llm_factory (defaults: openai/gpt-oss-20b via groq)
    llm = get_llm(role="router")
    
    complexity: Literal["SIMPLE", "COMPLEX"] = "COMPLEX"  # Default fallback
    try:
        raw_content = llm.generate(prompt)
        if "<complexity>SIMPLE</complexity>" in raw_content or "SIMPLE" in raw_content.upper():
            complexity = "SIMPLE"
        else:
            complexity = "COMPLEX"
    except Exception as exc:
        logger.error("[router] Router LLM failed: %s. Defaulting to COMPLEX.", exc)
        
    logger.info("[router] Classified complexity: %s", complexity)
    
    return {"query_complexity": complexity}

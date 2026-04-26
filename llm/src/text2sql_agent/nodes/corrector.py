"""Module containing the Corrector Node using the LLM factory."""

import re
import logging
from typing import Dict, Any

from text2sql_agent.core.state import AgentState
from text2sql_agent.core.llm_factory import get_llm

logger = logging.getLogger(__name__)

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
    error_feedback = state["execution_feedback"]
    bad_sql = state["generated_sql"]
    current_guideline = state.get("guideline", "")
    iteration_count = state.get("iteration_count", 0)
    
    logger.info("[corrector_node] Correction attempt #%d.", iteration_count + 1)
    
    # Asymmetric Architecture: Powerful reasoning model for Correction
    llm = get_llm(role="critic", provider="google", model_name="gemini-2.5-flash")
    
    # 1. Feedback Generation Phase
    feedback_prompt = f"""You are an Expert AI Database Architect.
Query: {question}
Schema: {schema}
Generated SQL: {bad_sql}
Execution Feedback: {error_feedback}

Analyze the error. Explain WHY the execution failed or resulted in a semantic mismatch, and provide clear strategic steps to fix it.
"""
    if current_guideline:
         feedback_prompt += f"\nAccount for past guidelines:\n{current_guideline}"
         
    try:
        analytical_feedback = llm.generate(feedback_prompt)
        logger.info("[corrector_node] Analytical feedback string built.")
    except Exception as exc:
        logger.error("[corrector_node] Critic feedback phase error: %s", exc)
        analytical_feedback = f"Error generating analytical feedback: {exc}"
        
    # 2. Correction Phase
    correction_prompt = f"""You are an Expert AI Database Architect.
Query: {question}
Schema: {schema}
Error Context: {error_feedback}
Analytical Guidelines: {analytical_feedback}

Rewrite the SQL query to perfectly answer the user's question and fix the errors.
Output ONLY the SQL code inside ```sql ... ``` block and nothing else."""

    corrected_sql = bad_sql
    try:
        content_corr = llm.generate(correction_prompt)
        match = re.search(r"```sql\n(.*?)```", content_corr, re.DOTALL | re.IGNORECASE)
        if match:
             corrected_sql = match.group(1).strip()
        else:
             corrected_sql = content_corr.strip().replace("\n", " ")
    except Exception as exc:
        logger.error("[corrector_node] Critic correction phase error: %s", exc)
        
    return {
        "generated_sql": corrected_sql,
        "guideline": current_guideline + "\n" + analytical_feedback,
        "iteration_count": iteration_count + 1
    }

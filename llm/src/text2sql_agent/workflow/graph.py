"""Module containing the core LangGraph state machine definition."""

import logging
from typing import Literal
from langgraph.graph import StateGraph, END

from text2sql_agent.core.state import AgentState
from text2sql_agent.nodes.explorer import explorer_node
from text2sql_agent.nodes.generator import generator_node
from text2sql_agent.nodes.evaluator import evaluator_node
from text2sql_agent.nodes.corrector import corrector_node

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 3

def should_continue(state: AgentState) -> Literal["corrector_node", "__end__"]:
    """
    Conditional routing logic for the graph.
    
    Args:
        state (AgentState): The current state.
        
    Returns:
        Literal["corrector_node", "__end__"]: The next node to route to.
    """
    feedback = state.get("execution_feedback", "")
    iterations = state.get("iteration_count", 0)

    if feedback == "SUCCESS":
        logger.info("[should_continue] Execution succeeded at iteration %d -> END.", iterations)
        return END

    if iterations >= MAX_ITERATIONS:
        logger.warning("[should_continue] Max iterations (%d) reached -> END.", MAX_ITERATIONS)
        return END

    logger.info("[should_continue] Execution failed (iter=%d) -> corrector_node.", iterations)
    return "corrector_node"

def compile_workflow():
    """
    Builds, compiles, and returns the LangGraph workflow.
    
    Returns:
        CompiledGraph: The executable state graph.
    """
    builder = StateGraph(AgentState)

    # Add Nodes
    builder.add_node("explorer_node", explorer_node)
    builder.add_node("generator_node", generator_node)
    builder.add_node("evaluator_node", evaluator_node)
    builder.add_node("corrector_node", corrector_node)

    # Define the default edges
    builder.set_entry_point("explorer_node")
    builder.add_edge("explorer_node", "generator_node")
    builder.add_edge("generator_node", "evaluator_node")

    # Define Conditional Edge after Evaluator
    builder.add_conditional_edges("evaluator_node", should_continue)

    # Ensure corrector cycles back to evaluator for another sandbox check
    builder.add_edge("corrector_node", "evaluator_node")

    graph = builder.compile()
    
    logger.info("=== MAGIC LangGraph Compiled Successfully ===")
    return graph

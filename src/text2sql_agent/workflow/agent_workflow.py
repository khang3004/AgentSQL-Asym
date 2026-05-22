"""AgentSQL LangGraph Workflow — Unified SOTA Text-to-SQL State Machine.

This is the single main orchestrator for the AgentSQL system. It wires together
all upgraded nodes into a coherent state machine:

Architecture:
    [router_node]
         │
    SIMPLE ──────────────────────────────────────────────► [sql_generator_node]
    COMPLEX ─► [schema_explorer_node] ──────────────────► [sql_generator_node]
                  • CHESS pruning (FAISS)                        │
                  • MCI metadata (MetadataExtractor)        [sql_validator_node]
                  • Context Assembly + MAGIC guidelines           │
                                                    SUCCESS ─────► [output_aligner_node] → END
                                                    any error ───► [sql_corrector_node]
                                                                         │ (loop, max 3 iter)
                                                                    [sql_validator_node]

Node roles:
    router_node       — ROUTER_MODEL/PROVIDER: classifies SIMPLE vs COMPLEX
    schema_explorer   — Offline: CHESS + MCI + Context Assembly (zero API cost)
    sql_generator     — GENERATOR_MODEL/PROVIDER: DDL vs Markdown dual synthesis
    sql_validator     — Offline SemanticErrorChecker + Reflection (reflector role)
    sql_corrector     — CORRECTOR_MODEL/PROVIDER: 4-state MAGIC targeted fix
    output_aligner    — ALIGNMENT_MODEL/PROVIDER: trims SELECT to match question
"""

import logging
from typing import Literal

from langgraph.graph import StateGraph, END

from ..core.state import AgentState
from ..nodes.router import router_node
from ..nodes.schema_explorer import explorer_node
from ..nodes.sql_generator import generator_node
from ..nodes.sql_validator import evaluator_node
from ..nodes.sql_corrector import corrector_node
from ..nodes.output_aligner import alignment_node

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 3


def route_complexity(
    state: AgentState,
) -> Literal["generator_node", "explorer_node"]:
    """Routes from router to generator (SIMPLE) or schema explorer (COMPLEX).

    Args:
        state (AgentState): Current workflow state.

    Returns:
        Literal: Next node name.
    """
    complexity = state.get("query_complexity", "COMPLEX")
    if complexity == "SIMPLE":
        logger.info(
            "[agent_workflow] SIMPLE query — bypassing schema explorer → generator."
        )
        return "generator_node"
    logger.info(
        "[agent_workflow] COMPLEX query — routing to schema_explorer."
    )
    return "explorer_node"


def should_continue(
    state: AgentState,
) -> Literal["corrector_node", "alignment_node"]:
    """Routes from validator to corrector (error) or aligner (success / max iter).

    Args:
        state (AgentState): Current workflow state.

    Returns:
        Literal: Next node name.
    """
    feedback = state.get("execution_feedback", "")
    iterations = state.get("iteration_count", 0)

    if feedback == "SUCCESS":
        logger.info(
            "[agent_workflow] Validation SUCCESS at iter=%d → output_aligner.",
            iterations,
        )
        return "alignment_node"

    if iterations >= MAX_ITERATIONS:
        logger.warning(
            "[agent_workflow] Max iterations (%d) reached → output_aligner (best effort).",
            MAX_ITERATIONS,
        )
        return "alignment_node"

    logger.info(
        "[agent_workflow] Validation failed (iter=%d) → sql_corrector.", iterations
    )
    return "corrector_node"


def compile_workflow():
    """Builds, compiles, and returns the unified AgentSQL LangGraph workflow.

    Returns:
        CompiledGraph: The executable state machine, ready for ``graph.invoke()``.
    """
    builder = StateGraph(AgentState)

    # Register all nodes
    builder.add_node("router_node", router_node)
    builder.add_node("explorer_node", explorer_node)
    builder.add_node("generator_node", generator_node)
    builder.add_node("evaluator_node", evaluator_node)
    builder.add_node("corrector_node", corrector_node)
    builder.add_node("alignment_node", alignment_node)

    # Entry point
    builder.set_entry_point("router_node")

    # router → generator (SIMPLE) or explorer (COMPLEX)
    builder.add_conditional_edges("router_node", route_complexity)

    # explorer → generator (always, for COMPLEX path)
    builder.add_edge("explorer_node", "generator_node")

    # generator → validator
    builder.add_edge("generator_node", "evaluator_node")

    # validator → corrector (error) or aligner (success)
    builder.add_conditional_edges("evaluator_node", should_continue)

    # corrector → validator (correction loop)
    builder.add_edge("corrector_node", "evaluator_node")

    # aligner → END
    builder.add_edge("alignment_node", END)

    graph = builder.compile()
    logger.info("=== AgentSQL LangGraph Workflow compiled successfully ===")
    return graph

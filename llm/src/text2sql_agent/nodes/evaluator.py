"""Module containing the Evaluator Node using EphemeralSandbox."""

import os
import logging
from typing import Dict, Any

from ..core.state import AgentState
from ..tools.sandbox import EphemeralSandbox

logger = logging.getLogger(__name__)


def evaluator_node(state: AgentState) -> Dict[str, Any]:
    """
    Evaluates the generated query against the ground truth using the Sandbox.

    Args:
        state (AgentState): The current state.

    Returns:
        Dict[str, Any]: A state update dict containing execution_feedback.
    """
    iter_count = state.get("iteration_count", 0)
    logger.info(
        "[evaluator_node] Checking SQL (iter=%d) against Ground Truth...", iter_count
    )

    db_path = state["db_path"]

    # Construct standard Database URI for SQLAlchemy
    if os.path.isabs(db_path):
        db_uri = f"sqlite:///{db_path}"
    else:
        # Resolve to absolute path dynamically
        abs_path = os.path.abspath(db_path)
        db_uri = f"sqlite:///{abs_path}"

    sandbox = EphemeralSandbox()
    feedback = sandbox.execute_and_compare(
        predicted_sql=state["generated_sql"],
        ground_truth_sql=state["ground_truth_sql"],
        db_uri=db_uri,
    )

    if feedback == "SUCCESS":
        logger.info("[evaluator_node] Execution & semantic match succeeded.")
    else:
        logger.warning("[evaluator_node] %s", feedback[:200])

    return {"execution_feedback": feedback}

"""Module containing the Explorer Node for LangGraph."""

import logging
from typing import Dict, Any

from text2sql_agent.core.state import AgentState
from text2sql_agent.tools.mcp_client import MCPDatabaseClient

logger = logging.getLogger(__name__)

def explorer_node(state: AgentState) -> Dict[str, Any]:
    """
    Explores the database schema based on the query.
    
    Args:
        state (AgentState): The current state of the workflow.
        
    Returns:
        Dict[str, Any]: A state update dict with the retrieved schema_context.
    """
    logger.info("[explorer_node] Fetching schema for db: %s", state.get("db_path", ""))
    
    mcp_client = MCPDatabaseClient()
    schema_context = mcp_client.get_relevant_schema(state["question"], state["db_path"])
    
    logger.info("[explorer_node] Schema extracted (%d chars).", len(schema_context))
    
    return {"schema_context": schema_context}

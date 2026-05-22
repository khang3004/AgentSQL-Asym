"""Module to define the structure of the Agent's state in LangGraph."""

from typing import NotRequired, TypedDict, Literal


""" Agentic Text-to-SQL workflow orchestrator built on LangGraph.

Architecture (Groq-only, Asymmetric):
  - Generator : openai/gpt-oss-120b via Groq  (Senior Data Engineer)
  - Corrector : openai/gpt-oss-20b  via Groq  (Syntax Fixer)
  - Fallback  : meta-llama/llama-4-scout-17b-16e-instruct (on key exhaustion)
  - Key rotation: instant cycle on HTTP 429, zero sleep between retries.

Workflow:
    schema_exploration → sql_generation → execution_sandbox
        ↑                                        |
        └─────── feedback_correction ←──── [ERROR]
                        |
                      [SUCCESS or max_iter]
                        ↓
                       END

Usage:
    from graph_orchestrator import compile_graph, AgentState

    graph = compile_graph()
    initial_state: AgentState = {
        "question": "How many customers are in the SME segment?",
        "db_path": "data_minidev/MINIDEV/dev_databases/debit_card_specializing/...",
        "schema_context": "",
        "generated_sql": "",
        "execution_feedback": "",
        "guideline": "",
        "iteration_count": 0,
    }
    result = graph.invoke(initial_state)
    print(result["generated_sql"])
"""


class AgentState(TypedDict):
    """
    Defines the shared state dictionary passed across LangGraph nodes.

    Attributes:
        question (str): The initial natural language query from the user.
        db_path (str): File path to the SQLite database being targeted.
        schema_context (str): String representation of the extracted database schema (DDL + sample data).
        generated_sql (str): The current generated SQL query candidate.
        ground_truth_sql (str): Ground truth SQL for execution validation (evaluation purposes).
        execution_feedback (str): The result or exception string arising from Sandbox execution.
        guideline (str): Historical lessons or correction guidelines derived from past failures.
        iteration_count (int): Counter tracking the number of generation/correction cycles.
    """

    question: str
    db_path: str
    schema_context: str
    generated_sql: str
    ground_truth_sql: str
    evidence: NotRequired[str]
    difficulty: NotRequired[str]
    execution_feedback: str
    guideline: str
    iteration_count: int
    query_complexity: NotRequired[Literal["SIMPLE", "COMPLEX"]]
    candidate_sqls: NotRequired[dict[str, str]]
    metadata_context: NotRequired[str]    # MCI JSON from MetadataExtractor (schema_explorer)
    reflection_error: NotRequired[str]    # Logical mismatch from Reflection (sql_validator)

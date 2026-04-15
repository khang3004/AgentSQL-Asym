"""
graph_orchestrator.py
=====================
Agentic Text-to-SQL workflow orchestrator built on LangGraph.

Architecture inspired by the MAGIC (Multi-Agent Generative Improvement Cycle) paper
(SynQo / Microsoft Research) and modernized with:
  - LangGraph StateGraph for deterministic, inspectable agent routing.
  - MCP (Model Context Protocol) stubs for schema exploration.
  - NanoClaw-style sandboxed SQL execution stubs.
  - Groq API for fast, low-latency SQL generation (LLaMA-4 Scout).
  - Gemini API for high-quality feedback and correction reasoning.

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

from __future__ import annotations

import sqlite3
import logging
from typing import Literal

from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_ITERATIONS: int = 3
"""Maximum correction iterations before the graph terminates unconditionally."""


# ---------------------------------------------------------------------------
# State Definition
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    """Typed state container passed between every node in the LangGraph workflow.

    This dict is the single source of truth for the entire agent pipeline. Every
    node receives the full state and returns a *partial* dict with only the keys
    it wishes to update — LangGraph merges those updates automatically.

    Attributes:
        question: The natural-language question posed by the end-user.
        db_path: Absolute path to the target SQLite database file (`.sqlite`).
        schema_context: Serialised DDL + sample-row information for tables that
            are relevant to ``question``. Populated by ``schema_exploration_node``.
        generated_sql: The most recently produced or corrected SQL query string.
            Updated by both ``sql_generation_node`` and ``feedback_correction_node``.
        execution_feedback: Outcome of running ``generated_sql`` against the
            sandbox database.  Either the literal string ``"SUCCESS"`` or a
            structured error log (exception type + message) for the correction
            agent to reason about.
        guideline: A distilled, task-specific correction guideline string
            derived from historic feedback trajectories (MAGIC Phase-1 output).
            Injected statically or produced by a separate guideline-generation
            pass.  Empty string means "no guideline available yet".
        iteration_count: Number of feedback-correction cycles completed so far.
            Compared against ``MAX_ITERATIONS`` in ``should_continue``.
    """

    question: str
    db_path: str
    schema_context: str
    generated_sql: str
    execution_feedback: str
    guideline: str
    iteration_count: int


# ---------------------------------------------------------------------------
# Node: Schema Exploration (MCP stub)
# ---------------------------------------------------------------------------


def schema_exploration_node(state: AgentState) -> dict:
    """Fetch and filter the database schema relevant to the user question.

    In production this node would invoke an MCP (Model Context Protocol) tool
    call that:
      1. Lists all tables in the target SQLite database.
      2. Scores each table for relevance to ``state["question"]`` via a fast
         embedding similarity search (e.g., ChromaDB + sentence-transformers).
      3. Returns DDL + three sample rows *only* for the top-k relevant tables —
         reducing prompt size and steering the generation node.

    The current implementation is a **synchronous SQLite stub** that retrieves
    the full schema unconditionally.  Replace the body of this function with
    a real MCP call when the MCP server is available.

    Args:
        state: The current ``AgentState`` snapshot provided by LangGraph.

    Returns:
        A partial state dict containing the updated ``schema_context`` key.

    Raises:
        sqlite3.OperationalError: If the database file at ``state["db_path"]``
            cannot be opened or queried.
    """
    logger.info(
        "[schema_exploration_node] Fetching schema for db: %s", state["db_path"]
    )

    # --- MCP stub: replace with real MCP tool invocation ---
    schema_lines: list[str] = []
    try:
        conn = sqlite3.connect(state["db_path"])
        cursor = conn.cursor()

        # Retrieve all CREATE TABLE statements
        cursor.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name;"
        )
        rows = cursor.fetchall()
        for table_name, create_sql in rows:
            if create_sql:
                schema_lines.append(f"-- Table: {table_name}\n{create_sql};\n")

                # Append up to 3 sample rows for value awareness
                try:
                    cursor.execute(f"SELECT * FROM [{table_name}] LIMIT 3;")  # noqa: S608
                    samples = cursor.fetchall()
                    if samples:
                        schema_lines.append(f"-- Sample rows from {table_name}:")
                        for row in samples:
                            schema_lines.append(f"--   {row}")
                        schema_lines.append("")
                except sqlite3.Error:
                    pass  # Non-fatal: skip sample rows for protected tables

        conn.close()
    except sqlite3.OperationalError as exc:
        logger.error("[schema_exploration_node] DB error: %s", exc)
        schema_lines.append(f"-- ERROR: could not open database: {exc}")
    # --------------------------------------------------------

    schema_context: str = "\n".join(schema_lines)
    logger.info(
        "[schema_exploration_node] Schema extracted (%d chars).", len(schema_context)
    )
    return {"schema_context": schema_context}


# ---------------------------------------------------------------------------
# Node: SQL Generation (Groq API stub)
# ---------------------------------------------------------------------------


def sql_generation_node(state: AgentState) -> dict:
    """Generate an initial SQL query from the user question using the Groq API.

    In production this node sends a structured chat-completion request to the
    Groq API (model: ``meta-llama/llama-4-scout-17b-16e-instruct``) with a
    zero-shot or few-shot prompt that includes:
      - ``state["schema_context"]``: filtered schema from the previous node.
      - ``state["question"]``: the natural-language question.
      - ``state["guideline"]``: optional MAGIC guideline to steer generation.

    The Groq call is intentionally **stubbed** here.  To activate it, import
    ``groq.Groq`` and replace the placeholder return value with a real API call
    following the pattern in ``llm/src/groq_request.py``.

    Args:
        state: The current ``AgentState`` snapshot provided by LangGraph.

    Returns:
        A partial state dict containing the updated ``generated_sql`` key and
        a reset ``execution_feedback`` (cleared so the sandbox always re-runs).
    """
    logger.info(
        "[sql_generation_node] Generating SQL for question: %s", state["question"]
    )

    import os
    import re
    from groq import Groq

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    prompt = f"""You are an expert SQL Generator.
Database Schema and Sample Rows:
{state["schema_context"]}

Question: {state["question"]}

Please generate a valid SQLite query to answer the question. 
Output ONLY the SQL code inside ```sql ... ``` block and nothing else.
"""
    if state["guideline"]:
        prompt += f"\nFollow these Correction Guidelines:\n{state['guideline']}"

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024,
        )
        content = response.choices[0].message.content

        # Simple extraction of SQL from markdown fences
        match = re.search(r"```sql\n(.*?)```", content, re.DOTALL | re.IGNORECASE)
        if match:
            generated_sql = match.group(1).strip()
        else:
            generated_sql = content.strip().replace("\n", " ")
    except Exception as exc:
        logger.error("[sql_generation_node] API error: %s", exc)
        generated_sql = f"SELECT 1; -- Error generating SQL: {exc}"
    # --------------------------------------------------

    logger.info("[sql_generation_node] Generated SQL: %s", generated_sql)
    return {
        "generated_sql": generated_sql,
        "execution_feedback": "",  # reset feedback so sandbox always re-evaluates
    }


# ---------------------------------------------------------------------------
# Node: Execution Sandbox (NanoClaw container stub)
# ---------------------------------------------------------------------------


def execution_sandbox_node(state: AgentState) -> dict:
    """Execute the generated SQL in a sandboxed environment and capture feedback.

    In production this node would:
      1. Dispatch ``state["generated_sql"]`` to a **NanoClaw ephemeral container**
         via its REST API — providing full process isolation and a hard wall-clock
         timeout (typically 30 s).
      2. Collect stdout / stderr and the raw result-set from the container.
      3. Return ``"SUCCESS"`` if execution completed without error, or a
         structured JSON error log (including exception type, line number, and
         the offending SQL fragment) for the feedback-correction cycle.

    The current implementation is a **native SQLite stub** that runs the SQL
    directly in-process.  This is safe for read-only SELECT queries but should
    NOT be used in production for untrusted SQL.

    Args:
        state: The current ``AgentState`` snapshot containing ``generated_sql``
            and ``db_path``.

    Returns:
        A partial state dict with the updated ``execution_feedback`` key.
        Value is ``"SUCCESS"`` on success, or an error string on failure.
    """
    sql: str = state["generated_sql"]
    db_path: str = state["db_path"]
    logger.info(
        "[execution_sandbox_node] Executing SQL (iter=%d): %s",
        state["iteration_count"],
        sql[:120],
    )

    # --- NanoClaw stub: replace with container API call ---
    feedback: str
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchmany(5)  # fetch sample to confirm executability
        conn.close()
        feedback = "SUCCESS"
        logger.info(
            "[execution_sandbox_node] Execution succeeded. Sample rows: %s", rows[:3]
        )
    except sqlite3.Error as exc:
        feedback = (
            f"SQLiteError({type(exc).__name__}): {exc} | OffendingSQL: {sql[:200]}"
        )
        logger.warning("[execution_sandbox_node] Execution failed: %s", feedback)
    # ------------------------------------------------------

    return {"execution_feedback": feedback}


# ---------------------------------------------------------------------------
# Node: Feedback & Correction (Gemini API stub)
# ---------------------------------------------------------------------------


def feedback_correction_node(state: AgentState) -> dict:
    """Analyse the execution error log and produce a corrected SQL query.

    This node implements the core of the MAGIC self-correction loop:
      1. **Feedback phase**: The error log in ``state["execution_feedback"]`` is
         analysed in the context of the schema, the original question, and the
         domain-specific guideline.  The analysis synthesises a concise natural-
         language explanation of *why* the SQL failed (e.g. wrong JOIN, missing
         CAST, incorrect aggregation).
      2. **Correction phase**: The feedback is used to produce a revised SQL
         statement.

    In production both phases are powered by the **Gemini API** (e.g.
    ``gemini-2.0-flash``) via ``google.generativeai``.  Gemini is chosen here for
    its superior instruction-following and long-context capabilities relative to
    the LLaMA generation model, which matches the MAGIC paper's design of using
    different models at different stages.

    The ``iteration_count`` is incremented here so the routing function
    ``should_continue`` can gate on the maximum number of attempts.

    Args:
        state: The current ``AgentState`` snapshot containing ``generated_sql``,
            ``execution_feedback``, ``schema_context``, ``guideline``, and
            ``iteration_count``.

    Returns:
        A partial state dict with the updated ``generated_sql`` (corrected SQL)
        and the incremented ``iteration_count``.

    Note:
        If both the Feedback and Correction phases are stubbed, this node simply
        appends a comment to the existing SQL to simulate a correction attempt.
        This is intentional — it allows the graph to be exercised end-to-end
        without live API keys.
    """
    iteration: int = state["iteration_count"] + 1
    error_log: str = state["execution_feedback"]
    original_sql: str = state["generated_sql"]

    logger.info(
        "[feedback_correction_node] Correction attempt #%d. Error: %s",
        iteration,
        error_log[:200],
    )

    import os
    import re
    import google.generativeai as genai

    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-2.5-flash")  # using latest gemini

    # --- Feedback phase ---
    feedback_prompt = f"""You are an Expert AI Database Architect.
Original Question: {state["question"]}
Failed SQL Query: {original_sql}
Execution Error: {error_log}

Please analyse the error. Explain concisely why the execution failed or might be semantically wrong, 
and suggest exactly how to fix it for SQLite.
"""
    if state["guideline"]:
        feedback_prompt += f"\nAlso, account for these previous historic guideliens:\n{state['guideline']}"

    try:
        feedback_response = model.generate_content(feedback_prompt)
        feedback_text: str = feedback_response.text
        logger.info("[feedback_correction_node] Feedback text generated.")
    except Exception as exc:
        logger.error("[feedback_correction_node] Gemini feedback error: %s", exc)
        feedback_text = f"API Error during feedback: {exc}"

    # --- Correction phase ---
    correction_prompt = f"""You are an Expert AI Database Architect.
Database Schema:
{state["schema_context"]}

Original Question: {state["question"]}
Failed SQL Query: {original_sql}
Expert Analysis Feedback: {feedback_text}

Based on the expert feedback, rewrite the SQL query so it works perfectly in SQLite. 
Output ONLY the SQL code inside ```sql ... ``` block and nothing else.
"""
    try:
        correction_response = model.generate_content(correction_prompt)
        content_corr = correction_response.text
        match = re.search(r"```sql\n(.*?)```", content_corr, re.DOTALL | re.IGNORECASE)
        if match:
            corrected_sql = match.group(1).strip()
        else:
            corrected_sql = content_corr.strip().replace("\n", " ")
    except Exception as exc:
        logger.error("[feedback_correction_node] Gemini correction error: %s", exc)
        corrected_sql = original_sql  # fallback on error
    # ------------------------------------------

    logger.info(
        "[feedback_correction_node] Corrected SQL (stub): %s", corrected_sql[:120]
    )
    return {
        "generated_sql": corrected_sql,
        "iteration_count": iteration,
    }


# ---------------------------------------------------------------------------
# Routing Function (Conditional Edge)
# ---------------------------------------------------------------------------


def should_continue(
    state: AgentState,
) -> Literal["feedback_correction_node", "__end__"]:
    """Determine the next node after the execution sandbox.

    Routing logic (mirrors MAGIC manager-agent termination criteria):
      - Route to ``END`` if ``execution_feedback`` is ``"SUCCESS"`` — the SQL is
        correct and no further correction is needed.
      - Route to ``END`` if ``iteration_count`` has reached ``MAX_ITERATIONS`` —
        prevents infinite loops when the model cannot self-correct within the
        budget.
      - Route to ``feedback_correction_node`` in all other cases (execution
        produced an error and the iteration budget is not yet exhausted).

    Args:
        state: The current ``AgentState`` snapshot after ``execution_sandbox_node``
            has returned.

    Returns:
        The string name of the next node, or the special ``END`` sentinel from
        ``langgraph.graph``.
    """
    feedback: str = state["execution_feedback"]
    iteration: int = state["iteration_count"]

    if feedback == "SUCCESS":
        logger.info(
            "[should_continue] Execution succeeded at iteration %d → END.", iteration
        )
        return "__end__"

    if iteration >= MAX_ITERATIONS:
        logger.warning(
            "[should_continue] Max iterations (%d) reached → END (best-effort SQL).",
            MAX_ITERATIONS,
        )
        return "__end__"

    logger.info(
        "[should_continue] Execution failed (iter=%d) → feedback_correction_node.",
        iteration,
    )
    return "feedback_correction_node"


# ---------------------------------------------------------------------------
# Graph Compilation
# ---------------------------------------------------------------------------


def compile_graph() -> StateGraph:
    """Build, wire, and compile the LangGraph ``StateGraph`` for Text-to-SQL.

    The compiled graph can be invoked directly with an ``AgentState`` dict via
    ``graph.invoke(initial_state)`` or streamed step-by-step via
    ``graph.stream(initial_state)``.

    Node registration order:
      1. ``schema_exploration_node``  — MCP schema fetch.
      2. ``sql_generation_node``      — Groq LLaMA SQL generation.
      3. ``execution_sandbox_node``   — NanoClaw SQL execution.
      4. ``feedback_correction_node`` — Gemini feedback + correction.

    Edge topology:
      - ``START`` → ``schema_exploration_node`` (implicit entry point).
      - ``schema_exploration_node`` → ``sql_generation_node``.
      - ``sql_generation_node`` → ``execution_sandbox_node``.
      - ``execution_sandbox_node`` → ``should_continue`` (conditional routing).
        - ``"SUCCESS"`` or max-iter → ``END``.
        - error + budget remaining → ``feedback_correction_node``.
      - ``feedback_correction_node`` → ``execution_sandbox_node`` (retry loop).

    Returns:
        A compiled ``StateGraph`` instance ready for invocation.
    """
    builder: StateGraph = StateGraph(AgentState)

    # --- Register nodes ---
    builder.add_node("schema_exploration_node", schema_exploration_node)
    builder.add_node("sql_generation_node", sql_generation_node)
    builder.add_node("execution_sandbox_node", execution_sandbox_node)
    builder.add_node("feedback_correction_node", feedback_correction_node)

    # --- Define edges ---
    # Entry point
    builder.set_entry_point("schema_exploration_node")

    # Linear forward edges
    builder.add_edge("schema_exploration_node", "sql_generation_node")
    builder.add_edge("sql_generation_node", "execution_sandbox_node")

    # Conditional branching after sandbox
    builder.add_conditional_edges(
        "execution_sandbox_node",
        should_continue,
        {
            "feedback_correction_node": "feedback_correction_node",
            "__end__": END,
        },
    )

    # Correction loop: retry execution after each correction
    builder.add_edge("feedback_correction_node", "execution_sandbox_node")

    return builder.compile()


# ---------------------------------------------------------------------------
# Entry-point smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import json
    import argparse
    from dotenv import load_dotenv

    # Load environment variables just in case it's run locally.
    # Docker already injects from .env.
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of questions to test in this execution.",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------
    # Quick smoke-test: iterate first N questions.
    # -----------------------------------------------------------------
    _DB_ROOT = os.path.join(
        os.path.dirname(__file__),
        "../../data_minidev/MINIDEV/dev_databases",
    )

    _MINI_DEV_JSON = os.path.join(
        os.path.dirname(__file__), "../../data_minidev/MINIDEV/mini_dev_sqlite.json"
    )
    with open(_MINI_DEV_JSON, encoding="utf-8") as _f:
        dataset = json.load(_f)

    # We test on exactly num_samples questions
    test_subset = dataset[: args.num_samples]

    logger.info("=== Compiling MAGIC Graph ===")
    graph = compile_graph()

    results_collection = []

    for idx, sample in enumerate(test_subset):
        logger.info("\n\n" + "=" * 50)
        logger.info(
            "Processing Sample %d / %d | Question ID: %s",
            idx + 1,
            args.num_samples,
            sample.get("question_id", idx),
        )

        db_id: str = sample["db_id"]
        db_path: str = os.path.join(_DB_ROOT, db_id, f"{db_id}.sqlite")

        initial_state: AgentState = {
            "question": sample["question"],
            "db_path": db_path,
            "schema_context": "",
            "generated_sql": "",
            "execution_feedback": "",
            "guideline": "",  # inject MAGIC guideline here when available
            "iteration_count": 0,
        }

        # Run the graph end-to-end
        final_state: AgentState = graph.invoke(initial_state)  # type: ignore[assignment]

        logger.info("=== Final Result for Sample %d ===", idx + 1)
        logger.info("Generated SQL  : %s", final_state["generated_sql"])
        logger.info("Exec Feedback  : %s", final_state["execution_feedback"])
        logger.info("Iterations     : %d", final_state["iteration_count"])

        # Save to collection
        results_collection.append(
            {
                "question_id": sample.get("question_id", idx),
                "question": final_state["question"],
                "db_id": db_id,
                "final_sql": final_state["generated_sql"],
                "execution_status": final_state["execution_feedback"],
                "iterations_used": final_state["iteration_count"],
            }
        )

    # Save output summarizing the run
    out_dir = os.path.join(os.path.dirname(__file__), "../exp_result/magic_output")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "test_10_samples_magic.json")

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_collection, f, indent=4)

    logger.info("========================================")
    logger.info("Smoke test complete! Results saved to: %s", out_file)

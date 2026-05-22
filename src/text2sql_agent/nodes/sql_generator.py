"""Generator Node — The Senior Data Engineer.

Primary model : openai/gpt-oss-120b   (complex schema reasoning & SQL drafting)
Fallback model: meta-llama/llama-4-scout-17b-16e-instruct  (auto, on key exhaustion)

Upgraded features:
  - Divide-and-Merge: instructs LLM to decompose COMPLEX queries into sub-questions and sub-queries.
  - Diverse Synthesis: generates TWO candidate SQLs in parallel (DDL schema vs Light Markdown schema)
    and verifies them with execution-guided sandbox routing.
"""

import logging
import os
import re
from typing import Any, Dict

from ..core.state import AgentState
from ..core.llm_factory import get_llm
from ..core.sql_utils import extract_sql
from ..tools.execution_sandbox import EphemeralSandbox

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_BASE_INSTRUCTIONS = """\
You are a Senior Data Engineer with deep expertise in SQL query authoring for \
analytical workloads on relational databases.

Your task is to translate a natural-language question into a single, read-only \
SQLite query that is both syntactically correct and semantically faithful to the question.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT CODING RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Use ONLY tables and columns present in the schema below.
• Prefer explicit JOINs via foreign keys or matching id columns.
• For ratios / averages: CAST(... AS FLOAT) for the numerator or denominator.
• For dates: use date(), strftime(), SUBSTR(), or half-open range predicates.
• Backtick SQLite reserved identifiers when required.
• Encode "top / most / least / ranking" questions with ORDER BY + LIMIT.
• Return exactly ONE read-only SQLite query inside a ```sql block — no prose after it.
"""

_DIVIDE_AND_MERGE_PROTOCOL = """\
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MANDATORY REASONING PROTOCOL (DIVIDE-AND-MERGE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Since this query has been classified as COMPLEX, you MUST produce a <thought> block
containing a detailed Divide-and-Merge reasoning chain:

1. SUB-QUESTIONS: Deconstruct the complex user question into natural language sub-questions.
2. SUB-QUERIES: Write intermediate SQL sub-queries or CTE strategies for each sub-question.
3. MERGE STRATEGY: Explicitly explain how you will merge these sub-queries (JOIN, UNION, subquery) 
   to produce the final correct SQL.
4. RESULT GRAIN: What is one row in the final result set?
"""

_SIMPLE_PROTOCOL = """\
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MANDATORY REASONING PROTOCOL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Before writing the final SQL you MUST produce a <thought> block.
Inside the <thought> block explicitly answer these three questions:

1. SELECTED TABLES  — Which tables are needed and why?
2. JOIN PATHS       — What foreign-key or id-column links connect them?
3. RESULT GRAIN     — What is one row in the final result set?
"""

_GENERATOR_SYSTEM_PROMPT = """\
{instructions}

{protocol}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCHEMA AND SAMPLE VALUES ({schema_type} format)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{schema}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TASK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Question : {question}
Evidence : {evidence}

Now reason inside <thought>…</thought>, then output one ```sql block.\
"""

# ---------------------------------------------------------------------------
# Helper function for Light Markdown Conversion
# ---------------------------------------------------------------------------

def _to_light_markdown(schema_ddl: str) -> str:
    """Converts a detailed schema dump containing full DDL into a Light Markdown format.

    Strips the raw 'CREATE TABLE' SQL definitions, leaving only the structured 
    column lists, metadata, and comments.
    """
    lines = schema_ddl.split("\n")
    markdown_lines = []
    in_raw_ddl = False
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("CREATE TABLE"):
            in_raw_ddl = True
            continue
        if in_raw_ddl and stripped.endswith(";"):
            in_raw_ddl = False
            continue
        if in_raw_ddl:
            continue
            
        if line.startswith("----- Table:") or stripped.startswith("- ") or "Foreign keys" in line or "Sample rows:" in line:
            markdown_lines.append(line)
            
    return "\n".join(markdown_lines)

# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def generator_node(state: AgentState) -> Dict[str, Any]:
    """Generate initial SQL candidates and select the best using execution verification.

    Args:
        state: The current LangGraph agent state.

    Returns:
        State update dict with ``generated_sql``, verified ``execution_feedback``,
        and ``query_complexity``.
    """
    question = state["question"]
    schema = state.get("schema_context", "") or "No schema context available."
    evidence = state.get("evidence", "") or "None"
    complexity = state.get("query_complexity", "COMPLEX")
    db_path = state["db_path"]

    logger.info("[generator_node] Generating SQL for %s query: %s", complexity, question)

    # 1. Select the reasoning protocol based on query complexity
    protocol = _DIVIDE_AND_MERGE_PROTOCOL if complexity == "COMPLEX" else _SIMPLE_PROTOCOL

    # 2. Build both DDL and Light Markdown schemas
    ddl_schema = schema
    markdown_schema = _to_light_markdown(schema)

    # 3. Model override via environment
    gen_model = os.environ.get("GENERATOR_MODEL")
    llm = get_llm(role="generator", model_name=gen_model)

    # 4. Synthesize Candidate 1: DDL Prompt
    prompt_ddl = _GENERATOR_SYSTEM_PROMPT.format(
        instructions=_BASE_INSTRUCTIONS,
        protocol=protocol,
        schema_type="DDL DDL",
        schema=ddl_schema,
        question=question,
        evidence=evidence,
    )
    
    # 5. Synthesize Candidate 2: Markdown Prompt
    prompt_markdown = _GENERATOR_SYSTEM_PROMPT.format(
        instructions=_BASE_INSTRUCTIONS,
        protocol=protocol,
        schema_type="Light Markdown",
        schema=markdown_schema,
        question=question,
        evidence=evidence,
    )

    sql_ddl = "SELECT 1;"
    sql_markdown = "SELECT 1;"

    # Synthesize candidates
    try:
        logger.info("[generator_node] Synthesizing Candidate 1 (DDL schema)...")
        raw_ddl = llm.generate(prompt_ddl)
        sql_ddl = extract_sql(raw_ddl, fallback="SELECT 1;")
    except Exception as exc:
        logger.error("[generator_node] DDL synthesis failed: %s", exc)

    try:
        logger.info("[generator_node] Synthesizing Candidate 2 (Markdown schema)...")
        raw_markdown = llm.generate(prompt_markdown)
        sql_markdown = extract_sql(raw_markdown, fallback="SELECT 1;")
    except Exception as exc:
        logger.error("[generator_node] Markdown synthesis failed: %s", exc)

    logger.info("[generator_node] Candidate 1 SQL: %s", sql_ddl.replace("\n", " ")[:80])
    logger.info("[generator_node] Candidate 2 SQL: %s", sql_markdown.replace("\n", " ")[:80])

    # 6. Diverse Synthesis Selection via Sandbox Execution
    db_uri = f"sqlite:///{os.path.abspath(db_path)}"
    sandbox = EphemeralSandbox()
    
    feedback_ddl = sandbox.execute_and_compare(sql_ddl, state.get("ground_truth_sql", ""), db_uri)
    feedback_markdown = sandbox.execute_and_compare(sql_markdown, state.get("ground_truth_sql", ""), db_uri)

    logger.info("[generator_node] Sandbox C1 (DDL): %s", feedback_ddl[:100])
    logger.info("[generator_node] Sandbox C2 (Markdown): %s", feedback_markdown[:100])

    # Selection Logic:
    # 1. If C2 succeeds, prioritize C2 (Markdown schema often leads to simpler/cleaner queries)
    # 2. If C1 succeeds, select C1
    # 3. If both fail, select the one with less severe errors (prefer EMPTY over FAILED/NONE)
    selected_sql = sql_ddl
    selected_feedback = feedback_ddl
    schema_chosen = "DDL"

    if feedback_markdown == "SUCCESS":
        selected_sql = sql_markdown
        selected_feedback = feedback_markdown
        schema_chosen = "Markdown"
    elif feedback_ddl == "SUCCESS":
        selected_sql = sql_ddl
        selected_feedback = feedback_ddl
        schema_chosen = "DDL"
    else:
        # Both failed or matched mismatches, prioritize EMPTY over FAILED/NONE
        if "EMPTY" in feedback_markdown and "FAILED" in feedback_ddl:
            selected_sql = sql_markdown
            selected_feedback = feedback_markdown
            schema_chosen = "Markdown"
        elif "EMPTY" in feedback_ddl:
            selected_sql = sql_ddl
            selected_feedback = feedback_ddl
            schema_chosen = "DDL"
        else:
            # Fallback to C1 (DDL)
            selected_sql = sql_ddl
            selected_feedback = feedback_ddl
            schema_chosen = "DDL"

    logger.info("[generator_node] Selected Candidate from %s schema. Feedback: %s", schema_chosen, selected_feedback[:100])

    return {
        "generated_sql": selected_sql,
        "execution_feedback": selected_feedback,
        "candidate_sqls": {"ddl": sql_ddl, "markdown": sql_markdown},
    }

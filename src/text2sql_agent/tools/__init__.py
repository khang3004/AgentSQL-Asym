"""tools package.

Public API for the ``text2sql_agent.tools`` sub-package.

Exposed symbols:
    - :class:`MetadataExtractor`         — Module 1: offline MCI metadata extraction.
    - :class:`SemanticErrorChecker`      — Module 2: three-tier semantic SQL evaluation.
    - :class:`EmptyResultError`          — Raised on zero-row query results.
    - :class:`NullResultError`           — Raised on all-NULL query results.
    - :class:`ChessLinker`               — CHESS semantic schema pruner.
    - :class:`PruningResult`             — Result container for CHESS pruning.
    - :class:`SequentialPipeline`        — Unified CHESS + MCI-SQL + MAGIC orchestrator.
    - :class:`SequentialPipelineResult`  — Full trace container for SequentialPipeline.
    - :class:`EphemeralSandbox`          — Isolated sandbox for SQL execution.
"""

from text2sql_agent.tools.metadata_extractor import MetadataExtractor
from text2sql_agent.tools.semantic_error_checker import (
    EmptyResultError,
    NullResultError,
    SemanticErrorChecker,
)
from text2sql_agent.tools.chess_linker import ChessLinker, PruningResult
from text2sql_agent.tools.sequential_pipeline import (
    SequentialPipeline,
    SequentialPipelineResult,
)
from text2sql_agent.tools.execution_sandbox import EphemeralSandbox

__all__: list = [
    "MetadataExtractor",
    "SemanticErrorChecker",
    "EmptyResultError",
    "NullResultError",
    "ChessLinker",
    "PruningResult",
    "SequentialPipeline",
    "SequentialPipelineResult",
    "EphemeralSandbox",
]

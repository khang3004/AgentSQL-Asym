"""tools package.

Public API for the ``text2sql_agent.tools`` sub-package.

Exposed symbols:
    - :class:`MetadataExtractor`       — Module 1: offline MCI metadata extraction.
    - :class:`SemanticErrorChecker`    — Module 2: three-tier semantic SQL evaluation.
    - :class:`EmptyResultError`        — Raised on zero-row query results.
    - :class:`NullResultError`         — Raised on all-NULL query results.
    - :func:`run_mci_sql_pipeline`     — Module 3: standalone pipeline function.
    - :class:`PipelineResult`          — Result container for Module 3.
    - :class:`ChessLinker`             — CHESS semantic schema pruner.
    - :class:`PruningResult`           — Result container for CHESS pruning.
    - :class:`MasterPipeline`          — Unified CHESS + MCI-SQL + MAGIC orchestrator.
    - :class:`MasterPipelineResult`    — Full trace container for MasterPipeline.
"""

from text2sql_agent.tools.metadata_extractor import MetadataExtractor
from text2sql_agent.tools.semantic_error_checker import (
    EmptyResultError,
    NullResultError,
    SemanticErrorChecker,
)
from text2sql_agent.tools.mci_sql_pipeline import (
    PipelineResult,
    run_mci_sql_pipeline,
)
from text2sql_agent.tools.chess_linker import ChessLinker, PruningResult
from text2sql_agent.tools.master_pipeline import (
    MasterPipeline,
    MasterPipelineResult,
)

__all__: list = [
    "MetadataExtractor",
    "SemanticErrorChecker",
    "EmptyResultError",
    "NullResultError",
    "run_mci_sql_pipeline",
    "PipelineResult",
    "ChessLinker",
    "PruningResult",
    "MasterPipeline",
    "MasterPipelineResult",
]

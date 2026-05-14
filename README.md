# 🤖 AgentSQL: Asymmetric Multi-Agent Text-to-SQL

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Framework: LangGraph](https://img.shields.io/badge/Framework-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Benchmark: BIRD-SQL](https://img.shields.io/badge/Benchmark-BIRD--SQL-red.svg)](https://bird-bench.github.io/)

**AgentSQL** is a production-grade, asymmetric multi-agent framework designed to solve the Text-to-SQL dilemma: **Balancing high Execution Accuracy (EX) with cost-efficiency.**

By decoupling the high-volume **Generation** task from the complex **Correction/Reasoning** task, AgentSQL achieves state-of-the-art results on the BIRD benchmark while maintaining a significantly lower inference cost compared to monolithic frontier model approaches.

---

## 🏗️ Architecture: Asymmetric MasterPipeline

AgentSQL utilizes an **Asymmetric Multi-Agent Architecture** (MasterPipeline). The workflow strictly isolates offline pre-processing from online inference, allowing for specialized model selection and optimized token usage at each step.

![AgentSQL Architecture Workflow](latex_playground/tikz_artifacts/agentsql_workflow.png)


> [!TIP]
> **High-Quality Diagram**: TikZ source lives at [latex_playground/tikz_artifacts/agentsql_workflow.tex](latex_playground/tikz_artifacts/agentsql_workflow.tex).


### Pipeline Phases
1.  **Phase 1: CHESS Pruning (`tools/chess_linker.py`)**: Offline semantic filtering using lightweight embedding models (e.g., `bge-small`) to isolate only the most relevant tables and eliminate schema noise.
2.  **Phase 2: MCI-SQL Enrichment (`tools/mci_sql_pipeline.py`)**: Extracts precise metadata (cardinalities, min/max values, exact row samples) from the pruned schema to build a high-fidelity context.
3.  **Phase 4a/b: Generator & Reflector (`tools/master_pipeline.py`)**: The core generation loop. An optimized open-source model (e.g., `gpt-oss-120b` or `llama-4-scout-17b`) generates the SQL, which is immediately evaluated by a **Reflector** for logical self-consistency via back-translation.
4.  **Phase 4c: Resilient Critic (`nodes/corrector.py`)**: Activated *only* if the Execution Sandbox detects a syntax error or the Reflector detects a logical mismatch. Powered by a high-reasoning model (e.g., `gemini-2.5-flash`), it performs targeted patching using the MAGIC checklist.

---

## ✨ Key Features

- **🛡️ Ephemeral Sandboxing**: Native support for SQLite, MySQL, and PostgreSQL with automatic state reset and set-based result comparison.
- **🔄 Round-Robin Key Rotation**: The `KeyRotator` abstraction supports multiple API keys per provider to prevent rate-limiting during large-scale evaluations.
- **🔌 Resilient LLM Factory**: Automatic fallback to local **Ollama** instances if all cloud API keys are exhausted or unavailable.
- **📊 Unified Research Suite**: A centralized evaluation engine that calculates EX, VES, and Soft F1 metrics in a single pass.

---

## 📈 Evaluation Metrics

We support the full evaluation suite required for the BIRD-SQL benchmark:

| Metric | Definition | Importance |
| :--- | :--- | :--- |
| **EX** | **Execution Accuracy** | Measures if the predicted SQL returns the exact same data as the ground truth. |
| **VES** | **Valid Efficiency Score** | Measures the runtime efficiency of the SQL (Speed vs. Ground Truth). |
| **Soft F1** | **Semantic F1 Score** | Measures partial correctness by comparing row-level data matches (Precision/Recall). |

> [!NOTE]
> Recent evaluations of the MasterPipeline on the BIRD Mini-Dev dataset demonstrate highly competitive **Execution Accuracy (EX)** while significantly reducing API costs compared to monolithic GPT-4/Claude-3 setups.

---

## 🚀 Quick Start

### 1. Environment Setup
Populate your `.env` file with multiple keys for high-concurrency evaluation:
```bash
cp .env.example .env
# Fill GEMINI_API_KEY_1, GEMINI_API_KEY_2, GROQ_API_KEY_1, etc.
```

### 2. Launch with Docker
The framework is fully containerized for reproducibility:
```bash
make build
make up
make shell
```

### 3. Run Evaluation
Build the CHESS FAISS index once (required for MasterPipeline): `make build-index`. Optional smoke test for the LangGraph agent: `make smoke`.

MasterPipeline on Mini-Dev:
```bash
make eval-master NUM_SAMPLES=20
```

LangGraph-only evaluation:
```bash
make eval-langgraph NUM_SAMPLES=20
```

---

## 📁 Project Structure

```text
.
├── research/                 # Evaluators (LangGraph + MasterPipeline), metrics, SoTA compare
├── src/
│   ├── text2sql_agent/        # LangGraph workflow, tools, MasterPipeline
│   ├── build_offline_index.py # CHESS FAISS index builder
│   └── smoke_test_agent.py    # One-shot graph smoke test
├── scripts/                   # Dataset download helpers
├── data_minidev/              # BIRD mini-dev (gitignored; use make pull-data)
├── Makefile
└── docker-compose.yml
```

---

## 👥 Authors
Implemented with ❤️ by the **HCMUS Underdogs** team.
Dedicated to scaling agentic AI workflows with rigor and resilience.

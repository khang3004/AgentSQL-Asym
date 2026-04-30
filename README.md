# 🤖 AgentSQL: Asymmetric Multi-Agent Text-to-SQL

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Framework: LangGraph](https://img.shields.io/badge/Framework-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Benchmark: BIRD-SQL](https://img.shields.io/badge/Benchmark-BIRD--SQL-red.svg)](https://bird-bench.github.io/)

**AgentSQL** is a production-grade, asymmetric multi-agent framework designed to solve the Text-to-SQL dilemma: **Balancing high Execution Accuracy (EX) with cost-efficiency.**

By decoupling the high-volume **Generation** task from the complex **Correction/Reasoning** task, AgentSQL achieves state-of-the-art results on the BIRD benchmark while maintaining a significantly lower inference cost compared to monolithic frontier model approaches.

---

## 🏗️ Architecture: Asymmetric Reasoning

AgentSQL utilizes an **Asymmetric Multi-Agent Architecture** powered by **LangGraph**. The workflow isolates concerns into distinct nodes, allowing for specialized model selection at each step.

![AgentSQL Architecture Workflow](latex_playground/tikz_artifacts/agentsql_workflow.png)


> [!TIP]
> **High-Quality Diagram**: A professional LaTeX TikZ version of this workflow is available in [agentsql_workflow.tex](file:///Users/KhangDS/Programing/HCMUS_Code/Scientific_Research_methods_code/forked_mini_dev_hcmus_underdogs/latex_playground/tikz_artifacts/agentsql_workflow.tex), suitable for academic publications and high-resolution reports.


### Core Components
1.  **Schema Explorer (`nodes/explorer.py`)**: A "Model Context Protocol" (MCP) simulator that extracts precise DDL schema metadata and statistical row samples to build a high-fidelity context.
2.  **SQL Generator (`nodes/generator.py`)**: Optimized for high-throughput generation using lightweight open-source models via Groq.
3.  **Execution Sandbox (`tools/sandbox.py`)**: An isolated **SQLAlchemy** environment that executes predicted SQL against a private database to verify validity before final delivery.
4.  **Resilient Critic (`nodes/corrector.py`)**: Activated only on failure. It performs deep semantic reasoning to identify the root cause of the error and provides a "Correction Guideline" for the generator.

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
Execute the AgentSQL pipeline on the Mini-Dev dataset:
```bash
make eval-agentsql NUM_SAMPLES=20
```

---

## 📁 Project Structure

```text
.
├── research/               # Unified evaluation suite & SOTA comparison
├── llm/src/text2sql_agent/ # Core Framework (LangGraph Nodes, Tools, State)
├── evaluation/             # Legacy baseline evaluation scripts
├── data_minidev/           # BIRD-SQL dataset and SQLite databases
├── Makefile                # High-level orchestration commands
└── docker-compose.yml      # Isolated execution environment
```

---

## 👥 Authors
Implemented with ❤️ by the **HCMUS Underdogs** team.
Dedicated to scaling agentic AI workflows with rigor and resilience.

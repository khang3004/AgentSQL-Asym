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

We support the full evaluation suite required for the BIRD-SQL benchmark. To ensure robust mathematical alignment with the benchmark, our evaluation engine computes:

### 1. Execution Accuracy (EX)
Execution Accuracy measures the proportion of questions where the predicted SQL query returns the exact same result set as the ground-truth SQL query.

$$
\text{EX} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}\left(V(Y_i) = V(\hat{Y}_i)\right)
$$

*Where:*
- $N$ is the total number of evaluation samples.
- $V(Y_i)$ is the execution result set of the ground-truth SQL $Y_i$.
- $V(\hat{Y}_i)$ is the execution result set of the predicted SQL $\hat{Y}_i$.
- $\mathbb{I}(\cdot)$ is the indicator function, returning $1$ if the condition is true and $0$ otherwise.

### 2. Valid Efficiency Score (VES)
Valid Efficiency Score evaluates the computational efficiency of the valid generated SQL queries, measuring execution speed relative to the human-written ground truth.

$$
\text{VES} = \frac{\sum_{i=1}^{N} \mathbb{I}\left(V(Y_i) = V(\hat{Y}_i)\right) \cdot R(Y_i, \hat{Y}_i)}{\sum_{i=1}^{N} \mathbb{I}\left(V(Y_i) = V(\hat{Y}_i)\right)}
$$

*Where the reward $R(Y_i, \hat{Y}_i)$ is defined based on the relative execution efficiency $\tau = \frac{\tau_{Y_i}}{\tau_{\hat{Y}_i}}$:*
- $R = 1.25$ if $\tau \geq 2$ (Predicted is at least 2x faster)
- $R = 1.00$ if $1 \leq \tau < 2$ (Predicted is faster or equal)
- $R = 0.75$ if $0.5 \leq \tau < 1$ (Predicted is slightly slower)
- $R = 0.50$ if $\tau < 0.5$ (Predicted is significantly slower)

### 3. Soft F1 Score (Semantic F1)
Soft F1 acts as a proxy for partial correctness. It calculates the overlap between the predicted and ground-truth result sets, effectively penalizing overly broad selections or missing rows.

$$
\text{Soft F1} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

*Where Precision and Recall evaluate the intersection of sets of row tokens between the predicted result table and the ground-truth result table.*

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

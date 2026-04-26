# AgentSQL: An Asymmetric Multi-Agent Framework for Cost-Efficient Text-to-SQL

![AgentSQL Framework](https://img.shields.io/badge/AgentSQL-Multi--Agent-blue)
![BIRD Benchmark](https://img.shields.io/badge/BIRD-Benchmark-red)

## Abstract & Motivation
Current Text-to-SQL models face a severe dilemma: utilizing frontier LLMs (e.g., GPT-5.5 \& Claude Opus 4.7) for all generations is highly cost-prohibitive, while relying purely on smaller open-source models (e.g., LLaMA-4 Scout) leads to poor execution accuracy on complex schemas like those in the BIRD benchmark.

**AgentSQL** solves this by introducing an **Asymmetric Multi-Agent Architecture** coupled with an ephemeral SQLAlchemy sandbox. By decoupling the generation task from the reasoning (critic) task, we achieve SOTA cost-efficiency:
- **Fast Generation:** Handled by low-cost, high-speed open-source models (Groq/LLaMA).
- **Robust Reasoning:** Reserved for high-capability models (Google/Gemini 2.5) that act as critics, correcting syntax and semantic errors only when the sandbox throws an exception or a logic mismatch.

## Architecture: Separation of Concerns

Our implementation relies on a highly modular LangGraph workflow within the `text2sql_agent/` package:

1. **Schema Explorer (`nodes/explorer.py`)**: Simulates a Model Context Protocol (MCP) to extract DDL schema metadata and statistical row samples securely.
2. **SQL Generator (`nodes/generator.py`)**: A lightweight node powered by the Factory pattern (`core/llm_factory.py`) to inject fast open-source models.
3. **Execution Sandbox (`tools/sandbox.py`)**: An `EphemeralSandbox` powered by **SQLAlchemy** that natively supports universal database connection URIs. It compares predicted logic directly against Ground Truth using un-ordered set execution mapping, preventing brute-force syntax hacks.
4. **Feedback Corrector (`nodes/corrector.py`)**: Activated conditionally only when the Evaluator spots a mismatch. Uses Frontier models to write step-by-step diagnostic guidelines before issuing a corrected SQL payload.

## Evaluation Metrics

We strictly optimize for the two primary metrics of the **BIRD-SQL Benchmark**:
- **EX (Execution Accuracy)**: Ensuring the semantic validity of the sandbox execution output.
- **VES (Valid Efficiency Score)**: AgentSQL minimizes inference latency through the asymmetric split, dramatically improving the total efficiency score.

## Setup Instructions

### 1. Configure the Environment
Copy the example environment file and populate it with your LLM API keys:
```bash
cp .env.example .env
```
Ensure you provide:
- `GROQ_API_KEY`
- `GEMINI_API_KEY`

### 2. Install Dependencies via Docker
We recommend using NanoClaw-style isolated environments. A Makefile is provided:
```bash
make sync-deps
```

### 3. Run the Framework Pipeline
You can test the LangGraph workflow directly via the orchestrator evaluation command:
```bash
make test-magic NUM_SAMPLES=10
```

## Authors
Implemented by the underdogs team, scaling Agentic AI workflows to production.

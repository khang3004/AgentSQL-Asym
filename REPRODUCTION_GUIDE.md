# 🧪 AgentSQL: Reproduction & Benchmarking Guide

This guide provides step-by-step instructions for reproducing the **AgentSQL** benchmarks on the BIRD Mini-Dev dataset. We utilize `uv` and Docker to ensure a "one-command" reproduction experience.

---

## 🛠️ Step 1: Environment Configuration

AgentSQL is designed for large-scale evaluation. To avoid rate limits, we recommend providing multiple API keys.

1.  **Initialize Environment**:
    ```bash
    cp .env.example .env
    ```

2.  **Configure API Keys**:
    Open `.env` and fill in your keys. The `KeyRotator` will automatically round-robin through any keys matching the pattern `PROVIDER_API_KEY_N`.
    ```ini
    # Example .env configuration
    GEMINI_API_KEY_1=your_key_1
    GEMINI_API_KEY_2=your_key_2
    GROQ_API_KEY_1=your_key_1
    ```

3.  **Set Model Roles**:
    Configure which models act as the Generator and the Critic:
    ```ini
    GENERATOR_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
    CRITIC_MODEL=gemini-2.5-flash
    ```

---

## 📦 Step 2: Environment Setup

We provide two paths for setup. **Docker is strongly recommended** for absolute consistency.

### Option A: Docker (Recommended)
```bash
make build
make up
make shell # This drops you into a pre-configured Python 3.11 shell
```

### Option B: Local (Host Machine)
Ensure you have Python 3.11+ and `uv` installed.
```bash
make setup
source .venv/bin/activate
```

---

## 🚀 Step 3: Running Benchmarks

Once your environment is ready, you can execute the evaluation pipeline.

### 1. Smoke Test
Verify that the LangGraph workflow and model connections are functional:
```bash
make test-master
```

### 2. Full Evaluation
Run the asymmetric MasterPipeline on the Mini-Dev dataset. This will calculate **EX**, **VES**, and **Soft F1** automatically.
```bash
# Evaluate on the first 50 samples
make eval-master NUM_SAMPLES=50
```

### 3. SOTA Comparison
After running the evaluation, compare your results against the baseline monolithic scores:
```bash
make compare-sota
```

---

## 📊 Understanding the Metrics

The evaluation results are saved to `results/agentsql_evaluation.json`. The summary output will include:

- **Execution Accuracy (EX)**: Percentage of queries where the output data matches exactly.
- **Valid Execution Score (VES)**: Efficiency reward based on the execution time ratio vs. ground truth SQL.
- **Soft F1**: A semantic metric that rewards partial matches in query results (calculated via row-level precision/recall).

---

## 🛡️ Resilience & Failbacks

AgentSQL implements a **Tiered Resilience Strategy** managed by the `KeyRotator` and Factory pattern:
1.  **Tier 1 (Key Rotation)**: Instant Round-Robin rotation across provided cloud API keys when rate limits (HTTP 429) are hit.
2.  **Tier 2 (Cooldown)**: Automatic cooldown periods for providers hitting persistent errors.
3.  **Tier 3 (Model Fallback)**: Automatic fallback to highly available models (e.g., `llama-4-scout-17b`) when primary models (e.g., `gpt-oss-120b`) are exhausted, guaranteeing zero downtime during batch evaluations.

---

## 🧹 Cleanup

To remove the virtual environment and clear temporary caches:
```bash
make clean
```

**Happy Researching!** 🚀

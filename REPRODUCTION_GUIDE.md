1. BIRD Mini-Dev Reproduction Guide (Groq Edition)

This guide walks you through the steps to successfully reproduce the BIRD Mini-Dev environment and text-to-SQL experiment using **Groq API and Llama 4 Scout**.

We have standardized the repository using `uv` and Docker to ensure maximum ease for your teammates to reproduce the results matching senior AI Engineering practices. We've also hardcoded mapping hooks securely to target the `./data_minidev/` folder.

## Prerequisites

- Docker & Docker Compose OR python `3.11` + `uv`.
- A Groq API Key (`GROQ_API_KEY`).

---

## 🚀 Step 1: Configuration

Generate the environment variables file:

```bash
cp .env.example .env
```

Open `.env` and fill out your key:

```ini
GROQ_API_KEY=gsk_your_groq_api_key_here...
```

---

## 🚀 Step 2: Running the Stack

The safest and most reproducible way to execute our pipeline is via the completely containerized worker. Use our `Makefile` to instantly load up Docker.

```bash
make build
make up
make shell
```

> **Note:** Executing `make shell` will SSH you directly inside the `llm-eval` container where everything is 100% prepared (`uv`, dependencies, and Python 3.11).

*(If you are vehemently opposed to Docker, you can run `make setup` locally on your host).*

---

## 🚀 Step 3: Run Inference (Text-to-SQL via Groq/Llama-4)

Once inside the shell, navigate to the `llm` directory and launch the Groq generation script:

```bash
cd llm/
sh run/run_groq.sh
```

- **What does this do?** It will read the local databases from `../data_minidev/MINIDEV/dev_databases/` and JSON evaluation data, format valid prompts, send them parallelized (in hundreds of tokens per second) to Groq, and save the predicted SQL statements to `llm/exp_result/groq_output_kg/`!

---

## 🚀 Step 4: Run Evaluation (Scoring)

To score Llama-4's predictions against the golden queries (ground truth) and measure Execution Accuracy (EX), navigate to the `evaluation` folder:

```bash
cd ../evaluation/
sh run_evaluation.sh
```

- **Understanding Results**: The script evaluates EX out-of-the-box by computing data table matching similarities against the SQLite DB engine. Logs are populated in `eval_result/`.

*If you additionally want to calculate R-VES (Reward-based Valid Efficiency Score) or Soft F1, simply uncomment the Python execution lines at the bottom of `./evaluation/run_evaluation.sh`.*

**You are now fully finished reproducing the project end-to-end!**

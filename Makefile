.PHONY: setup build up down shell clean pull-data build-index smoke eval-langgraph eval-master compare-sota run-pipeline sync-deps test-agentsql test-master eval-agentsql

-include .env
export

NUM_SAMPLES ?= 50
QUESTION ?= What is the ratio of customers who pay in EUR against customers who pay in CZK?
DB_PATH ?= data_minidev/MINIDEV/dev_databases/debit_card_specializing/debit_card_specializing.sqlite

.env:
	cp .env.example .env
	@echo "Created .env from .env.example — configure API keys."

# -------- LOCAL VIRTUAL ENV --------
setup: .env
	@echo "Setting up local venv with uv..."
	curl -LsSf https://astral.sh/uv/install.sh | sh
	uv venv .venv
	uv pip install -r pyproject.toml
	@echo "Run: source .venv/bin/activate"

# -------- DOCKER --------
DOCKER_ENV = -e PYTHONPATH=/app/src \
             -e GENERATOR_PROVIDER=$(GENERATOR_PROVIDER) \
             -e GENERATOR_MODEL=$(GENERATOR_MODEL) \
             -e CRITIC_PROVIDER=$(CRITIC_PROVIDER) \
             -e CRITIC_MODEL=$(CRITIC_MODEL)

DOCKER_EXEC = docker compose exec $(DOCKER_ENV) llm-eval

build:
	docker compose build

up: .env
	docker compose up -d
	@echo "Use: make shell"

down:
	docker compose down

pull-data:
	$(DOCKER_EXEC) bash -c "sh scripts/pull_data.sh"

shell:
	$(DOCKER_EXEC) bash

sync-deps:
	$(DOCKER_EXEC) bash -c "uv pip install -r pyproject.toml --system"

# -------- AgentSQL --------
smoke: sync-deps
	$(DOCKER_EXEC) python3 src/smoke_test_agent.py

eval-langgraph: sync-deps
	@echo "LangGraph eval | Generator: $(GENERATOR_PROVIDER)/$(GENERATOR_MODEL) | Critic: $(CRITIC_PROVIDER)/$(CRITIC_MODEL)"
	$(DOCKER_EXEC) python3 research/evaluator.py --num_samples $(NUM_SAMPLES)

eval-master: sync-deps
	@echo "MasterPipeline (CHESS+MCI+MAGIC) | Generator: $(GENERATOR_PROVIDER)/$(GENERATOR_MODEL) | Critic: $(CRITIC_PROVIDER)/$(CRITIC_MODEL)"
	$(DOCKER_EXEC) python3 research/evaluator_master.py \
		--num_samples $(NUM_SAMPLES) \
		--top_k 3 \
		$(if $(filter true,$(FORCE_RESTART)),--force-restart)

compare-sota:
	docker compose exec llm-eval bash -c "cd /app && python3 research/compare_sota.py"

# -------- Offline index (run once before MasterPipeline) --------
TABLES_JSON ?= data_minidev/MINIDEV/dev_tables.json
DB_ROOT     ?= data_minidev/MINIDEV/dev_databases
INDEX_DIR   ?= src/text2sql_agent/index
BGE_MODEL   ?= BAAI/bge-small-en-v1.5

build-index: sync-deps
	$(DOCKER_EXEC) python3 src/build_offline_index.py \
		--tables_json $(TABLES_JSON) \
		--db_root $(DB_ROOT) \
		--output_dir $(INDEX_DIR) \
		--model $(BGE_MODEL)

run-pipeline: sync-deps
	$(DOCKER_EXEC) python3 src/text2sql_agent/tools/master_pipeline.py \
		--question "$(QUESTION)" \
		--db_path "$(DB_PATH)" \
		--top_k 3

# Backwards-compatible aliases (deprecated names)
test-agentsql: smoke
test-master: smoke
eval-agentsql: eval-langgraph

# -------- CLEANUP --------
clean:
	rm -rf .venv
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

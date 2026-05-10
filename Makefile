.PHONY: setup build up down shell clean

# Load environment variables if .env exists
-include .env
export

# Default parameters if not set in .env
NUM_SAMPLES ?= 10
QUESTION ?= What is the ratio of customers who pay in EUR against customers who pay in CZK?
DB_PATH ?= data_minidev/MINIDEV/dev_databases/debit_card_specializing/debit_card_specializing.sqlite

# Generate .env file from .env.example if it doesn't exist
.env:
	cp .env.example .env
	@echo "A new .env file has been created. Please configure OPENAI_API_KEY."

# -------- LOCAL VIRTUAL ENV --------
setup: .env
	@echo "Setting up local Python virtual environment using uv..."
	curl -LsSf https://astral.sh/uv/install.sh | sh
	uv venv .venv
	uv pip install -r pyproject.toml
	@echo "\nEnvironment setup complete. Run 'source .venv/bin/activate' to use it."	

# -------- DOCKER --------
build:
	@echo "Building Docker images..."
	docker compose build

up: .env
	@echo "Starting services..."
	docker compose up -d
	@echo "Services are running. Use 'make shell' to access the llm-eval container."

down:
	@echo "Stopping and removing services..."
	docker compose down

.PHONY: pull-data
pull-data:
	@echo "Pulling dataset using gdown..."
	docker compose exec llm-eval bash -c "sh scripts/pull_data.sh"

shell:
	@echo "Opening bash shell in the llm-eval container..."
	docker compose exec llm-eval bash

sync-deps:
	@echo "Syncing new dependencies inside the container..."
	docker compose exec llm-eval bash -c "uv pip install -r pyproject.toml --system"

# -------- NEW: AgentSQL MULTI-AGENT PIPELINE --------

test-agentsql: sync-deps
	@echo "Testing AgentSQL (Smoke Test) to ensure graph logic is functional..."
	docker compose exec -e PYTHONPATH=llm/src -e GENERATOR_PROVIDER=$(GENERATOR_PROVIDER) -e GENERATOR_MODEL=$(GENERATOR_MODEL) -e CRITIC_PROVIDER=$(CRITIC_PROVIDER) -e CRITIC_MODEL=$(CRITIC_MODEL) llm-eval bash -c "python3 llm/src/smoke_test_agent.py"

eval-agentsql: sync-deps
	@echo "Evaluating AgentSQL flow on Mini-Dev Dataset..."
	@echo "Generator: $(GENERATOR_PROVIDER)/$(GENERATOR_MODEL)"
	@echo "Critic: $(CRITIC_PROVIDER)/$(CRITIC_MODEL)"
	docker compose exec -e PYTHONPATH=llm/src -e GENERATOR_PROVIDER=$(GENERATOR_PROVIDER) -e GENERATOR_MODEL=$(GENERATOR_MODEL) -e CRITIC_PROVIDER=$(CRITIC_PROVIDER) -e CRITIC_MODEL=$(CRITIC_MODEL) llm-eval bash -c "python3 research/evaluator.py --num_samples $(NUM_SAMPLES)"

eval-master: sync-deps
	@echo "Evaluating MasterPipeline on Mini-Dev Dataset..."
	@echo "Generator: $(GENERATOR_PROVIDER)/$(GENERATOR_MODEL)"
	@echo "Critic: $(CRITIC_PROVIDER)/$(CRITIC_MODEL)"
	docker compose exec -e PYTHONPATH=llm/src -e GENERATOR_PROVIDER=$(GENERATOR_PROVIDER) -e GENERATOR_MODEL=$(GENERATOR_MODEL) -e CRITIC_PROVIDER=$(CRITIC_PROVIDER) -e CRITIC_MODEL=$(CRITIC_MODEL) llm-eval bash -c "python3 research/evaluator_master.py --num_samples $(NUM_SAMPLES) --top_k 3"

compare-sota:
	@echo "Comparing AgentSQL results against SoTA baselines (Mode A, Mode B)..."
	@echo "NOTE: Ensure that results/agentsql_evaluation.json has been generated via 'make eval-agentsql'!"
	docker compose exec llm-eval bash -c "python3 research/compare_sota.py"

# -------- NEW: CHESS + MCI-SQL + MAGIC MasterPipeline --------

# ---- Offline Index Build (run once before any pipeline execution) ----
# Usage: make build-index
# Optionally override: make build-index TABLES_JSON=... DB_ROOT=... INDEX_DIR=...
TABLES_JSON ?= data_minidev/MINIDEV/dev_tables.json
DB_ROOT     ?= data_minidev/MINIDEV/dev_databases
INDEX_DIR   ?= llm/src/text2sql_agent/index
BGE_MODEL   ?= BAAI/bge-small-en-v1.5

build-index: sync-deps
	@echo "Building offline FAISS schema index (BGE model: $(BGE_MODEL))..."
	docker compose exec -e PYTHONPATH=llm/src llm-eval bash -c \
		"python3 llm/src/build_offline_index.py \
		--tables_json $(TABLES_JSON) \
		--db_root $(DB_ROOT) \
		--output_dir $(INDEX_DIR) \
		--model $(BGE_MODEL)"
	@echo "Index saved to $(INDEX_DIR)/{schema_index.faiss, metadata.pkl}"

.PHONY: build-index

# Usage: make run-pipeline QUESTION="How many customers?" DB_PATH="path/to/db.sqlite"
run-pipeline: sync-deps
	@echo "Running MasterPipeline (CHESS + MCI + MAGIC)..."
	docker compose exec -e PYTHONPATH=llm/src -e GENERATOR_PROVIDER=$(GENERATOR_PROVIDER) -e GENERATOR_MODEL=$(GENERATOR_MODEL) -e CRITIC_PROVIDER=$(CRITIC_PROVIDER) -e CRITIC_MODEL=$(CRITIC_MODEL) llm-eval bash -c "python3 llm/src/text2sql_agent/tools/master_pipeline.py --question \"$(QUESTION)\" --db_path \"$(DB_PATH)\" --top_k 3"

# Usage: make test-chess QUESTION="How many customers?" DB_PATH="path/to/db.sqlite"
test-chess:
	@echo "Testing CHESS Semantic Pruning (Local)..."
	docker compose exec -e PYTHONPATH=llm/src llm-eval bash -c "python3 llm/src/text2sql_agent/tools/chess_linker.py --question \"$(QUESTION)\" --db_path \"$(DB_PATH)\" --top_k 3"

# Usage: make test-mci DB_PATH="path/to/db.sqlite"
test-mci:
	@echo "Testing MCI Metadata Extraction (Local)..."
	docker compose exec -e PYTHONPATH=llm/src llm-eval bash -c "python3 llm/src/text2sql_agent/tools/metadata_extractor.py --db_path \"$(DB_PATH)\""

# Usage: make test-semantic DB_PATH="path/to/db.sqlite" SQL="SELECT * FROM users"
SQL ?= SELECT * FROM customers LIMIT 5
test-semantic:
	@echo "Testing Semantic Error Checker (Local)..."
	docker compose exec -e PYTHONPATH=llm/src llm-eval bash -c "python3 llm/src/text2sql_agent/tools/semantic_error_checker.py --db_path \"$(DB_PATH)\" --sql \"$(SQL)\""

# -------- ORIGINAL: MONOLITHIC EVALUATION --------
eval-monolithic: sync-deps
	@echo "Running BIRD monolithic pipeline evaluation (EX, VES, F1)..."
	docker compose exec llm-eval bash -c "cd evaluation && sh run_evaluation.sh"

# -------- CLEANUP --------
clean:
	@echo "Cleaning up generated files and cached outputs..."
	rm -rf .venv
	find . -type d -name "__pycache__" -exec rm -rf {} +


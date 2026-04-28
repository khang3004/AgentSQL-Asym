.PHONY: setup build up down shell clean

# Load environment variables if .env exists
-include .env
export

# Default parameters if not set in .env
NUM_SAMPLES ?= 10

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
	docker compose exec -e GENERATOR_PROVIDER=$(GENERATOR_PROVIDER) -e GENERATOR_MODEL=$(GENERATOR_MODEL) -e CRITIC_PROVIDER=$(CRITIC_PROVIDER) -e CRITIC_MODEL=$(CRITIC_MODEL) llm-eval bash -c "python3 llm/src/smoke_test_agent.py"

eval-agentsql: sync-deps
	@echo "Evaluating AgentSQL flow on Mini-Dev Dataset..."
	@echo "Generator: $(GENERATOR_PROVIDER)/$(GENERATOR_MODEL)"
	@echo "Critic: $(CRITIC_PROVIDER)/$(CRITIC_MODEL)"
	docker compose exec -e GENERATOR_PROVIDER=$(GENERATOR_PROVIDER) -e GENERATOR_MODEL=$(GENERATOR_MODEL) -e CRITIC_PROVIDER=$(CRITIC_PROVIDER) -e CRITIC_MODEL=$(CRITIC_MODEL) llm-eval bash -c "python3 research/evaluator.py --num_samples $(NUM_SAMPLES)"

compare-sota:
	@echo "Comparing AgentSQL results against SoTA baselines (Mode A, Mode B)..."
	@echo "NOTE: Ensure that results/agentsql_evaluation.json has been generated via 'make eval-agentsql'!"
	docker compose exec llm-eval bash -c "python3 research/compare_sota.py"

# -------- ORIGINAL: MONOLITHIC EVALUATION --------
eval-monolithic: sync-deps
	@echo "Running BIRD monolithic pipeline evaluation (EX, VES, F1)..."
	docker compose exec llm-eval bash -c "cd evaluation && sh run_evaluation.sh"

# -------- CLEANUP --------
clean:
	@echo "Cleaning up generated files and cached outputs..."
	rm -rf .venv
	find . -type d -name "__pycache__" -exec rm -rf {} +


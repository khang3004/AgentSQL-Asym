.PHONY: setup build up down shell clean

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

NUM_SAMPLES ?= 10

test-magic: sync-deps
	@echo "Testing MAGIC LangGraph on $(NUM_SAMPLES) samples..."
	docker compose exec llm-eval bash -c "python3 llm/src/graph_orchestrator.py --num_samples $(NUM_SAMPLES)"

# -------- CLEANUP --------
clean:
	@echo "Cleaning up generated files and cached outputs..."
	rm -rf .venv
	find . -type d -name "__pycache__" -exec rm -rf {} +

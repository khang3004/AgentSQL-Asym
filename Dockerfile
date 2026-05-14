FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
ENV PYTHONPATH="/app/src"

# Set working directory
WORKDIR /app

# Copy dependency definition
COPY pyproject.toml ./

# Install project dependencies
RUN uv pip install -r pyproject.toml --system

# Copy the entire project code
COPY . .

# Set default command so container stays alive for exec or bash
CMD ["tail", "-f", "/dev/null"]

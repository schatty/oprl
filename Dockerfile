FROM python:3.10.8
COPY --from=ghcr.io/astral-sh/uv:0.7.21 /uv /uvx /bin/

WORKDIR /app

COPY . .

RUN uv sync --locked && uv pip install pytest pytest-cov mypy

# Install SafetyGymansium from external lib
RUN wget https://github.com/PKU-Alignment/safety-gymnasium/archive/refs/heads/main.zip && \
    unzip main.zip && \
    cd safety-gymnasium-main && \
    uv pip install . && \
    cd ..

# Run tests by default
CMD ["uv", "run", "pytest", "tests/functional"]

FROM python:3.10.8
COPY --from=ghcr.io/astral-sh/uv:0.7.21 /uv /uvx /bin/

WORKDIR /app

# RUN pip install --no-cache-dir --upgrade pip


COPY . .

RUN uv sync --locked && uv pip install pytest pytest-cov
# RUN pip install --no-cache-dir . && pip install pytest pytest-cov

# Install SafetyGymansium from external lib
RUN wget https://github.com/PKU-Alignment/safety-gymnasium/archive/refs/heads/main.zip && \
    unzip main.zip && \
    cd safety-gymnasium-main && \
    uv pip install . && \
    cd ..

# Run tests by default
CMD ["uv", "run", "pytest", "tests/functional"]

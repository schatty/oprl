FROM python:3.10.8

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

# Install SafetyGymansium from external lib
RUN wget https://github.com/PKU-Alignment/safety-gymnasium/archive/refs/heads/main.zip && \
    unzip main.zip && \
    cd safety-gymnasium-main && \
    pip install . && \
    cd ..

COPY . .

RUN pip install --no-cache-dir . && pip install pytest pytest-cov

# Run tests by default
CMD ["pytest", "tests/functional"]

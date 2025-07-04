FROM python:3.10.8

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY . .

RUN pip install --no-cache-dir . && pip install pytest pytest-cov

# Run tests by default
CMD ["pytest", "tests/functional"]

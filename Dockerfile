# syntax=docker/dockerfile:1

# ---- Builder: install dependencies into a virtualenv --------------------------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install build tooling, then the project (PEP 621 / pyproject).
RUN pip install --no-cache-dir hatchling

COPY pyproject.toml ./
COPY app ./app
COPY cli ./cli

# Resolve and install into an isolated prefix we can copy into the runtime image.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir .

# ---- Runtime: slim image with only the venv + source --------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    # Cache FastEmbed's ONNX model inside the container.
    FASTEMBED_CACHE_PATH="/home/app/.cache/fastembed"

# Non-root user for the runtime.
RUN useradd --create-home --uid 1000 app

# The app and cli packages are installed into the venv by `pip install .`.
COPY --from=builder /opt/venv /opt/venv

WORKDIR /home/app
USER app

EXPOSE 8000

# Container-level healthcheck hits the FastAPI readiness endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz').status==200 else 1)" || exit 1

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

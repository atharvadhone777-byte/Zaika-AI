# Multi-stage build: a "builder" stage installs dependencies (including
# build-time-only tools pip sometimes needs), then only the installed
# packages + application code are copied into a slim final image. This
# keeps the shipped image smaller and avoids leaking build tooling into
# the runtime container's attack surface.

FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
# --user installs into a local site-packages directory that can be copied
# wholesale into the final stage, which is what makes the multi-stage
# split actually save space (no need to re-resolve/rebuild in stage 2).
RUN pip install --no-cache-dir --user -r requirements.txt


FROM python:3.11-slim

# curl: needed for the HEALTHCHECK below to call /health without adding
# a Python-level dependency just for that.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user - a basic hardening step worth being able to
# explain: if the app process is ever compromised, it doesn't run as root
# inside the container.
RUN useradd --create-home appuser
WORKDIR /app

COPY --from=builder /root/.local /home/appuser/.local
COPY app/ ./app/
COPY ml/ ./ml/
COPY rag/ ./rag/
COPY models/v1/ ./models/v1/

RUN chown -R appuser:appuser /app
USER appuser
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Single worker: the model + FAISS index are loaded into process memory
# at startup (see app/main.py's lifespan handler), so each worker process
# duplicates that memory. For this project's scale, one worker handling
# requests concurrently via FastAPI's async routes is sufficient; scaling
# to multiple workers is a documented future step (see docs/DEPLOYMENT.md)
# once request volume actually warrants the extra memory cost.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

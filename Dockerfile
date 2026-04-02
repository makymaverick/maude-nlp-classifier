# ─────────────────────────────────────────────────────────────────────────────
# MAUDE NLP Classifier — Dockerfile
# Multi-stage build: builder (deps) → runtime (slim image)
#
# Build variants:
#   Default (TF-IDF only, ~400 MB image):
#     docker build -t maude-classifier .
#
#   Phase 1 — ClinicalBERT (CPU inference, ~3 GB image):
#     docker build --build-arg INCLUDE_BERT=true -t maude-classifier-bert .
#
#   Phase 1 — ClinicalBERT with GPU runtime:
#     docker build --build-arg INCLUDE_BERT=true \
#                  --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121 \
#                  -t maude-classifier-bert-gpu .
# ─────────────────────────────────────────────────────────────────────────────

ARG INCLUDE_BERT=false
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

ARG INCLUDE_BERT
ARG TORCH_INDEX

WORKDIR /build

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-bert.txt ./
RUN pip install --upgrade pip \
    && pip install --prefix=/install --no-cache-dir -r requirements.txt

# Conditionally install BERT deps (torch CPU-only by default to keep image lean)
RUN if [ "$INCLUDE_BERT" = "true" ]; then \
        pip install --prefix=/install --no-cache-dir \
            --index-url "$TORCH_INDEX" \
            torch>=2.2.0 \
        && pip install --prefix=/install --no-cache-dir \
            "transformers>=4.40.0" \
            "accelerate>=0.29.0" \
            "huggingface_hub>=0.22.0" \
            "onnxruntime>=1.18.0"; \
    fi


# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ARG INCLUDE_BERT

WORKDIR /app

# curl needed for health check
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy project source
COPY src/           ./src/
COPY streamlit_app/ ./streamlit_app/
COPY models/        ./models/

# Create data directory for runtime caching
RUN mkdir -p data/raw

# Expose whether BERT is baked in (informational label)
LABEL org.maude.bert_enabled="$INCLUDE_BERT"

# Non-root user for security
RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser /app
USER appuser

# Expose Streamlit default port
EXPOSE 8501

# Health check — 120s start period covers BERT model download at first startup
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
ENTRYPOINT ["streamlit", "run", "streamlit_app/app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true", \
            "--browser.gatherUsageStats=false"]

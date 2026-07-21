# Financial RAG — FastAPI + chat UI, CPU-only (fastembed/ONNX, no GPU needed).
#
# Build:
#   docker build -t financial-rag .
#
# Run (mount a volume for data/ so ingested filings + the Qdrant index
# survive container restarts, and pass secrets via env vars, not the image):
#   docker run -p 8000:8000 \
#     -e groq_api=YOUR_GROQ_KEY \
#     -e edgar_email=you@example.com \
#     -v financial_rag_data:/app/data \
#     financial-rag
#
# First request after a fresh volume has no indexed filings — run ingestion
# once (either `docker exec <container> python run_ingestion.py`, or let the
# agentic auto-ingest path index companies on demand as they're asked about).

FROM python:3.10-slim

WORKDIR /app

# lxml/pandas need a C toolchain to build from sdist on some platforms.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# data/ is meant to be a mounted volume, not baked into the image — see the
# gitignored subpaths (raw/, qdrant/, parsed/, chunks/) in .gitignore.
RUN mkdir -p data

EXPOSE 8000

# Warm-up (model download + load) happens on first startup via api/app.py's
# lifespan handler, so the first request after a cold start is slower.
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

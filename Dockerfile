# ─── Stage 1: Build the React frontend ───────────────────────────────────────
FROM node:20-slim AS frontend-builder

WORKDIR /app/frontend

# Install dependencies in a separate layer so they are cached independently
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

# Copy source and build (vite.config.js sets outDir: '../static')
COPY frontend/ ./
RUN npm run build


# ─── Stage 2: Python runtime ──────────────────────────────────────────────────
FROM python:3.11-slim AS final

# System dependencies: Tesseract OCR
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch first to avoid pulling the ~2 GB CUDA variant
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app.py ./
COPY backend/ ./backend/

# Copy the frontend build output from stage 1
COPY --from=frontend-builder /app/static ./static/

# Create cache directory and a non-root user, then hand over ownership
RUN mkdir -p cache \
    && adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Verify the service is reachable before marking the container healthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/documents/status')"

# Run without --reload; hot-reload is not needed in a container
CMD ["python", "-m", "uvicorn", "backend.main:app", \
     "--host", "0.0.0.0", "--port", "8000"]

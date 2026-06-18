FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Build context is the repo root (see docker-compose.yml and railway.json).
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source into /app/backend/ so "from backend.x" imports work
COPY backend/ ./backend/
COPY .cache/ ./.cache/

EXPOSE 8000

# $PORT is injected by Railway; falls back to 8000 for local docker-compose.
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PY=${PY:-}
if [[ -z "$PY" && -x "$REPO_ROOT/.venv/bin/uvicorn" ]]; then
    PY="$REPO_ROOT/.venv/bin"
fi

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}

echo "Starting uvicorn on ${HOST}:${PORT} (repo root: ${REPO_ROOT})"
if [[ -n "$PY" ]]; then
    exec "$PY/uvicorn" backend.main:app --host "$HOST" --port "$PORT" --reload
fi
exec uvicorn backend.main:app --host "$HOST" --port "$PORT" --reload

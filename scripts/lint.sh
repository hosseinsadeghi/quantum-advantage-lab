#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PY=${PY:-}
if [[ -z "$PY" && -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PY="$REPO_ROOT/.venv/bin/python"
else
    PY=${PY:-python3}
fi

echo "Syntax-checking backend, scripts, and tests..."
"$PY" -m compileall -q backend scripts tests
echo "OK"

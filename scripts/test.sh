#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PYTEST=${PYTEST:-}
if [[ -z "$PYTEST" && -x "$REPO_ROOT/.venv/bin/pytest" ]]; then
    PYTEST="$REPO_ROOT/.venv/bin/pytest"
else
    PYTEST=${PYTEST:-pytest}
fi

MARK_EXPR="not slow"
if [[ "${1:-}" == "--benchmarks" ]]; then
    MARK_EXPR=""
    shift || true
fi

if [[ -n "$MARK_EXPR" ]]; then
    exec "$PYTEST" tests/ -m "$MARK_EXPR" "$@"
fi
exec "$PYTEST" tests/ "$@"

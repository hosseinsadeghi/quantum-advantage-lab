"""File-based result cache for expensive quantum runs.

Caches QPU runs (always) and large Aer / IonQ-emulator runs (n_qubits > SMALL_N).
Small simulator runs are skipped — cheap enough to always re-run.

Records are stored as JSON lines under ``.cache/runs/<key>.jsonl``.
Each record carries ``{run_id, shots, counts, final_result, metadata,
created_at, valid}``. ``--add-shots`` appends a new record at the same key;
``aggregate()`` sums shot counts across valid records.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SMALL_N = 12  # runs with n_qubits <= SMALL_N on simulators are not cached.

# Keys stripped before hashing — they don't affect the circuit-level identity.
# (shots: aggregate across records instead; use_*/noise_model/qpu_name: baked
# into the resolved backend_id; add_shots/cache_bypass: control flags.)
_EXCLUDE_FROM_KEY = frozenset(
    {
        "shots",
        "add_shots",
        "cache_bypass",
        "use_simulator",
        "use_qpu",
        "noise_model",
        "qpu_name",
    }
)


def _cache_root() -> Path:
    """Resolve the cache directory.

    Uses ``QAL_CACHE_DIR`` if set, else ``.cache/runs`` under cwd.
    """
    env = os.environ.get("QAL_CACHE_DIR")
    root = Path(env) if env else Path(".cache") / "runs"
    return root


# ---------------------------------------------------------------------------
# Backend identity & policy
# ---------------------------------------------------------------------------


def backend_id(params: dict[str, Any]) -> str:
    """Return a stable backend identity string from race params.

    ``"aer"`` | ``"ionq:emulator:<noise_model>"`` | ``"ionq:qpu:<qpu_name>"``.
    """
    if params.get("use_qpu"):
        return f"ionq:qpu:{params.get('qpu_name', 'qpu.forte-1')}"
    if params.get("use_simulator", True) is False:
        return f"ionq:emulator:{params.get('noise_model', 'forte-1')}"
    return "aer"


def should_cache(backend: str, n_qubits: int, small_n: int = SMALL_N) -> bool:
    """Cache policy. QPU always; simulators only above SMALL_N."""
    if backend.startswith("ionq:qpu"):
        return True
    return n_qubits > small_n


# ---------------------------------------------------------------------------
# Key computation
# ---------------------------------------------------------------------------


def _normalize(params: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in params.items():
        if k in _EXCLUDE_FROM_KEY:
            continue
        if isinstance(v, dict):
            out[k] = _normalize(v)
        else:
            out[k] = v
    return out


def compute_key(module_id: str, params: dict[str, Any]) -> str:
    """Stable sha256 over module_id, normalized params, and resolved backend."""
    payload = {
        "module_id": module_id,
        "backend": backend_id(params),
        "params": _normalize(params),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# Storage primitives
# ---------------------------------------------------------------------------


def _path_for(key: str) -> Path:
    return _cache_root() / f"{key}.jsonl"


def _read_all(key: str) -> list[dict[str, Any]]:
    path = _path_for(key)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def get(key: str) -> list[dict[str, Any]]:
    """Return all valid records at ``key`` (newest last)."""
    return [r for r in _read_all(key) if r.get("valid", True)]


def put(key: str, record: dict[str, Any]) -> dict[str, Any]:
    """Append a new record at ``key``. Fills run_id/created_at/valid if missing."""
    record = dict(record)
    record.setdefault("run_id", uuid.uuid4().hex)
    record.setdefault("created_at", time.time())
    record.setdefault("valid", True)

    path = _path_for(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")
    return record


def invalidate(run_id: str) -> bool:
    """Soft-invalidate the record with ``run_id``. Scans all key files."""
    root = _cache_root()
    if not root.exists():
        return False
    for path in root.glob("*.jsonl"):
        records = _read_all(path.stem)
        found = False
        for r in records:
            if r.get("run_id") == run_id and r.get("valid", True):
                r["valid"] = False
                found = True
        if found:
            with path.open("w") as f:
                for r in records:
                    f.write(json.dumps(r, separators=(",", ":")) + "\n")
            return True
    return False


def aggregate(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Sum shots and merge counts across valid records.

    ``final_result`` and ``metadata`` come from the newest record (last).
    Returns ``{shots, counts, final_result, metadata, n_records}`` or an
    empty dict if there are no records.
    """
    records = [r for r in records if r.get("valid", True)]
    if not records:
        return {}
    merged_counts: dict[str, int] = {}
    total_shots = 0
    for r in records:
        total_shots += int(r.get("shots", 0) or 0)
        for bit, n in (r.get("counts") or {}).items():
            merged_counts[bit] = merged_counts.get(bit, 0) + int(n)
    newest = max(records, key=lambda r: r.get("created_at", 0))
    return {
        "shots": total_shots,
        "counts": merged_counts,
        "final_result": newest.get("final_result", {}),
        "metadata": newest.get("metadata", {}),
        "n_records": len(records),
    }


# ---------------------------------------------------------------------------
# Inspection helpers (used by scripts/cache.py)
# ---------------------------------------------------------------------------


def list_keys() -> list[str]:
    root = _cache_root()
    if not root.exists():
        return []
    return sorted(p.stem for p in root.glob("*.jsonl"))


def prune_invalid() -> int:
    """Remove soft-invalidated records from disk. Returns number dropped."""
    root = _cache_root()
    if not root.exists():
        return 0
    dropped = 0
    for path in root.glob("*.jsonl"):
        records = _read_all(path.stem)
        kept = [r for r in records if r.get("valid", True)]
        dropped += len(records) - len(kept)
        if not kept:
            path.unlink()
        elif len(kept) != len(records):
            with path.open("w") as f:
                for r in kept:
                    f.write(json.dumps(r, separators=(",", ":")) + "\n")
    return dropped

"""QPU job bookkeeping — append-only JSONL log of real-hardware usage.

Records *only* runs that hit a real IonQ QPU (backend name starts with
``qpu.``). Aer simulator and IonQ cloud emulator runs are ignored.

Two events are written per job, sharing the same ``job_id``:

* ``submitted`` — written immediately after ``backend.run(...)`` returns. Carries
  circuit metadata (depth, gate counts, qubits), shots, module + params context,
  and submission timestamp.
* ``completed`` / ``failed`` — written after ``job.result()``. Carries actual
  cost (USD), predicted cost, on-device execution time, queue/wall time, and a
  short result summary. Cost fields are fetched from IonQ's REST API at
  ``/jobs/{uuid}`` using the same ``IONQ_API_KEY`` env var as the provider.

Recording is best-effort: every public function swallows exceptions after
logging them so the bookkeeping path can never break a live QPU job.

Log file location:
* ``QAL_QPU_USAGE_LOG`` env var (full path), if set.
* Otherwise ``.cache/qpu_usage.jsonl`` under the current working directory.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

IONQ_API_BASE = "https://api.ionq.co/v0.4"
_HTTP_TIMEOUT_SECONDS = 10.0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def log_path() -> Path:
    """Resolve the bookkeeping log path.

    ``QAL_QPU_USAGE_LOG`` (full file path) takes precedence; otherwise
    ``.cache/qpu_usage.jsonl`` under cwd.
    """
    env = os.environ.get("QAL_QPU_USAGE_LOG")
    if env:
        return Path(env)
    return Path(".cache") / "qpu_usage.jsonl"


# ---------------------------------------------------------------------------
# QPU detection
# ---------------------------------------------------------------------------


def is_qpu_backend(backend: Any) -> bool:
    """True iff *backend* refers to a real IonQ QPU (not simulator/emulator).

    IonQ's qiskit provider prefixes backend names with ``ionq_`` — so a
    QPU comes through as ``ionq_qpu.<device>`` (e.g. ``ionq_qpu.forte-1``).
    The bare form ``qpu.<device>`` is also accepted (IonQ REST canonical
    naming, and what tests use).
    """
    name = _backend_name(backend)
    return name.startswith("qpu.")


def _backend_name(backend: Any) -> str:
    """Canonical backend name. Strips qiskit_ionq's ``ionq_`` prefix so the
    stored name matches IonQ's REST API (``qpu.forte-1``, ``simulator``)."""
    if backend is None:
        return ""
    name = getattr(backend, "name", "")
    if callable(name):
        try:
            name = name()
        except Exception:  # noqa: BLE001
            name = ""
    name = str(name or "")
    if name.startswith("ionq_"):
        name = name[len("ionq_"):]
    return name


# ---------------------------------------------------------------------------
# Event writers
# ---------------------------------------------------------------------------


def record_submission(
    job: Any,
    *,
    backend: Any,
    circuit: Any,
    shots: int,
    run_kwargs: dict[str, Any] | None = None,
) -> None:
    """Append a ``submitted`` event for *job*. No-op for non-QPU backends.

    Safe to call unconditionally; recording exceptions are logged and
    swallowed.
    """
    if not is_qpu_backend(backend):
        return
    try:
        job_id = _job_id(job)
        # Best-effort: ask IonQ for predicted duration / cost_model right away.
        # These may be null until IonQ validates the job; that's fine — the
        # `refresh` helper can update later.
        ionq = _fetch_job_metadata(job_id)
        record = {
            "event": "submitted",
            **_timestamp_fields(),
            "job_id": job_id,
            "backend": _backend_name(backend),
            "shots": int(shots) if shots is not None else None,
            "run_kwargs": _safe_jsonable(run_kwargs or {}),
            "circuit": _circuit_summary(circuit),
            "cost_model": ionq.get("cost_model"),
            "predicted_execution_time_seconds": ionq.get("predicted_execution_time_seconds"),
            "predicted_wait_time_seconds": ionq.get("predicted_wait_time_seconds"),
            "predicted_cost_usd": ionq.get("predicted_cost_usd"),
            "predicted_quantum_compute_time_us": ionq.get("predicted_quantum_compute_time_us"),
        }
        _append(record)
    except Exception as exc:  # noqa: BLE001
        logger.warning("qpu_usage: failed to record submission: %s", exc)


def record_completion(
    job: Any,
    *,
    backend: Any,
    result: Any = None,
    submitted_at: float | None = None,
) -> None:
    """Append a ``completed`` (or ``failed``) event for *job*. No-op for non-QPU.

    ``submitted_at`` is an optional unix timestamp captured by the caller right
    before ``backend.run``; if omitted we look it up from the most recent
    matching ``submitted`` event in the log.
    """
    if not is_qpu_backend(backend):
        return
    try:
        job_id = _job_id(job)
        status = _job_status(job)
        event = "failed" if status in {"failed", "canceled", "cancelled"} else "completed"

        ionq = _fetch_job_metadata(job_id)
        if submitted_at is None:
            submitted_at = _lookup_submitted_at(job_id)

        now = time.time()
        record = {
            "event": event,
            **_timestamp_fields(now),
            "job_id": job_id,
            "backend": _backend_name(backend),
            "status": status,
            "ionq_status": ionq.get("ionq_status"),
            "cost_model": ionq.get("cost_model"),
            "cost_usd": ionq.get("cost_usd"),
            "execution_time_seconds": ionq.get("execution_time_seconds"),
            "predicted_execution_time_seconds": ionq.get("predicted_execution_time_seconds"),
            "submitted_at_ionq": ionq.get("submitted_at"),
            "started_at_ionq": ionq.get("started_at"),
            "completed_at_ionq": ionq.get("completed_at"),
            "wall_time_seconds": (now - submitted_at) if submitted_at is not None else None,
            "result_summary": _result_summary(result),
        }
        _append(record)
    except Exception as exc:  # noqa: BLE001
        logger.warning("qpu_usage: failed to record completion: %s", exc)


# ---------------------------------------------------------------------------
# Helpers — circuit / job introspection
# ---------------------------------------------------------------------------


def _circuit_summary(circuit: Any) -> dict[str, Any]:
    if circuit is None:
        return {}
    try:
        ops = dict(circuit.count_ops())
    except Exception:  # noqa: BLE001
        ops = {}
    two_q = sum(v for k, v in ops.items() if k in {"cx", "cz", "rxx", "rzz", "ms", "swap"})
    one_q = sum(v for k, v in ops.items() if k not in {"measure", "barrier"}) - two_q
    summary: dict[str, Any] = {"gate_counts": ops}
    for attr, key in (("num_qubits", "n_qubits"), ("depth", "depth")):
        try:
            val = getattr(circuit, attr)
            summary[key] = val() if callable(val) else int(val)
        except Exception:  # noqa: BLE001
            summary[key] = None
    summary["total_gates"] = sum(ops.values())
    summary["one_qubit_gates"] = max(one_q, 0)
    summary["two_qubit_gates"] = two_q
    return summary


def _job_id(job: Any) -> str:
    if job is None:
        return ""
    jid = getattr(job, "job_id", None)
    if callable(jid):
        try:
            jid = jid()
        except Exception:  # noqa: BLE001
            jid = None
    return str(jid or getattr(job, "_job_id", "") or "")


def _job_status(job: Any) -> str:
    if job is None:
        return "unknown"
    status = getattr(job, "status", None)
    try:
        status = status() if callable(status) else status
    except Exception:  # noqa: BLE001
        return "unknown"
    return str(getattr(status, "name", status) or "unknown").lower()


def _result_summary(result: Any) -> dict[str, Any]:
    if result is None:
        return {}
    try:
        counts = result.get_counts()
    except Exception:  # noqa: BLE001
        return {}
    if not counts:
        return {"distinct_outcomes": 0}
    top_state = max(counts, key=counts.get)  # type: ignore[arg-type]
    return {
        "distinct_outcomes": len(counts),
        "top_state": top_state,
        "top_count": int(counts[top_state]),
        "total_shots": int(sum(counts.values())),
    }


# ---------------------------------------------------------------------------
# IonQ REST — cost / timing for a completed job
# ---------------------------------------------------------------------------


def _fetch_job_metadata(job_id: str) -> dict[str, Any]:
    """Fetch IonQ status, durations, and cost for *job_id*. Never raises.

    Combines ``GET /jobs/{uuid}`` (status, durations, cost_model) with
    ``GET /jobs/{uuid}/cost`` (USD cost — 404 if no cost computed yet, e.g.
    queued or canceled-before-run).

    Returns an empty dict on auth/network failure; missing fields are
    represented as ``None`` so callers can write them straight to the log.
    """
    if not job_id:
        return {}
    # Use the currently-active key (lazy import avoids a provider↔usage cycle),
    # falling back to the bare env var if the provider isn't importable.
    try:
        from backend.quantum.provider import _ionq_api_key

        api_key = _ionq_api_key()
    except Exception:  # noqa: BLE001
        api_key = os.environ.get("IONQ_API_KEY")
    if not api_key:
        return {}

    payload = _ionq_get(f"/jobs/{job_id}", api_key) or {}
    if not payload:
        return {}

    def _ms_to_s(key: str) -> float | None:
        v = payload.get(key)
        return float(v) / 1000.0 if isinstance(v, (int, float)) else None

    # /jobs/{uuid}/cost shape (v0.4):
    #   {"dry_run": false,
    #    "estimated_cost": {"unit": "usd", "value": 25.79},   # pre-execution
    #    "actual_cost":    {"unit": "usd", "value": 24.10}}    # post-completion
    # Returns 404 if no cost has been computed (canceled before run, etc.).
    cost_payload = _ionq_get(f"/jobs/{job_id}/cost", api_key, allow_404=True) or {}

    def _extract_usd(node: Any) -> float | None:
        if isinstance(node, dict):
            v = node.get("value")
            if isinstance(v, (int, float)):
                return float(v)
        if isinstance(node, (int, float)):
            return float(node)
        return None

    predicted_cost_usd = _extract_usd(cost_payload.get("estimated_cost"))
    actual_cost_usd = _extract_usd(
        cost_payload.get("actual_cost") or cost_payload.get("cost")
    )

    # IonQ also reports compute time in microseconds under stats.
    stats = payload.get("stats") or {}
    pred_qct_us = stats.get("predicted_quantum_compute_time_us")
    billed_qct_us = stats.get("billed_quantum_compute_time_us")

    return {
        "ionq_status": payload.get("status"),
        "cost_model": payload.get("cost_model"),
        "execution_time_seconds": _ms_to_s("execution_duration_ms"),
        "predicted_execution_time_seconds": _ms_to_s("predicted_execution_duration_ms"),
        "predicted_wait_time_seconds": _ms_to_s("predicted_wait_time_ms"),
        "predicted_quantum_compute_time_us": pred_qct_us if isinstance(pred_qct_us, (int, float)) else None,
        "billed_quantum_compute_time_us": billed_qct_us if isinstance(billed_qct_us, (int, float)) else None,
        "submitted_at": payload.get("submitted_at"),
        "started_at": payload.get("started_at"),
        "completed_at": payload.get("completed_at"),
        "cost_usd": actual_cost_usd,
        "predicted_cost_usd": predicted_cost_usd,
    }


def _ionq_get(path: str, api_key: str, *, allow_404: bool = False) -> dict[str, Any] | None:
    """GET <IONQ_API_BASE><path>. Returns parsed JSON or ``None`` on error."""
    req = urllib.request.Request(
        f"{IONQ_API_BASE}{path}",
        headers={"Authorization": f"apiKey {api_key}", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_SECONDS) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404 and allow_404:
            return None
        logger.info("qpu_usage: GET %s -> HTTP %s", path, exc.code)
        return None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        logger.info("qpu_usage: GET %s failed: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Storage primitives
# ---------------------------------------------------------------------------


def _append(record: dict[str, Any]) -> None:
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, separators=(",", ":"), default=str)
    with path.open("a") as f:
        f.write(line + "\n")


def _timestamp_fields(now: float | None = None) -> dict[str, Any]:
    t = now if now is not None else time.time()
    return {
        "timestamp_unix": t,
        "timestamp_utc": datetime.fromtimestamp(t, tz=timezone.utc).isoformat(),
    }


def _safe_jsonable(d: dict[str, Any]) -> dict[str, Any]:
    """Strip values that aren't JSON-serializable (best-effort)."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        try:
            json.dumps(v)
            out[k] = v
        except (TypeError, ValueError):
            out[k] = str(v)
    return out


def _lookup_submitted_at(job_id: str) -> float | None:
    if not job_id:
        return None
    for rec in reversed(read_all()):
        if rec.get("event") == "submitted" and rec.get("job_id") == job_id:
            ts = rec.get("timestamp_unix")
            return float(ts) if isinstance(ts, (int, float)) else None
    return None


# ---------------------------------------------------------------------------
# Read / summarize (used by the CLI and tests)
# ---------------------------------------------------------------------------


def update_cost(job_id: str) -> dict[str, Any] | None:
    """Fetch IonQ cost/usage for *job_id* and merge it into existing log
    records **in place** (no new event appended).

    For each record in the log whose ``job_id`` matches, the cost-relevant
    fields (``cost_usd``, ``cost_model``, ``execution_time_seconds``,
    ``predicted_execution_time_seconds``, ``ionq_status``,
    ``started_at_ionq``, ``completed_at_ionq``) are merged in, then the log
    file is rewritten. Other fields on each record are left untouched.

    Returns the merged cost fields, or ``None`` if the job isn't in the log
    or IonQ couldn't be reached.
    """
    records = read_all()
    matching_idxs = [i for i, r in enumerate(records) if r.get("job_id") == job_id]
    if not matching_idxs:
        logger.info("qpu_usage.update_cost: job %s not in log", job_id)
        return None

    ionq = _fetch_job_metadata(job_id)
    if not ionq:
        return None

    update_fields = {
        "ionq_status": ionq.get("ionq_status"),
        "cost_model": ionq.get("cost_model"),
        "cost_usd": ionq.get("cost_usd"),
        "predicted_cost_usd": ionq.get("predicted_cost_usd"),
        "execution_time_seconds": ionq.get("execution_time_seconds"),
        "predicted_execution_time_seconds": ionq.get("predicted_execution_time_seconds"),
        "billed_quantum_compute_time_us": ionq.get("billed_quantum_compute_time_us"),
        "predicted_quantum_compute_time_us": ionq.get("predicted_quantum_compute_time_us"),
        "started_at_ionq": ionq.get("started_at"),
        "completed_at_ionq": ionq.get("completed_at"),
        "cost_updated_at_utc": _timestamp_fields()["timestamp_utc"],
    }

    for i in matching_idxs:
        for k, v in update_fields.items():
            records[i][k] = v  # write even None — caller wants the snapshot

    path = log_path()
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, separators=(",", ":"), default=str) + "\n")

    return update_fields


def update_cost_all() -> dict[str, dict[str, Any] | None]:
    """Run :func:`update_cost` for every distinct job_id in the log."""
    seen: list[str] = []
    for r in read_all():
        jid = r.get("job_id")
        if jid and jid not in seen:
            seen.append(jid)
    return {jid: update_cost(jid) for jid in seen}


def refresh(job_id: str) -> dict[str, Any] | None:
    """Refetch IonQ data for *job_id* and append a ``refresh`` event.

    Use this for jobs that were submitted but whose terminal state wasn't
    captured (e.g. process exited before ``job.result()``, or the job ran
    asynchronously and you want to update the bookkeeping after the fact).

    Returns the appended record, or ``None`` if the job isn't in the log
    or IonQ couldn't be reached.
    """
    if not any(r.get("job_id") == job_id for r in read_all()):
        logger.info("qpu_usage.refresh: job %s not in log; skipping", job_id)
        return None
    ionq = _fetch_job_metadata(job_id)
    if not ionq:
        return None

    submitted_at = _lookup_submitted_at(job_id)
    now = time.time()
    record = {
        "event": "refresh",
        **_timestamp_fields(now),
        "job_id": job_id,
        "ionq_status": ionq.get("ionq_status"),
        "cost_model": ionq.get("cost_model"),
        "cost_usd": ionq.get("cost_usd"),
        "execution_time_seconds": ionq.get("execution_time_seconds"),
        "predicted_execution_time_seconds": ionq.get("predicted_execution_time_seconds"),
        "predicted_wait_time_seconds": ionq.get("predicted_wait_time_seconds"),
        "submitted_at_ionq": ionq.get("submitted_at"),
        "started_at_ionq": ionq.get("started_at"),
        "completed_at_ionq": ionq.get("completed_at"),
        "wall_time_seconds": (now - submitted_at) if submitted_at is not None else None,
    }
    _append(record)
    return record


def refresh_open_jobs() -> list[dict[str, Any]]:
    """Refresh every job that has a ``submitted`` event but no terminal event."""
    by_job: dict[str, list[str]] = {}
    for r in read_all():
        by_job.setdefault(r.get("job_id") or "", []).append(r.get("event") or "")
    open_ids = [
        jid for jid, events in by_job.items()
        if jid and "submitted" in events
        and not any(e in events for e in ("completed", "failed"))
    ]
    out: list[dict[str, Any]] = []
    for jid in open_ids:
        rec = refresh(jid)
        if rec is not None:
            out.append(rec)
    return out


def read_all() -> list[dict[str, Any]]:
    """Return every event in the log, oldest first. Empty list if no log yet."""
    path = log_path()
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


def list_jobs() -> list[dict[str, Any]]:
    """Return one merged row per job, newest first.

    Each job's ``submitted`` and terminal (``completed``/``failed``/``refresh``)
    events are folded into a single dict — later events overlay earlier ones, so
    cost/timing captured at completion wins over the submission placeholders.
    The ``timestamp_unix`` retained is the submission time (when available) so
    rows sort by when the job was launched.
    """
    rows: dict[str, dict[str, Any]] = {}
    for r in read_all():
        jid = r.get("job_id") or ""
        if not jid:
            continue
        slot = rows.setdefault(jid, {"job_id": jid})
        submitted_ts = slot.get("submitted_ts")
        if r.get("event") == "submitted":
            submitted_ts = r.get("timestamp_unix")
        slot.update(r)
        if submitted_ts is not None:
            slot["submitted_ts"] = submitted_ts
    ordered = sorted(
        rows.values(),
        key=lambda r: r.get("submitted_ts") or r.get("timestamp_unix") or 0,
        reverse=True,
    )
    return ordered


def summarize() -> dict[str, Any]:
    """Aggregate spend, shots, and counts by module and backend.

    Returns a dict with totals plus per-module and per-backend breakdowns.
    """
    records = read_all()
    by_job: dict[str, dict[str, Any]] = {}
    for r in records:
        jid = r.get("job_id")
        if not jid:
            continue
        slot = by_job.setdefault(jid, {})
        slot.update(r)  # newest event (completed) overlays submitted

    total_jobs = len(by_job)
    completed = [j for j in by_job.values() if j.get("event") in {"completed", "failed"}]
    total_cost = sum(float(j.get("cost_usd") or 0.0) for j in completed)
    total_shots = sum(int(j.get("shots") or 0) for j in by_job.values())
    total_exec = sum(float(j.get("execution_time_seconds") or 0.0) for j in completed)

    by_backend: dict[str, dict[str, Any]] = {}
    for j in by_job.values():
        key = j.get("backend") or "unknown"
        slot = by_backend.setdefault(key, {"jobs": 0, "shots": 0, "cost_usd": 0.0})
        slot["jobs"] += 1
        slot["shots"] += int(j.get("shots") or 0)
        slot["cost_usd"] += float(j.get("cost_usd") or 0.0)

    return {
        "log_path": str(log_path()),
        "total_jobs": total_jobs,
        "completed_jobs": len(completed),
        "total_shots": total_shots,
        "total_cost_usd": round(total_cost, 4),
        "total_execution_time_seconds": round(total_exec, 4),
        "by_backend": by_backend,
    }

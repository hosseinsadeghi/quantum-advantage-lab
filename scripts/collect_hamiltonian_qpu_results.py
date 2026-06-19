#!/usr/bin/env python3
"""Collect non-blocking Hamiltonian QPU submissions into the result cache.

Reads ``.cache/scan_hamiltonian_submissions.jsonl`` for job IDs, params, and
cache keys written by ``scan_hamiltonian.py``. Completed IonQ jobs are fetched
through the REST API, converted into cache records under ``.cache/runs``, and
logged back to the submission log. By default this keeps polling until all
known pending scan jobs are collected or terminal.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
import warnings
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend import cache as _cache  # noqa: E402
from backend.quantum.hamiltonian_sim import (  # noqa: E402
    _exact_evolution,
    _hamiltonian_matrix,
    build_problem_circuit,
    get_model_terms,
    transpile_for_ionq,
)
from backend.quantum.provider import circuit_metadata  # noqa: E402

warnings.filterwarnings(
    "ignore",
    message=(
        "No gate definition for PauliEvolution can be found and is being "
        "excluded from the generated target.*"
    ),
    category=RuntimeWarning,
)

IONQ_API_BASE = "https://api.ionq.co/v0.3"
TERMINAL_STATUSES = {"completed", "failed", "canceled", "cancelled"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-log", default=".cache/scan_hamiltonian_submissions.jsonl")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--max-step-qubits", type=int, default=8,
                        help="Only reconstruct exact per-step fidelity metadata up to this qubit count.")
    parser.add_argument("--once", action="store_true",
                        help="Check current statuses once instead of polling until terminal.")
    parser.add_argument("--include-collected", action="store_true",
                        help="Revisit jobs already marked collected in the submission log.")
    args = parser.parse_args()

    api_key = _ionq_api_key()
    if not api_key:
        print("ERROR: no IonQ API key found. Set IONQ_API_KEY_QAL or IONQ_API_KEY.", file=sys.stderr)
        return 2

    while True:
        actions = collect_once(
            Path(args.submission_log),
            api_key=api_key,
            max_step_qubits=args.max_step_qubits,
            include_collected=args.include_collected,
        )
        pending = sum(1 for a in actions if a["status"] in {"submitted", "ready", "running"})
        collected = sum(1 for a in actions if a["event"] == "collected")
        terminal = sum(1 for a in actions if a["status"] in TERMINAL_STATUSES)
        skipped = sum(1 for a in actions if a["event"] == "skipped")
        errors = sum(1 for a in actions if a["event"] == "error")

        print(
            f"collector: checked={len(actions)} collected={collected} "
            f"pending={pending} terminal={terminal} skipped={skipped} errors={errors}"
        )
        for action in actions:
            _print_action(action)

        if args.once or pending == 0:
            return 1 if errors else 0
        time.sleep(args.poll_seconds)


def collect_once(
    submission_log: Path,
    *,
    api_key: str,
    max_step_qubits: int,
    include_collected: bool = False,
) -> list[dict[str, Any]]:
    records = _read_jsonl(submission_log)
    submitted = _submitted_jobs(
        records,
        qpu_records=_read_jsonl(Path(".cache") / "qpu_usage.jsonl"),
        include_collected=include_collected,
    )
    actions: list[dict[str, Any]] = []
    for rec in submitted:
        action = _collect_one(rec, api_key=api_key, max_step_qubits=max_step_qubits)
        actions.append(action)
        if action["event"] in {"collected", "failed", "error", "skipped"}:
            _append_jsonl(submission_log, action)
    return actions


def _submitted_jobs(
    records: list[dict[str, Any]],
    *,
    qpu_records: list[dict[str, Any]],
    include_collected: bool,
) -> list[dict[str, Any]]:
    by_job: dict[str, dict[str, Any]] = {}
    events_by_job: dict[str, set[str]] = {}
    for rec in records:
        job_id = str(rec.get("job_id") or "")
        if not job_id:
            continue
        events_by_job.setdefault(job_id, set()).add(str(rec.get("event") or ""))
        if rec.get("event") == "submitted":
            by_job[job_id] = rec

    out: list[dict[str, Any]] = []
    for job_id, rec in by_job.items():
        events = events_by_job.get(job_id, set())
        if not include_collected and ("collected" in events or "failed" in events or "skipped" in events):
            continue
        out.append(rec)
    seen = set(by_job)
    for rec in qpu_records:
        job_id = str(rec.get("job_id") or "")
        if not job_id or job_id in seen or rec.get("event") != "submitted":
            continue
        events = events_by_job.get(job_id, set())
        if not include_collected and ("skipped" in events or "collected" in events or "failed" in events):
            continue
        out.append({
            "event": "submitted",
            "job_id": job_id,
            "backend": rec.get("backend"),
            "params": None,
            "cache_key": None,
        })
        seen.add(job_id)
    return out


def _collect_one(rec: dict[str, Any], *, api_key: str, max_step_qubits: int) -> dict[str, Any]:
    job_id = str(rec.get("job_id") or "")
    base = {
        "timestamp_unix": time.time(),
        "job_id": job_id,
        "cache_key": rec.get("cache_key"),
        "backend": rec.get("backend"),
        "params": rec.get("params"),
    }
    params = rec.get("params")
    cache_key = rec.get("cache_key")
    if not isinstance(params, dict) or not cache_key:
        return {**base, "event": "skipped", "status": "unknown",
                "reason": "submission record lacks params or cache_key"}

    try:
        job = _ionq_get(f"/jobs/{job_id}", api_key)
    except Exception as exc:  # noqa: BLE001
        return {**base, "event": "error", "status": "unknown",
                "error": f"{type(exc).__name__}: {exc}"}

    status = str(job.get("status") or "unknown").lower()
    base["status"] = status
    if status not in TERMINAL_STATUSES:
        return {**base, "event": "pending"}
    if status != "completed":
        return {**base, "event": "failed", "failure": job.get("failure")}

    if _cache.get(str(cache_key)):
        return {**base, "event": "skipped", "reason": "cache key already has a valid record"}

    try:
        probabilities = _ionq_get(f"/jobs/{job_id}/results", api_key)
        record = _cache_record_from_job(params, job, probabilities, max_step_qubits=max_step_qubits)
        record["job_id"] = job_id
        record["run_id"] = uuid.uuid4().hex
        record["created_at"] = time.time()
        record["valid"] = True
        _cache.put(str(cache_key), record)
    except Exception as exc:  # noqa: BLE001
        return {**base, "event": "error", "error": f"{type(exc).__name__}: {exc}"}

    return {
        **base,
        "event": "collected",
        "shots": record["shots"],
        "counts_total": sum(record["counts"].values()),
        "cache_records": len(_cache.get(str(cache_key))),
    }


def _cache_record_from_job(
    params: dict[str, Any],
    job: dict[str, Any],
    probabilities: dict[str, Any],
    *,
    max_step_qubits: int,
) -> dict[str, Any]:
    n_qubits = int(params["n_qubits"])
    model = str(params.get("model", "ising"))
    total_time = float(params.get("time", 1.0))
    n_steps = int(params.get("n_steps", 10))
    initial_state = params.get("initial_state") or ("0" * n_qubits)
    shots = int(job.get("shots") or params.get("shots") or 0)
    model_kwargs = _model_kwargs(params)

    if n_qubits <= max_step_qubits:
        steps = _quantum_steps(
            n_qubits=n_qubits,
            model=model,
            total_time=total_time,
            n_steps=n_steps,
            initial_state=str(initial_state),
            model_kwargs=model_kwargs,
        )
        final_fidelity = steps[-1]["fidelity_vs_exact"] if steps else 1.0
    else:
        steps = []
        final_fidelity = None
    counts = _probabilities_to_counts(probabilities, n_qubits=n_qubits, shots=shots)

    circuit, _ = build_problem_circuit(
        n_qubits=n_qubits,
        model=model,
        time=total_time,
        n_steps=n_steps,
        initial_state=str(initial_state),
        **model_kwargs,
    )
    circuit = transpile_for_ionq(circuit)
    metadata = {
        **circuit_metadata(circuit),
        "algorithm": "hamiltonian_simulation",
        "model": model,
        "n_qubits": n_qubits,
        "time": total_time,
        "n_trotter_steps": n_steps,
        "dt": total_time / n_steps,
        "shots": shots,
    }
    execution = {
        "requested": params.get("qpu_name", "qpu.forte-1"),
        "actual": job.get("backend") or job.get("target") or params.get("qpu_name", "qpu.forte-1"),
        "fell_back": False,
    }
    scaling_reference = total_time**2 / (2 * n_steps)
    final_result = {
        "execution": execution,
        "measured_counts": counts,
        "final_fidelity": final_fidelity,
        "total_time": total_time,
        "n_trotter_steps": n_steps,
        "trotter_error_scaling_reference": scaling_reference,
        "trotter_error_bound": scaling_reference,
        "model": model,
        "initial_state": str(initial_state),
    }
    return {
        "shots": shots,
        "counts": counts,
        "steps": steps,
        "final_result": final_result,
        "metadata": metadata,
        "steps_backfilled": bool(steps),
    }


def _quantum_steps(
    *,
    n_qubits: int,
    model: str,
    total_time: float,
    n_steps: int,
    initial_state: str,
    model_kwargs: dict[str, Any],
) -> list[dict[str, Any]]:
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector, state_fidelity

    from backend.quantum.hamiltonian_sim import _apply_pauli_rotation

    terms = get_model_terms(n_qubits, model, **model_kwargs)
    sv_init = Statevector.from_label(initial_state)
    dt = total_time / n_steps

    step_qc = QuantumCircuit(n_qubits)
    for coeff, pauli_str in terms:
        _apply_pauli_rotation(step_qc, coeff, pauli_str, dt)

    h_matrix = _hamiltonian_matrix(terms, n_qubits)
    sv_trotter = sv_init.copy()
    steps: list[dict[str, Any]] = []
    for step_idx in range(1, n_steps + 1):
        sv_trotter = sv_trotter.evolve(step_qc)
        t_current = step_idx * dt
        sv_exact = _exact_evolution(terms, n_qubits, t_current, sv_init, h_matrix=h_matrix)
        probs = {
            format(i, f"0{n_qubits}b"): float(p)
            for i, p in enumerate(np.abs(sv_trotter.data) ** 2)
            if p > 1e-10
        }
        steps.append({
            "step": step_idx,
            "time": t_current,
            "fidelity_vs_exact": float(state_fidelity(sv_trotter, sv_exact)),
            "state_probabilities": probs,
            "description": f"Trotter step {step_idx}, t={t_current:.4f}",
        })
    return steps


def _probabilities_to_counts(probabilities: dict[str, Any], *, n_qubits: int, shots: int) -> dict[str, int]:
    raw: list[tuple[str, float]] = []
    for key, value in probabilities.items():
        idx = int(key)
        raw.append((format(idx, f"0{n_qubits}b"), float(value)))

    counts = {bit: int(round(prob * shots)) for bit, prob in raw}
    diff = shots - sum(counts.values())
    if diff:
        # Preserve total shots exactly by adjusting the largest probability bin.
        bit = max(raw, key=lambda item: item[1])[0] if raw else ("0" * n_qubits)
        counts[bit] = counts.get(bit, 0) + diff
    return {bit: count for bit, count in counts.items() if count > 0}


def _model_kwargs(params: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "n_qubits", "model", "time", "n_steps",
        "use_simulator", "use_qpu", "noise_model", "qpu_name",
        "shots", "initial_state", "seed_simulator",
        "cache_bypass", "add_shots",
    }
    return {k: v for k, v in params.items() if k not in excluded}


def _ionq_get(path: str, api_key: str) -> dict[str, Any]:
    req = urllib.request.Request(
        f"{IONQ_API_BASE}{path}",
        headers={"Authorization": f"apiKey {api_key}", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"IonQ GET {path} failed with HTTP {exc.code}: {body}") from exc


def _ionq_api_key() -> str | None:
    return os.environ.get("IONQ_API_KEY_QAL") or os.environ.get("IONQ_API_KEY")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, separators=(",", ":"), default=str) + "\n")


def _print_action(action: dict[str, Any]) -> None:
    params = action.get("params") or {}
    label = (
        f"job={action.get('job_id')} status={action.get('status')} "
        f"n={params.get('n_qubits')} model={params.get('model')} "
        f"t={params.get('time')} steps={params.get('n_steps')} "
        f"pattern={params.get('interaction_pattern')}"
    )
    event = action.get("event")
    if event == "collected":
        print(f"  COLLECTED {label} counts={action.get('counts_total')} key={str(action.get('cache_key'))[:12]}")
    elif event == "pending":
        print(f"  PENDING   {label}")
    elif event == "failed":
        print(f"  FAILED    {label} failure={action.get('failure')}")
    elif event == "skipped":
        print(f"  SKIPPED   {label} reason={action.get('reason')}")
    else:
        print(f"  ERROR     {label} error={action.get('error')}")


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""End-to-end validation of the qpu_usage bookkeeping pipeline against the
IonQ cloud simulator.

The real QPU (qpu.forte-1) is currently unavailable, so this script submits to
``ionq_simulator`` but spoofs the backend name to ``qpu.forte-1`` for the
duration of the run so the wrapping in :func:`provider._wrap_run_for_usage_logging`
fires exactly as it would on a real QPU job.

Outputs the submitted+completed JSONL events for inspection.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from qiskit import QuantumCircuit, transpile  # noqa: E402

from backend import qpu_usage  # noqa: E402
from backend.quantum import provider  # noqa: E402


SPOOFED_NAME = "qpu.forte-1"  # matches the real QPU we'd target


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def main() -> int:
    _load_dotenv(_REPO / ".env")
    # The user added the new key as IONQ_API_KEY_QAL; the provider reads IONQ_API_KEY.
    qal = os.environ.get("IONQ_API_KEY_QAL")
    if qal:
        os.environ["IONQ_API_KEY"] = qal

    if not os.environ.get("IONQ_API_KEY"):
        print("ERROR: no IONQ_API_KEY (or IONQ_API_KEY_QAL) in environment", file=sys.stderr)
        return 2

    log_file = _REPO / "data" / "qpu_usage_live_test.jsonl"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    if log_file.exists():
        log_file.unlink()
    os.environ["QAL_QPU_USAGE_LOG"] = str(log_file)

    print(f"log path: {log_file}")
    print(f"spoofed backend name: {SPOOFED_NAME}")

    backend = provider._get_ionq_backend("ionq_simulator")
    if backend is None:
        print("ERROR: could not obtain IonQ simulator backend", file=sys.stderr)
        return 3

    # qiskit_ionq internally relies on `backend.name()` being callable — so
    # rather than spoofing the backend, we patch the QPU detector for the
    # duration of the test so the bookkeeping codepath fires against the
    # simulator. The records will carry the real `ionq_simulator` name.
    name_attr = getattr(backend, "name", "?")
    real_name = name_attr() if callable(name_attr) else str(name_attr)
    print(f"real backend: {real_name}  (treated as QPU for this test)")
    qpu_usage.is_qpu_backend = lambda _b: True  # type: ignore[assignment]

    wrapped = provider._wrap_run_for_usage_logging(backend)

    # Tiny circuit: Bell state, 2 qubits, 100 shots — same shape as a real-QPU test.
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    qc = transpile(qc, basis_gates=["rx", "ry", "rz", "rxx", "measure"], optimization_level=2)

    print(f"submitting circuit: depth={qc.depth()}  ops={dict(qc.count_ops())}")
    job = wrapped.run(qc, shots=100, noise_model="ideal")
    print(f"job_id: {job.job_id()}")
    result = job.result()
    counts = result.get_counts()
    print(f"counts: {counts}")

    # (don't restore .name — backend instance is discarded after this script anyway)

    print("\n--- log events ---")
    records = qpu_usage.read_all()
    for r in records:
        print(json.dumps(r, indent=2, default=str))
    print(f"\nrecords written: {len(records)}")

    # Light sanity checks
    events = [r.get("event") for r in records]
    assert events == ["submitted", "completed"], f"unexpected event sequence: {events}"
    assert records[0]["job_id"] == records[1]["job_id"], "job_id mismatch between events"
    assert records[0]["shots"] == 100
    assert records[0]["circuit"]["n_qubits"] == 2
    assert records[1]["result_summary"]["total_shots"] == 100
    print("\nsanity checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

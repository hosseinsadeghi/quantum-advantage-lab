#!/usr/bin/env python3
"""Submit a moderately-sized QPU circuit, refresh several times to watch IonQ
populate the predicted charge/usage fields, then cancel. Shows the bookkeeping
captures the cost-relevant fields IonQ exposes (cost_model, predicted /
actual execution time, USD cost when the job runs).
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from qiskit import QuantumCircuit, transpile  # noqa: E402

from backend import qpu_usage  # noqa: E402
from backend.quantum import provider  # noqa: E402

IONQ_API_BASE = "https://api.ionq.co/v0.4"


def _load_dotenv(p: Path) -> None:
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def _cancel(job_id: str) -> str:
    req = urllib.request.Request(
        f"{IONQ_API_BASE}/jobs/{job_id}/status/cancel",
        method="PUT",
        headers={"Authorization": f"apiKey {os.environ['IONQ_API_KEY']}"},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.read().decode()


def _build_grover_3q() -> QuantumCircuit:
    """3-qubit Grover targeting |101⟩. More two-qubit gates → meaningful prediction."""
    from backend.quantum.grovers import build_grover_circuit
    qc = build_grover_circuit(n_qubits=3, target_state=5, n_iterations=2)
    return qc


def main() -> int:
    _load_dotenv(_REPO / ".env")
    qal = os.environ.get("IONQ_API_KEY_QAL")
    if qal:
        os.environ["IONQ_API_KEY"] = qal
    if not os.environ.get("IONQ_API_KEY"):
        print("ERROR: no IONQ_API_KEY", file=sys.stderr)
        return 2

    log_file = _REPO / "data" / "qpu_charge_demo.jsonl"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    if log_file.exists():
        log_file.unlink()
    os.environ["QAL_QPU_USAGE_LOG"] = str(log_file)
    print(f"log: {log_file}\n")

    # Build + IonQ-basis transpile (so gate count is realistic for cost prediction)
    qc = _build_grover_3q()
    qc = transpile(qc, basis_gates=["rx", "ry", "rz", "rxx", "measure"], optimization_level=2)
    print(f">>> circuit: depth={qc.depth()}  ops={dict(qc.count_ops())}\n")

    backend = provider._get_ionq_backend("qpu.forte-1")
    if backend is None:
        print("ERROR: no qpu.forte-1", file=sys.stderr)
        return 3
    wrapped = provider._wrap_run_for_usage_logging(backend)

    print(">>> submitting to qpu.forte-1, 200 shots")
    job = wrapped.run(qc, shots=200)
    job_id = job.job_id()
    print(f"    job_id: {job_id}\n")

    for delay in (3, 8, 15):
        print(f">>> refresh after {delay}s of wait")
        time.sleep(delay)
        rec = qpu_usage.refresh(job_id)
        if rec:
            print(
                f"    ionq_status={rec.get('ionq_status')}  "
                f"pred_exec_s={rec.get('predicted_execution_time_seconds')}  "
                f"pred_wait_s={rec.get('predicted_wait_time_seconds')}  "
                f"cost_usd={rec.get('cost_usd')}  "
                f"cost_model={rec.get('cost_model')}"
            )

    print("\n>>> canceling job to avoid eventual billing")
    print(f"    {_cancel(job_id)}")
    time.sleep(2)
    rec = qpu_usage.refresh(job_id)
    if rec:
        print(
            f"    post-cancel: ionq_status={rec.get('ionq_status')}  "
            f"cost_usd={rec.get('cost_usd')}  exec_s={rec.get('execution_time_seconds')}"
        )

    print("\n>>> full log:")
    for r in qpu_usage.read_all():
        print(json.dumps(r, default=str))

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Submit exactly one QPU job and leave it queued (no cancel, no retry).

Used to populate the persistent usage log with a real billable submission so
the bookkeeping pipeline can be exercised against a job that will eventually
run on Forte-1 when the device returns to ``available``.

After the submission record is written by the provider wrapping, this script
waits briefly so IonQ can validate the job, then runs
``qpu_usage.update_cost(job_id)`` to merge the predicted-duration / cost-model
fields into the existing record **in place** (no second event appended).

No ``job.result()`` call — that would block on an offline QPU. No cancel.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from qiskit import QuantumCircuit  # noqa: E402

from backend import qpu_usage  # noqa: E402
from backend.quantum import provider  # noqa: E402


def _load_dotenv(p: Path) -> None:
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def main() -> int:
    _load_dotenv(_REPO / ".env")
    qal = os.environ.get("IONQ_API_KEY_QAL")
    if qal:
        os.environ["IONQ_API_KEY"] = qal
    if not os.environ.get("IONQ_API_KEY"):
        print("ERROR: no IONQ_API_KEY", file=sys.stderr)
        return 2

    log_file = _REPO / "data" / "qpu_usage.jsonl"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    os.environ["QAL_QPU_USAGE_LOG"] = str(log_file)
    print(f"log: {log_file}")
    print(f"(existing records preserved; appending one submission)\n")

    backend = provider._get_ionq_backend("qpu.forte-1")
    if backend is None:
        print("ERROR: no qpu.forte-1 backend", file=sys.stderr)
        return 3
    wrapped = provider._wrap_run_for_usage_logging(backend)

    # Smallest meaningful job — Bell state, 100 shots.
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    print(f">>> circuit: 2-qubit Bell state, depth={qc.depth()}, "
          f"ops={dict(qc.count_ops())}, shots=100\n")

    print(">>> submitting (one attempt, no retry)")
    try:
        job = wrapped.run(qc, shots=100)
    except Exception as exc:
        print(f"SUBMISSION FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 4
    job_id = job.job_id()
    print(f"    job_id: {job_id}")
    print("    NOTE: job is queued; not canceled; will run when Forte-1 is back.\n")

    print(">>> waiting 8s for IonQ to validate the job")
    time.sleep(8)

    print(">>> update_cost (in-place merge of predicted fields)")
    diff = qpu_usage.update_cost(job_id)
    if diff is None:
        print("    update_cost returned None")
    else:
        print(json.dumps(diff, indent=2, default=str))
    print()

    print(">>> final record for this job in the log:")
    for r in qpu_usage.read_all():
        if r.get("job_id") == job_id:
            print(json.dumps(r, indent=2, default=str))

    return 0


if __name__ == "__main__":
    sys.exit(main())

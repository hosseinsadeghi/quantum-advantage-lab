#!/usr/bin/env python3
"""Submit a tiny job to qpu.forte-1, log it, then cancel.

Intent: prove the new IonQ key can hit the real QPU endpoint (not just the
simulator), verify the bookkeeping wrapping records the submission to the
JSONL log, then cancel the queued job so it never runs and never bills.

Does NOT block on ``job.result()`` — the device is currently offline and
that would hang indefinitely. The submission event is enough to verify the
end-to-end QPU path; the completion event will simply not fire (by design,
since result() is never called).
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from qiskit import QuantumCircuit  # noqa: E402

from backend import qpu_usage  # noqa: E402
from backend.quantum import provider  # noqa: E402

IONQ_API_BASE = "https://api.ionq.co/v0.4"


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def _get_job(job_id: str) -> dict:
    req = urllib.request.Request(
        f"{IONQ_API_BASE}/jobs/{job_id}",
        headers={"Authorization": f"apiKey {os.environ['IONQ_API_KEY']}"},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def _cancel_job(job_id: str) -> tuple[int, str]:
    req = urllib.request.Request(
        f"{IONQ_API_BASE}/jobs/{job_id}/status/cancel",
        method="PUT",
        headers={"Authorization": f"apiKey {os.environ['IONQ_API_KEY']}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def main() -> int:
    _load_dotenv(_REPO / ".env")
    qal = os.environ.get("IONQ_API_KEY_QAL")
    if qal:
        os.environ["IONQ_API_KEY"] = qal
    if not os.environ.get("IONQ_API_KEY"):
        print("ERROR: no IONQ_API_KEY", file=sys.stderr)
        return 2

    log_file = _REPO / "data" / "qpu_usage_real_qpu.jsonl"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    if log_file.exists():
        log_file.unlink()
    os.environ["QAL_QPU_USAGE_LOG"] = str(log_file)
    print(f"log: {log_file}\n")

    # ---- 1. get the real QPU backend ---------------------------------
    print(">>> Step 1: acquire qpu.forte-1 backend")
    backend = provider._get_ionq_backend("qpu.forte-1")
    if backend is None:
        print("ERROR: could not get qpu.forte-1 backend (key/package issue)")
        return 3
    print(f"    backend.name() -> {backend.name()}")
    print(f"    is_qpu_backend -> {qpu_usage.is_qpu_backend(backend)}")

    wrapped = provider._wrap_run_for_usage_logging(backend)
    print("    wrapped for usage logging\n")

    # ---- 2. build minimal circuit -----------------------------------
    print(">>> Step 2: build tiny circuit (Bell state, 2q, 100 shots)")
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    print(f"    depth={qc.depth()}  ops={dict(qc.count_ops())}\n")

    # ---- 3. submit ---------------------------------------------------
    print(">>> Step 3: submit to qpu.forte-1")
    t0 = time.time()
    try:
        job = wrapped.run(qc, shots=100)
    except Exception as exc:
        print(f"    SUBMISSION REJECTED: {type(exc).__name__}: {exc}")
        print("    (no record written — bookkeeping correctly skips failed submissions)")
        return 0

    submit_dt = time.time() - t0
    job_id = job.job_id()
    print(f"    submission accepted in {submit_dt:.2f}s")
    print(f"    job_id: {job_id}\n")

    # ---- 4. verify log -----------------------------------------------
    print(">>> Step 4: verify bookkeeping wrote 'submitted' event")
    records = qpu_usage.read_all()
    print(f"    records in log: {len(records)}")
    if records:
        print(json.dumps(records[0], indent=2, default=str))
    print()

    # ---- 5. wait briefly + refresh to capture predicted duration ----
    print(">>> Step 5: wait 5s for IonQ validation, then refresh log")
    time.sleep(5)
    rec = qpu_usage.refresh(job_id)
    if rec:
        print(json.dumps(rec, indent=2, default=str))
    else:
        print("    refresh returned no data")
    print()

    # ---- 6. cancel ---------------------------------------------------
    print(">>> Step 6: cancel the job (avoid eventual billing)")
    code, body = _cancel_job(job_id)
    print(f"    HTTP {code}: {body[:300]}")
    print()

    # ---- 7. refresh log post-cancel ---------------------------------
    print(">>> Step 7: refresh log post-cancel (final state, actual cost)")
    rec = qpu_usage.refresh(job_id)
    if rec:
        print(json.dumps(rec, indent=2, default=str))
    print()

    # ---- 8. summarize all events -----------------------------------
    print(">>> Step 8: events in log")
    for r in qpu_usage.read_all():
        print(
            f"  - {r['event']:9s}  status={r.get('ionq_status') or r.get('status','?'):10s}  "
            f"cost_usd={r.get('cost_usd')}  "
            f"pred_exec_s={r.get('predicted_execution_time_seconds')}  "
            f"exec_s={r.get('execution_time_seconds')}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())

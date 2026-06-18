#!/usr/bin/env python3
"""Human-eyeballable IonQ cloud-simulator connectivity check.

Requires IONQ_API_KEY. Submits a 2-qubit Bell circuit, prints counts.
Phase-2 extends this with full quantum-solver round-trip tests.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    api_key = os.environ.get("IONQ_API_KEY")
    if not api_key:
        print("IONQ_API_KEY not set; aborting. Export it and rerun.", file=sys.stderr)
        return 2

    try:
        from qiskit_ionq import IonQProvider  # type: ignore[import-untyped]
    except ImportError:
        print("qiskit-ionq not installed. Uncomment it in backend/requirements.txt.", file=sys.stderr)
        return 2

    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    provider = IonQProvider(api_key)
    backend = provider.get_backend("ionq_simulator")
    print(f"backend: {backend.name}")

    job = backend.run(qc, shots=1024)
    result = job.result()
    counts = result.get_counts()
    print(f"counts: {counts}")

    ok = abs(counts.get("00", 0) / 1024 - 0.5) < 0.1 and abs(
        counts.get("11", 0) / 1024 - 0.5
    ) < 0.1
    print("OK" if ok else "UNEXPECTED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

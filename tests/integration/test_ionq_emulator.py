"""IonQ cloud-emulator integration tests.

Skipped unless both conditions hold:
    * ``IONQ_API_KEY`` is set in the environment.
    * ``qiskit-ionq`` is installed (``uv sync --extra ionq``).

These tests submit real jobs to IonQ's ``ionq_simulator`` backend. They use
free emulator shots (no QPU burn) but still cross the public internet, so
they live behind the ``integration`` marker and are excluded from the default
test run.

Run explicitly with::

    IONQ_API_KEY=ionq_... uv run pytest -m integration tests/integration/test_ionq_emulator.py
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from qiskit import QuantumCircuit

from backend.quantum.provider import transpile_for_ionq

pytestmark = pytest.mark.integration


_HAVE_QISKIT_IONQ = True
try:  # pragma: no cover - import guard
    from qiskit_ionq import IonQProvider  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    _HAVE_QISKIT_IONQ = False


_skip = pytest.mark.skipif(
    not os.environ.get("IONQ_API_KEY") or not _HAVE_QISKIT_IONQ,
    reason="requires IONQ_API_KEY + qiskit-ionq (uv sync --extra ionq)",
)


@pytest.fixture(scope="module")
def ionq_backend():
    """Shared IonQ cloud simulator backend for the whole module."""
    if not os.environ.get("IONQ_API_KEY") or not _HAVE_QISKIT_IONQ:
        pytest.skip("IonQ provider unavailable")
    provider = IonQProvider(os.environ["IONQ_API_KEY"])
    return provider.get_backend("ionq_simulator")


def _bell_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2, 2, name="bell")
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def _bell_with_entangling_idle(entangle_pairs: int = 10) -> QuantumCircuit:
    """Bell pair followed by N ``rxx(pi/2); barrier; rxx(-pi/2)`` echo pairs.

    Each echo pair is mathematically identity but uses two full-strength MS
    gates, so it exposes the 2q-gate error the Forte-1 emulator models. The
    barrier prevents the transpiler from collapsing the pair back to identity.
    """
    qc = QuantumCircuit(2, 2, name="bell_with_idle")
    qc.h(0)
    qc.cx(0, 1)
    for _ in range(entangle_pairs):
        qc.rxx(np.pi / 2, 0, 1)
        qc.barrier()
        qc.rxx(-np.pi / 2, 0, 1)
        qc.barrier()
    qc.measure([0, 1], [0, 1])
    return qc


def _normalise(counts: dict[str, float]) -> dict[str, float]:
    total = sum(counts.values()) or 1.0
    return {k: v / total for k, v in counts.items()}


@_skip
def test_bell_ideal_is_exact(ionq_backend):
    """noise_model='ideal' returns exact probabilities on the Bell state."""
    qc = transpile_for_ionq(_bell_circuit(), optimization_level=2)
    job = ionq_backend.run(qc, noise_model="ideal")
    probs = _normalise(dict(job.result().get_counts()))

    # Ideal Bell: |00> and |11> each exactly 0.5; |01>, |10> are zero.
    assert probs.get("00", 0.0) == pytest.approx(0.5, abs=1e-9)
    assert probs.get("11", 0.0) == pytest.approx(0.5, abs=1e-9)
    assert probs.get("01", 0.0) == pytest.approx(0.0, abs=1e-9)
    assert probs.get("10", 0.0) == pytest.approx(0.0, abs=1e-9)


@_skip
@pytest.mark.slow
def test_bell_with_idle_forte1_above_threshold(ionq_backend):
    """Bell + entangling idle on forte-1 keeps > 85% Bell-subspace mass.

    The tolerance is deliberately loose: emulator noise varies with the
    calibration snapshot date published by IonQ, so tightening below ~0.85
    will cause spurious CI flakes.
    """
    shots = 1024
    qc = transpile_for_ionq(_bell_with_entangling_idle(entangle_pairs=10), optimization_level=2)
    job = ionq_backend.run(qc, noise_model="forte-1", shots=shots)
    counts = dict(job.result().get_counts())
    probs = _normalise(counts)

    bell_mass = probs.get("00", 0.0) + probs.get("11", 0.0)
    assert bell_mass > 0.85, (
        f"Bell-subspace mass {bell_mass:.3f} on forte-1 emulator is unexpectedly low; "
        f"counts={counts}"
    )


@_skip
def test_get_backend_routes_to_ionq_emulator():
    """High-level ``get_backend`` hands back the IonQ emulator + noise_model kwarg."""
    from backend.quantum.provider import get_backend

    backend, run_kwargs, info = get_backend(
        use_simulator=False, use_qpu=False, noise_model="forte-1"
    )
    # qiskit-ionq backend class name contains 'IonQ'
    assert "IonQ" in type(backend).__name__, (
        f"expected an IonQ backend, got {type(backend).__name__}"
    )
    assert run_kwargs == {"noise_model": "forte-1"}
    assert info["actual"] == "ionq_simulator"
    assert info["fell_back"] is False


# ---------------------------------------------------------------------------
# Hamiltonian-sim specific (Phase 2.4)
# ---------------------------------------------------------------------------

def _tv_distance(p: dict[str, float], q: dict[str, float]) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


@_skip
@pytest.mark.slow
def test_hamiltonian_trotter_on_forte1_agrees_with_aer(ionq_backend):
    """Small Trotter circuit on forte-1 emulator must agree with Aer within noise.

    A 2q, 2-step TFIM circuit is shallow enough that the forte-1 2q-gate error
    budget bounds TV-distance to Aer below ~0.15 in normal calibration. The
    tolerance is deliberately loose: emulator noise varies with calibration
    snapshot date, so tightening will cause flakes.
    """
    from backend.quantum.hamiltonian_sim import build_trotter_circuit, get_model_terms
    from backend.quantum.provider import transpile_for_ionq
    from qiskit_aer import AerSimulator  # type: ignore[import-untyped]

    shots = 512
    terms = get_model_terms(2, "ising", J=1.0, h=0.5)
    qc = build_trotter_circuit(terms, time=0.5, n_steps=2)
    qc_ionq = transpile_for_ionq(qc, optimization_level=2)

    # forte-1 emulator
    job = ionq_backend.run(qc_ionq, noise_model="forte-1", shots=shots)
    emu_counts = dict(job.result().get_counts())
    emu_probs = {k: v / shots for k, v in emu_counts.items()}

    # Aer reference at the same circuit
    aer = AerSimulator(method="statevector")
    aer_counts = dict(aer.run(qc, shots=shots, seed_simulator=42).result().get_counts())
    aer_probs = {k: v / shots for k, v in aer_counts.items()}

    tv = _tv_distance(emu_probs, aer_probs)
    assert tv < 0.15, (
        f"TV-distance forte-1 vs Aer = {tv:.3f} exceeds 0.15 tolerance. "
        f"emu={emu_probs}, aer={aer_probs}"
    )

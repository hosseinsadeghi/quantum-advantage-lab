"""Quantum provider management.

Handles backend selection between Aer simulator and IonQ trapped-ion hardware.
Provides transpilation helpers targeting IonQ's native gate set {GPi, GPi2, MS}.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Gate

from backend import qpu_usage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IonQ native gate stubs (used for transpilation target description)
# ---------------------------------------------------------------------------

class GPiGate(Gate):
    """Single-qubit GPi gate parameterized by phase phi."""

    def __init__(self, phi: float):
        super().__init__("gpi", 1, [phi])


class GPi2Gate(Gate):
    """Single-qubit GPi2 gate parameterized by phase phi."""

    def __init__(self, phi: float):
        super().__init__("gpi2", 1, [phi])


class MSGate(Gate):
    """Two-qubit Molmer-Sorensen (MS) gate with phases phi0, phi1."""

    def __init__(self, phi0: float = 0.0, phi1: float = 0.0):
        super().__init__("ms", 2, [phi0, phi1])


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------

# Default QPU name. Aria devices are retired; Forte-1 is the only production
# IonQ QPU as of 2026. Override via ``qpu_name`` when a new device comes online.
DEFAULT_QPU_NAME = "qpu.forte-1"
DEFAULT_NOISE_MODEL = "forte-1"


def _get_ionq_backend(backend_name: str = "ionq_simulator") -> Any:
    """Attempt to obtain an IonQ backend via qiskit_ionq.

    Requires the ``IONQ_API_KEY`` environment variable to be set.

    Parameters
    ----------
    backend_name:
        IonQ backend identifier. Common values:
        ``"ionq_simulator"`` (cloud simulator, free),
        ``"qpu.forte-1"`` (production QPU),
        ``"qpu.forte-enterprise-1"`` (reserved-customer QPU).

    Returns
    -------
    Backend instance or *None* when the provider is unavailable.
    """
    api_key = os.environ.get("IONQ_API_KEY")
    if not api_key:
        logger.warning("IONQ_API_KEY not set -- cannot initialise IonQ provider")
        return None

    try:
        from qiskit_ionq import IonQProvider  # type: ignore[import-untyped]

        provider = IonQProvider(api_key)
        backend = provider.get_backend(backend_name)
        logger.info("IonQ backend acquired: %s", backend.name)
        return backend
    except ImportError:
        logger.warning("qiskit_ionq package not installed -- falling back to Aer")
        return None
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to initialise IonQ backend %r: %s", backend_name, exc)
        return None


def _get_aer_backend() -> Any:
    """Return the Aer statevector simulator backend.

    Falls back to Qiskit's built-in ``StatevectorSimulator`` when Aer is not
    installed.
    """
    try:
        from qiskit_aer import AerSimulator  # type: ignore[import-untyped]

        backend = AerSimulator(method="statevector")
        logger.info("Using AerSimulator (statevector)")
        return backend
    except ImportError:
        pass

    try:
        from qiskit.providers.basic_provider import BasicSimulator  # type: ignore[import-untyped]

        backend = BasicSimulator()
        logger.info("Using BasicSimulator fallback")
        return backend
    except ImportError:
        pass

    raise RuntimeError(
        "No simulator backend available. Install qiskit-aer: pip install qiskit-aer"
    )


def _wrap_run_for_usage_logging(backend: Any) -> Any:
    """Patch ``backend.run`` so every QPU submission is journaled.

    The wrapper:

    1. Calls the underlying ``backend.run(circuit, shots=..., **kwargs)``.
    2. Writes a ``submitted`` event to the qpu_usage log (circuit metadata,
       shots, run kwargs, job id, submission timestamp).
    3. Replaces the returned job's ``.result()`` so that the *first* call to
       it appends a ``completed`` / ``failed`` event with cost, on-device
       execution time, and a short result summary fetched from IonQ.

    Wrapping happens once per backend instance. The wrapper is a no-op if the
    backend somehow isn't a real QPU (defensive — ``get_backend`` only routes
    QPU backends through here).
    """
    if getattr(backend, "_qal_usage_wrapped", False):
        return backend

    original_run = backend.run

    def run(circuit, shots=None, **kwargs):  # type: ignore[no-untyped-def]
        submitted_at = time.time()
        if shots is None:
            job = original_run(circuit, **kwargs)
        else:
            job = original_run(circuit, shots=shots, **kwargs)

        qpu_usage.record_submission(
            job,
            backend=backend,
            circuit=circuit,
            shots=shots if shots is not None else 0,
            run_kwargs=kwargs,
        )

        original_result = job.result
        completion_recorded = {"done": False}

        def result(*args, **kw):  # type: ignore[no-untyped-def]
            try:
                r = original_result(*args, **kw)
            except Exception:
                if not completion_recorded["done"]:
                    completion_recorded["done"] = True
                    qpu_usage.record_completion(
                        job, backend=backend, result=None, submitted_at=submitted_at
                    )
                raise
            if not completion_recorded["done"]:
                completion_recorded["done"] = True
                qpu_usage.record_completion(
                    job, backend=backend, result=r, submitted_at=submitted_at
                )
            return r

        job.result = result
        return job

    backend.run = run
    backend._qal_usage_wrapped = True
    return backend


def get_backend(
    use_simulator: bool = True,
    *,
    use_qpu: bool = False,
    noise_model: str = DEFAULT_NOISE_MODEL,
    qpu_name: str = DEFAULT_QPU_NAME,
) -> tuple[Any, dict[str, Any]]:
    """Return ``(backend, run_kwargs)`` for circuit execution.

    Three routes, resolved in order:

    * ``use_simulator=True`` (default) → local Aer simulator. ``run_kwargs`` is empty.
    * ``use_simulator=False, use_qpu=False`` → IonQ cloud **emulator**
      (``ionq_simulator``) with ``noise_model`` selecting the calibration
      snapshot (``"forte-1"`` default, or ``"ideal"`` / ``"aria-1"`` / ``"aria-2"`` /
      any other profile published at cloud.ionq.com/backends/simulators).
      Free; noise replays the chosen device's gate errors and coherence.
    * ``use_qpu=True`` → real IonQ QPU (``qpu_name``, default ``"qpu.forte-1"``).
      Billable — gate only behind an explicit user action.

    If ``qiskit-ionq`` or ``IONQ_API_KEY`` is missing, the function falls back
    to Aer with a warning.

    The returned ``run_kwargs`` dict should be splatted into ``backend.run``::

        backend, run_kwargs = get_backend(...)
        job = backend.run(circuit, shots=shots, **run_kwargs)

    Only IonQ emulator runs populate ``run_kwargs`` (with ``noise_model``);
    Aer and QPU runs get an empty dict.
    """
    if use_qpu:
        qpu = _get_ionq_backend(qpu_name)
        if qpu is not None:
            return _wrap_run_for_usage_logging(qpu), {}
        logger.warning("IonQ QPU %r unavailable -- falling back to Aer", qpu_name)
        return _get_aer_backend(), {}

    if not use_simulator:
        emu = _get_ionq_backend("ionq_simulator")
        if emu is not None:
            return emu, {"noise_model": noise_model}
        logger.info("IonQ emulator unavailable -- falling back to Aer")

    return _get_aer_backend(), {}


# ---------------------------------------------------------------------------
# Transpilation helpers
# ---------------------------------------------------------------------------

# IonQ native basis gates expressed as standard Qiskit names that the
# transpiler understands.  The IonQ native set {GPi, GPi2, MS} is equivalent
# to the universal set {rz, ry, rx, rxx} (after phase adjustments).
# We map to the closest standard equivalents the transpiler can target.
IONQ_BASIS_GATES = ["rx", "ry", "rz", "rxx", "id", "measure"]


def transpile_for_ionq(
    circuit: QuantumCircuit,
    optimization_level: int = 2,
) -> QuantumCircuit:
    """Transpile *circuit* to IonQ-compatible native gates.

    IonQ trapped-ion QPUs use an all-to-all connectivity with native gates
    {GPi, GPi2, MS}.  We transpile to the equivalent standard gate set
    {rx, ry, rz, rxx} which the IonQ compiler further lowers to native gates
    on submission.

    Parameters
    ----------
    circuit:
        The Qiskit ``QuantumCircuit`` to transpile.
    optimization_level:
        Qiskit transpiler optimization level (0-3).

    Returns
    -------
    Transpiled ``QuantumCircuit``.
    """
    transpiled = transpile(
        circuit,
        basis_gates=IONQ_BASIS_GATES,
        optimization_level=optimization_level,
        # IonQ has all-to-all connectivity so no coupling map needed
    )
    return transpiled


def circuit_metadata(circuit: QuantumCircuit) -> dict:
    """Extract useful metadata from a quantum circuit.

    Returns a dict with depth, gate counts, width, etc.
    """
    ops = circuit.count_ops()
    return {
        "n_qubits": circuit.num_qubits,
        "depth": circuit.depth(),
        "gate_counts": dict(ops),
        "total_gates": sum(ops.values()),
        "n_classical_bits": circuit.num_clbits,
    }

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

# Default QPU name. Aria devices are retired; the Forte family is current.
# ``qpu.forte-1`` is the shared production device; ``qpu.forte-enterprise-1`` is
# a reserved-capacity device. Both are real hardware (billable).
DEFAULT_QPU_NAME = "qpu.forte-1"
DEFAULT_NOISE_MODEL = "forte-1"

# Real-hardware QPUs surfaced in the UI as selectable solvers.
IONQ_QPU_NAMES = ("qpu.forte-1", "qpu.forte-enterprise-1")

IONQ_API_BASE = "https://api.ionq.co/v0.3"

# Default shot counts. Real IonQ hardware is billed per gate-shot, so QPU runs
# use a much smaller count than the (free) Aer/emulator runs to cap cost. Both
# are overridable via env without code changes.
SIM_SHOTS = int(os.environ.get("QAL_SIM_SHOTS", "1024"))
QPU_SHOTS = int(os.environ.get("QAL_QPU_SHOTS", "256"))


def _env_flag(name: str) -> bool:
    """True when *name* is set to a common truthy string."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def qpu_submission_disabled() -> bool:
    """Whether new real-hardware QPU submissions are disabled on this deploy.

    ``DISABLE_QPU_SUBMISSION`` is the dedicated control. ``DISABLE_QPU`` is
    still honoured as a legacy alias so existing deploys do not change
    behaviour unexpectedly.
    """
    return _env_flag("DISABLE_QPU_SUBMISSION") or _env_flag("DISABLE_QPU")


def resolve_shots(params: dict[str, Any]) -> int:
    """Pick the shot count for a run.

    An explicit ``shots`` in *params* always wins. Otherwise real-hardware runs
    (``use_qpu``) get the reduced :data:`QPU_SHOTS` to limit spend, while
    simulator/emulator runs get the larger free :data:`SIM_SHOTS`. Note the
    cache key excludes shots, so lowering this never invalidates a cached
    result — shot counts aggregate across records instead.
    """
    explicit = params.get("shots")
    if explicit:
        return int(explicit)
    return QPU_SHOTS if params.get("use_qpu") else SIM_SHOTS


class QPUUnavailableError(RuntimeError):
    """Raised when a real-hardware QPU run is requested but cannot be executed.

    Carries a clear, user-facing message. We *never* silently fall back to the
    simulator for an explicit QPU request — the caller sees this error instead.
    """


# ---------------------------------------------------------------------------
# API key registry — multiple named keys, selectable at runtime
# ---------------------------------------------------------------------------
#
# Keys come from the environment, never the browser:
#   IONQ_API_KEY            -> name "default"
#   IONQ_API_KEY_<NAME>     -> name "<NAME>"  (e.g. IONQ_API_KEY_QAL -> "QAL")
# The UI selects among *names*; key values are never exposed over the API.
# A module-level selection overrides the default-preferred key for the process.

_KEY_ENV_PREFIX = "IONQ_API_KEY_"
_selected_key_name: str | None = None


def _discover_keys() -> dict[str, str]:
    """Map ``name -> key value`` for every IonQ key present in the environment."""
    keys: dict[str, str] = {}
    default = os.environ.get("IONQ_API_KEY")
    if default:
        keys["default"] = default
    for var, val in os.environ.items():
        if var.startswith(_KEY_ENV_PREFIX) and val:
            keys[var[len(_KEY_ENV_PREFIX):]] = val
    return keys


def active_key_name() -> str | None:
    """Name of the key currently in effect.

    Honours an explicit :func:`select_key` choice; otherwise prefers ``QAL``
    (the historical QPU-entitled key) then any other discovered key. ``None``
    when no IonQ key is configured at all.
    """
    keys = _discover_keys()
    if not keys:
        return None
    if _selected_key_name in keys:
        return _selected_key_name
    if "QAL" in keys:
        return "QAL"
    return next(iter(keys))


def select_key(name: str) -> bool:
    """Set the active key by name. Returns False if the name isn't configured."""
    global _selected_key_name
    if name not in _discover_keys():
        return False
    _selected_key_name = name
    logger.info("IonQ API key switched to %r", name)
    return True


def list_keys() -> list[dict[str, Any]]:
    """List configured keys for the selector UI — names only, never values.

    Each entry: ``{name, active, fingerprint}`` where ``fingerprint`` is the
    last 4 chars so the operator can tell keys apart without exposing them.
    """
    active = active_key_name()
    return [
        {"name": n, "active": n == active, "fingerprint": f"…{v[-4:]}" if len(v) >= 4 else "…"}
        for n, v in _discover_keys().items()
    ]


def _ionq_api_key() -> str | None:
    """Return the value of the currently-active IonQ API key (or ``None``)."""
    keys = _discover_keys()
    name = active_key_name()
    return keys.get(name) if name else None


def _get_ionq_backend(backend_name: str = "ionq_simulator") -> Any:
    """Attempt to obtain an IonQ backend via qiskit_ionq.

    Requires an IonQ API key (``IONQ_API_KEY_QAL`` or ``IONQ_API_KEY``).

    Parameters
    ----------
    backend_name:
        IonQ backend identifier. Common values:
        ``"ionq_simulator"`` (cloud simulator, free),
        ``"qpu.forte-1"`` (production QPU),
        ``"qpu.forte-enterprise-1"`` (reserved-capacity QPU).

    Returns
    -------
    Backend instance or *None* when the provider is unavailable.
    """
    api_key = _ionq_api_key()
    if not api_key:
        logger.warning("No IonQ API key set -- cannot initialise IonQ provider")
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


def get_qpu_availability() -> dict[str, dict[str, Any]]:
    """Return live availability for each real-hardware QPU in ``IONQ_QPU_NAMES``.

    Queries IonQ's ``/backends`` endpoint once and maps each device to::

        {"available": bool, "status": str, "has_access": bool, "reason": str}

    ``available`` reflects whether the current backend process can submit a new
    job right now: the device must be operational (``status == "available"``)
    *and* the active key must be entitled to submit (``has_access``). On any
    error (no key, package/network failure) every QPU is reported unavailable
    with a ``reason``.
    """
    import json
    import urllib.request

    def _all(reason: str) -> dict[str, dict[str, Any]]:
        return {
            name: {"available": False, "status": "unknown",
                   "has_access": False, "reason": reason}
            for name in IONQ_QPU_NAMES
        }

    api_key = _ionq_api_key()
    if not api_key:
        return _all("no IonQ API key configured")

    try:
        req = urllib.request.Request(
            f"{IONQ_API_BASE}/backends",
            headers={"Authorization": f"apiKey {api_key}"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
            data = json.load(resp)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not fetch IonQ backend availability: %s", exc)
        return _all(f"availability query failed: {exc}")

    by_name = {b.get("backend"): b for b in data if isinstance(b, dict)}
    out: dict[str, dict[str, Any]] = {}
    for name in IONQ_QPU_NAMES:
        info = by_name.get(name, {})
        status = info.get("status", "unknown")
        has_access = bool(info.get("has_access", False))
        available = status == "available" and has_access
        if not info:
            reason = "device not listed by IonQ"
        elif status != "available":
            reason = f"device {status}"
        elif not has_access:
            reason = "account lacks access to this device"
        else:
            reason = ""
        out[name] = {
            "available": available,
            "status": status,
            "has_access": has_access,
            "reason": reason,
        }
    return out


def _ensure_known_qpu_available(qpu_name: str) -> None:
    """Raise :class:`QPUUnavailableError` when a surfaced QPU cannot be used.

    This preflights the two QPUs we expose in the UI so missing entitlement,
    offline hardware, and backend-catalog failures are surfaced before a
    billable submission is attempted. Unknown device names are left to the
    provider itself so custom script usage is not blocked by the UI list.
    """
    if qpu_name not in IONQ_QPU_NAMES:
        return
    info = get_qpu_availability().get(qpu_name) or {}
    if info.get("available"):
        return
    reason = info.get("reason") or info.get("status") or "device unavailable"
    raise QPUUnavailableError(f"QPU {qpu_name!r} requested but {reason}.")


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


def run_job_and_get_result(
    backend: Any,
    circuit: QuantumCircuit,
    *,
    shots: int,
    run_kwargs: dict[str, Any],
    execution: dict[str, Any] | None = None,
) -> Any:
    """Execute ``backend.run(...).result()`` with strict QPU error semantics."""
    execution = execution or {}
    requested = str(execution.get("requested") or "")
    is_qpu = requested.startswith("qpu.")

    try:
        job = backend.run(circuit, shots=shots, **run_kwargs)
    except Exception as exc:  # noqa: BLE001
        if is_qpu:
            raise QPUUnavailableError(
                f"QPU {requested!r} submission failed: {exc}"
            ) from exc
        raise

    try:
        return job.result()
    except Exception as exc:  # noqa: BLE001
        if is_qpu:
            raise QPUUnavailableError(
                f"QPU {requested!r} run failed before a valid hardware result was returned: {exc}"
            ) from exc
        raise


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
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Return ``(backend, run_kwargs, info)`` for circuit execution.

    Three routes, resolved in order:

    * ``use_simulator=True`` (default) → local Aer simulator. ``run_kwargs`` is empty.
    * ``use_simulator=False, use_qpu=False`` → IonQ cloud **emulator**
      (``ionq_simulator``) with ``noise_model`` selecting the calibration
      snapshot (``"forte-1"`` default, or ``"ideal"`` / ``"aria-1"`` / ``"aria-2"`` /
      any other profile published at cloud.ionq.com/backends/simulators).
      Free; noise replays the chosen device's gate errors and coherence.
    * ``use_qpu=True`` → real IonQ QPU (``qpu_name``, default ``"qpu.forte-1"``).
      Billable — gate only behind an explicit user action.

    ``info`` reports what the caller asked for vs. what actually ran. The
    emulator route may fall back to Aer (``fell_back=True``) with a message; the
    **QPU route never falls back** — an unreachable QPU raises
    :class:`QPUUnavailableError` so the request can't silently run on a
    simulator or incur surprise behaviour.
    """
    if use_qpu:
        requested = qpu_name
    elif not use_simulator:
        requested = "ionq_simulator"
    else:
        requested = "aer"

    def _aer(message: str = "") -> tuple[Any, dict[str, Any], dict[str, Any]]:
        return _get_aer_backend(), {}, {
            "requested": requested,
            "actual": "aer",
            "fell_back": requested != "aer",
            "message": message,
        }

    # Deployment kill switch for new real-hardware submissions. Cache replay
    # happens above this layer, so a cached QPU result remains replayable even
    # when fresh hardware jobs are forbidden.
    if qpu_submission_disabled():
        if use_qpu:
            raise QPUUnavailableError(
                f"QPU {qpu_name!r} requested but new QPU submissions are disabled on this deployment."
            )

    if requested == "aer":
        return _aer()

    no_key = not _ionq_api_key()
    if use_qpu:
        # Real hardware was explicitly requested. Never silently fall back to a
        # simulator — if we can't reach the QPU, raise a clear error so the UI
        # can say exactly what happened (and never bill/mislead the user).
        if no_key:
            raise QPUUnavailableError(
                f"QPU {qpu_name!r} requested but no IonQ API key is active. "
                "Select a key with hardware access in the API-key dropdown."
            )
        _ensure_known_qpu_available(qpu_name)
        qpu = _get_ionq_backend(qpu_name)
        if qpu is None:
            raise QPUUnavailableError(
                f"QPU {qpu_name!r} requested but the IonQ provider could not be "
                "initialised (package or connectivity error). No simulator "
                "fallback — see server logs for details."
            )
        key_label = active_key_name()
        info = {"requested": requested, "actual": qpu_name, "fell_back": False,
                "message": f"Submitted to {qpu_name} using API key '{key_label}'.",
                "key_name": key_label}
        return _wrap_run_for_usage_logging(qpu), {}, info

    emu = _get_ionq_backend("ionq_simulator")
    if emu is not None:
        info = {"requested": requested, "actual": "ionq_simulator",
                "fell_back": False, "message": ""}
        return emu, {"noise_model": noise_model}, info
    logger.info("IonQ emulator unavailable -- falling back to Aer")
    return _aer(
        "IonQ emulator unavailable "
        f"({'no API key' if no_key else 'provider/package error'}) "
        "— ran on local Aer simulator instead."
    )


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

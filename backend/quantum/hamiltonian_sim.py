"""Hamiltonian Simulation via Trotterization.

First-order product-formula (Lie-Trotter) decomposition for Ising and
Heisenberg spin-chain models, with per-step fidelity tracking.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, Statevector, state_fidelity

from backend.quantum.provider import (
    circuit_metadata,
    get_backend,
    transpile_for_ionq,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model Hamiltonians (returned as list of Pauli terms)
# ---------------------------------------------------------------------------

# Each term is (coefficient, pauli_string) where pauli_string uses "I","X","Y","Z"
PauliTerm = tuple[float, str]


def _ising_terms(n_qubits: int, J: float = 1.0, h: float = 0.5) -> list[PauliTerm]:
    """Transverse-field Ising model on a 1D chain.

    H = -J * sum_{i} Z_i Z_{i+1}  -  h * sum_{i} X_i
    """
    terms: list[PauliTerm] = []

    # ZZ interactions (nearest-neighbour)
    for i in range(n_qubits - 1):
        pauli = ["I"] * n_qubits
        pauli[i] = "Z"
        pauli[i + 1] = "Z"
        terms.append((-J, "".join(pauli)))

    # Transverse field
    for i in range(n_qubits):
        pauli = ["I"] * n_qubits
        pauli[i] = "X"
        terms.append((-h, "".join(pauli)))

    return terms


def _heisenberg_terms(
    n_qubits: int, Jx: float = 1.0, Jy: float = 1.0, Jz: float = 1.0
) -> list[PauliTerm]:
    """Heisenberg XXX/XXZ model on a 1D chain.

    H = sum_{i} [ Jx X_i X_{i+1} + Jy Y_i Y_{i+1} + Jz Z_i Z_{i+1} ]
    """
    terms: list[PauliTerm] = []

    for i in range(n_qubits - 1):
        for pauli_char, coupling in [("X", Jx), ("Y", Jy), ("Z", Jz)]:
            pauli = ["I"] * n_qubits
            pauli[i] = pauli_char
            pauli[i + 1] = pauli_char
            terms.append((coupling, "".join(pauli)))

    return terms


def get_model_terms(
    n_qubits: int,
    model: str = "ising",
    **kwargs: float,
) -> list[PauliTerm]:
    """Return Hamiltonian terms for the specified model.

    Parameters
    ----------
    n_qubits:
        Number of spins.
    model:
        ``"ising"`` or ``"heisenberg"``.
    **kwargs:
        Coupling constants forwarded to the model builder.
    """
    model = model.lower()
    if model == "ising":
        return _ising_terms(n_qubits, **kwargs)
    elif model == "heisenberg":
        return _heisenberg_terms(n_qubits, **kwargs)
    else:
        raise ValueError(f"Unknown model {model!r}. Supported: ising, heisenberg")


# ---------------------------------------------------------------------------
# Single Pauli-term exponentiation
# ---------------------------------------------------------------------------

def _apply_pauli_rotation(
    circuit: QuantumCircuit,
    coeff: float,
    pauli_str: str,
    dt: float,
) -> None:
    """Append exp(-i * coeff * dt * P) for a single Pauli string P.

    Uses the standard CNOT-staircase decomposition:
    1. Basis-change rotations to map X->Z, Y->Z
    2. CNOT cascade to compute parity in an ancilla-free manner
    3. Rz rotation on the last active qubit
    4. Undo CNOT cascade and basis change
    """
    n = len(pauli_str)
    active_qubits: list[int] = []
    bases: list[str] = []

    for i, p in enumerate(pauli_str):
        if p != "I":
            active_qubits.append(i)
            bases.append(p)

    if not active_qubits:
        # Pure identity -- global phase, skip
        return

    angle = 2 * coeff * dt  # exp(-i * theta/2 * Z) uses Rz(theta)

    # Single-qubit Pauli: direct rotation
    if len(active_qubits) == 1:
        q = active_qubits[0]
        b = bases[0]
        if b == "X":
            circuit.rx(angle, q)
        elif b == "Y":
            circuit.ry(angle, q)
        elif b == "Z":
            circuit.rz(angle, q)
        return

    # Multi-qubit: basis change
    for q, b in zip(active_qubits, bases):
        if b == "X":
            circuit.h(q)
        elif b == "Y":
            circuit.rx(np.pi / 2, q)
            # Sdg . H also works, but Rx(pi/2) is cleaner

    # CNOT staircase
    for i in range(len(active_qubits) - 1):
        circuit.cx(active_qubits[i], active_qubits[i + 1])

    # Rz on last active qubit
    circuit.rz(angle, active_qubits[-1])

    # Undo CNOT staircase
    for i in range(len(active_qubits) - 2, -1, -1):
        circuit.cx(active_qubits[i], active_qubits[i + 1])

    # Undo basis change
    for q, b in zip(active_qubits, bases):
        if b == "X":
            circuit.h(q)
        elif b == "Y":
            circuit.rx(-np.pi / 2, q)


# ---------------------------------------------------------------------------
# Trotter circuit builder
# ---------------------------------------------------------------------------

def build_trotter_circuit(
    hamiltonian_terms: list[PauliTerm],
    time: float,
    n_steps: int,
) -> QuantumCircuit:
    """Build a first-order Trotter circuit for exp(-i H t).

    Parameters
    ----------
    hamiltonian_terms:
        List of ``(coefficient, pauli_string)`` tuples.
    time:
        Total evolution time.
    n_steps:
        Number of Trotter steps (higher = more accurate).

    Returns
    -------
    ``QuantumCircuit`` implementing the approximate time evolution.
    """
    if not hamiltonian_terms:
        raise ValueError("hamiltonian_terms must be non-empty")

    n_qubits = len(hamiltonian_terms[0][1])
    dt = time / n_steps

    qc = QuantumCircuit(n_qubits, n_qubits, name="Trotter")

    for step in range(n_steps):
        for coeff, pauli_str in hamiltonian_terms:
            _apply_pauli_rotation(qc, coeff, pauli_str, dt)
        qc.barrier()

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


# ---------------------------------------------------------------------------
# Exact evolution (for fidelity reference)
# ---------------------------------------------------------------------------

def _hamiltonian_matrix(
    hamiltonian_terms: list[PauliTerm], n_qubits: int
) -> np.ndarray:
    """Return the dense Hamiltonian matrix built from Pauli terms.

    Cached so per-step fidelity loops don't rebuild it every iteration.
    """
    from qiskit.quantum_info import SparsePauliOp

    labels = [p for _, p in hamiltonian_terms]
    coeffs = [c for c, _ in hamiltonian_terms]
    del n_qubits  # inferred from label length by SparsePauliOp
    return SparsePauliOp.from_list(list(zip(labels, coeffs))).to_matrix()


def _exact_evolution(
    hamiltonian_terms: list[PauliTerm],
    n_qubits: int,
    time: float,
    initial_state: Statevector,
    *,
    h_matrix: np.ndarray | None = None,
) -> Statevector:
    """Compute exact time evolution via matrix exponentiation.

    ``h_matrix``: optional precomputed Hamiltonian matrix. Pass this when
    calling in a tight per-step loop to skip the ``SparsePauliOp.to_matrix``
    rebuild on every iteration.
    """
    from scipy.linalg import expm  # type: ignore[import-untyped]

    H = _hamiltonian_matrix(hamiltonian_terms, n_qubits) if h_matrix is None else h_matrix
    U = expm(-1j * H * time)
    return Statevector(U @ initial_state.data)


# ---------------------------------------------------------------------------
# Runner with per-step fidelity
# ---------------------------------------------------------------------------

def run_simulation(
    n_qubits: int,
    model: str = "ising",
    time: float = 1.0,
    n_steps: int = 10,
    use_simulator: bool = True,
    use_qpu: bool = False,
    noise_model: str = "forte-1",
    qpu_name: str = "qpu.forte-1",
    shots: int = 1024,
    initial_state: str | None = None,
    seed_simulator: int | None = None,
    **model_kwargs: float,
) -> dict[str, Any]:
    """Run Hamiltonian simulation and return per-Trotter-step fidelity.

    Parameters
    ----------
    n_qubits:
        Number of qubits / spins.
    model:
        ``"ising"`` or ``"heisenberg"``.
    time:
        Total simulation time.
    n_steps:
        Number of Trotter steps.
    use_simulator:
        Use local Aer simulator when True.
    shots:
        Number of measurement shots for the final circuit.
    initial_state:
        Bitstring for the initial computational-basis state (e.g. ``"1010"``).
        Defaults to ``"0...0"``.
    **model_kwargs:
        Coupling constants forwarded to the model.

    Returns
    -------
    dict with keys:
        steps : list[dict]
            Fidelity with exact evolution and state probabilities per Trotter step.
        final_result : dict
            Final fidelity, measured counts.
        metadata : dict
            Circuit information.
    """
    terms = get_model_terms(n_qubits, model, **model_kwargs)

    if initial_state is None:
        initial_state = "0" * n_qubits
    if len(initial_state) != n_qubits:
        raise ValueError("initial_state length must match n_qubits")

    sv_init = Statevector.from_label(initial_state)
    dt = time / n_steps

    steps: list[dict[str, Any]] = []

    # Track Trotter-evolved state step by step
    sv_trotter = sv_init.copy()

    # Build one Trotter step as a circuit (no measurements)
    step_qc = QuantumCircuit(n_qubits)

    # Prepare initial state
    init_qc = QuantumCircuit(n_qubits)
    for i, bit in enumerate(reversed(initial_state)):
        if bit == "1":
            init_qc.x(i)

    for coeff, pauli_str in terms:
        _apply_pauli_rotation(step_qc, coeff, pauli_str, dt)

    # Build the Hamiltonian matrix once; per-step _exact_evolution reuses it.
    h_matrix = _hamiltonian_matrix(terms, n_qubits)

    for step_idx in range(1, n_steps + 1):
        sv_trotter = sv_trotter.evolve(step_qc)

        # Exact evolution up to this time for fidelity comparison
        t_current = step_idx * dt
        sv_exact = _exact_evolution(
            terms, n_qubits, t_current, sv_init, h_matrix=h_matrix
        )

        fidelity = float(state_fidelity(sv_trotter, sv_exact))
        probs = {
            format(i, f"0{n_qubits}b"): float(p)
            for i, p in enumerate(np.abs(sv_trotter.data) ** 2)
            if p > 1e-10
        }

        steps.append({
            "step": step_idx,
            "time": t_current,
            "fidelity_vs_exact": fidelity,
            "state_probabilities": probs,
            "description": f"Trotter step {step_idx}, t={t_current:.4f}",
        })

    # Build and run the full circuit with measurements
    full_circuit = QuantumCircuit(n_qubits, n_qubits)

    # Prepare initial state
    for i, bit in enumerate(reversed(initial_state)):
        if bit == "1":
            full_circuit.x(i)

    # Append full Trotter evolution
    trotter_body = build_trotter_circuit(terms, time, n_steps)
    # Remove measurements from trotter_body (we add our own)
    trotter_no_meas = trotter_body.remove_final_measurements(inplace=False)
    full_circuit.compose(trotter_no_meas, inplace=True)
    full_circuit.measure(range(n_qubits), range(n_qubits))

    if not use_simulator:
        full_circuit = transpile_for_ionq(full_circuit)

    backend, run_kwargs, execution = get_backend(
        use_simulator=use_simulator,
        use_qpu=use_qpu,
        noise_model=noise_model,
        qpu_name=qpu_name,
    )
    if seed_simulator is not None and use_simulator:
        run_kwargs = {**run_kwargs, "seed_simulator": seed_simulator}
    transpiled = transpile(full_circuit, backend=backend)
    job = backend.run(transpiled, shots=shots, **run_kwargs)
    result = job.result()
    counts = {str(k): int(v) for k, v in result.get_counts().items()}

    final_fidelity = steps[-1]["fidelity_vs_exact"] if steps else 1.0

    # The t^2/(2r) shape matches the first-order Trotter error *scaling*; the
    # prefactor is not the tight Childs-Su bound. Keep both keys for
    # backward-compat with any consumer that reads the old name.
    scaling_reference = time**2 / (2 * n_steps)

    return {
        "steps": steps,
        "final_result": {
            "execution": execution,
            "measured_counts": counts,
            "final_fidelity": final_fidelity,
            "total_time": time,
            "n_trotter_steps": n_steps,
            "trotter_error_scaling_reference": scaling_reference,
            "trotter_error_bound": scaling_reference,
            "model": model,
            "initial_state": initial_state,
        },
        "metadata": {
            **circuit_metadata(full_circuit),
            "algorithm": "hamiltonian_simulation",
            "model": model,
            "n_qubits": n_qubits,
            "time": time,
            "n_trotter_steps": n_steps,
            "dt": dt,
            "shots": shots,
        },
    }

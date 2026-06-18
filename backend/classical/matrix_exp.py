"""Classical Matrix Exponentiation for Hamiltonian Simulation.

Computes exact time evolution via scipy.linalg.expm, serving as the
classical counterpart to Trotterized quantum simulation.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy.linalg import expm  # type: ignore[import-untyped]

from backend.quantum.hamiltonian_sim import get_model_terms


# ---------------------------------------------------------------------------
# Hamiltonian construction (mirrors backend.quantum.hamiltonian_sim)
# ---------------------------------------------------------------------------

_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _pauli_string_matrix(pauli_str: str) -> np.ndarray:
    """Compute the full matrix for a multi-qubit Pauli string."""
    result = _PAULI[pauli_str[0]]
    for char in pauli_str[1:]:
        result = np.kron(result, _PAULI[char])
    return result


def _build_hamiltonian(
    n_qubits: int,
    model: str = "ising",
    **kwargs: float,
) -> np.ndarray:
    """Build the Hamiltonian matrix for the specified model."""
    dim = 2 ** n_qubits
    H = np.zeros((dim, dim), dtype=complex)
    for coeff, pauli_str in get_model_terms(n_qubits, model, **kwargs):
        H += coeff * _pauli_string_matrix(pauli_str)
    return H


# ---------------------------------------------------------------------------
# Classical simulation runner
# ---------------------------------------------------------------------------

def run_classical_simulation(
    n_qubits: int,
    model: str = "ising",
    time_total: float = 1.0,
    n_steps: int = 10,
    initial_state: str | None = None,
    **model_kwargs: float,
) -> dict[str, Any]:
    """Simulate Hamiltonian time evolution via classical matrix exponentiation.

    This is the classical counterpart to Trotterized quantum simulation.
    It computes the exact unitary U = exp(-iHt) at each time step using
    ``scipy.linalg.expm`` and tracks the computational cost.

    Parameters
    ----------
    n_qubits:
        Number of qubits / spins.
    model:
        ``"ising"`` or ``"heisenberg"``.
    time_total:
        Total evolution time.
    n_steps:
        Number of time steps at which to record the state.
    initial_state:
        Bitstring for the initial computational-basis state (e.g. ``"1010"``).
        Defaults to ``"0...0"``.
    **model_kwargs:
        Coupling constants forwarded to the model builder.

    Returns
    -------
    dict with keys:
        steps : list[dict]
            State probabilities and timing at each time step.
        final_result : dict
            Final state information.
        metadata : dict
            Timing, matrix size, and computational cost estimates.
    """
    dim = 2 ** n_qubits
    matrix_size = dim

    if initial_state is None:
        initial_state = "0" * n_qubits
    if len(initial_state) != n_qubits:
        raise ValueError("initial_state length must match n_qubits")

    # Build initial state vector
    init_index = int(initial_state, 2)
    psi_init = np.zeros(dim, dtype=complex)
    psi_init[init_index] = 1.0

    wall_start = time.perf_counter()

    # Build Hamiltonian
    ham_start = time.perf_counter()
    H = _build_hamiltonian(n_qubits, model, **model_kwargs)
    ham_end = time.perf_counter()
    ham_build_ms = (ham_end - ham_start) * 1000

    dt = time_total / n_steps

    steps: list[dict[str, Any]] = []

    # FLOPs estimate: matrix exponentiation of dim x dim matrix is roughly
    # O(dim^3) for the Pade approximation used by scipy
    flops_per_expm = dim ** 3
    # Matrix-vector multiply is O(dim^2)
    flops_per_matvec = dim ** 2

    psi = psi_init.copy()

    for step_idx in range(1, n_steps + 1):
        step_start = time.perf_counter()

        t_current = step_idx * dt

        # Compute U = exp(-i H dt) and evolve
        U = expm(-1j * H * dt)
        psi = U @ psi

        # Normalize to guard against floating-point drift
        norm = np.linalg.norm(psi)
        if norm > 1e-15:
            psi /= norm

        # State probabilities
        probs = (np.abs(psi) ** 2).tolist()

        step_end = time.perf_counter()
        step_wall_ms = (step_end - step_start) * 1000

        # Filter to significant probabilities for readability
        state_probs = {
            format(i, f"0{n_qubits}b"): p
            for i, p in enumerate(probs)
            if p > 1e-10
        }

        steps.append({
            "step": step_idx,
            "time": t_current,
            "state_probs": state_probs,
            "wall_time_ms": step_wall_ms,
            "matrix_size": matrix_size,
            "flops_estimate": flops_per_expm + flops_per_matvec,
            "description": f"Classical expm step {step_idx}, t={t_current:.4f}",
        })

    wall_end = time.perf_counter()
    total_wall_ms = (wall_end - wall_start) * 1000

    # Final distribution
    final_probs = steps[-1]["state_probs"] if steps else {}

    return {
        "steps": steps,
        "final_result": {
            "final_state_probs": final_probs,
            "total_time": time_total,
            "n_steps": n_steps,
            "model": model,
            "initial_state": initial_state,
            "matrix_dimension": dim,
        },
        "metadata": {
            "algorithm": "classical_matrix_exponentiation",
            "model": model,
            "n_qubits": n_qubits,
            "matrix_dimension": dim,
            "time": time_total,
            "n_steps": n_steps,
            "dt": dt,
            "total_wall_time_ms": total_wall_ms,
            "hamiltonian_build_ms": ham_build_ms,
            "total_flops_estimate": n_steps * (flops_per_expm + flops_per_matvec),
            "complexity": f"O(n_steps * {dim}^3) -- exponential in n_qubits",
            "memory_bytes_estimate": dim * dim * 16,  # complex128
        },
    }

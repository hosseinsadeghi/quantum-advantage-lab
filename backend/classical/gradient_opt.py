"""Classical Gradient-Based Molecular Optimization.

Classical counterpart to VQE: constructs the same Hamiltonian matrix and
finds the ground-state energy using Nelder-Mead optimization over a
parameterized trial wavefunction.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy.optimize import minimize  # type: ignore[import-untyped]


# ---------------------------------------------------------------------------
# Hamiltonian construction (same matrices as the quantum VQE module)
# ---------------------------------------------------------------------------

def _pauli_matrix(label: str) -> np.ndarray:
    """Return the matrix for a single-character Pauli label."""
    matrices = {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }
    return matrices[label]


def _pauli_string_matrix(pauli_str: str) -> np.ndarray:
    """Compute the full matrix for a multi-qubit Pauli string via Kronecker products."""
    result = _pauli_matrix(pauli_str[0])
    for char in pauli_str[1:]:
        result = np.kron(result, _pauli_matrix(char))
    return result


def _build_hamiltonian_matrix(molecule: str) -> np.ndarray:
    """Build the Hamiltonian matrix for the specified molecule.

    Uses the same coefficients as ``backend.quantum.vqe``.
    """
    molecule = molecule.upper()

    if molecule == "H2":
        terms = [
            ("II", -1.0523732),
            ("IZ",  0.3979374),
            ("ZI", -0.3979374),
            ("ZZ", -0.0112801),
            ("XX",  0.1809312),
            ("YY",  0.1809312),
        ]
    elif molecule == "LIH":
        terms = [
            ("IIII", -7.4983),
            ("IIIZ",  0.2252),
            ("IIZI", -0.2252),
            ("IZII",  0.1722),
            ("ZIII", -0.1722),
            ("IIZZ",  0.1209),
            ("IZIZ",  0.0455),
            ("ZIZI",  0.0455),
            ("ZZII",  0.1209),
            ("IIXX",  0.0454),
            ("IIYY",  0.0454),
            ("XXII",  0.0454),
            ("YYII",  0.0454),
            ("IZZI",  0.0657),
            ("ZIIZ",  0.0657),
            ("ZZZZ",  0.0089),
        ]
    else:
        raise ValueError(
            f"Unknown molecule {molecule!r}. Supported: H2, LiH"
        )

    n_qubits = len(terms[0][0])
    dim = 2 ** n_qubits
    H = np.zeros((dim, dim), dtype=complex)
    for pauli_str, coeff in terms:
        H += coeff * _pauli_string_matrix(pauli_str)

    return H


# ---------------------------------------------------------------------------
# Trial wavefunction
# ---------------------------------------------------------------------------

def _trial_wavefunction(params: np.ndarray, dim: int) -> np.ndarray:
    """Construct a normalized trial wavefunction from parameters.

    Uses a parameterized unitary acting on |0...0> built from Givens
    rotations.  The number of parameters equals ``dim - 1`` (angles that
    parameterize the unit sphere in C^dim via real-valued rotations).
    """
    # Start from |0>
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0

    # Apply a sequence of Givens-like rotations to explore the Hilbert space
    n_params = len(params)
    for k in range(min(n_params, dim - 1)):
        i = k
        j = k + 1
        theta = params[k]
        c, s = np.cos(theta), np.sin(theta)
        new_i = c * state[i] - s * state[j]
        new_j = s * state[i] + c * state[j]
        state[i] = new_i
        state[j] = new_j

    # Normalize (should already be normalized, but guard against numerics)
    norm = np.linalg.norm(state)
    if norm > 1e-15:
        state /= norm

    return state


# ---------------------------------------------------------------------------
# Classical optimization runner
# ---------------------------------------------------------------------------

def run_classical_optimization(
    molecule: str = "H2",
    max_iterations: int = 200,
    initial_params: np.ndarray | None = None,
) -> dict[str, Any]:
    """Find the molecular ground-state energy using classical Nelder-Mead optimization.

    This is the classical counterpart to VQE.  It constructs the same
    Hamiltonian matrix and optimizes a parameterized trial wavefunction
    to minimize the energy expectation value.

    Parameters
    ----------
    molecule:
        ``"H2"`` (2 qubits, 4x4 matrix) or ``"LiH"`` (4 qubits, 16x16 matrix).
    max_iterations:
        Maximum number of optimizer iterations.
    initial_params:
        Initial parameter vector.  If *None*, random initialization is used.

    Returns
    -------
    dict with keys:
        steps : list[dict]
            Energy and parameter snapshot per optimizer iteration.
        final_result : dict
            Optimal energy, parameters, convergence status.
        metadata : dict
            Timing and algorithmic information.
    """
    H = _build_hamiltonian_matrix(molecule)
    dim = H.shape[0]
    n_qubits = int(np.log2(dim))
    n_params = dim - 1  # Givens rotation parameters

    if initial_params is None:
        rng = np.random.default_rng(42)
        initial_params = rng.uniform(-np.pi, np.pi, size=n_params)

    # Known exact ground-state energies for reference
    exact_energies = {"H2": -1.8572750, "LIH": -7.8825378}
    exact_gs = exact_energies.get(molecule.upper())

    energy_history: list[dict[str, Any]] = []
    iteration_counter = [0]
    best_energy = [float("inf")]

    wall_start = time.perf_counter()

    def objective(params: np.ndarray) -> float:
        iter_start = time.perf_counter()

        psi = _trial_wavefunction(params, dim)
        energy = float(np.real(psi.conj() @ H @ psi))

        iter_end = time.perf_counter()
        iteration_counter[0] += 1

        if energy < best_energy[0]:
            best_energy[0] = energy

        converged = False
        if exact_gs is not None:
            converged = abs(energy - exact_gs) < 1.6e-3  # chemical accuracy

        energy_history.append({
            "iteration": iteration_counter[0],
            "energy": energy,
            "params": params.tolist(),
            "converged": converged,
            "wall_time_ms": (iter_end - iter_start) * 1000,
        })

        return energy

    opt_result = minimize(
        objective,
        initial_params,
        method="Nelder-Mead",
        options={
            "maxiter": max_iterations,
            "xatol": 1e-6,
            "fatol": 1e-8,
            "adaptive": True,
        },
    )

    wall_end = time.perf_counter()
    total_wall_ms = (wall_end - wall_start) * 1000

    optimal_energy = float(opt_result.fun)
    optimal_params = opt_result.x.tolist()

    return {
        "steps": energy_history,
        "final_result": {
            "optimal_energy": optimal_energy,
            "optimal_params": optimal_params,
            "molecule": molecule,
            "n_iterations": iteration_counter[0],
            "converged": opt_result.success,
            "exact_ground_state_energy": exact_gs,
            "chemical_accuracy": (
                abs(optimal_energy - exact_gs) < 1.6e-3 if exact_gs else None
            ),
        },
        "metadata": {
            "algorithm": "classical_nelder_mead",
            "molecule": molecule,
            "n_qubits": n_qubits,
            "matrix_dimension": dim,
            "n_params": n_params,
            "optimizer": "Nelder-Mead",
            "max_iterations": max_iterations,
            "total_wall_time_ms": total_wall_ms,
            "complexity": f"O(iterations * {dim}^2) for {dim}x{dim} matrix",
        },
    }

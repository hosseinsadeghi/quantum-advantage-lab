"""Variational Quantum Eigensolver (VQE).

Hardware-efficient ansatz designed for IonQ's all-to-all trapped-ion
connectivity, with COBYLA optimisation and per-iteration energy tracking.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector

from backend.quantum.provider import (
    circuit_metadata,
    get_backend,
    run_job_and_get_result,
    transpile_for_ionq,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hamiltonian construction
# ---------------------------------------------------------------------------

_HAMILTONIANS: dict[str, dict[str, Any]] = {}


def _h2_hamiltonian() -> SparsePauliOp:
    """Minimal 2-qubit qubit-mapped H2 Hamiltonian (STO-3G, BK transform).

    H = g0*II + g1*ZI + g2*IZ + g3*ZZ + g4*XX + g5*YY

    Coefficients from Kandala et al., Nature 549, 242 (2017).
    """
    coeffs = {
        "II": -1.0523732,
        "IZ":  0.3979374,
        "ZI": -0.3979374,
        "ZZ": -0.0112801,
        "XX":  0.1809312,
        "YY":  0.1809312,
    }
    labels = list(coeffs.keys())
    values = [coeffs[l] for l in labels]
    return SparsePauliOp.from_list(list(zip(labels, values)))


def _lih_hamiltonian() -> SparsePauliOp:
    """Simplified 4-qubit LiH Hamiltonian (leading terms only).

    A reduced representation keeping the dominant Pauli terms from the
    qubit-mapped molecular Hamiltonian.
    """
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
    return SparsePauliOp.from_list(terms)


def build_hamiltonian(molecule: str = "H2") -> SparsePauliOp:
    """Return a qubit Hamiltonian for the requested molecule.

    Parameters
    ----------
    molecule:
        ``"H2"`` (2 qubits) or ``"LiH"`` (4 qubits).
    """
    molecule = molecule.upper()
    builders = {
        "H2": _h2_hamiltonian,
        "LIH": _lih_hamiltonian,
    }
    if molecule not in builders:
        raise ValueError(
            f"Unknown molecule {molecule!r}. Supported: {list(builders)}"
        )
    return builders[molecule]()


# ---------------------------------------------------------------------------
# Hardware-efficient ansatz
# ---------------------------------------------------------------------------

def build_vqe_ansatz(
    n_qubits: int,
    depth: int,
    params: np.ndarray | None = None,
) -> QuantumCircuit:
    """Build a hardware-efficient ansatz exploiting all-to-all connectivity.

    Each layer consists of:
    1. Ry-Rz rotations on every qubit (2 * n_qubits params per layer)
    2. All-to-all entangling via a ladder of RXX gates (native on IonQ)

    Parameters
    ----------
    n_qubits:
        Number of qubits.
    depth:
        Number of variational layers.
    params:
        Flat array of variational parameters.  Expected length:
        ``depth * 2 * n_qubits + 2 * n_qubits`` (final rotation layer).
        If *None*, parameters are initialised to zero.

    Returns
    -------
    Parameterized ``QuantumCircuit``.
    """
    n_params_per_layer = 2 * n_qubits
    total_params = depth * n_params_per_layer + n_params_per_layer  # +final rotations

    if params is None:
        params = np.zeros(total_params)
    if len(params) != total_params:
        raise ValueError(
            f"Expected {total_params} parameters, got {len(params)}"
        )

    qc = QuantumCircuit(n_qubits, name="VQE_Ansatz")
    idx = 0

    for layer in range(depth):
        # Single-qubit rotations
        for q in range(n_qubits):
            qc.ry(params[idx], q)
            idx += 1
            qc.rz(params[idx], q)
            idx += 1

        # All-to-all entangling: RXX between all pairs (native on IonQ via MS gate)
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                qc.rxx(np.pi / 4, i, j)

        qc.barrier()

    # Final rotation layer
    for q in range(n_qubits):
        qc.ry(params[idx], q)
        idx += 1
        qc.rz(params[idx], q)
        idx += 1

    return qc


# ---------------------------------------------------------------------------
# Energy evaluation
# ---------------------------------------------------------------------------

def _evaluate_energy(
    ansatz: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    params: np.ndarray,
    n_qubits: int,
    depth: int,
    backend: Any,
    use_statevector: bool = True,
    run_kwargs: dict[str, Any] | None = None,
    execution: dict[str, Any] | None = None,
    shots: int = 4096,
) -> float:
    """Evaluate <psi(params)|H|psi(params)> for the given parameters."""
    bound_circuit = build_vqe_ansatz(n_qubits, depth, params)

    if use_statevector:
        sv = Statevector.from_label("0" * n_qubits)
        sv = sv.evolve(bound_circuit)
        energy = float(np.real(sv.expectation_value(hamiltonian)))
    else:
        # Shot-based estimation via Pauli grouping
        energy = 0.0
        for pauli_label, coeff in zip(
            hamiltonian.paulis.to_labels(), hamiltonian.coeffs
        ):
            if set(pauli_label) == {"I"}:
                energy += float(np.real(coeff))
                continue

            meas_circuit = bound_circuit.copy()
            meas_circuit.add_register(
                *[
                    # already has classical register? skip
                ]
            )
            # Build measurement basis rotation
            cr = meas_circuit
            cb = QuantumCircuit(n_qubits, n_qubits)
            for i, p in enumerate(reversed(pauli_label)):
                if p == "X":
                    cb.h(i)
                elif p == "Y":
                    cb.sdg(i)
                    cb.h(i)
            cb.measure(range(n_qubits), range(n_qubits))
            full = bound_circuit.compose(cb)

            transpiled = transpile(full, backend=backend)
            result = run_job_and_get_result(
                backend,
                transpiled,
                shots=shots,
                run_kwargs=run_kwargs or {},
                execution=execution,
            )
            counts = result.get_counts()

            # Expectation from counts
            total_shots = sum(counts.values())
            exp_val = 0.0
            for bitstring, count in counts.items():
                # Parity of measured qubits where Pauli is not I
                parity = 0
                for i, p in enumerate(reversed(pauli_label)):
                    if p != "I":
                        parity += int(bitstring[n_qubits - 1 - i])
                sign = (-1) ** (parity % 2)
                exp_val += sign * count / total_shots
            energy += float(np.real(coeff)) * exp_val

    return energy


# ---------------------------------------------------------------------------
# VQE runner
# ---------------------------------------------------------------------------

def run_vqe(
    molecule: str = "H2",
    n_layers: int = 3,
    max_iterations: int = 100,
    use_simulator: bool = True,
    use_qpu: bool = False,
    noise_model: str = "forte-1",
    qpu_name: str = "qpu.forte-1",
    initial_params: np.ndarray | None = None,
    shots: int = 4096,
) -> dict[str, Any]:
    """Run VQE optimisation with COBYLA and return iteration-by-iteration energies.

    Returns
    -------
    dict with keys:
        steps : list[dict]
            Energy and parameter snapshot per optimiser iteration.
        final_result : dict
            Optimal energy, optimal parameters, molecule info.
        metadata : dict
            Circuit depth, gate counts, etc.
    """
    hamiltonian = build_hamiltonian(molecule)
    n_qubits = hamiltonian.num_qubits
    n_params_per_layer = 2 * n_qubits
    total_params = n_layers * n_params_per_layer + n_params_per_layer

    if initial_params is None:
        rng = np.random.default_rng(42)
        initial_params = rng.uniform(-np.pi, np.pi, size=total_params)

    backend, run_kwargs, execution = get_backend(
        use_simulator=use_simulator,
        use_qpu=use_qpu,
        noise_model=noise_model,
        qpu_name=qpu_name,
    )
    use_statevector = use_simulator  # use exact expectation for simulator

    energy_history: list[dict[str, Any]] = []
    iteration_counter = [0]

    def objective(params: np.ndarray) -> float:
        energy = _evaluate_energy(
            ansatz=None,  # rebuilt inside
            hamiltonian=hamiltonian,
            params=params,
            n_qubits=n_qubits,
            depth=n_layers,
            backend=backend,
            use_statevector=use_statevector,
            run_kwargs=run_kwargs,
            execution=execution,
            shots=shots,
        )
        iteration_counter[0] += 1
        energy_history.append({
            "iteration": iteration_counter[0],
            "energy": energy,
            "params": params.tolist(),
        })
        return energy

    # COBYLA optimisation
    from scipy.optimize import minimize  # type: ignore[import-untyped]

    opt_result = minimize(
        objective,
        initial_params,
        method="COBYLA",
        options={"maxiter": max_iterations, "rhobeg": 0.5},
    )

    optimal_energy = float(opt_result.fun)
    optimal_params = opt_result.x.tolist()

    # Build final circuit for metadata
    final_circuit = build_vqe_ansatz(n_qubits, n_layers, np.array(optimal_params))
    if not use_simulator:
        final_circuit = transpile_for_ionq(final_circuit)

    # Known exact ground-state energies for reference
    exact_energies = {"H2": -1.8572750, "LIH": -7.8825378}
    exact_gs = exact_energies.get(molecule.upper())

    return {
        "steps": energy_history,
        "final_result": {
            "execution": execution,
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
            **circuit_metadata(final_circuit),
            "algorithm": "vqe",
            "molecule": molecule,
            "n_layers": n_layers,
            "n_params": total_params,
            "optimizer": "COBYLA",
            "max_iterations": max_iterations,
        },
    }

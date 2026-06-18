"""Grover's Search Algorithm.

Builds and executes parameterized Grover circuits with step-by-step
amplitude tracking for visualization.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

from backend.quantum.provider import (
    circuit_metadata,
    get_backend,
    run_job_and_get_result,
    transpile_for_ionq,
)

# ---------------------------------------------------------------------------
# Oracle construction
# ---------------------------------------------------------------------------

def _build_oracle(n_qubits: int, target_state: int) -> QuantumCircuit:
    """Build a phase-flip oracle that marks *target_state*.

    The oracle applies a phase of -1 to the computational basis state
    ``|target_state>``.  Implemented via multi-controlled Z using an ancilla-free
    decomposition.

    Parameters
    ----------
    n_qubits:
        Number of qubits (search-space size is 2**n_qubits).
    target_state:
        Integer representation of the marked state (0 <= target_state < 2**n_qubits).
    """
    if target_state < 0 or target_state >= 2**n_qubits:
        raise ValueError(
            f"target_state {target_state} out of range for {n_qubits} qubits"
        )

    oracle = QuantumCircuit(n_qubits, name="Oracle")

    # Flip qubits where the target bit is 0 so that the target becomes |11...1>
    target_bits = format(target_state, f"0{n_qubits}b")[::-1]  # LSB first
    for i, bit in enumerate(target_bits):
        if bit == "0":
            oracle.x(i)

    # Multi-controlled Z = H on last qubit -> MCX -> H on last qubit
    if n_qubits == 1:
        oracle.z(0)
    elif n_qubits == 2:
        oracle.cz(0, 1)
    else:
        oracle.h(n_qubits - 1)
        oracle.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        oracle.h(n_qubits - 1)

    # Undo the X flips
    for i, bit in enumerate(target_bits):
        if bit == "0":
            oracle.x(i)

    return oracle


# ---------------------------------------------------------------------------
# Diffusion operator
# ---------------------------------------------------------------------------

def _build_diffusion(n_qubits: int) -> QuantumCircuit:
    """Build the Grover diffusion (inversion about the mean) operator."""
    diffusion = QuantumCircuit(n_qubits, name="Diffusion")

    diffusion.h(range(n_qubits))
    diffusion.x(range(n_qubits))

    # Multi-controlled Z
    if n_qubits == 1:
        diffusion.z(0)
    elif n_qubits == 2:
        diffusion.cz(0, 1)
    else:
        diffusion.h(n_qubits - 1)
        diffusion.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        diffusion.h(n_qubits - 1)

    diffusion.x(range(n_qubits))
    diffusion.h(range(n_qubits))

    return diffusion


# ---------------------------------------------------------------------------
# Circuit builder
# ---------------------------------------------------------------------------

def optimal_iterations(n_qubits: int) -> int:
    """Return the optimal number of Grover iterations: floor(pi/4 * sqrt(N))."""
    N = 2**n_qubits
    return max(1, int(math.floor(math.pi / 4 * math.sqrt(N))))


def build_grover_circuit(
    n_qubits: int,
    target_state: int,
    n_iterations: int | None = None,
) -> QuantumCircuit:
    """Build a complete Grover search circuit.

    Parameters
    ----------
    n_qubits:
        Number of search qubits.
    target_state:
        Integer label of the marked item.
    n_iterations:
        Override the number of Grover iterations.  When *None* the optimal
        count ``floor(pi/4 * sqrt(2**n_qubits))`` is used.

    Returns
    -------
    A ``QuantumCircuit`` with measurement gates appended.
    """
    if n_iterations is None:
        n_iterations = optimal_iterations(n_qubits)

    oracle = _build_oracle(n_qubits, target_state)
    diffusion = _build_diffusion(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits, name="Grover")

    # Uniform superposition
    qc.h(range(n_qubits))
    qc.barrier()

    for _ in range(n_iterations):
        qc.compose(oracle, inplace=True)
        qc.barrier()
        qc.compose(diffusion, inplace=True)
        qc.barrier()

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


# ---------------------------------------------------------------------------
# Runner with intermediate amplitudes
# ---------------------------------------------------------------------------

def _statevector_amplitudes(sv: Statevector, n_qubits: int) -> dict[str, float]:
    """Convert a Statevector into a dict of bitstring -> probability."""
    probs = sv.probabilities_dict()
    # Ensure all basis states are present
    full: dict[str, float] = {}
    for i in range(2**n_qubits):
        label = format(i, f"0{n_qubits}b")
        full[label] = float(probs.get(label, 0.0))
    return full


def run_grovers(
    n_qubits: int,
    target_state: int,
    use_simulator: bool = True,
    use_qpu: bool = False,
    noise_model: str = "forte-1",
    qpu_name: str = "qpu.forte-1",
    shots: int = 1024,
) -> dict[str, Any]:
    """Execute Grover's algorithm and return step-by-step results.

    Returns
    -------
    dict with keys:
        steps : list[dict]
            Each entry contains ``iteration``, ``amplitudes`` (probability
            distribution), and ``target_probability``.
        final_result : dict
            Contains ``measured_counts``, ``top_result``, ``success_probability``.
        metadata : dict
            Circuit depth, gate counts, etc.
    """
    n_iter = optimal_iterations(n_qubits)
    oracle = _build_oracle(n_qubits, target_state)
    diffusion = _build_diffusion(n_qubits)
    target_label = format(target_state, f"0{n_qubits}b")

    steps: list[dict[str, Any]] = []

    # Build circuit incrementally and snapshot amplitudes via statevector sim
    sv = Statevector.from_label("0" * n_qubits)

    # Apply initial Hadamards
    init_qc = QuantumCircuit(n_qubits)
    init_qc.h(range(n_qubits))
    sv = sv.evolve(init_qc)

    # Record initial uniform distribution
    amps = _statevector_amplitudes(sv, n_qubits)
    steps.append({
        "iteration": 0,
        "amplitudes": amps,
        "target_probability": amps.get(target_label, 0.0),
        "description": "Initial uniform superposition",
    })

    for it in range(1, n_iter + 1):
        sv = sv.evolve(oracle)
        sv = sv.evolve(diffusion)
        amps = _statevector_amplitudes(sv, n_qubits)
        steps.append({
            "iteration": it,
            "amplitudes": amps,
            "target_probability": amps.get(target_label, 0.0),
            "description": f"After Grover iteration {it}",
        })

    # Build the full measurement circuit and run on the chosen backend
    full_circuit = build_grover_circuit(n_qubits, target_state, n_iterations=n_iter)

    if not use_simulator:
        full_circuit = transpile_for_ionq(full_circuit)

    backend, run_kwargs, execution = get_backend(
        use_simulator=use_simulator,
        use_qpu=use_qpu,
        noise_model=noise_model,
        qpu_name=qpu_name,
    )
    transpiled = transpile(full_circuit, backend=backend)
    result = run_job_and_get_result(
        backend,
        transpiled,
        shots=shots,
        run_kwargs=run_kwargs,
        execution=execution,
    )
    counts = result.get_counts()

    # Determine winner
    top_result = max(counts, key=counts.get)  # type: ignore[arg-type]
    success_prob = counts.get(target_label, 0) / shots

    return {
        "steps": steps,
        "final_result": {
            "execution": execution,
            "measured_counts": counts,
            "top_result": top_result,
            "target_state": target_label,
            "success_probability": success_prob,
            "n_iterations": n_iter,
            "optimal_iterations": n_iter,
        },
        "metadata": {
            **circuit_metadata(full_circuit),
            "algorithm": "grovers_search",
            "n_qubits": n_qubits,
            "search_space_size": 2**n_qubits,
            "shots": shots,
        },
    }

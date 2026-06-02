"""Discrete-Time Quantum Walks.

Implements coined quantum walks on cycle and complete graphs with
step-by-step position distribution tracking.
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
    transpile_for_ionq,
)

# ---------------------------------------------------------------------------
# Coin operators
# ---------------------------------------------------------------------------

def _hadamard_coin(circuit: QuantumCircuit, coin_qubit: int) -> None:
    """Apply Hadamard coin to the coin qubit."""
    circuit.h(coin_qubit)


def _grover_coin(circuit: QuantumCircuit, coin_qubits: list[int]) -> None:
    """Apply the Grover diffusion coin on multiple coin qubits.

    The Grover coin is  2|s><s| - I  where |s> is the uniform superposition.
    For a single coin qubit this reduces to the Hadamard.
    """
    n = len(coin_qubits)
    if n == 1:
        circuit.h(coin_qubits[0])
        return

    # Implement as H^n . (2|0><0| - I) . H^n
    for q in coin_qubits:
        circuit.h(q)
    for q in coin_qubits:
        circuit.x(q)

    # Multi-controlled Z
    if n == 2:
        circuit.cz(coin_qubits[0], coin_qubits[1])
    else:
        circuit.h(coin_qubits[-1])
        circuit.mcx(coin_qubits[:-1], coin_qubits[-1])
        circuit.h(coin_qubits[-1])

    for q in coin_qubits:
        circuit.x(q)
    for q in coin_qubits:
        circuit.h(q)


# ---------------------------------------------------------------------------
# Shift operators
# ---------------------------------------------------------------------------

def _cycle_shift(
    circuit: QuantumCircuit,
    coin_qubit: int,
    position_qubits: list[int],
) -> None:
    """Conditional increment/decrement on a cycle graph (mod 2^n).

    If coin=|0> shift position right (+1 mod N).
    If coin=|1> shift position left  (-1 mod N).

    Implemented as a conditional binary adder/subtractor.
    """
    n_pos = len(position_qubits)

    # --- Shift right when coin=|0> ---
    # Flip coin so |0> -> |1> to use as control
    circuit.x(coin_qubit)

    # Controlled increment: cascade of multi-controlled X from MSB to LSB
    for i in range(n_pos - 1, -1, -1):
        controls = [coin_qubit] + position_qubits[:i]
        if len(controls) == 1:
            circuit.cx(controls[0], position_qubits[i])
        elif len(controls) == 2:
            circuit.ccx(controls[0], controls[1], position_qubits[i])
        else:
            circuit.mcx(controls, position_qubits[i])

    circuit.x(coin_qubit)  # undo flip

    # --- Shift left when coin=|1> ---
    # Controlled decrement = controlled increment of (N-x) complement
    # Equivalent: flip positions, controlled increment, flip positions
    for q in position_qubits:
        circuit.x(q)

    for i in range(n_pos - 1, -1, -1):
        controls = [coin_qubit] + position_qubits[:i]
        if len(controls) == 1:
            circuit.cx(controls[0], position_qubits[i])
        elif len(controls) == 2:
            circuit.ccx(controls[0], controls[1], position_qubits[i])
        else:
            circuit.mcx(controls, position_qubits[i])

    for q in position_qubits:
        circuit.x(q)


def _complete_shift(
    circuit: QuantumCircuit,
    coin_qubits: list[int],
    position_qubits: list[int],
) -> None:
    """Shift operator for a complete graph.

    On a complete graph of N vertices, the walker can move to any other vertex.
    We encode the coin with log2(N) qubits.  The shift swaps the position
    register with the coin register (controlled SWAP network).
    """
    n = min(len(coin_qubits), len(position_qubits))
    for i in range(n):
        circuit.swap(coin_qubits[i], position_qubits[i])


# ---------------------------------------------------------------------------
# Circuit builder
# ---------------------------------------------------------------------------

def build_walk_circuit(
    n_qubits: int,
    n_steps: int,
    graph_type: str = "cycle",
) -> QuantumCircuit:
    """Build a discrete-time quantum walk circuit.

    Parameters
    ----------
    n_qubits:
        Number of position qubits.  The walker lives on 2**n_qubits nodes.
    n_steps:
        Number of walk steps to apply.
    graph_type:
        ``"cycle"`` -- walk on a cycle graph (1 coin qubit, Hadamard coin).
        ``"complete"`` -- walk on the complete graph (n_qubits coin qubits,
        Grover coin).

    Returns
    -------
    ``QuantumCircuit`` with measurements on the position register.
    """
    graph_type = graph_type.lower()
    if graph_type not in ("cycle", "complete"):
        raise ValueError(f"Unsupported graph_type: {graph_type!r}")

    if graph_type == "cycle":
        n_coin = 1
    else:
        n_coin = n_qubits  # complete graph needs log2(N) coin qubits

    total_qubits = n_coin + n_qubits
    qc = QuantumCircuit(total_qubits, n_qubits, name=f"QWalk_{graph_type}")

    coin_qubits = list(range(n_coin))
    position_qubits = list(range(n_coin, n_coin + n_qubits))

    # Initial state: walker at position 0, coin in superposition
    if graph_type == "cycle":
        qc.h(coin_qubits[0])
    else:
        for cq in coin_qubits:
            qc.h(cq)

    qc.barrier()

    for step in range(n_steps):
        # Coin
        if graph_type == "cycle":
            _hadamard_coin(qc, coin_qubits[0])
        else:
            _grover_coin(qc, coin_qubits)

        # Shift
        if graph_type == "cycle":
            _cycle_shift(qc, coin_qubits[0], position_qubits)
        else:
            _complete_shift(qc, coin_qubits, position_qubits)

        qc.barrier()

    # Measure position register
    qc.measure(position_qubits, range(n_qubits))
    return qc


# ---------------------------------------------------------------------------
# Runner with per-step distributions
# ---------------------------------------------------------------------------

def _position_distribution(
    sv: Statevector,
    n_coin: int,
    n_position: int,
) -> dict[int, float]:
    """Trace out coin qubits and return position probabilities.

    Parameters
    ----------
    sv: full statevector over coin + position registers.
    n_coin: number of coin qubits (indices 0..n_coin-1).
    n_position: number of position qubits.
    """
    probs = np.abs(sv.data) ** 2
    n_total = n_coin + n_position
    N_pos = 2**n_position
    N_coin = 2**n_coin

    pos_probs: dict[int, float] = {i: 0.0 for i in range(N_pos)}
    for idx, p in enumerate(probs):
        # Qiskit uses little-endian bit ordering
        # idx encodes |q_{n-1} ... q_1 q_0>
        # coin qubits are 0..n_coin-1 (least significant in the index)
        pos_bits = (idx >> n_coin) & (N_pos - 1)
        pos_probs[pos_bits] += float(p)

    return pos_probs


def run_quantum_walk(
    n_qubits: int,
    n_steps: int,
    graph_type: str = "cycle",
    use_simulator: bool = True,
    use_qpu: bool = False,
    noise_model: str = "forte-1",
    qpu_name: str = "qpu.forte-1",
    shots: int = 1024,
) -> dict[str, Any]:
    """Execute a quantum walk and return per-step position distributions.

    Returns
    -------
    dict with keys:
        steps : list[dict]
            Per-step position probability distributions.
        final_result : dict
            Final measured counts and dominant position.
        metadata : dict
            Circuit information.
    """
    graph_type = graph_type.lower()
    n_coin = 1 if graph_type == "cycle" else n_qubits
    total_qubits = n_coin + n_qubits

    coin_qubits = list(range(n_coin))
    position_qubits = list(range(n_coin, n_coin + n_qubits))

    steps: list[dict[str, Any]] = []

    # Statevector simulation for intermediate distributions
    sv = Statevector.from_label("0" * total_qubits)

    # Initial coin superposition
    init_qc = QuantumCircuit(total_qubits)
    if graph_type == "cycle":
        init_qc.h(coin_qubits[0])
    else:
        for cq in coin_qubits:
            init_qc.h(cq)
    sv = sv.evolve(init_qc)

    pos_dist = _position_distribution(sv, n_coin, n_qubits)
    steps.append({
        "step": 0,
        "position_distribution": pos_dist,
        "description": "Initial state",
    })

    for t in range(1, n_steps + 1):
        step_qc = QuantumCircuit(total_qubits)

        # Coin
        if graph_type == "cycle":
            _hadamard_coin(step_qc, coin_qubits[0])
        else:
            _grover_coin(step_qc, coin_qubits)

        # Shift
        if graph_type == "cycle":
            _cycle_shift(step_qc, coin_qubits[0], position_qubits)
        else:
            _complete_shift(step_qc, coin_qubits, position_qubits)

        sv = sv.evolve(step_qc)
        pos_dist = _position_distribution(sv, n_coin, n_qubits)
        steps.append({
            "step": t,
            "position_distribution": pos_dist,
            "description": f"After step {t}",
        })

    # Full circuit execution with measurements
    full_circuit = build_walk_circuit(n_qubits, n_steps, graph_type)

    if not use_simulator:
        full_circuit = transpile_for_ionq(full_circuit)

    backend, run_kwargs, execution = get_backend(
        use_simulator=use_simulator,
        use_qpu=use_qpu,
        noise_model=noise_model,
        qpu_name=qpu_name,
    )
    transpiled = transpile(full_circuit, backend=backend)
    job = backend.run(transpiled, shots=shots, **run_kwargs)
    result = job.result()
    counts = result.get_counts()

    top_position = max(counts, key=counts.get)  # type: ignore[arg-type]

    return {
        "steps": steps,
        "final_result": {
            "execution": execution,
            "measured_counts": counts,
            "top_position": top_position,
            "n_steps": n_steps,
            "graph_type": graph_type,
            "n_nodes": 2**n_qubits,
        },
        "metadata": {
            **circuit_metadata(full_circuit),
            "algorithm": "quantum_walk",
            "graph_type": graph_type,
            "n_position_qubits": n_qubits,
            "n_coin_qubits": n_coin,
            "n_steps": n_steps,
            "shots": shots,
        },
    }

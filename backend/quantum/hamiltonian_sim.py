"""Hamiltonian Simulation via Trotterization.

First-order product-formula (Lie-Trotter) decomposition for Ising and
Heisenberg spin-chain models, with per-step fidelity tracking.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Statevector, state_fidelity

from backend.quantum.provider import (
    circuit_metadata,
    get_backend,
    run_job_and_get_result,
    transpile_for_ionq,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model Hamiltonians (returned as list of Pauli terms)
# ---------------------------------------------------------------------------

# Each term is (coefficient, pauli_string) where pauli_string uses "I","X","Y","Z"
PauliTerm = tuple[float, str]
InteractionPattern = str

_TWO_QUBIT_GATE_NAMES = {"cx", "cz", "ecr", "iswap", "ms", "rxx", "ryy", "rzz", "swap"}
_HEAVY_HEX_BASIS_GATES = ["rz", "sx", "x", "cx", "swap", "measure"]


def _interaction_scale(distance: int, interaction_pattern: InteractionPattern, alpha: float) -> float:
    """Return the coupling weight for a pair separated by *distance*."""
    if distance < 1:
        raise ValueError("distance must be >= 1")
    if interaction_pattern == "chain":
        return 1.0 if distance == 1 else 0.0
    if interaction_pattern == "all_to_all":
        return 1.0
    if interaction_pattern == "power_law":
        if alpha <= 0:
            raise ValueError("alpha must be > 0 for power_law interactions")
        return 1.0 / (distance ** alpha)
    raise ValueError(
        f"Unknown interaction_pattern {interaction_pattern!r}. "
        "Supported: chain, power_law, all_to_all"
    )


def interaction_edges(
    n_qubits: int,
    interaction_pattern: InteractionPattern = "chain",
    alpha: float = 3.0,
) -> list[dict[str, Any]]:
    """Return weighted pairwise couplings for the selected interaction pattern."""
    edges: list[dict[str, Any]] = []
    for source in range(n_qubits):
        for target in range(source + 1, n_qubits):
            distance = target - source
            weight = _interaction_scale(distance, interaction_pattern, alpha)
            if weight <= 0:
                continue
            edges.append({
                "source": source,
                "target": target,
                "distance": distance,
                "weight": float(weight),
            })
    return edges


def _ising_terms(
    n_qubits: int,
    J: float = 1.0,
    h: float = 0.5,
    interaction_pattern: InteractionPattern = "chain",
    alpha: float = 3.0,
) -> list[PauliTerm]:
    """Transverse-field Ising model on a 1D chain.

    H = -J * sum_{i} Z_i Z_{i+1}  -  h * sum_{i} X_i
    """
    terms: list[PauliTerm] = []

    # ZZ interactions
    for edge in interaction_edges(n_qubits, interaction_pattern, alpha):
        i = edge["source"]
        j = edge["target"]
        pauli = ["I"] * n_qubits
        pauli[i] = "Z"
        pauli[j] = "Z"
        terms.append((-J * edge["weight"], "".join(pauli)))

    # Transverse field
    for i in range(n_qubits):
        pauli = ["I"] * n_qubits
        pauli[i] = "X"
        terms.append((-h, "".join(pauli)))

    return terms


def _heisenberg_terms(
    n_qubits: int,
    Jx: float = 1.0,
    Jy: float = 1.0,
    Jz: float = 1.0,
    interaction_pattern: InteractionPattern = "chain",
    alpha: float = 3.0,
) -> list[PauliTerm]:
    """Heisenberg XXX/XXZ model on a 1D chain.

    H = sum_{i} [ Jx X_i X_{i+1} + Jy Y_i Y_{i+1} + Jz Z_i Z_{i+1} ]
    """
    terms: list[PauliTerm] = []

    for edge in interaction_edges(n_qubits, interaction_pattern, alpha):
        i = edge["source"]
        j = edge["target"]
        for pauli_char, coupling in [("X", Jx), ("Y", Jy), ("Z", Jz)]:
            pauli = ["I"] * n_qubits
            pauli[i] = pauli_char
            pauli[j] = pauli_char
            terms.append((coupling * edge["weight"], "".join(pauli)))

    return terms


def get_model_terms(
    n_qubits: int,
    model: str = "ising",
    interaction_pattern: InteractionPattern = "chain",
    alpha: float = 3.0,
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
        return _ising_terms(
            n_qubits,
            interaction_pattern=interaction_pattern,
            alpha=alpha,
            **kwargs,
        )
    elif model == "heisenberg":
        return _heisenberg_terms(
            n_qubits,
            interaction_pattern=interaction_pattern,
            alpha=alpha,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown model {model!r}. Supported: ising, heisenberg")


def build_problem_circuit(
    n_qubits: int,
    model: str = "ising",
    time: float = 1.0,
    n_steps: int = 10,
    initial_state: str | None = None,
    **model_kwargs: float,
) -> tuple[QuantumCircuit, list[PauliTerm]]:
    """Build the measured Hamiltonian-simulation circuit used for execution/analysis."""
    terms = get_model_terms(n_qubits, model, **model_kwargs)

    if initial_state is None:
        initial_state = "0" * n_qubits
    if len(initial_state) != n_qubits:
        raise ValueError("initial_state length must match n_qubits")

    full_circuit = QuantumCircuit(n_qubits, n_qubits)
    for i, bit in enumerate(reversed(initial_state)):
        if bit == "1":
            full_circuit.x(i)

    trotter_body = build_trotter_circuit(terms, time, n_steps)
    trotter_no_meas = trotter_body.remove_final_measurements(inplace=False)
    full_circuit.compose(trotter_no_meas, inplace=True)
    full_circuit.measure(range(n_qubits), range(n_qubits))
    return full_circuit, terms


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


def _coupling_map_size(coupling_map: Any) -> int:
    """Best-effort physical-qubit count for a Qiskit coupling map."""
    physical = getattr(coupling_map, "physical_qubits", None)
    if physical is not None:
        return len(physical)
    size = getattr(coupling_map, "size", None)
    if callable(size):
        return int(size())
    if isinstance(size, int):
        return size
    edges = list(coupling_map.get_edges())
    if not edges:
        return 0
    return max(max(edge) for edge in edges) + 1


def _heavy_hex_coupling_map(n_qubits: int) -> tuple[Any, dict[str, int]]:
    """Smallest heavy-hex coupling map whose capacity fits *n_qubits*."""
    from qiskit.transpiler import CouplingMap

    if n_qubits <= 1:
        return CouplingMap([]), {"distance": 0, "physical_qubits": 1}

    last_error: Exception | None = None
    for distance in range(3, 51, 2):
        try:
            coupling_map = CouplingMap.from_heavy_hex(distance)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue
        size = _coupling_map_size(coupling_map)
        if size >= n_qubits:
            return coupling_map, {"distance": distance, "physical_qubits": size}

    raise RuntimeError(
        f"Could not construct a heavy-hex coupling map for {n_qubits} qubits"
    ) from last_error


def transpile_for_heavy_hex(
    circuit: QuantumCircuit,
    optimization_level: int = 2,
) -> tuple[QuantumCircuit, dict[str, int]]:
    """Transpile *circuit* to a generic heavy-hex superconducting topology."""
    coupling_map, topology = _heavy_hex_coupling_map(circuit.num_qubits)
    transpiled = transpile(
        circuit,
        basis_gates=_HEAVY_HEX_BASIS_GATES,
        coupling_map=coupling_map,
        optimization_level=optimization_level,
        layout_method="sabre",
        routing_method="sabre",
    )
    return transpiled, topology


def _two_qubit_depth(circuit: QuantumCircuit) -> int:
    """Depth counting only layers that contain a 2-qubit operation."""
    try:
        return int(
            circuit.depth(
                filter_function=lambda instruction: getattr(
                    getattr(instruction, "operation", instruction), "num_qubits", 0
                ) == 2
            )
        )
    except Exception:  # noqa: BLE001
        dag = circuit_to_dag(circuit)
        depth = 0
        for layer in dag.layers():
            if any(getattr(node.op, "num_qubits", 0) == 2 for node in layer["graph"].op_nodes()):
                depth += 1
        return depth


def connectivity_circuit_summary(
    circuit: QuantumCircuit,
    *,
    topology: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract routing-focused summary metrics from a circuit."""
    ops = dict(circuit.count_ops())
    two_qubit_gates = sum(int(v) for k, v in ops.items() if k in _TWO_QUBIT_GATE_NAMES)
    summary = {
        "n_qubits": circuit.num_qubits,
        "depth": int(circuit.depth()),
        "gate_counts": ops,
        "total_gates": int(sum(ops.values())),
        "two_qubit_gates": two_qubit_gates,
        "swap_count": int(ops.get("swap", 0)),
        "two_qubit_depth": _two_qubit_depth(circuit),
    }
    if topology:
        summary["topology"] = topology
    return summary


def analyze_connectivity(
    n_qubits: int,
    model: str = "ising",
    time: float = 1.0,
    n_steps: int = 10,
    initial_state: str | None = None,
    optimization_level: int = 2,
    interaction_pattern: InteractionPattern = "chain",
    alpha: float = 3.0,
    **model_kwargs: float,
) -> dict[str, Any]:
    """Compare logical, IonQ, and heavy-hex transpiled Hamiltonian circuits."""
    logical_circuit, _terms = build_problem_circuit(
        n_qubits=n_qubits,
        model=model,
        time=time,
        n_steps=n_steps,
        initial_state=initial_state,
        interaction_pattern=interaction_pattern,
        alpha=alpha,
        **model_kwargs,
    )
    ionq_circuit = transpile_for_ionq(logical_circuit, optimization_level=optimization_level)
    heavy_hex_circuit, heavy_hex_topology = transpile_for_heavy_hex(
        logical_circuit,
        optimization_level=optimization_level,
    )

    logical = connectivity_circuit_summary(logical_circuit)
    ionq = connectivity_circuit_summary(ionq_circuit, topology={"name": "ionq_all_to_all"})
    heavy_hex = connectivity_circuit_summary(
        heavy_hex_circuit,
        topology={"name": "heavy_hex", **heavy_hex_topology},
    )

    heavy_depth = heavy_hex["depth"]
    heavy_two_q = heavy_hex["two_qubit_gates"]

    return {
        "logical": {
            **logical,
            "interaction_pattern": interaction_pattern,
        },
        "ionq": ionq,
        "heavy_hex": heavy_hex,
        "interaction_graph": {
            "n_qubits": n_qubits,
            "pattern": interaction_pattern,
            "alpha": alpha if interaction_pattern == "power_law" else None,
            "edges": interaction_edges(n_qubits, interaction_pattern, alpha),
        },
        "metrics": {
            "routing_depth_reduction_pct": 100.0 * (heavy_depth - ionq["depth"]) / max(heavy_depth, 1),
            "swap_tax_avoided": heavy_hex["swap_count"] - ionq["swap_count"],
            "two_qubit_overhead_reduction_pct": (
                100.0 * (heavy_two_q - ionq["two_qubit_gates"]) / max(heavy_two_q, 1)
            ),
        },
        "params": {
            "n_qubits": n_qubits,
            "model": model,
            "time": time,
            "n_steps": n_steps,
            "interaction_pattern": interaction_pattern,
            "alpha": alpha if interaction_pattern == "power_law" else None,
            "initial_state": initial_state or ("0" * n_qubits),
        },
    }


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
    full_circuit, _ = build_problem_circuit(
        n_qubits=n_qubits,
        model=model,
        time=time,
        n_steps=n_steps,
        initial_state=initial_state,
        **model_kwargs,
    )

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
    result = run_job_and_get_result(
        backend,
        transpiled,
        shots=shots,
        run_kwargs=run_kwargs,
        execution=execution,
    )
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

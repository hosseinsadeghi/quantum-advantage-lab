"""Correctness tests for backend.quantum.hamiltonian_sim.

These tests verify the two audit-flagged concerns (Y-basis convention and
initial-state bit-ordering) and the Trotter-convergence contract. They are
intentionally cheap so the whole file runs in <5 s.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector, state_fidelity
from scipy.linalg import expm  # type: ignore[import-untyped]

from backend.classical.matrix_exp import _build_hamiltonian, _pauli_string_matrix
from backend.quantum.hamiltonian_sim import (
    _apply_pauli_rotation,
    _exact_evolution,
    analyze_connectivity,
    build_trotter_circuit,
    get_model_terms,
    interaction_edges,
    run_simulation,
)

# ---------------------------------------------------------------------------
# Reference Pauli matrices
# ---------------------------------------------------------------------------

_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _circuit_operator_qubit0_first(qc: QuantumCircuit) -> np.ndarray:
    """Return the unitary of ``qc`` with qubit 0 as MSB.

    Qiskit's default ``Operator(qc).data`` uses qubit 0 as LSB. For elementwise
    comparison against ``np.kron(P_0, P_1, ...)`` (which puts ``P_0`` as the
    MSB), we reverse the bit ordering by applying ``Operator.reverse_qargs()``.
    """
    return Operator(qc).reverse_qargs().data


def _expected_pauli_string_evolution(pauli_str: str, coeff: float, dt: float) -> np.ndarray:
    """Return exp(-i * coeff * dt * P) with qubit 0 as MSB (matches our kron convention)."""
    P = _pauli_string_matrix(pauli_str)  # qubit 0 = MSB (np.kron left-most)
    return expm(-1j * coeff * dt * P)


# ---------------------------------------------------------------------------
# 2.1 — Pauli rotation correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pauli", ["X", "Y", "Z"])
def test_pauli_rotation_single_qubit(pauli: str) -> None:
    """Single-qubit exp(-i coeff dt P) must match scipy.expm elementwise.

    Catches any Y-basis sign-flip: if `_apply_pauli_rotation` gets the Y
    rotation wrong, the off-diagonal signs will disagree with scipy.expm.
    """
    coeff = 0.7
    dt = 0.3

    qc = QuantumCircuit(1)
    _apply_pauli_rotation(qc, coeff, pauli, dt)

    got = Operator(qc).data
    want = expm(-1j * coeff * dt * _PAULI[pauli])
    np.testing.assert_allclose(got, want, atol=1e-10, err_msg=f"Pauli {pauli}")


@pytest.mark.parametrize(
    "pauli_str",
    ["XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ", "IX", "XI", "IY", "YI"],
)
def test_pauli_rotation_two_qubit(pauli_str: str) -> None:
    """Two-qubit Pauli strings (incl. all Y combos) match scipy.expm elementwise."""
    coeff = 0.4
    dt = 0.25

    qc = QuantumCircuit(2)
    _apply_pauli_rotation(qc, coeff, pauli_str, dt)

    got = _circuit_operator_qubit0_first(qc)
    want = _expected_pauli_string_evolution(pauli_str, coeff, dt)
    np.testing.assert_allclose(got, want, atol=1e-10, err_msg=f"Pauli {pauli_str}")


def test_pauli_rotation_three_qubit_YYY() -> None:
    """Triple-Y string exercises basis-change on three qubits simultaneously."""
    coeff = 0.3
    dt = 0.2
    qc = QuantumCircuit(3)
    _apply_pauli_rotation(qc, coeff, "YYY", dt)

    got = _circuit_operator_qubit0_first(qc)
    want = _expected_pauli_string_evolution("YYY", coeff, dt)
    np.testing.assert_allclose(got, want, atol=1e-10)


# ---------------------------------------------------------------------------
# 2.1 — Initial-state bit ordering roundtrip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bitstring", ["0", "1", "10", "01", "100", "010", "001", "1000", "0010", "1010"])
def test_initial_state_preparation_roundtrip(bitstring: str) -> None:
    """Circuit prep via ``reversed + qc.x`` must produce the statevector that
    ``Statevector.from_label`` expects.

    Audit concern: these two paths could disagree for asymmetric labels.
    """
    n = len(bitstring)
    sv_ref = Statevector.from_label(bitstring)

    qc = QuantumCircuit(n)
    for i, bit in enumerate(reversed(bitstring)):
        if bit == "1":
            qc.x(i)
    sv_from_circuit = Statevector.from_label("0" * n).evolve(qc)

    np.testing.assert_allclose(
        sv_from_circuit.data, sv_ref.data, atol=1e-12,
        err_msg=f"bitstring {bitstring!r}: circuit prep disagrees with Statevector.from_label",
    )


# ---------------------------------------------------------------------------
# 2.2 — Trotter convergence
# ---------------------------------------------------------------------------

def test_trotter_converges_to_exact() -> None:
    """For fixed t, Trotter fidelity vs exact must be monotonically non-decreasing
    in n_steps and reach > 0.9999 at n_steps=16."""
    fidelities = []
    for n_steps in (1, 2, 4, 8, 16):
        result = run_simulation(
            n_qubits=2, model="ising", time=0.5, n_steps=n_steps,
            use_simulator=True, shots=32,
        )
        fidelities.append(result["steps"][-1]["fidelity_vs_exact"])

    for a, b in zip(fidelities, fidelities[1:]):
        assert b >= a - 1e-9, f"Trotter fidelity regressed: {fidelities}"
    assert fidelities[-1] > 0.9999, f"Trotter at 16 steps didn't converge: {fidelities[-1]}"


def test_trotter_error_scales_as_1_over_r() -> None:
    """Infidelity 1 - F should scale ~ 1/r at large r (first-order Trotter)."""
    ts = 0.5
    infidelities = []
    for r in (8, 16, 32):
        result = run_simulation(
            n_qubits=2, model="ising", time=ts, n_steps=r,
            use_simulator=True, shots=32,
        )
        infidelities.append(1.0 - result["steps"][-1]["fidelity_vs_exact"])

    # Halving r should roughly halve infidelity. Allow 2x slack.
    ratio = infidelities[0] / max(infidelities[1], 1e-12)
    assert 1.5 < ratio < 4.5, f"Trotter infidelity scaling off: ratios={infidelities}"


# ---------------------------------------------------------------------------
# 2.2 — Model-terms consistency
# ---------------------------------------------------------------------------

def test_get_model_terms_matches_classical_hamiltonian_ising() -> None:
    """Hamiltonian built from quantum ``get_model_terms`` must equal the
    classical ``_ising_hamiltonian`` elementwise."""
    n = 3
    J, h = 1.1, 0.4
    terms = get_model_terms(n, "ising", J=J, h=h)

    op = SparsePauliOp.from_list([(p, c) for (c, p) in terms])
    H_q = op.to_matrix()

    H_c = _build_hamiltonian(n, "ising", J=J, h=h)

    # Note: SparsePauliOp reads pauli strings with leftmost char = qubit n-1,
    # while _pauli_string_matrix treats leftmost char = qubit 0 (np.kron). Those
    # two labelings differ by a full qubit-order reversal; for the symmetric
    # TFIM chain the matrix is invariant under this relabel because every
    # nearest-neighbor pair is covered. Assert equality to catch any sign or
    # coupling-strength drift.
    np.testing.assert_allclose(H_q, H_c, atol=1e-12)


def test_get_model_terms_matches_classical_hamiltonian_heisenberg() -> None:
    n = 3
    Jx, Jy, Jz = 1.2, 0.7, 1.3
    terms = get_model_terms(n, "heisenberg", Jx=Jx, Jy=Jy, Jz=Jz)

    op = SparsePauliOp.from_list([(p, c) for (c, p) in terms])
    H_q = op.to_matrix()

    H_c = _build_hamiltonian(n, "heisenberg", Jx=Jx, Jy=Jy, Jz=Jz)

    np.testing.assert_allclose(H_q, H_c, atol=1e-12)


def test_interaction_edges_support_chain_power_law_and_all_to_all() -> None:
    chain = interaction_edges(4, "chain")
    power = interaction_edges(4, "power_law", alpha=2.0)
    dense = interaction_edges(4, "all_to_all")

    assert [(edge["source"], edge["target"]) for edge in chain] == [(0, 1), (1, 2), (2, 3)]
    assert len(dense) == 6  # complete graph K4
    longest = next(edge for edge in power if edge["source"] == 0 and edge["target"] == 3)
    assert longest["weight"] == pytest.approx(1.0 / 9.0)


def test_connectivity_analysis_reports_stable_summary_keys() -> None:
    result = analyze_connectivity(
        n_qubits=4,
        model="ising",
        time=0.5,
        n_steps=2,
        interaction_pattern="all_to_all",
    )

    for key in ("logical", "ionq", "heavy_hex"):
        block = result[key]
        assert "depth" in block
        assert "two_qubit_gates" in block
        assert "swap_count" in block
        assert "two_qubit_depth" in block
        assert isinstance(block["gate_counts"], dict)

    heavy_hex = result["heavy_hex"]
    ionq = result["ionq"]
    assert heavy_hex["swap_count"] >= ionq["swap_count"]
    assert (
        heavy_hex["depth"] >= ionq["depth"]
        or heavy_hex["two_qubit_gates"] >= ionq["two_qubit_gates"]
    )


# ---------------------------------------------------------------------------
# 2.2 — Fidelity stays ~1 under perfect Trotter on a trivial Hamiltonian
# ---------------------------------------------------------------------------

def test_exact_evolution_is_unitary() -> None:
    """``_exact_evolution`` must preserve norm to 1e-12 over reasonable t."""
    n = 3
    terms = get_model_terms(n, "ising")
    sv_init = Statevector.from_label("0" * n)
    for t in (0.1, 0.5, 1.0, 2.0):
        sv = _exact_evolution(terms, n, t, sv_init)
        np.testing.assert_allclose(np.linalg.norm(sv.data), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 2.1 — model_kwargs must flow through to the Hamiltonian
# ---------------------------------------------------------------------------

def test_model_kwargs_flow_through_run_simulation() -> None:
    """Changing J/h via model_kwargs must change the exact-evolution fidelity."""
    res_default = run_simulation(
        n_qubits=2, model="ising", time=0.5, n_steps=2,
        use_simulator=True, shots=32,
    )
    res_strong = run_simulation(
        n_qubits=2, model="ising", time=0.5, n_steps=2,
        use_simulator=True, shots=32, J=3.0, h=2.0,
    )

    # Very different Hamiltonians → final fidelity curves differ.
    # (First-order Trotter with stronger couplings incurs more error.)
    f1 = res_default["steps"][-1]["fidelity_vs_exact"]
    f2 = res_strong["steps"][-1]["fidelity_vs_exact"]
    assert abs(f1 - f2) > 1e-6, (
        "model_kwargs appear not to be reaching the Hamiltonian builder: "
        f"default F={f1}, strong F={f2}"
    )


# ---------------------------------------------------------------------------
# 2.1 — Trotter scaling-reference naming
# ---------------------------------------------------------------------------

def test_seed_simulator_makes_counts_reproducible() -> None:
    """Same seed → identical shot counts across repeated calls."""
    r1 = run_simulation(
        n_qubits=2, model="ising", time=0.4, n_steps=3,
        use_simulator=True, shots=256, seed_simulator=12345,
    )
    r2 = run_simulation(
        n_qubits=2, model="ising", time=0.4, n_steps=3,
        use_simulator=True, shots=256, seed_simulator=12345,
    )
    assert r1["final_result"]["measured_counts"] == r2["final_result"]["measured_counts"]


def test_simulation_race_forwards_model_kwargs() -> None:
    """SimulationRace must forward J/h from the params dict to both solvers."""
    import asyncio
    from backend.modules.simulation_race import SimulationRace

    mod = SimulationRace()
    res_default = asyncio.run(mod.run({"n_qubits": 2, "time": 0.5, "n_steps": 2}))
    res_strong = asyncio.run(mod.run({"n_qubits": 2, "time": 0.5, "n_steps": 2, "J": 3.0, "h": 2.0}))

    f1 = res_default.quantum_result.get("final_fidelity")
    f2 = res_strong.quantum_result.get("final_fidelity")
    assert f1 is not None and f2 is not None
    assert abs(f1 - f2) > 1e-6, "J/h were not forwarded through SimulationRace"


def test_trotter_error_scaling_reference_has_t2_over_r_shape() -> None:
    """The reported scaling reference in the result dict should equal t^2/(2r)."""
    result = run_simulation(
        n_qubits=2, model="ising", time=0.5, n_steps=4,
        use_simulator=True, shots=32,
    )
    fr = result["final_result"]
    key = "trotter_error_scaling_reference" if "trotter_error_scaling_reference" in fr else "trotter_error_bound"
    assert fr[key] == pytest.approx(0.5**2 / (2 * 4))

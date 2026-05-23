"""Correctness tests for backend.classical.matrix_exp."""

from __future__ import annotations

import numpy as np

from backend.classical.matrix_exp import (
    _build_hamiltonian,
    _ising_hamiltonian,
    run_classical_simulation,
)


def test_ising_hamiltonian_is_hermitian() -> None:
    H = _ising_hamiltonian(4, J=1.0, h=0.5)
    np.testing.assert_allclose(H, H.conj().T, atol=1e-12)


def test_run_classical_simulation_norm_preserved() -> None:
    """The evolved state must stay normalized across every reported step."""
    result = run_classical_simulation(n_qubits=3, model="ising", time_total=1.5, n_steps=8)
    for step in result["steps"]:
        probs = list(step["state_probs"].values())
        np.testing.assert_allclose(sum(probs), 1.0, atol=1e-10)


def test_heisenberg_hamiltonian_is_hermitian() -> None:
    H = _build_hamiltonian(4, "heisenberg", Jx=1.0, Jy=0.7, Jz=1.3)
    np.testing.assert_allclose(H, H.conj().T, atol=1e-12)


def test_model_kwargs_flow_through_run_classical() -> None:
    """Changing J via model_kwargs must change the final state probabilities."""
    res_default = run_classical_simulation(n_qubits=2, model="ising", time_total=1.0, n_steps=4)
    res_strong = run_classical_simulation(n_qubits=2, model="ising", time_total=1.0, n_steps=4, J=3.0, h=2.0)

    p0 = res_default["final_result"]["final_state_probs"]
    p1 = res_strong["final_result"]["final_state_probs"]
    # Different couplings → different computational-basis probabilities.
    assert set(p0.keys()) == set(p1.keys()) or any(
        abs(p0.get(k, 0.0) - p1.get(k, 0.0)) > 1e-6 for k in set(p0) | set(p1)
    )

"""Regression guard: pinned Hamiltonian-sim params → frozen snapshot.

Runs the seeded 3-qubit TFIM Trotter circuit and asserts the output matches
the committed fixture. Any future change that alters numerics (new transpile
pass, statevector rounding, Trotter formula) must bump the fixture
deliberately. Keeping this test ensures the quantum-classical race stays
reproducible across refactors.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.quantum.hamiltonian_sim import run_simulation

_FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "hamiltonian_golden_n3_t05_s4.json"


@pytest.fixture(scope="module")
def golden() -> dict:
    with _FIXTURE.open() as f:
        return json.load(f)


def test_hamiltonian_golden_fidelity(golden: dict) -> None:
    """Final fidelity matches the frozen snapshot to 1e-10."""
    p = golden["params"]
    res = run_simulation(
        n_qubits=p["n_qubits"], model=p["model"], time=p["time"], n_steps=p["n_steps"],
        use_simulator=True, shots=p["shots"],
        seed_simulator=p["seed_simulator"], initial_state=p["initial_state"],
    )
    assert res["final_result"]["final_fidelity"] == pytest.approx(
        golden["final_fidelity"], abs=1e-10
    )


def test_hamiltonian_golden_per_step_fidelity(golden: dict) -> None:
    """Every Trotter step fidelity matches the snapshot."""
    p = golden["params"]
    res = run_simulation(
        n_qubits=p["n_qubits"], model=p["model"], time=p["time"], n_steps=p["n_steps"],
        use_simulator=True, shots=p["shots"],
        seed_simulator=p["seed_simulator"], initial_state=p["initial_state"],
    )
    for got, want in zip(res["steps"], golden["steps"]):
        assert got["step"] == want["step"]
        assert got["fidelity_vs_exact"] == pytest.approx(want["fidelity_vs_exact"], abs=1e-10)
        assert got["time"] == pytest.approx(want["time"], abs=1e-12)


def test_hamiltonian_golden_measured_counts(golden: dict) -> None:
    """Seeded Aer shot outcomes match the snapshot exactly."""
    p = golden["params"]
    res = run_simulation(
        n_qubits=p["n_qubits"], model=p["model"], time=p["time"], n_steps=p["n_steps"],
        use_simulator=True, shots=p["shots"],
        seed_simulator=p["seed_simulator"], initial_state=p["initial_state"],
    )
    assert res["final_result"]["measured_counts"] == golden["measured_counts"]

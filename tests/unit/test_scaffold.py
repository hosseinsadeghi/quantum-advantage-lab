"""Smoke test that the repo imports cleanly under the test harness."""

from __future__ import annotations


def test_modules_registry_loads():
    from backend.modules import MODULES

    assert set(MODULES) == {
        "grovers_search",
        "quantum_walks",
        "vqe",
        "hamiltonian_sim",
    }


def test_provider_aer_backend_available():
    from backend.quantum.provider import get_backend

    backend, run_kwargs, info = get_backend(use_simulator=True)
    assert backend is not None
    assert run_kwargs == {}
    assert info["actual"] == "aer"
    assert info["fell_back"] is False
    assert "simulator" in backend.name.lower() or "basic" in backend.name.lower()

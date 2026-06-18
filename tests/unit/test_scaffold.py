"""Smoke test that the repo imports cleanly under the test harness."""

from __future__ import annotations

import pytest


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


def test_provider_qpu_without_key_raises(monkeypatch):
    from backend.quantum.provider import QPUUnavailableError, get_backend

    monkeypatch.delenv("IONQ_API_KEY", raising=False)
    monkeypatch.delenv("IONQ_API_KEY_QAL", raising=False)

    with pytest.raises(QPUUnavailableError, match="qpu\\.forte-1.*no IonQ API key is active"):
        get_backend(use_simulator=False, use_qpu=True, qpu_name="qpu.forte-1")


def test_provider_qpu_init_failure_raises(monkeypatch):
    from backend.quantum import provider

    monkeypatch.setenv("IONQ_API_KEY", "fake")
    monkeypatch.setattr(
        provider,
        "get_qpu_availability",
        lambda: {
            "qpu.forte-1": {
                "available": True,
                "status": "available",
                "has_access": True,
                "reason": "",
            },
        },
    )
    monkeypatch.setattr(provider, "_get_ionq_backend", lambda _name: None)

    with pytest.raises(provider.QPUUnavailableError, match="qpu\\.forte-1.*could not be initialised"):
        provider.get_backend(use_simulator=False, use_qpu=True, qpu_name="qpu.forte-1")


def test_provider_qpu_submission_disabled_deploy_raises(monkeypatch):
    from backend.quantum.provider import QPUUnavailableError, get_backend

    monkeypatch.setenv("IONQ_API_KEY", "fake")
    monkeypatch.setenv("DISABLE_QPU_SUBMISSION", "true")

    with pytest.raises(QPUUnavailableError, match="qpu\\.forte-1.*new QPU submissions are disabled"):
        get_backend(use_simulator=False, use_qpu=True, qpu_name="qpu.forte-1")


def test_provider_legacy_disable_qpu_alias_still_raises(monkeypatch):
    from backend.quantum.provider import QPUUnavailableError, get_backend

    monkeypatch.setenv("IONQ_API_KEY", "fake")
    monkeypatch.setenv("DISABLE_QPU", "true")

    with pytest.raises(QPUUnavailableError, match="qpu\\.forte-1.*new QPU submissions are disabled"):
        get_backend(use_simulator=False, use_qpu=True, qpu_name="qpu.forte-1")

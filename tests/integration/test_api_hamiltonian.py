"""FastAPI integration tests for the Hamiltonian-sim race module.

Uses the in-process ``TestClient`` / ``WebSocketTestSession`` from FastAPI —
no external services required. These run as part of the default test pass.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_health(client: TestClient) -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_modules_listing_contains_hamiltonian_sim(client: TestClient) -> None:
    r = client.get("/api/modules")
    assert r.status_code == 200
    ids = {m["id"] for m in r.json()}
    assert "hamiltonian_sim" in ids


def test_module_defaults_shape(client: TestClient) -> None:
    r = client.get("/api/modules/hamiltonian_sim/defaults")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "hamiltonian_sim"
    assert "default_params" in body
    dp = body["default_params"]
    for key in ("n_qubits", "model", "time", "n_steps", "use_simulator", "use_qpu", "noise_model"):
        assert key in dp, f"missing default_params key {key!r}"


def test_post_race_hamiltonian_small(client: TestClient) -> None:
    r = client.post(
        "/api/race/hamiltonian_sim",
        json={"params": {"n_qubits": 3, "n_steps": 4, "time": 0.5}},
    )
    assert r.status_code == 200
    body = r.json()

    assert body["module"] == "hamiltonian_sim"
    assert len(body["quantum"]["steps"]) == 4
    assert len(body["classical"]["steps"]) == 4

    final_fidelity = body["quantum"]["result"].get("final_fidelity")
    assert final_fidelity is not None and final_fidelity > 0.9, (
        f"final_fidelity below threshold: {final_fidelity}"
    )


def test_post_race_forwards_model_kwargs(client: TestClient) -> None:
    """Passing J/h through the API must affect the quantum result."""
    base = client.post(
        "/api/race/hamiltonian_sim",
        json={"params": {"n_qubits": 2, "n_steps": 2, "time": 0.5}},
    ).json()
    strong = client.post(
        "/api/race/hamiltonian_sim",
        json={"params": {"n_qubits": 2, "n_steps": 2, "time": 0.5, "J": 3.0, "h": 2.0}},
    ).json()

    f1 = base["quantum"]["result"]["final_fidelity"]
    f2 = strong["quantum"]["result"]["final_fidelity"]
    assert abs(f1 - f2) > 1e-6, "J/h did not propagate from API params to the Hamiltonian"


def test_connectivity_analysis_endpoint_shape(client: TestClient) -> None:
    r = client.post(
        "/api/analysis/hamiltonian/connectivity",
        json={"params": {"n_qubits": 4, "time": 0.5, "n_steps": 2, "interaction_pattern": "chain"}},
    )
    assert r.status_code == 200
    body = r.json()

    assert set(body) >= {"logical", "ionq", "heavy_hex", "metrics", "interaction_graph"}
    assert body["interaction_graph"]["pattern"] == "chain"
    for key in ("logical", "ionq", "heavy_hex"):
        assert "depth" in body[key]
        assert "two_qubit_gates" in body[key]
        assert "swap_count" in body[key]


def test_connectivity_chain_has_low_swap_overhead(client: TestClient) -> None:
    r = client.post(
        "/api/analysis/hamiltonian/connectivity",
        json={"params": {"n_qubits": 3, "time": 0.5, "n_steps": 2, "interaction_pattern": "chain"}},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["heavy_hex"]["swap_count"] == 0
    assert body["metrics"]["swap_tax_avoided"] == 0


def test_connectivity_all_to_all_shows_more_routing_pressure_on_heavy_hex(client: TestClient) -> None:
    r = client.post(
        "/api/analysis/hamiltonian/connectivity",
        json={"params": {"n_qubits": 5, "time": 0.5, "n_steps": 2, "interaction_pattern": "all_to_all"}},
    )
    assert r.status_code == 200
    body = r.json()

    heavy_hex = body["heavy_hex"]
    ionq = body["ionq"]
    assert (
        heavy_hex["depth"] > ionq["depth"]
        or heavy_hex["two_qubit_gates"] > ionq["two_qubit_gates"]
        or heavy_hex["swap_count"] > ionq["swap_count"]
    )


def test_post_race_unknown_module_404(client: TestClient) -> None:
    r = client.post("/api/race/does_not_exist", json={"params": {}})
    assert r.status_code == 404


def test_post_race_qpu_failure_returns_error(client: TestClient, monkeypatch) -> None:
    monkeypatch.delenv("IONQ_API_KEY", raising=False)
    monkeypatch.delenv("IONQ_API_KEY_QAL", raising=False)

    r = client.post(
        "/api/race/hamiltonian_sim",
        json={"params": {"n_qubits": 2, "n_steps": 2, "time": 0.5, "use_simulator": False, "use_qpu": True}},
    )
    assert r.status_code == 503
    assert "qpu.forte-1" in r.json()["detail"]
    assert "no IonQ API key is active" in r.json()["detail"]


def test_ws_race_hamiltonian_streams_steps(client: TestClient) -> None:
    """WS stream must yield step messages and a final complete message whose
    ``steps_count`` matches the number of step messages actually seen."""
    with client.websocket_connect("/ws/race/hamiltonian_sim") as ws:
        ws.send_text(json.dumps({"n_qubits": 3, "n_steps": 3, "time": 0.5}))

        q_steps: list[dict] = []
        c_steps: list[dict] = []
        complete: dict | None = None

        while True:
            msg = ws.receive_json()
            t = msg["type"]
            if t == "quantum_step":
                q_steps.append(msg["data"])
            elif t == "classical_step":
                c_steps.append(msg["data"])
            elif t == "complete":
                complete = msg["data"]
                break
            elif t == "error":
                pytest.fail(f"server reported error: {msg['data']}")

        assert complete is not None
        assert len(q_steps) == complete["quantum"]["steps_count"]
        assert len(c_steps) == complete["classical"]["steps_count"]
        assert complete["quantum"]["final_result"]["final_fidelity"] > 0.9


def test_ws_race_qpu_failure_sends_error_not_complete(client: TestClient, monkeypatch) -> None:
    monkeypatch.delenv("IONQ_API_KEY", raising=False)
    monkeypatch.delenv("IONQ_API_KEY_QAL", raising=False)

    with client.websocket_connect("/ws/race/hamiltonian_sim") as ws:
        ws.send_text(json.dumps({
            "n_qubits": 2,
            "n_steps": 2,
            "time": 0.5,
            "use_simulator": False,
            "use_qpu": True,
        }))
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "qpu.forte-1" in msg["data"]["message"]
        assert "no IonQ API key is active" in msg["data"]["message"]


def test_ws_race_unknown_module_closes(client: TestClient) -> None:
    """Connecting to an unknown module must close (4004) before any message."""
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect("/ws/race/does_not_exist") as ws:
            ws.receive_text()
    assert exc_info.value.code == 4004

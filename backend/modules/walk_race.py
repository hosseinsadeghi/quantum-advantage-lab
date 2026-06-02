"""Quantum Walk vs Classical Random Walk race module."""

from __future__ import annotations

from typing import Any

from backend.modules.base import RaceModule
from backend.quantum.quantum_walks import run_quantum_walk
from backend.classical.random_walk import run_classical_walk


class WalkRace(RaceModule):
    module_id = "quantum_walks"
    title = "Quantum Walk vs Classical Random Walk"
    description = (
        "Compare a quantum walk on a graph against a classical random walk. "
        "Quantum walks spread quadratically faster and can be used to "
        "accelerate graph-based algorithms."
    )
    default_params: dict[str, Any] = {
        "n_qubits": 4,
        "n_steps": 10,
        "graph_type": "cycle",
        "n_trials": 1000,
        "use_simulator": True,
        "use_qpu": False,
        "noise_model": "forte-1",
    }

    def run_quantum(self, params: dict[str, Any]) -> dict[str, Any]:
        return run_quantum_walk(
            n_qubits=params.get("n_qubits", 4),
            n_steps=params.get("n_steps", 10),
            graph_type=params.get("graph_type", "cycle"),
            use_simulator=params.get("use_simulator", True),
            use_qpu=params.get("use_qpu", False),
            noise_model=params.get("noise_model", "forte-1"),
            qpu_name=params.get("qpu_name", "qpu.forte-1"),
        )

    def run_classical(self, params: dict[str, Any]) -> dict[str, Any]:
        n_qubits = params.get("n_qubits", 4)
        n_nodes = 2 ** n_qubits
        return run_classical_walk(
            n_nodes=n_nodes,
            n_steps=params.get("n_steps", 10),
            graph_type=params.get("graph_type", "cycle"),
            n_trials=params.get("n_trials", 1000),
        )

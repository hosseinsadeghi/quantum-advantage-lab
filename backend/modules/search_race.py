"""Grover's Search vs Linear Search race module."""

from __future__ import annotations

from typing import Any

from backend.modules.base import RaceModule
from backend.quantum.grovers import run_grovers
from backend.quantum.provider import resolve_shots
from backend.classical.linear_search import run_linear_search


class SearchRace(RaceModule):
    module_id = "grovers_search"
    title = "Grover's Search vs Linear Search"
    description = (
        "Compare Grover's quantum search algorithm (O(sqrt(N))) "
        "against classical linear search (O(N)). Grover's achieves a "
        "provable quadratic speedup for unstructured search."
    )
    default_params: dict[str, Any] = {
        "n_qubits": 4,
        "target_state": 7,
        "use_simulator": True,
        "use_qpu": False,
        "noise_model": "forte-1",
    }

    def run_quantum(self, params: dict[str, Any]) -> dict[str, Any]:
        return run_grovers(
            n_qubits=params.get("n_qubits", 4),
            target_state=params.get("target_state", 7),
            use_simulator=params.get("use_simulator", True),
            use_qpu=params.get("use_qpu", False),
            noise_model=params.get("noise_model", "forte-1"),
            qpu_name=params.get("qpu_name", "qpu.forte-1"),
            shots=resolve_shots(params),
        )

    def run_classical(self, params: dict[str, Any]) -> dict[str, Any]:
        n_qubits = params.get("n_qubits", 4)
        n_items = 2 ** n_qubits
        target = params.get("target_state", 7)
        return run_linear_search(n_items=n_items, target=target)

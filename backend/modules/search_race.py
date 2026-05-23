"""Grover's Search vs Linear Search race module — placeholder.

Registers the race so it appears in the module listing, but both solvers
raise ``NotImplementedError`` until the underlying algorithms are filled in.
"""

from __future__ import annotations

from typing import Any

from backend.modules.base import RaceModule
from backend.quantum.grovers import run_grovers
from backend.classical.linear_search import run_linear_search


class SearchRace(RaceModule):
    module_id = "grovers_search"
    title = "Grover's Search vs Linear Search"
    description = (
        "Compare Grover's quantum search algorithm (O(sqrt(N))) against "
        "classical linear search (O(N)). Coming soon."
    )
    default_params: dict[str, Any] = {
        "n_qubits": 4,
        "target_state": 7,
        "use_simulator": True,
        "use_qpu": False,
        "noise_model": "forte-1",
    }

    def run_quantum(self, params: dict[str, Any]) -> dict[str, Any]:
        return run_grovers(**params)

    def run_classical(self, params: dict[str, Any]) -> dict[str, Any]:
        return run_linear_search(**params)

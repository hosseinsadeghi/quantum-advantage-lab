"""Quantum Walk vs Random Walk race module — placeholder.

Registers the race so it appears in the module listing, but both solvers
raise ``NotImplementedError`` until the underlying algorithms are filled in.
"""

from __future__ import annotations

from typing import Any

from backend.modules.base import RaceModule
from backend.quantum.quantum_walks import run_quantum_walk
from backend.classical.random_walk import run_classical_walk


class WalkRace(RaceModule):
    module_id = "quantum_walks"
    title = "Quantum Walk vs Random Walk"
    description = (
        "Compare a discrete-time coined quantum walk against a Monte Carlo "
        "classical random walk. Coming soon."
    )
    default_params: dict[str, Any] = {
        "n_qubits": 4,
        "n_steps": 10,
        "graph_type": "cycle",
        "use_simulator": True,
        "use_qpu": False,
        "noise_model": "forte-1",
    }

    def run_quantum(self, params: dict[str, Any]) -> dict[str, Any]:
        return run_quantum_walk(**params)

    def run_classical(self, params: dict[str, Any]) -> dict[str, Any]:
        return run_classical_walk(**params)
